package com.convaiinnovations.laya;

import com.convaiinnovations.laya.config.AgentConfig;
import com.convaiinnovations.laya.decode.Decoder;
import com.convaiinnovations.laya.hooks.HookCall;
import com.convaiinnovations.laya.hooks.HookRegistry;
import com.convaiinnovations.laya.hooks.Hooks;
import com.convaiinnovations.laya.hooks.PredictContext;
import com.convaiinnovations.laya.infer.InferenceSession;
import com.convaiinnovations.laya.onnx.LayaSession;
import com.convaiinnovations.laya.sequence.Collator;
import com.convaiinnovations.laya.sequence.SequenceBuilder;
import com.convaiinnovations.laya.sequence.WindowPlan;
import com.convaiinnovations.laya.tokenizer.Tokenizer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The runtime: ask several typed questions about one piece of evidence, in one forward pass.
 *
 * <p>laya is a "System 1" decision model -- a bidirectional encoder with a typed head, not a
 * generator. Every question about the same state becomes one row of a single batch, so four
 * questions about a document cost one batched encode rather than four round trips.
 *
 * <pre>{@code
 * try (Agent agent = Agent.open(modelDir, graphDir)) {
 *     Map<String, Question> questions = new LinkedHashMap<>();
 *     questions.put("intent", Question.choice("What does the customer want?", criteria));
 *     questions.put("urgent", Question.noul("This needs a human today."));
 *     Prediction p = agent.predict(email, questions, "en");
 *     if (p.usage().truncated()) {
 *         // the state did not fit: usage says how much was dropped, and from which questions
 *     }
 * }
 * }</pre>
 *
 * <p>Pass questions in a {@link LinkedHashMap}: a choice's options are positional, so iteration
 * order is part of what is asked.
 *
 * <p>One session is not safe for concurrent {@code predict} calls unless ONNX Runtime is
 * configured for it; hold one {@code Agent} per worker, or serialise access.
 */
public final class Agent implements AutoCloseable, Predictor, BatchPredictor {

    private final Tokenizer tokenizer;
    private final AgentConfig config;
    private final InferenceSession session;
    private final int padId;
    private final String modelName;
    /**
     * This agent's hooks, which a caller may edit at any point in its life.
     *
     * <p>Final and never replaced: a call takes a snapshot of the list, so installing a hook
     * mid-flight cannot disturb one, and handing out the registry rather than copying hook
     * methods onto {@code Agent} keeps six lifecycle names off the surface of a class whose
     * subject is prediction.
     */
    private final HookRegistry hooks = new HookRegistry();
    /**
     * This agent's own lifecycle, not the session's.
     *
     * <p>Tracked here rather than asked of the session: "is this agent usable" is the agent's
     * question, and putting it on {@link InferenceSession} would widen a seam that exists to be
     * narrow -- a stub would have to answer it with something, and a default answer of "open"
     * would be a lie the moment it was not.
     */
    private volatile boolean closed;

    private Agent(Tokenizer tokenizer, AgentConfig config, InferenceSession session, int padId,
                  String modelName) {
        this.tokenizer = tokenizer;
        this.config = config;
        this.session = session;
        this.padId = padId;
        this.modelName = modelName;
    }

    /**
     * Opens a checkpoint.
     *
     * @param modelDirectory holds {@code rl_agent_config.json} and {@code tokenizer/}
     * @param graphDirectory holds {@code laya.onnx}, or {@code encoder.onnx} and {@code head.onnx}
     */
    public static Agent open(Path modelDirectory, Path graphDirectory) throws IOException {
        // 0 means "let ONNX Runtime choose", which is what the Python runtime does. Pinning one
        // thread here ran the forward pass on a single core and cost about 3x.
        return open(modelDirectory, graphDirectory, 0);
    }

    /** Opens a checkpoint with an explicit intra-op thread count. */
    public static Agent open(Path modelDirectory, Path graphDirectory, int threads)
            throws IOException {
        Path name = modelDirectory.getFileName();
        return using(Tokenizer.fromModelDirectory(modelDirectory),
                AgentConfig.fromModelDirectory(modelDirectory),
                LayaSession.open(graphDirectory, threads),
                name == null ? null : name.toString());
    }

    /**
     * Assembles an agent from parts, for a caller that already holds them -- a router sharing one
     * tokenizer across checkpoints, or a test driving the batching and usage accounting through a
     * stub {@link InferenceSession} instead of a 1.2 GB graph.
     */
    public static Agent using(Tokenizer tokenizer, AgentConfig config, InferenceSession session) {
        return using(tokenizer, config, session, null);
    }

    /**
     * Assembles an agent from parts, naming the checkpoint.
     *
     * @param modelName what {@link PredictContext#model()} reports to a hook, or null. A router
     *     serving several checkpoints through one tokenizer is the caller that needs it: without
     *     a name, a hook watching every prediction in a process cannot say which checkpoint
     *     answered
     */
    public static Agent using(Tokenizer tokenizer, AgentConfig config, InferenceSession session,
                              String modelName) {
        // Padding is masked out of attention but still embedded, so it has to be a real id.
        // No silent fallback to the SEP id. The reference hands `tok.pad_token_id` straight to
        // the collator, so a checkpoint naming no pad_token fails the request outright -- and a
        // mis-exported checkpoint that fails loudly on one runtime must not answer quietly on the
        // other. (The graph masks padding, so the substituted id did not itself move the numbers;
        // the divergence was that one runtime accepted a configuration the other rejects.)
        int padId = tokenizer.padId().orElseThrow(() -> new IllegalStateException(
                "this checkpoint names no pad_token, so a batch cannot be padded; the reference "
                + "refuses the same checkpoint"));
        return new Agent(tokenizer, config, session, padId, modelName);
    }

    /**
     * The checkpoint name a hook sees, or null when the agent was assembled without one.
     *
     * <p>{@link #open} takes it from the model directory's own name; that is the only thing a
     * checkpoint on disk is identified by here, since {@code rl_agent_config.json} does not
     * carry one.
     */
    public String modelName() {
        return modelName;
    }

    /**
     * This agent's hooks: install, remove, and the policy for a throwing or slow one.
     *
     * <pre>{@code
     * agent.hooks().addHook(tracer).raiseErrors(false);
     * }</pre>
     */
    public HookRegistry hooks() {
        return hooks;
    }

    /** The checkpoint's tokenizer, for callers that want to measure a state's token cost. */
    public Tokenizer tokenizer() {
        return tokenizer;
    }

    /** The checkpoint's budgets and temperatures. */
    public AgentConfig config() {
        return config;
    }

    /** Asks every question about one state, with no language override. */
    @Override
    public Prediction predict(Object state, Map<String, Question> questions) {
        return predict(state, questions, null);
    }

    /**
     * Asks every question about one state in one forward pass.
     *
     * @param language a tag whose prefix before {@code -} may select a temperature override, or null
     */
    public Prediction predict(Object state, Map<String, Question> questions, String language) {
        // `singletonList`, not `List.of`, so that a null state reaches predictBatch's own check
        // and is refused there with a message rather than by a bare NullPointerException from
        // the list factory. The comment here used to claim a null state was legitimate because
        // "Python serialises None to the text null" -- which is backwards: the reference refuses
        // it for exactly that reason, since answering it would be a confident decision about the
        // four characters "null", byte-identical to passing the string.
        return predictBatch(Collections.singletonList(state), questions, language, 0, false).get(0);
    }

    /**
     * Asks every question about every state, one row per (state, question).
     *
     * <p>Length sorting is off, which is the Python runtime's default too. It never changes an
     * answer -- it only cuts padding -- but leaving it off keeps the batched path's grouping
     * predictable, and a caller who wants the throughput can ask for it.
     */
    @Override
    public List<Prediction> predictBatch(List<?> states, Map<String, Question> questions) {
        return predictBatch(states, questions, null, 0, false);
    }

    /**
     * Asks every question about every state.
     *
     * @param batchSize    rows' worth of states per graph call, or 0 for all of them at once.
     *                     This bounds peak memory, not the answers
     * @param sortByLength group states of similar length into the same call, which cuts padding.
     *                     It changes nothing about the answers: every row is sliced back to its
     *                     own option count and the results are returned in the caller's order
     * @return one prediction per state, in the order the states were given
     */
    public List<Prediction> predictBatch(List<?> states, Map<String, Question> questions,
                                         String language, int batchSize, boolean sortByLength) {
        return predictBatch(states, questions, language, batchSize, sortByLength, HookCall.none());
    }

    /**
     * Asks every question about every state, with hooks around the call.
     *
     * <p>The hooks are this agent's installed ones, after any process-wide defaults, with
     * {@code call}'s own appended last — see {@link Hooks#compose}. A start hook may rewrite the
     * states, the questions or the token budget, or answer the call outright with
     * {@link PredictContext#skip}, in which case the model is never reached. An end hook may
     * replace the results, and what it leaves is what this returns.
     *
     * <p>The checks on the states and the questions happen AFTER the start hooks, deliberately,
     * and that is the reference's order too: only the value that survives the hooks is
     * validated, so a hook that normalises a caller's loose input is allowed to do its job.
     *
     * @param call per-call hooks and policy overrides, or {@link HookCall#none()}
     */
    public List<Prediction> predictBatch(List<?> states, Map<String, Question> questions,
                                         String language, int batchSize, boolean sortByLength,
                                         HookCall call) {
        return predictBatch(states, questions, language, batchSize, sortByLength, call, null,
                null);
    }

    /**
     * {@link #predictBatch(List, Map, String, int, boolean, HookCall)} with the token budgets
     * seeded on the context, so this agent's own start hooks can still replace them. For
     * {@link Router#predictBatch(List)}, which carries a per-request budget.
     */
    List<Prediction> predictBatch(List<?> states, Map<String, Question> questions,
                                  String language, int batchSize, boolean sortByLength,
                                  HookCall call, Integer maxLen, Integer headMaxLen) {
        // Before any hook: a closed agent cannot answer, and dispatching a start hook that then
        // watched the call fail would report a prediction that was never going to happen.
        if (closed) {
            throw new IllegalStateException(
                    "this agent is closed; open a new one rather than reusing it");
        }
        PredictContext ctx = new PredictContext(states, questions, modelName, this, maxLen,
                headMaxLen);
        return Hooks.around(hooks.composeFor(call), ctx, hooks.policyFor(call),
                (hooked, asked, tokens, headTokens) ->
                        infer(hooked, asked, language, batchSize, sortByLength, tokens,
                                headTokens));
    }

    /**
     * The forward passes, over whatever the start hooks left to ask.
     *
     * <p>The reference's {@code _infer_batch}: everything a hook cannot change once it has run.
     * Separate from {@link #predictBatch} so that {@link #predictLong} can reach it without
     * dispatching a hook chain — see that method for why it must not — and so the hook wrapper
     * itself is one shared, tested piece rather than a sequence repeated per entry point.
     *
     * @param maxLen     the per-call token budget a hook set, or null for the checkpoint's
     * @param headMaxLen the per-call head budget a hook set, or null for the checkpoint's
     */
    private List<Prediction> infer(List<?> states, Map<String, Question> questions,
                                   String language, int batchSize, boolean sortByLength,
                                   Integer maxLen, Integer headMaxLen) {
        for (int i = 0; i < states.size(); i++) {
            // Refused, as both Python backends refuse it: `serialize_state(None)` is
            // `json.dumps(None)`, so a missing state would otherwise be answered as a decision
            // about the literal text "null" -- byte-identical to passing the string, and at full
            // confidence. A caller whose state field is absent should hear about it.
            if (states.get(i) == null) {
                throw new IllegalArgumentException(
                        "state at index " + i + " is null; pass a string, a map or a list");
            }
        }
        if (states.isEmpty()) {
            return List.of();
        }
        if (questions.isEmpty()) {
            // Empty answers, zero usage, no tokenization and no forward pass -- which is what
            // the reference returns. Throwing here meant the two runtimes could not be swapped
            // under a question set derived from a filter: an empty schema or a disabled rule set
            // is a well-formed request with a well-formed empty answer.
            List<Prediction> empty = new ArrayList<>(states.size());
            for (int i = 0; i < states.size(); i++) {
                empty.add(new Prediction(Prediction.MODEL, Map.of(),
                        new Usage(0, 0, 0, 0, false, List.of(), Map.of())));
            }
            return List.copyOf(empty);
        }
        // Snapshotted once, at entry. The map is the caller's, and the sequences are built before
        // the graph runs while the answers are labelled after it: reading it twice let a mutation
        // in between attach one question's labels to another question's logits, with no exception
        // and nothing in `usage` to show it. Python snapshots for the same reason.
        Map<String, Question> asked = new LinkedHashMap<>(questions);
        for (Map.Entry<String, Question> entry : asked.entrySet()) {
            if (entry.getValue() == null) {
                throw new IllegalArgumentException("question " + entry.getKey() + " is null");
            }
        }
        int tokenBudget = maxLen == null ? config.maxLen() : maxLen;
        int headBudget = headMaxLen == null ? config.headMaxLen() : headMaxLen;
        List<String> questionIds = new ArrayList<>(asked.keySet());
        int chunk = batchSize > 0 ? batchSize : states.size();
        // Sorting only pays off when it can actually reorder across more than one call, which
        // mirrors the Python runtime's own guard.
        boolean reorder = sortByLength && chunk > 1 && chunk < states.size();
        // Bound the tokenized lookahead independently of the input size: sort within windows of
        // eight calls rather than over the whole input, so a million states do not have to be
        // encoded before the first one runs.
        int window = reorder ? chunk * 8 : chunk;

        Prediction[] out = new Prediction[states.size()];
        for (int start = 0; start < states.size(); start += window) {
            int end = Math.min(states.size(), start + window);
            List<List<SequenceBuilder.Sequence>> encoded = new ArrayList<>(end - start);
            for (int i = start; i < end; i++) {
                encoded.add(encodeState(states.get(i), questionIds, asked, tokenBudget,
                        headBudget));
            }
            List<Integer> order = new ArrayList<>(encoded.size());
            for (int i = 0; i < encoded.size(); i++) {
                order.add(i);
            }
            if (reorder) {
                order.sort((a, b) -> Integer.compare(longestRow(encoded.get(a)),
                        longestRow(encoded.get(b))));
            }
            for (int offset = 0; offset < order.size(); offset += chunk) {
                List<Integer> indices = order.subList(offset, Math.min(order.size(), offset + chunk));
                List<Collator.Item> rows = new ArrayList<>();
                for (int index : indices) {
                    List<SequenceBuilder.Sequence> built = encoded.get(index);
                    for (int q = 0; q < questionIds.size(); q++) {
                        rows.add(new Collator.Item(built.get(q).ids(), built.get(q).markers(),
                                asked.get(questionIds.get(q)).type().code()));
                    }
                }
                InferenceSession.Output output = session.run(Collator.collate(rows, padId));
                int row = 0;
                for (int index : indices) {
                    List<SequenceBuilder.Sequence> built = encoded.get(index);
                    out[start + index] = assemble(built, questionIds, asked, output, row,
                            language);
                    row += questionIds.size();
                }
            }
        }
        return List.of(out);
    }

    /** Scans a state longer than one sequence, with the checkpoint's own window and stride. */
    public LongPrediction predictLong(Object state, Map<String, Question> questions) {
        return predictLong(state, questions, null, null, null, 0);
    }

    /**
     * Evaluates questions over a state longer than the context window.
     *
     * <p>{@link #predict} truncates a state that exceeds {@code max_len} to a single window and
     * drops the rest silently. This tokenizes the state once, splits it into overlapping token
     * windows, decodes each back to text, scores them all through {@link #predictBatch} -- so the
     * windows share graph calls rather than costing one each -- and combines the per-window
     * answers:
     *
     * <ul>
     *   <li>{@code noul}: P(true) is the MAX over windows. The statement holds if any window
     *       supports it.</li>
     *   <li>{@code choice} and {@code score}: the answer from the single most-confident window,
     *       so a localized signal is not out-voted by the many neutral windows a long document is
     *       mostly made of.</li>
     * </ul>
     *
     * <p>Ties go to the EARLIEST window, which is what the reference's {@code max} over a range
     * does. The returned probability and confidence are the deciding window's, not a calibrated
     * number for the document; {@link LongPrediction.Window} says which window that was.
     *
     * <p>A state that already fits one window is answered in one call, with no window
     * attribution and {@code windows == 1}.
     *
     * <p><b>No hook runs here.</b> Not the per-call ones — there is no {@link HookCall}
     * parameter — and not the installed or process-wide ones either: the scan reaches the
     * forward passes through the same private path {@link #predictBatch} wraps, rather than
     * through {@code predictBatch} itself. That is a decision, not an omission.
     *
     * <p>A hook's whole power is to rewrite what is asked, and a scan is SIZED before any hook
     * could run. The reference lets hooks into {@code predict_long} and then spends four pieces
     * of machinery keeping them honest: a probe appended after the caller's own start hooks to
     * record what inference was actually handed, a budget check run after the chain to catch a
     * hook that added options the windows were not sized for, and two separate reporting paths
     * for a hook that answered the document outright and one that rewrote the states it was
     * scanned over. Without those, a hook that adds one option silently re-truncates every
     * window — the tail of each reaches no model while the reported span says it did — and a
     * hook that answers the call leaves {@code windows} claiming a scan that never happened.
     *
     * <p>Running hooks here without that machinery would be worse than not running them, so
     * this does not. Nothing is silently dropped: {@link #predictBatch} takes a {@link HookCall},
     * and a caller who needs hooks over a long state can window it themselves and hand the
     * windows to that.
     *
     * @param window     state tokens per window, or null for the per-question budget. Capped at
     *                   the room the questions leave -- see {@link WindowPlan}
     * @param stride     token step between windows, or null for a 50% overlap of the EFFECTIVE
     *                   window
     * @param batchSize  windows per graph call, or 0 to let {@link WindowPlan#batchCap} bound it
     * @throws IllegalArgumentException for a null state or questions, and for a window or stride
     *     {@link WindowPlan} refuses
     */
    public LongPrediction predictLong(Object state, Map<String, Question> questions,
                                      Integer window, Integer stride, String language,
                                      int batchSize) {
        if (closed) {
            throw new IllegalStateException(
                    "this agent is closed; open a new one rather than reusing it");
        }
        // Refused rather than serialised, as predict and predictBatch refuse it: a missing state
        // would otherwise be answered as a decision about the literal text "null".
        if (state == null) {
            throw new IllegalArgumentException("state must not be null; pass a string, a map or "
                    + "a list");
        }
        if (questions == null) {
            throw new IllegalArgumentException("questions must not be null");
        }

        int maxLen = config.maxLen();
        int headMaxLen = config.headMaxLen();
        WindowPlan.Budget plan = WindowPlan.budget(tokenizer, List.copyOf(questions.values()),
                maxLen, headMaxLen, window, stride);

        // The mask string is replaced, not stripped: a state containing the checkpoint's own mask
        // token would otherwise add markers the head did not ask for, and the option spans
        // downstream are found BY those markers.
        String text = SequenceBuilder.serializeState(state).replace(maskToken(), " ");
        int[] stateIds = tokenizer.encode(text);

        if (stateIds.length <= plan.window()) {
            // Identical to a plain call, with no windowing overhead and no window attribution --
            // nothing decided between windows because there was only one.
            Prediction single = infer(Collections.singletonList(state), questions, language, 0,
                    false, null, null).get(0);
            Map<String, LongPrediction.Windowed> answers = new LinkedHashMap<>();
            for (Map.Entry<String, Answer> entry : single.answers().entrySet()) {
                answers.put(entry.getKey(),
                        new LongPrediction.Windowed(entry.getValue(), null));
            }
            Usage one = single.usage();
            // `one.outputTokens()` carried through, not forced to zero. The MULTI-window path
            // zeroes it because the reference does; the single-window path returns the plain
            // call's usage as it was, and this head emits no tokens anyway -- so the two agree
            // today, and hardcoding it would make them disagree the moment that changed.
            return new LongPrediction(single.model(), answers,
                    new LongPrediction.LongUsage(one.inputTokens(), one.outputTokens(),
                            one.stateTokens(), one.stateTokensDropped(),
                            one.truncated() ? 1 : 0, one.truncatedQuestions(),
                            one.collapsedOptions(), 1));
        }

        List<String> scanned = new ArrayList<>();
        List<Integer> starts = new ArrayList<>();
        for (int at = 0; at < stateIds.length; at += plan.stride()) {
            int end = Math.min(at + plan.window(), stateIds.length);
            // Decoded back to text so predictBatch re-tokenizes each window as an ordinary
            // state; the default 50% overlap absorbs any boundary drift on re-tokenization.
            scanned.add(tokenizer.decode(java.util.Arrays.copyOfRange(stateIds, at, end)));
            starts.add(at);
            if (end >= stateIds.length) {
                break;
            }
        }

        Integer cap = WindowPlan.batchCap(scanned.size(), plan.window(),
                Math.max(64, maxLen - headMaxLen - 8), batchSize > 0 ? batchSize : null);
        List<Prediction> results = infer(scanned, questions, language,
                cap == null ? 0 : cap, false, null, null);
        if (results.size() != scanned.size()) {
            throw new IllegalStateException(String.format(
                    "the state was split into %d windows and the batch returned %d results",
                    scanned.size(), results.size()));
        }

        return aggregate(List.copyOf(questions.keySet()), results, starts, plan.window(),
                stateIds.length);
    }

    /**
     * Combines the per-window answers into one, naming the window that decided each.
     *
     * <p>Package-private and static so a test can drive it with recorded per-window results. The
     * rules it implements -- a tie going to the earliest window, {@code noul} chosen on P(true)
     * while a choice is chosen on confidence -- cannot be exercised through a real graph, because
     * a model cannot be asked to produce a tie or a particular deciding window on demand. Driving
     * it directly is what makes those rules testable at all.
     *
     * @param starts      first token of each window, parallel to {@code results}
     * @param window      the EFFECTIVE window, which is what the reported span is measured in
     * @param stateTokens the tokenized state's length, which clamps the final window's end
     */
    static LongPrediction aggregate(List<String> questionIds, List<Prediction> results,
                                    List<Integer> starts, int window, int stateTokens) {
        if (results.isEmpty()) {
            throw new IllegalArgumentException("a scan with no windows has nothing to aggregate");
        }
        if (starts.size() != results.size()) {
            throw new IllegalArgumentException(String.format(
                    "%d window starts for %d results", starts.size(), results.size()));
        }
        Map<String, LongPrediction.Windowed> answers = new LinkedHashMap<>();
        for (String id : questionIds) {
            int best = 0;
            double bestKey = Double.NEGATIVE_INFINITY;
            for (int at = 0; at < results.size(); at++) {
                Answer candidate = results.get(at).answer(id);
                if (candidate == null) {
                    throw new IllegalArgumentException(
                            "window " + at + " has no answer for question " + id);
                }
                // noul is chosen on P(true) and everything else on the calibrated confidence --
                // a DIFFERENT field, not the same number under two names.
                double key = candidate instanceof Answer.Noul noul
                        ? noul.noul()
                        : candidate.answerConfidence();
                // Strictly greater, so a tie keeps the EARLIEST window, as the reference does.
                if (key > bestKey) {
                    bestKey = key;
                    best = at;
                }
            }
            int start = starts.get(best);
            answers.put(id, new LongPrediction.Windowed(results.get(best).answer(id),
                    new LongPrediction.Window(best, start,
                            Math.min(start + window, stateTokens), results.size())));
        }
        return new LongPrediction(Prediction.MODEL, answers, aggregateUsage(results));
    }

    /**
     * Usage across the windows of one scan.
     *
     * <p>Numeric fields SUM, which makes {@code truncated} a count of windows that truncated
     * rather than a flag. {@code collapsedOptions} is MERGED rather than replaced: it is keyed by
     * question and set only on the windows where option spans actually collapsed, so replacing it
     * left a caller holding whichever collapsing window came last -- and the deciding window is
     * the most confident one, not the last one. {@code truncatedQuestions} is the last window's
     * list, carried as the reference carries it, which is why the count can be above zero while
     * the list is empty.
     */
    private static LongPrediction.LongUsage aggregateUsage(List<Prediction> results) {
        int inputTokens = 0;
        int stateTokens = 0;
        int dropped = 0;
        int truncatedWindows = 0;
        List<String> truncatedQuestions = List.of();
        Map<String, Usage.CollapsedOptions> collapsed = new LinkedHashMap<>();
        for (Prediction result : results) {
            Usage usage = result.usage();
            inputTokens += usage.inputTokens();
            stateTokens += usage.stateTokens();
            dropped += usage.stateTokensDropped();
            truncatedWindows += usage.truncated() ? 1 : 0;
            truncatedQuestions = usage.truncatedQuestions();
            collapsed.putAll(usage.collapsedOptions());
        }
        // Zero, not a sum: this head emits no tokens, and summing zeros across windows would
        // still be zero while implying the field meant something per window.
        return new LongPrediction.LongUsage(inputTokens, 0, stateTokens, dropped,
                truncatedWindows, truncatedQuestions, collapsed, results.size());
    }

    /**
     * Builds every question's sequence for one state, tokenizing the state once.
     *
     * <p>Once, not once per question: that is what {@code build_sequence}'s pre-tokenized state
     * parameter exists for, and a document is usually far longer than the question asked about it.
     *
     * @param maxLen     the budget in force for this call, the checkpoint's unless a hook moved it
     * @param headMaxLen the head budget in force for this call, likewise
     */
    private List<SequenceBuilder.Sequence> encodeState(Object state, List<String> questionIds,
                                                       Map<String, Question> questions,
                                                       int maxLen, int headMaxLen) {
        int[] stateIds = tokenizer.encode(
                SequenceBuilder.serializeState(state).replace(maskToken(), " "), -1);
        List<SequenceBuilder.Sequence> built = new ArrayList<>(questionIds.size());
        for (String id : questionIds) {
            Question question = questions.get(id);
            // truncateLeft for a LIST state, which is the reference's rule
            // (`truncate_left = isinstance(state, list)`). A conversation is serialised
            // newest-last, so cutting from the right throws away the current turn and answers
            // about the opening of the conversation instead. Measured on a 120-turn history: the
            // kept window started at message 0 where the reference started mid-message-95, and
            // the answers moved -- one option's probability by 3.8x -- while `usage` stayed
            // byte-identical, so nothing a caller can read revealed it.
            boolean truncateLeft = state instanceof List;
            SequenceBuilder.Sequence sequence = SequenceBuilder.build(tokenizer, state, question,
                    maxLen, headMaxLen, null, truncateLeft, stateIds);
            int defined = question.renderOptions().size();
            if (sequence.markers().length != defined) {
                // Markers sit at absolute positions and the sequence is then cut to `max_len`, so
                // the ones past it are dropped. Answering anyway returns a distribution over the
                // options that happened to survive, with the rest absent from the answer and
                // unchoosable, while `usage.truncated` blames the state instead. Python refuses the
                // request; so does this, naming both knobs because LOWERING `head_max_len` can fix
                // it while raising it makes it worse.
                throw new IllegalArgumentException(String.format(
                        "question %s: only %d of its %d option markers fit in max_len=%d with "
                        + "head_max_len=%d spent on the question; lower head_max_len, raise "
                        + "max_len, or use fewer options",
                        id, sequence.markers().length, defined, maxLen, headMaxLen));
            }
            built.add(sequence);
        }
        return built;
    }

    private Prediction assemble(List<SequenceBuilder.Sequence> built, List<String> questionIds,
                                Map<String, Question> questions, InferenceSession.Output output,
                                int firstRow, String language) {
        Map<String, Answer> answers = new LinkedHashMap<>();
        int inputTokens = 0;
        int dropped = 0;
        List<String> truncatedQuestions = new ArrayList<>();
        Map<String, Usage.CollapsedOptions> collapsed = new LinkedHashMap<>();
        for (int q = 0; q < questionIds.size(); q++) {
            String id = questionIds.get(q);
            SequenceBuilder.Sequence sequence = built.get(q);
            int row = firstRow + q;
            // The FILTERED marker count: a head that overran `max_len` loses markers, and the
            // answer must be derived over the options that survived into the sequence.
            int optionCount = sequence.markers().length;
            float[] actionProbabilities = Decoder.actionProbabilities(output.actLogits()[row]);
            answers.put(id, Decoder.decode(questions.get(id), output.logits()[row], optionCount,
                    actionProbabilities, null, language, config));

            // Every row carries the state, so the state's tokens are counted once per question --
            // which is what the forward pass actually costs.
            inputTokens += sequence.ids().length;
            dropped = Math.max(dropped, sequence.truncation().stateTokensDropped());
            if (sequence.truncation().truncated()) {
                truncatedQuestions.add(id);
            }
            SequenceBuilder.Stats stats = sequence.stats();
            if (stats.optionsDistinct() < stats.options()) {
                collapsed.put(id, new Usage.CollapsedOptions(stats.options(),
                        stats.optionsDistinct(), stats.tokensPerOption()));
            }
        }
        Usage usage = new Usage(inputTokens, 0, built.get(0).truncation().stateTokens(), dropped,
                dropped > 0, List.copyOf(truncatedQuestions),
                Collections.unmodifiableMap(collapsed));
        return new Prediction(Prediction.MODEL, Collections.unmodifiableMap(answers), usage);
    }

    private static int longestRow(List<SequenceBuilder.Sequence> built) {
        int longest = 0;
        for (SequenceBuilder.Sequence sequence : built) {
            longest = Math.max(longest, sequence.ids().length);
        }
        return longest;
    }

    private String maskToken() {
        String mask = tokenizer.maskToken();
        if (mask == null) {
            throw new IllegalStateException(
                    "this checkpoint's tokenizer_config.json names no mask_token");
        }
        return mask;
    }

    /** Closes the session. Idempotent, as {@code AutoCloseable} requires. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        session.close();
    }
}
