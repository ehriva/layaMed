package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import com.convaiinnovations.laya.sequence.SequenceBuilder;
import com.convaiinnovations.laya.sequence.WindowPlan;
import com.convaiinnovations.laya.tokenizer.Tokenizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/**
 * {@code predictLong} against {@code fixtures/predict_long.json}, recorded from
 * {@code ONNXAgent.predict_long}.
 *
 * <p>The model is not what is under test here, and could not be: the aggregation rules are chosen
 * by values a case has to control. A tie between two windows has to be constructed, the deciding
 * window has to be put somewhere specific, and {@code noul} is chosen on P(true) while a choice is
 * chosen on confidence -- a DIFFERENT field, which only shows when the two disagree. So the
 * generator stubs inference, drives the real {@code predict_long} over it, and records both what
 * it handed to inference and what it made of the answers.
 *
 * <p>That splits cleanly in three, and each part is checked against the reference separately:
 *
 * <ul>
 *   <li>the SPLIT -- the window texts the reference handed to inference, which pins the state
 *       tokenization, the window offsets and the per-window decode together;</li>
 *   <li>the AGGREGATION -- {@link Agent#aggregate}, driven with the recorded per-window results;
 *   </li>
 *   <li>the whole path end to end, over a stub session, which is the only one of the three that
 *       can show the pieces are wired to each other.</li>
 * </ul>
 */
final class PredictLongTest {

    @SuppressWarnings("unchecked")
    private static Map<String, Object> family() {
        return (Map<String, Object>) (Map<?, ?>) Fixtures.load("predict_long.json");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> forCheckpoint(String name) {
        Map<String, Object> by = (Map<String, Object>) family().get("by");
        Map<String, Object> entry = (Map<String, Object>) by.get(name);
        Assumptions.assumeTrue(entry != null && !entry.containsKey("skipped"),
                "the " + name + " checkpoint was not present when the fixture was recorded");
        return entry;
    }

    private static Tokenizer open(String name) {
        java.nio.file.Path model = Fixtures.checkpoint(name);
        Assumptions.assumeTrue(model != null, Fixtures.missingCheckpoint(name));
        try {
            return Tokenizer.fromModelDirectory(model);
        } catch (java.io.IOException problem) {
            throw new IllegalStateException("cannot open the " + name + " tokenizer", problem);
        }
    }

    // ------------------------------------------------------------------ the split

    @TestFactory
    @DisplayName("the scan hands inference the windows the reference handed it")
    @SuppressWarnings("unchecked")
    List<DynamicTest> splitMatchesTheReference() {
        List<DynamicTest> tests = new ArrayList<>();
        for (String name : List.of("english", "multilingual")) {
            Map<String, Object> entry = forCheckpoint(name);
            Map<String, Object> states = (Map<String, Object>) entry.get("states");
            Map<String, Object> splits = (Map<String, Object>) entry.get("splits");
            List<Object> cases = (List<Object>) entry.get("cases");
            int maxLen = number(entry.get("max_len"));
            int headMaxLen = number(entry.get("head_max_len"));
            Tokenizer tok = open(name);
            for (Object raw : cases) {
                Map<String, Object> one = (Map<String, Object>) raw;
                if (one.get("split_key") == null) {
                    continue;                      // answered in one call; nothing was scanned
                }
                tests.add(dynamicTest(name + " " + one.get("case"), () -> {
                    String state = (String) states.get(one.get("state_key"));
                    List<Object> want = (List<Object>) splits.get(one.get("split_key"));
                    assertNotNull(want, "no recorded split for " + one.get("split_key"));

                    List<Question> questions = questionsOf(one);
                    WindowPlan.Budget plan = WindowPlan.budget(tok, questions,
                            maxLen, headMaxLen,
                            optional(one.get("window")), optional(one.get("stride")));
                    // The budget the reference computed, before anything downstream of it is
                    // compared: a split that matched with a different budget would be a
                    // coincidence, not parity.
                    assertEquals(number(one.get("budget")), plan.window(), "effective window");

                    int[] ids = tok.encode(SequenceBuilder.serializeState(state)
                            .replace(tok.maskToken(), " "));
                    assertEquals(number(one.get("state_tokens")), ids.length, "state tokens");

                    List<String> got = new ArrayList<>();
                    List<Integer> starts = new ArrayList<>();
                    for (int at = 0; at < ids.length; at += plan.stride()) {
                        int end = Math.min(at + plan.window(), ids.length);
                        got.add(tok.decode(Arrays.copyOfRange(ids, at, end)));
                        starts.add(at);
                        if (end >= ids.length) {
                            break;
                        }
                    }
                    assertEquals(want.size(), got.size(), "window count");
                    for (int at = 0; at < want.size(); at++) {
                        assertEquals(want.get(at), got.get(at), "window " + at);
                    }
                    assertEquals(ints(one.get("starts")), starts, "window starts");
                }));
            }
        }
        assertTrue(tests.size() >= 20,
                () -> "both checkpoints' scanned cases, got " + tests.size());
        return tests;
    }

    // ------------------------------------------------------------------ the aggregation

    @TestFactory
    @DisplayName("every aggregation rule matches, including the ties")
    @SuppressWarnings("unchecked")
    List<DynamicTest> aggregationMatchesTheReference() {
        List<DynamicTest> tests = new ArrayList<>();
        int withTies = 0;
        for (String name : List.of("english", "multilingual")) {
            Map<String, Object> entry = forCheckpoint(name);
            for (Object raw : (List<Object>) entry.get("cases")) {
                Map<String, Object> one = (Map<String, Object>) raw;
                if (one.get("per_window") == null) {
                    continue;                      // answered in one call
                }
                if (((String) one.get("case")).contains("tie")) {
                    withTies++;
                }
                tests.add(dynamicTest(name + " " + one.get("case"), () -> {
                    List<Object> perWindow = (List<Object>) one.get("per_window");
                    List<Prediction> results = new ArrayList<>();
                    for (Object window : perWindow) {
                        results.add(predictionOf((Map<String, Object>) window));
                    }
                    Map<String, Object> questions =
                            (Map<String, Object>) one.get("questions");

                    LongPrediction got = Agent.aggregate(List.copyOf(questions.keySet()), results,
                            ints(one.get("starts")), number(one.get("budget")),
                            number(one.get("state_tokens")));

                    Map<String, Object> want =
                            (Map<String, Object>) one.get("result");
                    Map<String, Object> wantAnswers = (Map<String, Object>) want.get("answers");
                    assertEquals(wantAnswers.keySet(), got.answers().keySet(), "question ids");

                    for (Map.Entry<String, Object> expected : wantAnswers.entrySet()) {
                        String id = expected.getKey();
                        Map<String, Object> a = (Map<String, Object>) expected.getValue();
                        Answer answer = got.answer(id);
                        assertEquals(a.get("type"), answer.type(), id + " type");
                        assertEquals(dbl(a.get("answer_confidence")), answer.answerConfidence(),
                                1e-12, id + " answer_confidence");
                        assertEquals(dbl(a.get("confidence")), answer.confidence(), 1e-12,
                                id + " confidence");
                        if (answer instanceof Answer.Noul noul) {
                            assertEquals(dbl(a.get("noul")), noul.noul(), 1e-12, id + " noul");
                        } else if (answer instanceof Answer.Choice choice) {
                            assertEquals(a.get("choice"), choice.choice(), id + " choice");
                        } else if (answer instanceof Answer.Score score) {
                            assertEquals(dbl(a.get("score")), score.score(), 1e-12, id + " score");
                        }

                        Map<String, Object> w = (Map<String, Object>) a.get("window");
                        LongPrediction.Window window = got.window(id).orElseThrow();
                        assertEquals(number(w.get("index")), window.index(), id + " window index");
                        assertEquals(number(w.get("token_start")), window.tokenStart(),
                                id + " token_start");
                        assertEquals(number(w.get("token_end")), window.tokenEnd(),
                                id + " token_end");
                        assertEquals(number(w.get("count")), window.count(), id + " count");
                    }

                    // Usage: the fields this port models. The reference's usage dict is open and
                    // carries anything `predict_batch` put there -- its `backend` key is a string
                    // the last window wins -- while `LongUsage` is a closed record, so an unknown
                    // field has nowhere to go and is deliberately not carried.
                    Map<String, Object> wantUsage = (Map<String, Object>) want.get("usage");
                    LongPrediction.LongUsage usage = got.usage();
                    assertEquals(number(wantUsage.get("input_tokens")), usage.inputTokens(),
                            "input_tokens summed");
                    assertEquals(0, usage.outputTokens(), "output_tokens zeroed for a scan");
                    assertEquals(number(wantUsage.get("state_tokens")), usage.stateTokens(),
                            "state_tokens summed");
                    assertEquals(number(wantUsage.get("state_tokens_dropped")),
                            usage.stateTokensDropped(), "state_tokens_dropped summed");
                    // A COUNT of windows that truncated, not a flag -- which is why it can be
                    // above zero while truncatedQuestions is empty.
                    assertEquals(number(wantUsage.get("truncated")), usage.truncatedWindows(),
                            "truncated is a window count");
                    assertEquals(wantUsage.get("truncated_questions"), usage.truncatedQuestions(),
                            "truncated_questions is the LAST window's list");
                    assertEquals(number(wantUsage.get("windows")), usage.windows(), "windows");

                    Map<String, Object> wantOptions =
                            (Map<String, Object>) wantUsage.get("options");
                    if (wantOptions != null) {
                        // MERGED across windows, not replaced: it is set only on the windows where
                        // spans actually collapsed, and the deciding window is the most confident
                        // one, not the last one.
                        assertEquals(wantOptions.keySet(), usage.collapsedOptions().keySet(),
                                "collapsed options merged across windows");
                    }
                }));
            }
        }
        final int ties = withTies;
        assertTrue(ties >= 4,
                () -> "the tie cases are the ones a port can silently get backwards and there are "
                        + ties);
        return tests;
    }

    // ------------------------------------------------------------------ the whole path

    @Test
    @DisplayName("a state that fits one window is answered without windowing")
    void singleWindowPath(@org.junit.jupiter.api.io.TempDir java.nio.file.Path root)
            throws Exception {
        TinyCheckpoint.write(root, 64, 24);
        TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
        try (Agent agent = TinyCheckpoint.agent(root, session)) {
            Map<String, Question> questions = Map.of(
                    "holds", Question.noul("Does it hold?"));
            LongPrediction got = agent.predictLong("short", questions);
            assertEquals(1, got.usage().windows(), "one window, the model read all of it");
            assertTrue(got.window("holds").isEmpty(),
                    "no window decided, because there was only one");
            assertNotNull(got.answer("holds"), "and it is still answered");
            assertEquals(1, session.batches.size(), "one graph call");
        }
    }

    @Test
    @DisplayName("a long state is scanned, and every window span lands inside the state")
    void multiWindowPath(@org.junit.jupiter.api.io.TempDir java.nio.file.Path root)
            throws Exception {
        TinyCheckpoint.write(root, 64, 24);
        TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
        try (Agent agent = TinyCheckpoint.agent(root, session)) {
            Map<String, Question> questions = new LinkedHashMap<>();
            questions.put("holds", Question.noul("Does it hold?"));
            questions.put("intent", Question.choice("What now?",
                    TinyCheckpoint.ordered("refund", "money back", "replace", "a new one")));
            StringBuilder state = new StringBuilder();
            for (int i = 0; i < 200; i++) {
                state.append("clause ").append(i).append(" says something. ");
            }
            LongPrediction got = agent.predictLong(state.toString(), questions);

            assertTrue(got.usage().windows() > 1,
                    () -> "a long state must be scanned, got " + got.usage().windows());

            // The window COUNT, not just "more than one". The scan steps by the STRIDE, and a
            // scan that stepped by the window instead would still produce several windows, still
            // cover the state and still pass every other assertion here -- it would just stop
            // overlapping, so a span sitting across a boundary would be read whole by nothing.
            // Measured: a mutant swapping the step survived all 49 cases before this.
            WindowPlan.Budget plan = WindowPlan.budget(agent.tokenizer(),
                    List.copyOf(questions.values()), agent.config().maxLen(),
                    agent.config().headMaxLen(), null, null);
            int tokens = stateTokens(agent, state.toString());
            int expected = 0;
            for (int at = 0; at < tokens; at += plan.stride()) {
                expected++;
                if (at + plan.window() >= tokens) {
                    break;
                }
            }
            assertEquals(expected, got.usage().windows(),
                    () -> "a scan of " + tokens + " tokens at window " + plan.window()
                            + " stride " + plan.stride());
            assertTrue(plan.stride() < plan.window(),
                    "the default stride overlaps, which is what lets a span near a boundary land "
                    + "whole inside some window");
            assertEquals(questions.keySet(), got.answers().keySet(), "every question answered");
            for (String id : questions.keySet()) {
                LongPrediction.Window window = got.window(id).orElseThrow(
                        () -> new AssertionError(id + " has no deciding window"));
                assertEquals(got.usage().windows(), window.count(), id + " window count");
                assertTrue(window.index() >= 0 && window.index() < window.count(),
                        id + " index inside the scan");
                assertTrue(window.tokens() > 0, id + " span is non-empty");
                // The clamp: no span may name a token past the end of the state.
                assertTrue(window.tokenEnd() <= stateTokens(agent, state.toString()),
                        id + " span ends inside the state");
            }
        }
    }

    @Test
    @DisplayName("aggregate refuses input it cannot aggregate")
    void aggregateRefusesImpossibleInput() {
        assertThrows(IllegalArgumentException.class,
                () -> Agent.aggregate(List.of("q"), List.of(), List.of(), 10, 100),
                "a scan with no windows has nothing to aggregate");
        Prediction one = new Prediction(Prediction.MODEL,
                Map.of("q", new Answer.Noul(0.5, 0.5, 0.5, 0.0)),
                new Usage(1, 0, 1, 0, false, List.of(), Map.of()));
        assertThrows(IllegalArgumentException.class,
                () -> Agent.aggregate(List.of("q"), List.of(one), List.of(0, 7), 10, 100),
                "two starts for one result is a caller error, not something to guess at");
        assertThrows(IllegalArgumentException.class,
                () -> Agent.aggregate(List.of("missing"), List.of(one), List.of(0), 10, 100),
                "a question no window answered cannot be aggregated silently");
    }

    @Test
    @DisplayName("a null state or questions map is refused")
    void nullArguments(@org.junit.jupiter.api.io.TempDir java.nio.file.Path root)
            throws Exception {
        TinyCheckpoint.write(root, 64, 24);
        try (Agent agent = TinyCheckpoint.agent(root, new TinyCheckpoint.RecordingSession())) {
            Map<String, Question> questions = Map.of("holds", Question.noul("Does it hold?"));
            assertThrows(IllegalArgumentException.class,
                    () -> agent.predictLong(null, questions),
                    "a null state would otherwise be answered as a decision about the literal word null");
            assertThrows(IllegalArgumentException.class,
                    () -> agent.predictLong("text", null),
                    "a null questions map is a caller error");
        }
    }

    // ------------------------------------------------------------------ helpers

    private static int stateTokens(Agent agent, String state) {
        return agent.tokenizer().encode(
                SequenceBuilder.serializeState(state)
                        .replace(agent.tokenizer().maskToken(), " ")).length;
    }

    /** The recorded public question specs, as {@link Question} values. */
    @SuppressWarnings("unchecked")
    private static List<Question> questionsOf(Map<String, Object> one) {
        List<Question> out = new ArrayList<>();
        Map<String, Object> specs = (Map<String, Object>) one.get("questions");
        for (Object raw : specs.values()) {
            Map<String, Object> spec = (Map<String, Object>) raw;
            String instructions = (String) spec.get("instructions");
            Object criteria = spec.get("criteria");
            switch ((String) spec.get("type")) {
                case "choice" -> out.add(Question.choice(instructions,
                        (Map<String, Object>) criteria));
                case "score" -> out.add(Question.score(instructions, (List<Object>) criteria));
                default -> out.add(Question.noul(instructions));
            }
        }
        return out;
    }

    /** One recorded window's result, as a {@link Prediction}. */
    @SuppressWarnings("unchecked")
    private static Prediction predictionOf(Map<String, Object> window) {
        Map<String, Answer> answers = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry
                : ((Map<String, Object>) window.get("answers")).entrySet()) {
            answers.put(entry.getKey(), answerOf((Map<String, Object>) entry.getValue()));
        }
        Map<String, Object> usage = (Map<String, Object>) window.get("usage");
        Map<String, Usage.CollapsedOptions> collapsed = new LinkedHashMap<>();
        Map<String, Object> options = (Map<String, Object>) usage.get("options");
        if (options != null) {
            for (Map.Entry<String, Object> entry : options.entrySet()) {
                Map<String, Object> o = (Map<String, Object>) entry.getValue();
                Integer per = o.get("tokens_per_option") == null ? null
                        : number(o.get("tokens_per_option"));
                collapsed.put(entry.getKey(), new Usage.CollapsedOptions(
                        number(o.get("total")), number(o.get("distinct")), per));
            }
        }
        return new Prediction(Prediction.MODEL, answers, new Usage(
                number(usage.get("input_tokens")), number(usage.get("output_tokens")),
                number(usage.get("state_tokens")), number(usage.get("state_tokens_dropped")),
                Boolean.TRUE.equals(usage.get("truncated")),
                (List<String>) (List<?>) usage.get("truncated_questions"), collapsed));
    }

    @SuppressWarnings("unchecked")
    private static Answer answerOf(Map<String, Object> a) {
        double confidence = dbl(a.get("confidence"));
        double answerConfidence = dbl(a.get("answer_confidence"));
        double act = dbl(((Map<String, Object>) a.get("action")).get("act_probability"));
        switch ((String) a.get("type")) {
            case "noul":
                return new Answer.Noul(dbl(a.get("noul")), confidence, answerConfidence, act);
            case "choice": {
                Map<String, Double> probabilities = new LinkedHashMap<>();
                ((Map<String, Object>) a.get("probabilities"))
                        .forEach((k, v) -> probabilities.put(k, dbl(v)));
                return new Answer.Choice((String) a.get("choice"), probabilities,
                        confidence, answerConfidence, act);
            }
            default: {
                Map<String, Double> probabilities = new LinkedHashMap<>();
                ((Map<String, Object>) a.get("probabilities"))
                        .forEach((k, v) -> probabilities.put(k, dbl(v)));
                Map<String, String> legend = new LinkedHashMap<>();
                ((Map<String, Object>) a.get("legend"))
                        .forEach((k, v) -> legend.put(k, (String) v));
                return new Answer.Score(dbl(a.get("score")), legend, probabilities,
                        confidence, answerConfidence, act);
            }
        }
    }

    private static int number(Object value) {
        return ((Number) value).intValue();
    }

    private static double dbl(Object value) {
        return ((Number) value).doubleValue();
    }

    private static Integer optional(Object value) {
        return value == null ? null : ((Number) value).intValue();
    }

    @SuppressWarnings("unchecked")
    private static List<Integer> ints(Object value) {
        List<Integer> out = new ArrayList<>();
        for (Object each : (List<Object>) value) {
            out.add(((Number) each).intValue());
        }
        return out;
    }

    @Test
    @DisplayName("the recorded corpus still covers the branches this family exists for")
    @SuppressWarnings("unchecked")
    void corpusStillCoversTheBranches() {
        for (String name : List.of("english", "multilingual")) {
            Map<String, Object> entry = forCheckpoint(name);
            List<Object> cases = (List<Object>) entry.get("cases");
            assertTrue(cases.size() >= 12, () -> name + " lost cases: " + cases.size());
            int single = 0;
            int capped = 0;
            int multi = 0;
            for (Object raw : cases) {
                Map<String, Object> one = (Map<String, Object>) raw;
                if (Boolean.TRUE.equals(one.get("single_call"))) {
                    single++;
                } else {
                    multi++;
                }
                if (one.get("batch_cap_used") != null) {
                    capped++;
                }
            }
            final int s = single;
            final int c = capped;
            final int m = multi;
            assertTrue(s >= 1, () -> name + " has no single-window case");
            assertTrue(m >= 10, () -> name + " has only " + m + " scanned cases");
            assertTrue(c >= 1, () -> name + " never reaches the batch cap");
        }
        assertFalse(family().isEmpty(), "the family is present");
        assertNull(family().get("skipped"), "and was not skipped wholesale");
    }
}
