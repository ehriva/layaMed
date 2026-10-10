package com.convaiinnovations.laya.hooks;

import com.convaiinnovations.laya.Prediction;
import com.convaiinnovations.laya.Predictor;
import com.convaiinnovations.laya.Question;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * One call, as every hook of that call sees it — and as a hook may change it.
 *
 * <p>Mutable, which nothing else in this port is. That is the point: a hook that could only read
 * would be a logger, and the reference's contract is that a start hook may rewrite the states,
 * the questions and the token budget, and that an end hook may rewrite the results. A record
 * cannot express any of that.
 *
 * <p>One context exists per call and is shared by every event of it, so {@link #runId} is what
 * lets a tracer pair a start with its end without threading state of its own. Identity is
 * identity: the reference declares {@code eq=False} so two contexts are never equal and one can
 * sit in a set keyed by the call rather than by its contents, which is what a Java object does
 * by default — so there is no {@code equals} or {@code hashCode} here, on purpose.
 *
 * <p>Not thread-safe, and not meant to be: hooks run in order, on the calling thread or, for an
 * {@link AsyncHook}, on its executor while the caller waits. The exception is a hook that
 * overruns a {@link Hooks.Policy#timeout} or an {@code AsyncHook} deadline, which keeps running
 * after its deadline expires while the call moves on without it — see {@link #states(List)} for
 * what this class does about that and what it cannot do.
 *
 * <p>Two of the reference's fields are absent. {@code router} and {@code decision} carry a
 * {@code RouteDecision} to an {@code on_route} hook. {@link com.convaiinnovations.laya.Router}
 * dispatches {@code on_route} with the chosen checkpoint in {@link #model()} and no states,
 * questions or decision, so a hook here observes the route but cannot replace it, which the
 * reference allows. What the reference splits across {@code agent} and {@code router} is one field
 * here, {@link #predictor}, because both are a {@link Predictor}.
 */
public final class PredictContext {

    private final String runId = UUID.randomUUID().toString().replace("-", "");
    private final long startedAt = System.nanoTime();
    private final String model;
    private final Predictor predictor;

    // Volatile, every one of them. A hook that overran its deadline keeps running on another
    // thread, and after `Thread.join` times out there is no happens-before edge between that
    // thread's writes and the calling thread's reads -- so a plain field could hand the caller a
    // torn view, or a stale one indefinitely. Volatile does not make the call CORRECT under a
    // late writer, which is what `abandoned` below is for; it makes what the reader sees be a
    // value somebody actually wrote.
    private volatile List<Object> states;
    private volatile Map<String, Question> questions;
    private volatile List<Prediction> results;
    private volatile Integer maxLen;
    private volatile Integer headMaxLen;
    private volatile Hooks.Totals usage;
    private volatile Double elapsedMs;
    private volatile Throwable error;

    /**
     * Threads the call stopped waiting for -- on a deadline or an interrupt -- whose writes are
     * therefore refused.
     *
     * <p>Empty unless the call stopped waiting for some hook, and at most one entry each time it
     * did. A set rather than a flag because that can happen more than once in one chain.
     */
    private final Set<Thread> abandoned = ConcurrentHashMap.newKeySet();

    /**
     * A context for one call.
     *
     * <p>Neither {@code states} nor {@code questions} may be null, and each is refused by NAME.
     * A context copies both at construction — it owns what every hook of the call then edits —
     * so a null cannot be carried through to be refused later, the way the reference's plain
     * dataclass carries it to its own {@code "questions must be a dict of question id ->
     * definition"} check. The refusal is the same refusal, moved to where this port can make it.
     *
     * <p>This is a deliberate behaviour change, and it is worth naming because it is small and
     * invisible: before the hooks were wired in, {@code predictBatch(List.of(), null)} returned
     * an empty list, because the empty-state short-circuit ran before anything looked at the
     * questions. It now throws. The alternative — tolerating a null and treating it as "no
     * questions" — would make {@code predictBatch(states, null)} answer a real batch with empty
     * answers and zero usage instead of refusing it, which is the failure mode a typed API
     * exists to prevent. The reference refuses it too, for empty states as much as for any
     * other; returning a list there was this port's own divergence, not a ported behaviour.
     *
     * @param model      the resolved checkpoint name, or null when the caller named none
     * @param predictor  the agent or router answering this call, or null
     * @param maxLen     a per-call token budget, or null for the checkpoint's own
     * @param headMaxLen a per-call head budget, or null for the checkpoint's own
     * @throws IllegalArgumentException for a null {@code states} or {@code questions}
     */
    public PredictContext(List<?> states, Map<String, Question> questions, String model,
                          Predictor predictor, Integer maxLen, Integer headMaxLen) {
        if (states == null) {
            throw new IllegalArgumentException("states must not be null");
        }
        if (questions == null) {
            throw new IllegalArgumentException("questions must not be null");
        }
        this.states = new ArrayList<>(states);
        this.questions = new LinkedHashMap<>(questions);
        this.model = model;
        this.predictor = predictor;
        this.maxLen = maxLen;
        this.headMaxLen = headMaxLen;
    }

    /** A context with no budget overrides. */
    public PredictContext(List<?> states, Map<String, Question> questions, String model,
                          Predictor predictor) {
        this(states, questions, model, predictor, null, null);
    }

    /** Shared by every hook of one call, 32 hex characters, as the reference's {@code uuid4().hex}. */
    public String runId() {
        return runId;
    }

    /** The resolved checkpoint name, or null. */
    public String model() {
        return model;
    }

    /** The agent or router answering this call, or null. */
    public Predictor predictor() {
        return predictor;
    }

    /** The states this call will be answered over. */
    public List<Object> states() {
        return states;
    }

    /**
     * Replaces the states. Inference is handed what the LAST start hook left here.
     *
     * <p>Refused from a thread whose {@link Hooks.Policy#timeout} already expired, and so is
     * every other mutator here. This is a DELIBERATE DIVERGENCE from the reference, which has no
     * equivalent and cannot cheaply get one.
     *
     * <p>Why it exists: a hook that overruns its deadline is abandoned, not stopped — neither
     * runtime can interrupt a thread that will not cooperate — and it is still holding this
     * object while the call carries on without it. Measured on this port before the guard, over
     * 60 calls whose start hook overran a 50&nbsp;ms deadline by 30&nbsp;ms and then assigned
     * results: 60 of 60 late writes were accepted, and {@code usage} then described a different
     * answer from {@code results} in 60 of 60, because {@link Hooks#around} had already totalled
     * the usage of the answer it saw. With the guard, the same 60 calls refuse 60 of 60 and
     * disagree 0 times. The reference has the identical hazard, so this is the port being
     * stricter rather than the port being different for its own sake.
     *
     * <p>What it does NOT fix: a write already PAST this check when the deadline expires still
     * lands. Measured on the same harness with the overrun cut to 1–5&nbsp;ms, so that the write
     * and the deadline collide: 2 to 17 of 60 late writes were still accepted. The window shrinks
     * from "the whole remainder of the call" to "the few instructions between the check and the
     * store", which is a narrowing, not a proof — and the guard is cheap enough to be worth the
     * narrowing on its own.
     *
     * @throws IllegalStateException when the call has already stopped waiting for this thread
     */
    public void states(List<?> replacement) {
        refuseIfAbandoned("states");
        this.states = new ArrayList<>(replacement);
    }

    /** The questions this call will ask. */
    public Map<String, Question> questions() {
        return questions;
    }

    /**
     * Replaces the questions, keeping the given iteration order — a choice's options are positional.
     *
     * @throws IllegalStateException when the call has already stopped waiting for this thread; see
     *     {@link #states(List)}
     */
    public void questions(Map<String, Question> replacement) {
        refuseIfAbandoned("questions");
        this.questions = new LinkedHashMap<>(replacement);
    }

    /**
     * The answers, or null when the model has not run.
     *
     * <p>Null and empty are different states of a call: "inference has not happened" against
     * "inference returned nothing". A start hook reads this to tell whether an earlier hook has
     * already answered, and {@link Hooks#around} reads it to decide whether to run the model at
     * all, so collapsing the two would make an empty batch skip inference forever.
     */
    public List<Prediction> results() {
        return results;
    }

    /**
     * Replaces the results. From an end hook this is what the caller receives.
     *
     * @throws IllegalStateException when the call has already stopped waiting for this thread; see
     *     {@link #states(List)}
     */
    public void results(List<Prediction> replacement) {
        refuseIfAbandoned("results");
        this.results = replacement;
    }

    /** A per-call token budget, or null for the checkpoint's own. */
    public Integer maxLen() {
        return maxLen;
    }

    /**
     * Overrides the token budget for this call.
     *
     * @throws IllegalStateException when the call has already stopped waiting for this thread; see
     *     {@link #states(List)}
     */
    public void maxLen(Integer replacement) {
        refuseIfAbandoned("maxLen");
        this.maxLen = replacement;
    }

    /** A per-call head budget, or null for the checkpoint's own. */
    public Integer headMaxLen() {
        return headMaxLen;
    }

    /**
     * Overrides the head budget for this call.
     *
     * @throws IllegalStateException when the call has already stopped waiting for this thread; see
     *     {@link #states(List)}
     */
    public void headMaxLen(Integer replacement) {
        refuseIfAbandoned("headMaxLen");
        this.headMaxLen = replacement;
    }

    /** The call's totalled usage, set before the end hooks run, or null when it failed. */
    public Hooks.Totals usage() {
        return usage;
    }

    /** Set by {@link Hooks#around} once the results are known. */
    void usage(Hooks.Totals totals) {
        this.usage = totals;
    }

    /** Wall time for the whole call, set before the end hooks run. */
    public Double elapsedMs() {
        return elapsedMs;
    }

    /** {@link System#nanoTime} at construction, for a hook measuring its own slice of the call. */
    public long startedAt() {
        return startedAt;
    }

    /** Set by {@link Hooks#around} once the call is over, failed or not. */
    void elapsedMs(double value) {
        this.elapsedMs = value;
    }

    /** What the call failed with, or null. Set before the error and end hooks run. */
    public Throwable error() {
        return error;
    }

    /** Set by {@link Hooks#around} before the error hooks run. */
    void error(Throwable failure) {
        this.error = failure;
    }

    /**
     * Answers the call from a start hook: inference is skipped and the end hooks still run.
     *
     * <p>{@code results} replaces the WHOLE call, so it carries one entry per state in
     * {@link #states}, in that order — the shape {@code predictBatch} returns. A hook fires once
     * per call and a call can carry many states. The one other accepted shape is a single entry
     * for the whole call.
     *
     * <p>The count is checked HERE, against that contract, so a wrong one fails inside the hook
     * under the caller's own {@link Hooks.Policy#raiseErrors} rather than downstream. In the
     * reference, downstream was where it surfaced: a router indexed {@code results[0]} of an
     * empty list, which the server mapped to a 500, and a batch returned a shorter list than it
     * was given states — quietly dropping rows the caller was about to zip against.
     *
     * <p>Zero results for zero states is accepted, because zero is both "one per state" and
     * "nothing", and an empty call has nothing to answer.
     *
     * @throws IllegalArgumentException when the count is neither 1 nor one per state
     * @throws IllegalStateException when the call has already stopped waiting for this thread; see
     *     {@link #states(List)}
     */
    public void skip(List<Prediction> results) {
        refuseIfAbandoned("skip");
        if (results.size() != 1 && results.size() != states.size()) {
            throw new IllegalArgumentException(String.format(
                    "ctx.skip() takes one result for the whole call or one per state in "
                    + "ctx.states (%d); got %d", states.size(), results.size()));
        }
        this.results = results;
    }

    /**
     * Cuts {@code runner} off from this call: nothing it writes from here on is accepted.
     *
     * <p>Called the moment the call stops waiting for a hook, on its deadline or on an interrupt.
     * See {@link #states(List)}.
     */
    void abandon(Thread runner) {
        abandoned.add(runner);
    }

    /**
     * Undoes one {@link #abandon} once the overrunning task has returned, so a pooled thread that
     * runs this call's next callback is not refused for the previous one's overrun. Only the
     * caller that abandoned a thread may release it.
     */
    void release(Thread runner) {
        abandoned.remove(runner);
    }

    private void refuseIfAbandoned(String what) {
        if (!abandoned.isEmpty() && abandoned.contains(Thread.currentThread())) {
            throw new IllegalStateException(String.format(
                    "laya: hook thread %s was abandoned when the call stopped waiting for it; "
                    + "this call has moved on, so ctx.%s() is refused rather than applied to it",
                    Thread.currentThread().getName(), what));
        }
    }

    @Override
    public String toString() {
        return "PredictContext[runId=" + runId + ", states=" + states.size()
                + ", questions=" + questions.size()
                + ", results=" + (results == null ? "none" : String.valueOf(results.size()))
                + ", model=" + model + "]";
    }
}
