package com.convaiinnovations.laya.hooks;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * The hooks and policy overrides for ONE call, appended after whatever the agent already has.
 *
 * <pre>{@code
 * agent.predictBatch(states, questions, null, 0, false,
 *         HookCall.of(Hooks.onPredictStart(ctx -> ctx.maxLen(256)))
 *                 .raiseErrors(false));
 * }</pre>
 *
 * <p>One type rather than five parameters, because the reference's five — {@code hooks},
 * {@code on_predict_start}, {@code on_predict_end}, {@code hooks_raise} and
 * {@code hooks_timeout} — would otherwise be five overloads of every predicting method, and a
 * caller who wants only the third of them should not have to name the other four.
 *
 * <p>{@code raiseErrors} and {@code timeout} are boxed because null means "do not override, use
 * the agent's", which is a third state a {@code boolean} and a bare {@link Duration} default
 * cannot carry.
 */
public record HookCall(List<Hook> hooks, List<Consumer<PredictContext>> onPredictStart,
                       List<Consumer<PredictContext>> onPredictEnd, Boolean raiseErrors,
                       Duration timeout) {

    private static final HookCall NONE =
            new HookCall(List.of(), List.of(), List.of(), null, null);

    public HookCall {
        // normalise, not List.copyOf: both copy and both refuse a null entry, but only one of
        // them says which argument was wrong. `List.copyOf` threw a bare NullPointerException
        // from inside the JDK, so "a hooks entry must not be null" was unreachable from `of` and
        // `andThen` -- the two ways a caller actually builds one of these.
        hooks = Hooks.normalise(hooks, null, null);
        onPredictStart = List.copyOf(onPredictStart);
        onPredictEnd = List.copyOf(onPredictEnd);
        timeout = Hooks.validateTimeout(timeout);
    }

    /** No per-call hooks and no overrides: the agent's own configuration, unchanged. */
    public static HookCall none() {
        return NONE;
    }

    /** One hook, for this call only. */
    public static HookCall of(Hook hook) {
        return NONE.andThen(hook);
    }

    /** Several hooks, for this call only, in the given order. */
    public static HookCall of(List<? extends Hook> hooks) {
        // Handed over unfiltered, so the canonical constructor above is what copies and what
        // refuses a null entry. `List.copyOf` here threw before it could.
        return new HookCall(new ArrayList<>(hooks), List.of(), List.of(), null, null);
    }

    /** This call with one more hook appended. */
    public HookCall andThen(Hook hook) {
        List<Hook> grown = new ArrayList<>(hooks);
        grown.add(hook);
        return new HookCall(grown, onPredictStart, onPredictEnd, raiseErrors, timeout);
    }

    /** This call with one more start callback, which runs after every hook object. */
    public HookCall onStart(Consumer<PredictContext> callback) {
        List<Consumer<PredictContext>> grown = new ArrayList<>(onPredictStart);
        grown.add(callback);
        return new HookCall(hooks, grown, onPredictEnd, raiseErrors, timeout);
    }

    /** This call with one more end callback, which runs after every start callback. */
    public HookCall onEnd(Consumer<PredictContext> callback) {
        List<Consumer<PredictContext>> grown = new ArrayList<>(onPredictEnd);
        grown.add(callback);
        return new HookCall(hooks, onPredictStart, grown, raiseErrors, timeout);
    }

    /** This call overriding the agent's {@code raiseErrors}. */
    public HookCall raiseErrors(boolean value) {
        return new HookCall(hooks, onPredictStart, onPredictEnd, value, timeout);
    }

    /** This call overriding the agent's per-hook timeout. Null means "no override", not "no limit". */
    public HookCall timeout(Duration value) {
        return new HookCall(hooks, onPredictStart, onPredictEnd, raiseErrors, value);
    }

    /** Whether this adds nothing to the agent's own hooks and policy. */
    public boolean isEmpty() {
        return hooks.isEmpty() && onPredictStart.isEmpty() && onPredictEnd.isEmpty()
                && raiseErrors == null && timeout == null;
    }
}
