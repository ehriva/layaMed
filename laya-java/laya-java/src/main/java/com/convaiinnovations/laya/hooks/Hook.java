package com.convaiinnovations.laya.hooks;

/**
 * The one place a caller's own code runs inside a prediction.
 *
 * <p>Override only the events you need; the rest are no-ops. A hook may watch a call, rewrite
 * what it is about to ask, or answer it outright with {@link PredictContext#skip}.
 *
 * <pre>{@code
 * final class Tracer implements Hook {
 *     @Override public void onPredictEnd(PredictContext ctx) {
 *         span.record(ctx.usage().inputTokens(), ctx.elapsedMs());
 *     }
 * }
 * agent.hooks().addHook(new Tracer());
 * }</pre>
 *
 * <p>This is ONE type where the reference has two. {@code laya.hooks.Hook} is a structural
 * {@code Protocol} — a hook is anything with at least one of the six methods — and
 * {@code BaseHook} is the concrete no-op class to subclass when you would rather not implement
 * by shape. An interface with default methods is both at once, so there is nothing for a second
 * type to be. The consequence is that the reference's "missing methods are skipped" becomes "the
 * default does nothing", which is the same behaviour from the outside: {@link Hooks#dispatch}
 * calling a default no-op and the reference finding no attribute to call both leave the context
 * untouched and nothing in the log.
 *
 * <p>{@link AsyncHook} runs a hook on an {@link java.util.concurrent.Executor} and waits for it,
 * for a callback that must run on a particular thread. It is not a way to make a slow hook
 * free: the prediction still waits, as the reference's does.
 *
 * <p>A hook runs on the calling thread, on the thread a {@link Hooks.Policy#timeout()} runs it
 * on, or on an {@code AsyncHook}'s executor, and in each case the caller waits for it until a
 * deadline expires -- so until it overruns, it is as thread-safe as the call around it. Install
 * one that is not, and set {@link HookRegistry#concurrent(boolean)} to false to have dispatch
 * serialise it.
 */
public interface Hook {

    /**
     * Before inference, with the states and questions the call was made with.
     *
     * <p>Rewriting {@link PredictContext#states} or {@link PredictContext#questions} changes what
     * is asked; {@link PredictContext#skip} answers the call without the model running at all.
     */
    default void onPredictStart(PredictContext ctx) {
    }

    /**
     * After inference, on the success path and the failure path alike.
     *
     * <p>{@link PredictContext#results} is the answer and may be replaced; on a failure it is
     * null and {@link PredictContext#error} says why.
     */
    default void onPredictEnd(PredictContext ctx) {
    }

    /** After a router has chosen a checkpoint, before it is asked anything. */
    default void onRoute(PredictContext ctx) {
    }

    /** After a checkpoint has been loaded. */
    default void onLoad(PredictContext ctx) {
    }

    /** After a checkpoint has been evicted. */
    default void onEvict(PredictContext ctx) {
    }

    /**
     * When the call failed, before {@link #onPredictEnd}.
     *
     * <p>Throwing from here does not replace the failure that triggered it: see
     * {@link Hooks#around}.
     */
    default void onError(PredictContext ctx) {
    }
}
