package com.convaiinnovations.laya.hooks;

import static java.util.Objects.requireNonNull;

import com.convaiinnovations.laya.Prediction;
import com.convaiinnovations.laya.Question;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.locks.Lock;
import java.util.function.Consumer;

/**
 * Composing, dispatching and bounding hooks — the parts of {@code laya.hooks} that are not a type.
 *
 * <p>Three things here are contracts rather than implementation details, and each is pinned by
 * {@code fixtures/hooks.json}:
 *
 * <ul>
 *   <li><b>Order.</b> {@link #compose} is process-wide defaults, then the hooks installed on the
 *       agent, then the per-call ones; {@link #dispatch} calls them in that order. A port that
 *       composes the other way round still runs every hook and fails nothing — except that the
 *       tracer a caller installed to watch what their per-call hook did now runs first and sees
 *       nothing.</li>
 *   <li><b>{@link PredictContext#skip} does not stop the chain.</b> It assigns the results; it is
 *       {@link #around} reading them afterwards that skips inference. Every later start hook
 *       still runs, and can overwrite the answer.</li>
 *   <li><b>{@link Policy#raiseErrors}.</b> True fails the request at the first throwing hook and
 *       the rest of the chain never runs. False reports and continues, which is what a telemetry
 *       hook needs — it must not be able to fail a request.</li>
 * </ul>
 */
public final class Hooks {

    private static final System.Logger LOG =
            System.getLogger(Hooks.class.getPackage().getName());

    private static final Object DEFAULTS_MUTEX = new Object();
    private static List<Hook> defaults = List.of();
    private static final ThreadLocal<Boolean> SKIP_DEFAULTS = ThreadLocal.withInitial(() -> false);

    private Hooks() {
    }

    /**
     * The six lifecycle events, in the reference's own order.
     *
     * <p>The order is data, not presentation: {@code HOOK_EVENTS} is what the reference's
     * refusal messages enumerate, and a port that reorders it describes a different API.
     */
    public enum Event {
        /** Before inference. */
        PREDICT_START("on_predict_start"),
        /** After inference, success or failure. */
        PREDICT_END("on_predict_end"),
        /** After a router chose a checkpoint. */
        ROUTE("on_route"),
        /** After a checkpoint was loaded. */
        LOAD("on_load"),
        /** After a checkpoint was evicted. */
        EVICT("on_evict"),
        /** When the call failed, before {@link #PREDICT_END}. */
        ERROR("on_error");

        private final String wireName;

        Event(String wireName) {
            this.wireName = wireName;
        }

        /** The name the reference calls this event, which is what a cross-language caller reads. */
        public String wireName() {
            return wireName;
        }

        void callOn(Hook hook, PredictContext ctx) {
            switch (this) {
                case PREDICT_START -> hook.onPredictStart(ctx);
                case PREDICT_END -> hook.onPredictEnd(ctx);
                case ROUTE -> hook.onRoute(ctx);
                case LOAD -> hook.onLoad(ctx);
                case EVICT -> hook.onEvict(ctx);
                case ERROR -> hook.onError(ctx);
            }
        }
    }

    /**
     * One call's totalled token usage, as a hook sees it.
     *
     * <p>Not {@link com.convaiinnovations.laya.Usage}: that describes ONE state, including what
     * its budget dropped and which question was blamed. This is the sum over the call, which is
     * the only number a hook can report without re-deriving per-state attribution the reference
     * does not give it either.
     *
     * <p>{@code long}, not {@code int}, and that is a DELIBERATE DIVERGENCE from the reference's
     * shape rather than a transcription slip. A {@link com.convaiinnovations.laya.Usage} counts
     * one state in an {@code int}; summing those into an {@code int} wraps silently, and the
     * wrap was measured — three results of 1,000,000,000 input tokens each totalled
     * {@code -1294967296}. The reference cannot produce that number, because a Python
     * {@code int} does not overflow, so a port that reported it would be the only one of the two
     * runtimes lying to a metering hook about what a call cost. A {@code long} holds the sum of
     * {@link Integer#MAX_VALUE} over more than four billion results, which is past the point
     * where the batch itself is the problem.
     */
    public record Totals(long inputTokens, long outputTokens) {
    }

    /**
     * What dispatch does with a hook that throws, one that overruns, and one that is not
     * re-entrant.
     *
     * @param raiseErrors true fails the call at the first throwing hook; false reports through
     *     {@code onFailure} and runs the rest of the chain
     * @param lock        held across each hook call, for hooks that are not safe to run
     *     concurrently, or null to run them unguarded
     * @param timeout     bounds EACH hook call, or null for no limit. A hook that overruns
     *     raises — see {@link #dispatch} for what that does and does not protect
     * @param onFailure   where a swallowed failure is reported when {@code raiseErrors} is
     *     false. Never null; {@link #reporting()} is the default and logs at WARNING
     */
    public record Policy(boolean raiseErrors, Lock lock, Duration timeout,
                         Consumer<String> onFailure) {

        public Policy {
            timeout = validateTimeout(timeout);
            if (onFailure == null) {
                throw new IllegalArgumentException(
                        "a policy needs somewhere to report a swallowed hook failure; pass "
                        + "Hooks.reporting() for the default");
            }
        }

        /** Fail the call on a throwing hook, no lock, no timeout. The reference's defaults. */
        public static Policy raising() {
            return new Policy(true, null, null, reporting());
        }

        /** This policy with {@code raiseErrors} replaced. */
        public Policy raiseErrors(boolean value) {
            return new Policy(value, lock, timeout, onFailure);
        }

        /** This policy with the serialising lock replaced. */
        public Policy lock(Lock value) {
            return new Policy(raiseErrors, value, timeout, onFailure);
        }

        /** This policy with the per-hook timeout replaced. */
        public Policy timeout(Duration value) {
            return new Policy(raiseErrors, lock, value, onFailure);
        }

        /** This policy reporting swallowed failures somewhere else — a test, or a metric. */
        public Policy onFailure(Consumer<String> value) {
            return new Policy(raiseErrors, lock, timeout, value);
        }
    }

    /** The default failure sink: one WARNING line per swallowed hook failure. */
    public static Consumer<String> reporting() {
        return message -> LOG.log(System.Logger.Level.WARNING, message);
    }

    /** What inference {@link #around} runs when no hook has answered the call. */
    @FunctionalInterface
    public interface Inference {

        /**
         * Answers the call.
         *
         * @param states     the states the start hooks left on the context, never empty
         * @param questions  the questions they left, never empty of meaning
         * @param maxLen     a per-call token budget, or null for the checkpoint's own
         * @param headMaxLen a per-call head budget, or null for the checkpoint's own
         */
        List<Prediction> run(List<Object> states, Map<String, Question> questions,
                             Integer maxLen, Integer headMaxLen);
    }

    /**
     * Wraps a hook object around a plain {@code onPredictStart} callback.
     *
     * <p>{@link Hook} has six methods, so it is not a functional interface and a lambda cannot
     * be one. This is the reference's {@code on_predict_start=} parameter: the common case is a
     * caller who wants one event, and making them write an anonymous class for it is the kind of
     * friction that stops a tracer being added at all.
     */
    public static Hook onPredictStart(Consumer<PredictContext> callback) {
        requireCallback(callback, "onPredictStart");
        return new _StartAdapter(callback);
    }

    /** Wraps a hook object around a plain {@code onPredictEnd} callback. */
    public static Hook onPredictEnd(Consumer<PredictContext> callback) {
        requireCallback(callback, "onPredictEnd");
        return new _EndAdapter(callback);
    }

    /**
     * The reference's {@code _StartAdapter}, name included.
     *
     * <p>A leading underscore is not a Java convention and this is the only place in the port
     * that uses one. It is here because the class name is DATA: {@link #dispatch} and
     * {@link #call} quote it in the two lines they report, the reference quotes
     * {@code type(hook).__name__} and {@code method.__qualname__} in the same two, and
     * {@link #call} promises a caller grepping their logs across the runtimes finds the same
     * text. These two adapters are the advertised low-friction way to add one callback, so they
     * are the hooks most likely to be the ones named in a failure line.
     *
     * <p>A NAMED class rather than the anonymous one this was: {@link Class#getSimpleName} of an
     * anonymous class is the EMPTY STRING, so both lines read {@code laya: hook .on_predict_start
     * exceeded 0.05s} — a blank where the reference puts {@code _StartAdapter.on_predict_start}.
     * The hook a {@code raiseErrors(false)} policy exists to protect is a telemetry hook, and it
     * could not be identified from its own failure line.
     */
    private static final class _StartAdapter implements Hook {

        private final Consumer<PredictContext> callback;

        _StartAdapter(Consumer<PredictContext> callback) {
            this.callback = callback;
        }

        @Override
        public void onPredictStart(PredictContext ctx) {
            callback.accept(ctx);
        }

        @Override
        public String toString() {
            return "onPredictStart(" + callback + ")";
        }
    }

    /** The reference's {@code _EndAdapter}, name included — see {@link _StartAdapter}. */
    private static final class _EndAdapter implements Hook {

        private final Consumer<PredictContext> callback;

        _EndAdapter(Consumer<PredictContext> callback) {
            this.callback = callback;
        }

        @Override
        public void onPredictEnd(PredictContext ctx) {
            callback.accept(ctx);
        }

        @Override
        public String toString() {
            return "onPredictEnd(" + callback + ")";
        }
    }

    private static void requireCallback(Consumer<PredictContext> callback, String name) {
        if (callback == null) {
            throw new IllegalArgumentException(name + " must not be null");
        }
    }

    /**
     * One ordered list from a hook list and the two convenience callbacks.
     *
     * <p>Hooks first, in their own order, then the start callbacks, then the end callbacks —
     * so a {@code onPredictEnd} callback never runs before a hook object's own end method.
     *
     * <p>Three of the reference's refusals have no counterpart here and are not omissions: a
     * class rather than an instance, an object implementing none of the six events, and an
     * event attribute that is not callable are all unrepresentable once {@link Hook} is a type.
     * The fixture records them anyway, so that a reference that stopped refusing them would be
     * caught rather than quietly leaving this paragraph wrong. A null entry IS representable,
     * and is refused here.
     */
    public static List<Hook> normalise(List<? extends Hook> hooks,
                                       List<? extends Consumer<PredictContext>> onPredictStart,
                                       List<? extends Consumer<PredictContext>> onPredictEnd) {
        List<Hook> out = new ArrayList<>();
        if (hooks != null) {
            for (Hook hook : hooks) {
                if (hook == null) {
                    throw new IllegalArgumentException("a hooks entry must not be null");
                }
                out.add(hook);
            }
        }
        if (onPredictStart != null) {
            for (Consumer<PredictContext> callback : onPredictStart) {
                out.add(onPredictStart(callback));
            }
        }
        if (onPredictEnd != null) {
            for (Consumer<PredictContext> callback : onPredictEnd) {
                out.add(onPredictEnd(callback));
            }
        }
        return List.copyOf(out);
    }

    /**
     * The effective hook list for one call: defaults, then installed, then per-call.
     *
     * <p>The defaults are read HERE, at call time rather than at construction, so a hook set
     * after an agent was built still applies to it.
     */
    public static List<Hook> compose(List<? extends Hook> installed, HookCall call) {
        HookCall perCall = call == null ? HookCall.none() : call;
        List<Hook> out = new ArrayList<>(SKIP_DEFAULTS.get() ? List.of() : defaultHooks());
        if (installed != null) {
            out.addAll(installed);
        }
        out.addAll(normalise(perCall.hooks(), perCall.onPredictStart(), perCall.onPredictEnd()));
        return List.copyOf(out);
    }

    /**
     * The process-wide hooks, in order. Empty unless {@link #setDefaultHooks} was called.
     *
     * <p>A mutable COPY, as the reference returns: editing it is editing a list of your own, not
     * the process-wide one. {@link #withoutDefaultHooks} does not affect this — it suppresses the
     * defaults for a call, and a read that lied about what is installed would make a scope
     * impossible to debug from inside.
     */
    public static List<Hook> defaultHooks() {
        synchronized (DEFAULTS_MUTEX) {
            return new ArrayList<>(defaults);
        }
    }

    /**
     * Replaces the process-wide default hooks.
     *
     * <p>Defaults run before installed and per-call hooks for every agent in the process, so a
     * tracer or a metrics hook does not have to be threaded through every construction.
     */
    public static void setDefaultHooks(List<? extends Hook> hooks) {
        List<Hook> normalised = normalise(hooks, null, null);
        synchronized (DEFAULTS_MUTEX) {
            defaults = normalised;
        }
    }

    /**
     * Appends one hook to the process-wide defaults.
     *
     * <p>Process-wide and mutable, with no automatic restore: whatever is set here applies to
     * every agent in the JVM until something clears it. A test that sets a default and does not
     * clear it in teardown pollutes every test that runs after it, and the pollution surfaces as
     * a failure somewhere else. {@link #clearDefaultHooks} in an {@code @AfterEach} is the whole
     * discipline, and this port's own suites do exactly that.
     */
    public static void addDefaultHook(Hook hook) {
        List<Hook> normalised = normalise(singleton(hook), null, null);
        synchronized (DEFAULTS_MUTEX) {
            List<Hook> grown = new ArrayList<>(defaults);
            grown.addAll(normalised);
            defaults = List.copyOf(grown);
        }
    }

    /**
     * A one-element list that tolerates null, so {@link #normalise} is what refuses it.
     *
     * <p>{@code List.of(hook)} would throw a bare {@link NullPointerException} from inside the
     * JDK first, which made the "a hooks entry must not be null" message unreachable from every
     * single-hook entry point — {@link #addDefaultHook}, {@link HookRegistry#addHook} and
     * {@link HookCall#andThen} — and handed the caller a stack trace naming neither the argument
     * nor what was wrong with it.
     */
    static List<Hook> singleton(Hook hook) {
        List<Hook> one = new ArrayList<>(1);
        one.add(hook);
        return one;
    }

    /** Removes every process-wide default hook. */
    public static void clearDefaultHooks() {
        synchronized (DEFAULTS_MUTEX) {
            defaults = List.of();
        }
    }

    /**
     * Switches the process-wide defaults off for this thread, until the scope is closed.
     *
     * <pre>{@code
     * try (var ignored = Hooks.withoutDefaultHooks()) {
     *     agent.predict(state, questions);   // installed and per-call hooks only
     * }
     * }</pre>
     *
     * <p>Javac's {@code -Xlint:try} reports an unreferenced resource, so a build with
     * {@code -Werror} needs {@code @SuppressWarnings("try")} on the enclosing method. That is a
     * known javac wart about the idiom, not about this method.
     *
     * <p>The scopes NEST. Closing one restores the value it found, not {@code false}, so an
     * inner scope closing does not switch the defaults back on inside an outer one that is still
     * open. That is the reference's {@code _SKIP_DEFAULTS.reset(token)}, which nests by
     * construction; a {@link ThreadLocal} has no token, so the captured value is the token.
     *
     * <p>This is the reference's {@code _SKIP_DEFAULTS}, which a router enters so that a scan
     * made of many internal predictions fires a process-wide hook once for the scan rather than
     * once per forward pass. There it is a {@code contextvars.ContextVar} and here it is a
     * {@link ThreadLocal}, and the difference is real: a {@code ContextVar} is copied into an
     * asyncio task, while a {@code ThreadLocal} is not inherited by a thread the scope starts.
     * Work handed to another thread inside the scope therefore sees the defaults again.
     *
     * <p>One place in this port DOES run a hook off the calling thread — {@link Policy#timeout}
     * runs each hook on a {@code laya-hook-timeout} thread so the wait can be bounded — and an
     * earlier version of this paragraph said there was no such place. It still does not bite,
     * but for a different reason than "there is nowhere for it to": this flag is read by
     * {@link #compose}, on the calling thread, before anything is dispatched, so the hook thread
     * never consults it. The reference copies the whole context into its runner thread
     * ({@code contextvars.copy_context()}) rather than relying on that, which is why a
     * {@code ContextVar} a CALLER set is visible to a hook there and a {@link ThreadLocal} one
     * is not visible here. That difference is real and unported.
     */
    public static DefaultsScope withoutDefaultHooks() {
        // The PREVIOUS value, not false: the scopes nest, and an inner one closing must not
        // cancel an outer one's suppression. This is the reference's `_SKIP_DEFAULTS.reset(token)`
        // written out -- a ThreadLocal has no token, so the token is the captured value.
        boolean previous = SKIP_DEFAULTS.get();
        SKIP_DEFAULTS.set(true);
        return () -> SKIP_DEFAULTS.set(previous);
    }

    /**
     * What {@link #withoutDefaultHooks} hands back.
     *
     * <p>Its own type rather than {@link AutoCloseable} so that {@code close} declares no
     * checked exception: restoring a thread-local cannot fail, and a {@code throws Exception}
     * on the resource would push a {@code catch} into every caller for a failure that does not
     * exist.
     */
    @FunctionalInterface
    public interface DefaultsScope extends AutoCloseable {
        @Override
        void close();
    }

    /**
     * Returns a positive timeout, or null for no limit.
     *
     * <p>A non-positive timeout is refused here rather than left to the wait: a zero or negative
     * deadline returns before the hook has started, so the outcome of a fast hook under one is a
     * race. The reference also refuses a NaN and an infinity, which a {@link Duration} cannot
     * hold — those two refusals are unreachable here rather than unimplemented.
     */
    public static Duration validateTimeout(Duration value) {
        if (value == null) {
            return null;
        }
        if (value.isZero() || value.isNegative()) {
            throw new IllegalArgumentException(
                    "a hook timeout must be a positive duration or null; got " + value);
        }
        return value;
    }

    /**
     * Sums the per-state usage so a hook sees one total for the call.
     *
     * <p>The reference coerces as it goes — a missing usage block, a null one, a missing key, a
     * null value and a float all become an int — because a result there is a plain dict a hook
     * may have rewritten. Here a {@link Prediction} carries a {@link com.convaiinnovations.laya.Usage}
     * whose fields are primitive ints, so every one of those coercions is unreachable. They are
     * still pinned by the fixture, which records what each degenerate shape totals to, because
     * the Java side has to produce the same number from the representable equivalent.
     */
    public static Totals aggregateUsage(List<Prediction> results) {
        // Summed as longs: see Totals for the measured wrap an int accumulator produced, and for
        // why reproducing it would be fidelity to a shape rather than to a behaviour.
        long input = 0;
        long output = 0;
        for (Prediction result : results) {
            input += result.usage().inputTokens();
            output += result.usage().outputTokens();
        }
        return new Totals(input, output);
    }

    /**
     * Calls {@code event} on every hook, in order.
     *
     * <p>{@code policy.raiseErrors()} false reports through {@link Policy#onFailure} and
     * continues. {@code policy.lock()} serialises dispatch for hooks that are not re-entrant.
     * {@code policy.timeout()} bounds each hook call: an overrunning hook raises, or reports
     * when {@code raiseErrors} is false.
     *
     * <p>A timed-out hook KEEPS RUNNING. Neither Java nor Python can interrupt a thread that
     * will not cooperate, so what the deadline bounds is THE WAIT — not the request, and not the
     * process. The distinction is not pedantry and an earlier version of this paragraph had it
     * wrong. The abandoned thread still holds the live, mutable {@link PredictContext} of a call
     * that has moved on without it, and it can still write to it: measured over 60 identical
     * calls with a hook that overran a 50&nbsp;ms deadline and then assigned results, the late
     * write was accepted 60 times out of 60 and {@code ctx.usage} ended up describing a
     * different answer from {@code ctx.results} in 60 of them — because {@link #around} had
     * already totalled the usage of the answer it saw. In the runs where the write landed inside
     * the handful of microseconds before {@code around} read the results back, the CALLER was
     * handed the abandoned hook's answer instead. That is the reference's hazard too — it has
     * the same abandoned thread and the same shared dataclass.
     *
     * <p>What this port does about it, and the reference cannot easily: the thread is MARKED as
     * abandoned the moment its deadline expires, and {@link PredictContext} refuses a write from
     * a thread that carries that mark: the same 60 calls then disagree 0 times. See
     * {@link PredictContext#states(List)} for the one window it does not close. A hook that
     * blocks forever still leaks a thread per call, which is why the timeout is opt-in.
     *
     * <p>{@code raiseErrors} governs a {@link RuntimeException} and nothing else. An
     * {@link Error} is rethrown whatever the policy says, which is the reference's rule in Java
     * terms: it catches {@code Exception} and deliberately not {@code BaseException}, so a
     * {@code KeyboardInterrupt} or a {@code SystemExit} is never swallowed by a telemetry hook.
     * An {@link OutOfMemoryError} reported as a line of text and then carried on from is the
     * same mistake.
     */
    public static void dispatch(List<? extends Hook> hooks, Event event, PredictContext ctx,
                                Policy policy) {
        for (Hook hook : hooks) {
            try {
                // Read ONCE into a local. `policy.lock()` is a record accessor today, so the two
                // calls this replaces did return the same monitor -- but nothing in the type says
                // they must, and `lock()` on one instance with `unlock()` on another leaves the
                // first held with no owner able to release it. CodeQL's `java/unreleased-lock`
                // flagged exactly that shape here, in the one file where a stuck lock would wedge
                // every hooked call rather than one. Neither the pattern scanner nor a 14-mutant
                // sweep saw it: it is a property of which paths reach the unlock, not of a line.
                Lock lock = policy.lock();
                if (lock == null) {
                    call(hook, event, ctx, policy.timeout());
                } else {
                    lock.lock();
                    try {
                        call(hook, event, ctx, policy.timeout());
                    } finally {
                        lock.unlock();
                    }
                }
            } catch (RuntimeException problem) {
                if (policy.raiseErrors()) {
                    throw problem;
                }
                policy.onFailure().accept(String.format("laya: hook %s.%s failed: %s",
                        hookName(hook), event.wireName(), problem.getMessage()));
            }
        }
    }

    private static void call(Hook hook, Event event, PredictContext ctx, Duration timeout) {
        if (timeout == null) {
            event.callOn(hook, ctx);
            return;
        }
        Throwable[] box = new Throwable[1];
        Thread runner = new Thread(() -> {
            try {
                event.callOn(hook, ctx);
            } catch (RuntimeException | Error problem) {
                box[0] = problem;
            }
        }, "laya-hook-timeout");
        runner.setDaemon(true);
        runner.start();
        try {
            runner.join(timeout.toMillis(), timeout.toNanosPart() % 1_000_000);
        } catch (InterruptedException interrupted) {
            // Re-asserted BEFORE wrapping, because catching InterruptedException clears the
            // flag: a caller up the stack that polls `Thread.interrupted()` to decide whether to
            // shut down would otherwise be told it was never asked to.
            Thread.currentThread().interrupt();
            // The call stops waiting here, so the hook is cut off exactly as on a deadline.
            ctx.abandon(runner);
            throw new IllegalStateException("interrupted while waiting for hook "
                    + hookName(hook) + "." + event.wireName(), interrupted);
        }
        if (runner.isAlive()) {
            // The thread is not coming back under our control, so it is cut off from the call
            // instead: anything it writes to the context from here on is refused. See dispatch.
            ctx.abandon(runner);
            // Seconds, not `Duration.toString`'s "PT0.05S". The reference names the hook as
            // `<class>.<method>` and the deadline in seconds, and a caller grepping their logs
            // across the two runtimes should find the same line -- which is what `hookName` and
            // `seconds` are for, and what each of them documents it used to get wrong.
            throw new HookTimeoutException(String.format("laya: hook %s.%s exceeded %ss",
                    hookName(hook), event.wireName(), seconds(timeout)));
        }
        if (box[0] instanceof RuntimeException problem) {
            throw problem;
        }
        if (box[0] instanceof Error problem) {
            throw problem;
        }
    }

    /**
     * The name the two reported lines call a hook, which is the reference's own choice of name.
     *
     * <p>{@link Class#getSimpleName} alone is not it. For an ANONYMOUS class it is the empty
     * string, so {@code "laya: hook %s.%s failed"} came out as {@code laya: hook .on_predict_start
     * failed} — the one line a swallowed telemetry failure produces, naming nothing. Python has
     * no anonymous class and its {@code __qualname__} always names something, so the reference
     * cannot reach this case at all. A local or anonymous class falls back to the binary name's
     * last segment, {@code Enclosing$1}, which is ugly and is still a name.
     *
     * <p>Public because {@link Policy#onFailure} is a caller's own sink: a sink that wants to
     * attribute a failure to a hook should not have to re-derive the name this port already
     * picked, and get a blank for an anonymous one.
     */
    public static String hookName(Hook hook) {
        Class<?> type = hook.getClass();
        String simple = type.getSimpleName();
        if (!simple.isEmpty()) {
            return simple;
        }
        String binary = type.getName();
        int lastDot = binary.lastIndexOf('.');
        return lastDot < 0 ? binary : binary.substring(lastDot + 1);
    }

    /**
     * A deadline in seconds, formatted as the reference's {@code %g} formats it.
     *
     * <p>Not {@code %s} on a {@code double}, which is what this was: that prints {@code 1.0} for
     * a one-second deadline where CPython's {@code %g} prints {@code 1}, so the one line the
     * javadoc on {@link #call} promises would be greppable across both runtimes differed on the
     * commonest value there is. Measured in both: {@code 'laya: hook X exceeded %gs' % 1} is
     * {@code exceeded 1s}, and {@code String.format("exceeded %ss", 1.0)} is
     * {@code exceeded 1.0s}.
     *
     * <p>Java's own {@code %g} is not C's and cannot be used: it neither strips trailing zeros
     * nor switches to a fixed-point form, so it renders a one-second deadline as
     * {@code 1.00000}. This is C's rule written out — six significant digits, the exponent form
     * outside {@code [1e-4, 1e6)}, trailing zeros removed. {@code fixtures/hooks.json} records
     * CPython's own rendering per recorded deadline — {@code 1}, {@code 2}, {@code 10},
     * {@code 60}, {@code 1.5}, {@code 0.05} and {@code 1e-07} — and a test compares this method
     * against that field rather than against a string somebody typed. Checked by hand over a
     * wider sweep as well: 23 values including {@code 0.1}, {@code 0.25}, {@code 3600},
     * {@code 86400}, {@code 123456.789} and {@code 1e-09}, byte-identical on all 23.
     *
     * <p>Public for the same reason as {@link #hookName}: a caller logging a deadline of their
     * own should be able to produce the line this port produces, rather than one that differs
     * from it in the same way it used to differ from the reference.
     */
    public static String seconds(Duration timeout) {
        double value = timeout.toNanos() / 1e9;
        // %.5e is six significant digits, and its exponent is the one AFTER rounding -- which is
        // what the branch below has to read, or 9.999999 lands in the wrong form.
        String scientific = String.format(Locale.ROOT, "%.5e", value);
        int marker = scientific.indexOf('e');
        if (marker < 0) {
            // Unreachable from a `Duration`, whose `toNanos()` is a long, so `value` is always
            // finite -- but this method is PUBLIC so a caller's own sink can emit the same line,
            // and `%.5e` of a non-finite double renders "NaN"/"Infinity", neither of which holds
            // an 'e'. `substring(0)` would then hand the whole word to `Integer.parseInt`.
            // CodeQL raised the uncaught NumberFormatException; this makes the guard explicit
            // rather than resting on an invariant the signature does not state.
            return scientific;
        }
        int exponent = Integer.parseInt(scientific.substring(marker + 1));
        if (exponent < -4 || exponent >= 6) {
            return trimZeros(scientific.substring(0, marker))
                    + String.format(Locale.ROOT, "e%s%02d", exponent < 0 ? "-" : "+",
                            Math.abs(exponent));
        }
        return trimZeros(String.format(Locale.ROOT, "%." + (5 - exponent) + "f", value));
    }

    private static String trimZeros(String decimal) {
        if (decimal.indexOf('.') < 0) {
            return decimal;
        }
        int end = decimal.length();
        while (end > 0 && decimal.charAt(end - 1) == '0') {
            end--;
        }
        if (end > 0 && decimal.charAt(end - 1) == '.') {
            end--;
        }
        return decimal.substring(0, end);
    }

    /** A hook that outran {@link Policy#timeout}. Unchecked, so the usual policy governs it. */
    public static final class HookTimeoutException extends RuntimeException {

        private static final long serialVersionUID = 1L;

        HookTimeoutException(String message) {
            super(message);
        }
    }

    /**
     * Runs one call with its hooks around it: start, inference or a hook's answer, then error and
     * end.
     *
     * <p>Extracted rather than written inline in {@code Agent}, because the sequence below is the
     * whole observable contract of a hooked call and it is the same for anything that answers
     * questions. Keeping it in one place is also what lets it be tested against the reference
     * with a stub inference, which is how the fixture records it: whether the model ran at all is
     * then an observable rather than something inferred from the payload.
     *
     * <p>What happens, in order:
     *
     * <ol>
     *   <li>{@link Event#PREDICT_START} over every hook.</li>
     *   <li>If no hook assigned results, inference runs over whatever the hooks left on the
     *       context — the states, the questions and the two budget overrides. An EMPTY state list
     *       answers with an empty list without reaching inference, since there is nothing to
     *       collate.</li>
     *   <li>On a failure, {@link Event#ERROR} runs with the failure on the context. A hook that
     *       throws here is ATTACHED to the real failure as a suppressed exception, never
     *       substituted for it: the thing that broke the request must be what the caller
     *       catches.</li>
     *   <li>{@link Event#PREDICT_END} runs either way, after {@code elapsedMs} and — when there
     *       are results — the totalled {@code usage} are on the context. A hook that throws here
     *       fails the call if nothing else had, and is attached to the existing failure if
     *       something had.</li>
     * </ol>
     *
     * <p>The last two steps are written out rather than put in a {@code finally} block, and that
     * is not style: a Java {@code finally} that throws REPLACES the pending exception, which is
     * precisely the masking the reference goes out of its way to prevent.
     *
     * @return the results on the context, which is the hooks' last word on the call
     */
    public static List<Prediction> around(List<? extends Hook> hooks, PredictContext ctx,
                                          Policy policy, Inference inference) {
        Throwable raised = null;
        try {
            dispatch(hooks, Event.PREDICT_START, ctx, policy);
            if (ctx.results() == null) {
                if (ctx.states().isEmpty()) {
                    ctx.results(List.of());
                } else {
                    ctx.results(inference.run(ctx.states(), ctx.questions(), ctx.maxLen(),
                            ctx.headMaxLen()));
                }
            }
        } catch (RuntimeException | Error problem) {
            raised = problem;
            ctx.error(problem);
            try {
                dispatch(hooks, Event.ERROR, ctx, policy);
            } catch (RuntimeException | Error hookFailure) {
                // Attached, never substituted: the thing that broke the request is what the
                // caller has to catch, and a failing observer must not be able to hide it.
                attach(problem, hookFailure);
            }
        }

        ctx.elapsedMs((System.nanoTime() - ctx.startedAt()) / 1_000_000.0);
        if (ctx.results() != null) {
            ctx.usage(aggregateUsage(ctx.results()));
        }
        try {
            dispatch(hooks, Event.PREDICT_END, ctx, policy);
        } catch (RuntimeException | Error hookFailure) {
            if (raised == null) {
                throw hookFailure;
            }
            attach(raised, hookFailure);
        }

        if (raised instanceof RuntimeException problem) {
            throw problem;
        }
        if (raised instanceof Error problem) {
            throw problem;
        }
        return ctx.results();
    }

    /**
     * Reports a failure that happened before {@code on_predict_start} could fire -- routing or
     * loading, which the reference runs inside its predict {@code try} -- as {@code on_error} and
     * then {@code on_predict_end}, with {@code elapsedMs} measured from {@code startedAtNanos}.
     * A hook failure is attached to {@code problem}, never substituted.
     *
     * <p>For a {@link com.convaiinnovations.laya.Predictor} that must do work before its start
     * event, as {@code Router} does. Pass a context no hook has seen: one that already carries
     * results or an error is refused, though a context mid-call without either is not detected.
     */
    public static void failedBeforeStart(List<? extends Hook> hooks, PredictContext ctx,
                                         Policy policy, Throwable problem, long startedAtNanos) {
        requireNonNull(problem, "problem");
        if (ctx.results() != null || ctx.error() != null) {
            throw new IllegalStateException(
                    "failedBeforeStart needs a fresh context; this one already has an outcome");
        }
        ctx.error(problem);
        try {
            dispatch(hooks, Event.ERROR, ctx, policy);
        } catch (RuntimeException | Error hookFailure) {
            attach(problem, hookFailure);
        }
        ctx.elapsedMs((System.nanoTime() - startedAtNanos) / 1_000_000.0);
        try {
            dispatch(hooks, Event.PREDICT_END, ctx, policy);
        } catch (RuntimeException | Error hookFailure) {
            attach(problem, hookFailure);
        }
    }

    /**
     * {@code addSuppressed}, except when a hook rethrew the failure it was shown: a throwable
     * cannot suppress itself, and trying replaces the real failure with an
     * {@code IllegalArgumentException}.
     */
    private static void attach(Throwable problem, Throwable hookFailure) {
        if (hookFailure != problem) {
            problem.addSuppressed(hookFailure);
        }
    }

    /**
     * {@link #around} for many requests that share forward passes: each keeps its own context and
     * its own start and end, while {@code inference} answers the ones still pending in one go.
     *
     * <p>The reference's {@code Router.predict_batch} sequence. Every context starts, in order,
     * before any inference; {@code inference} must assign {@code results} on each context it is
     * handed. Contexts then end in <b>reverse</b> of the order they started, so a hook that sets
     * something in start and resets it in end unwinds the last one first. On a failure every
     * started context fails with it, a skipped one included, and gets the error event before its
     * end; a hook throwing on a failed context is attached to the failure. Every context gets its
     * end even when another's end hook throws; the first such throw is raised afterwards.
     *
     * @param inference answers the contexts no start hook answered; not called when there are none
     */
    public static void aroundGroup(List<? extends Hook> hooks, List<PredictContext> contexts,
                                   Policy policy, Consumer<List<PredictContext>> inference) {
        List<PredictContext> started = new ArrayList<>(contexts.size());
        Throwable raised = null;
        try {
            for (PredictContext ctx : contexts) {
                started.add(ctx);
                dispatch(hooks, Event.PREDICT_START, ctx, policy);
            }
            List<PredictContext> pending = new ArrayList<>();
            for (PredictContext ctx : started) {
                if (ctx.results() != null) {
                    continue;
                }
                if (ctx.states().isEmpty()) {
                    ctx.results(List.of());
                } else {
                    pending.add(ctx);
                }
            }
            if (!pending.isEmpty()) {
                inference.accept(List.copyOf(pending));
            }
            // Totalled before any end, so a malformed result fails the whole group rather than
            // escaping after some of its requests have already ended.
            List<Totals> totals = new ArrayList<>(started.size());
            for (PredictContext ctx : started) {
                totals.add(ctx.results() == null ? null : aggregateUsage(ctx.results()));
            }
            for (int i = 0; i < started.size(); i++) {
                started.get(i).usage(totals.get(i));
            }
        } catch (RuntimeException | Error problem) {
            raised = problem;
        }

        double now = System.nanoTime();
        for (PredictContext ctx : started) {
            ctx.elapsedMs((now - ctx.startedAt()) / 1_000_000.0);
            if (raised != null) {
                ctx.error(raised);
            }
        }
        Throwable firstEndFailure = null;
        for (int i = started.size() - 1; i >= 0; i--) {
            PredictContext ctx = started.get(i);
            List<Event> events = raised == null ? List.of(Event.PREDICT_END)
                    : List.of(Event.ERROR, Event.PREDICT_END);
            for (Event event : events) {
                try {
                    dispatch(hooks, event, ctx, policy);
                } catch (RuntimeException | Error hookFailure) {
                    if (raised != null) {
                        attach(raised, hookFailure);
                    } else if (firstEndFailure == null) {
                        firstEndFailure = hookFailure;
                    }
                }
            }
        }

        Throwable failure = raised != null ? raised : firstEndFailure;
        if (failure instanceof RuntimeException problem) {
            throw problem;
        }
        if (failure instanceof Error problem) {
            throw problem;
        }
    }
}
