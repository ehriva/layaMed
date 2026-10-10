package com.convaiinnovations.laya.hooks;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

/**
 * The hooks installed on one agent, and what dispatch does with them.
 *
 * <p>A list that can be edited while the agent is alive, so a tracer can be attached to a
 * long-lived agent without rebuilding it. A call takes a SNAPSHOT — {@link Hooks#compose} copies
 * — so adding or removing a hook never disturbs a call already in flight.
 *
 * <p>In the reference this is a mixin that {@code Agent}, {@code Router} and {@code ONNXAgent}
 * inherit. {@link com.convaiinnovations.laya.Agent} is final and already implements two
 * interfaces, and single inheritance makes a mixin the wrong shape here anyway: the registry is
 * reached through {@code agent.hooks()} instead, which also keeps six hook methods off the
 * surface of a class whose subject is prediction.
 */
public final class HookRegistry {

    private final Object mutex = new Object();
    private final ReentrantLock serial = new ReentrantLock();

    private volatile List<Hook> installed = List.of();
    private volatile boolean raiseErrors = true;
    private volatile boolean concurrent = true;
    private volatile Duration timeout;

    /** The installed hooks, in order, as a snapshot. */
    public List<Hook> hooks() {
        return installed;
    }

    /** Installs one hook. Returns this, for chaining. */
    public HookRegistry addHook(Hook hook) {
        // Hooks.singleton, not List.of: see that method for why a null has to reach normalise.
        return addHooks(Hooks.singleton(hook));
    }

    /** Installs several hooks, in order. Returns this, for chaining. */
    public HookRegistry addHooks(List<? extends Hook> hooks) {
        List<Hook> normalised = Hooks.normalise(hooks, null, null);
        if (normalised.isEmpty()) {
            return this;
        }
        synchronized (mutex) {
            List<Hook> grown = new ArrayList<>(installed);
            grown.addAll(normalised);
            installed = List.copyOf(grown);
        }
        return this;
    }

    /**
     * Removes a hook by identity, every copy of it.
     *
     * @return whether it was installed at all
     */
    public boolean removeHook(Hook hook) {
        synchronized (mutex) {
            List<Hook> remaining = new ArrayList<>(installed.size());
            for (Hook candidate : installed) {
                if (candidate != hook) {
                    remaining.add(candidate);
                }
            }
            if (remaining.size() == installed.size()) {
                return false;
            }
            installed = List.copyOf(remaining);
            return true;
        }
    }

    /**
     * Installs hooks for the duration of a block, then removes them.
     *
     * <pre>{@code
     * try (var ignored = agent.hooks().hooksInstalled(tracer)) {
     *     agent.predict(state, questions);
     * }
     * }</pre>
     *
     * <p>A build with {@code -Xlint:try -Werror} needs {@code @SuppressWarnings("try")} on the
     * enclosing method: javac reports an unreferenced resource for every try-with-resources
     * whose value is only a scope, which is what this is.
     */
    public Scope hooksInstalled(Hook... hooks) {
        return hooksInstalled(List.of(hooks));
    }

    /** Installs hooks for the duration of a block, then removes them. */
    public Scope hooksInstalled(List<? extends Hook> hooks) {
        List<Hook> added = Hooks.normalise(hooks, null, null);
        addHooks(added);
        return new Scope(added);
    }

    /** What {@link HookRegistry#hooksInstalled(Hook...)} hands back; closing it removes what it added. */
    public final class Scope implements AutoCloseable {

        private final List<Hook> added;
        private boolean closed;

        private Scope(List<Hook> added) {
            this.added = added;
        }

        /**
         * Removes ONE occurrence of each hook this scope added, the most recent match.
         *
         * <p>Not "every copy by identity", and not "restore a snapshot". Both are wrong, and
         * the reference carries the scars of each. Removing by identity alone took a hook the
         * application had installed before the block along with the block's own, and it stayed
         * gone — while the contract is that a block restores the previous list. Restoring a
         * snapshot is wrong in two other ways: with two overlapping scopes the first exit
         * reinstates its snapshot and so removes the second scope's hook, and a hook added
         * inside the block is discarded because it is not in the snapshot either. Taking one
         * occurrence per hook the scope added leaves everything else in place.
         *
         * <p>The scan runs BACKWARDS, taking the most recent match, because installation
         * appends and a hook that was already there sits earlier than this scope's copy of it.
         * That is the reference's direction, and it is OBSERVABLE — an earlier version of this
         * paragraph claimed it was not, on the grounds that the copies are the same object. They
         * are, so the MULTISET left behind is identical either way; the ORDER is not, and order
         * is the one thing about a hook list that is a contract. With {@code addHook(a)},
         * {@code addHook(x)} and then {@code hooksInstalled(a)} the list is {@code [a, x, a]}:
         * dropping the last match leaves {@code [a, x]} and dropping the first leaves
         * {@code [x, a]}, so a forward scan would reorder two hooks the caller installed itself
         * and change which one sees the other's work. A test pins the direction.
         *
         * <p>Idempotent, as {@link AutoCloseable} asks: closing twice would otherwise take a
         * second copy that belongs to someone else.
         */
        @Override
        public void close() {
            synchronized (mutex) {
                if (closed) {
                    return;
                }
                closed = true;
                List<Hook> current = new ArrayList<>(installed);
                for (Hook hook : added) {
                    for (int at = current.size() - 1; at >= 0; at--) {
                        if (current.get(at) == hook) {
                            current.remove(at);
                            break;
                        }
                    }
                }
                installed = List.copyOf(current);
            }
        }
    }

    /** Whether a throwing hook fails the call. True by default, as the reference. */
    public boolean raiseErrors() {
        return raiseErrors;
    }

    /** Sets whether a throwing hook fails the call, or is reported and stepped over. */
    public HookRegistry raiseErrors(boolean value) {
        this.raiseErrors = value;
        return this;
    }

    /** Whether hooks may run unguarded. True by default. */
    public boolean concurrent() {
        return concurrent;
    }

    /**
     * Sets whether hooks may run unguarded.
     *
     * <p>False serialises every dispatch on this agent behind one re-entrant lock, for a hook
     * that is not safe to run from two calls at once. It does not make a hook atomic with the
     * inference around it.
     */
    public HookRegistry concurrent(boolean value) {
        this.concurrent = value;
        return this;
    }

    /** The per-hook deadline, or null for no limit. */
    public Duration timeout() {
        return timeout;
    }

    /** Sets the per-hook deadline. Null removes it. */
    public HookRegistry timeout(Duration value) {
        this.timeout = Hooks.validateTimeout(value);
        return this;
    }

    /** The effective hook list for one call: defaults, then these, then the call's own. */
    public List<Hook> composeFor(HookCall call) {
        return Hooks.compose(installed, call);
    }

    /** The effective policy for one call, with the call's overrides applied over this agent's. */
    public Hooks.Policy policyFor(HookCall call) {
        HookCall perCall = call == null ? HookCall.none() : call;
        boolean raising = perCall.raiseErrors() == null ? raiseErrors : perCall.raiseErrors();
        Duration deadline = perCall.timeout() == null ? timeout : perCall.timeout();
        return new Hooks.Policy(raising, concurrent ? null : serial, deadline, Hooks.reporting());
    }
}
