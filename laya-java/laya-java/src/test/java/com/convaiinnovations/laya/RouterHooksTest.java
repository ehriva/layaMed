package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.Router.Checkpoint;
import com.convaiinnovations.laya.hooks.Hook;
import com.convaiinnovations.laya.hooks.Hooks;
import com.convaiinnovations.laya.hooks.PredictContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * A Router-level hook sees what no agent can: which checkpoint was chosen, and when one was built
 * or dropped. Before this, {@code Router} dispatched nothing at all.
 */
class RouterHooksTest {

    /** Records every event with the model it carried. */
    private static final class Trace implements Hook {

        final List<String> events = new ArrayList<>();

        private void note(String event, PredictContext ctx) {
            events.add(event + ":" + ctx.model());
        }

        @Override
        public void onRoute(PredictContext ctx) {
            note("route", ctx);
        }

        @Override
        public void onLoad(PredictContext ctx) {
            note("load", ctx);
        }

        @Override
        public void onEvict(PredictContext ctx) {
            note("evict", ctx);
        }

        @Override
        public void onPredictStart(PredictContext ctx) {
            note("start", ctx);
        }

        @Override
        public void onPredictEnd(PredictContext ctx) {
            note("end", ctx);
        }

        @Override
        public void onError(PredictContext ctx) {
            note("error", ctx);
        }
    }

    private static final class StubAgents implements Router.AgentFactory {

        private final Path root;
        int builds;

        StubAgents(Path root) {
            this.root = root;
        }

        @Override
        public Agent create(Checkpoint checkpoint) throws IOException {
            builds++;
            return TinyCheckpoint.agent(root, new TinyCheckpoint.RecordingSession());
        }
    }

    private static Map<String, Question> questions() {
        return Map.of("urgent", Question.noul("Needs a human."));
    }

    @Test
    @DisplayName("route() dispatches on_route once, naming the checkpoint it chose")
    void routeDispatches(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(new StubAgents(root)).build()) {
            router.hooks().addHook(trace);
            router.route("hello", questions());
            assertEquals(1, trace.events.size(), trace.events.toString());
            assertTrue(trace.events.get(0).startsWith("route:"), trace.events.toString());
        }
    }

    @Test
    @DisplayName("load() dispatches on_load for a build and stays quiet for a cache hit")
    void loadDispatchesOnlyOnBuild(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        StubAgents agents = new StubAgents(root);
        try (Router router = Router.builder().agents(agents).maxLoaded(2).build()) {
            router.hooks().addHook(trace);
            router.load("english");
            assertEquals(List.of("load:english"), trace.events);
            router.load("english");                    // resident: no second build, no second event
            assertEquals(List.of("load:english"), trace.events);
            assertEquals(1, agents.builds);
        }
    }

    @Test
    @DisplayName("an eviction forced by maxLoaded dispatches on_evict for the victim")
    void evictionDispatches(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(new StubAgents(root)).maxLoaded(1).build()) {
            router.hooks().addHook(trace);
            router.load("english");
            router.load("multilingual");               // maxLoaded 1, so english must go
            assertTrue(trace.events.contains("evict:english"), trace.events.toString());
            // The reference's order: the victim's on_evict, then the build's on_load.
            assertEquals(List.of("load:english", "evict:english", "load:multilingual"),
                    trace.events);
        }
    }

    @Test
    @DisplayName("unload dispatches on_evict for what it freed")
    void unloadDispatches(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(new StubAgents(root)).build()) {
            router.load("english");
            router.hooks().addHook(trace);             // installed after the load
            assertEquals(List.of(Checkpoint.ENGLISH), router.unload("english"));
            assertEquals(List.of("evict:english"), trace.events);
        }
    }

    @Test
    @DisplayName("predict dispatches one start/end pair for the whole route-and-answer call")
    void predictDispatchesOnePair(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(new StubAgents(root)).build()) {
            router.hooks().addHook(trace);
            router.predict("hello", questions());
            assertEquals(1, trace.events.stream().filter(e -> e.startsWith("start:")).count(),
                    trace.events.toString());
            assertEquals(1, trace.events.stream().filter(e -> e.startsWith("end:")).count(),
                    trace.events.toString());
            // The reference's order: route and load run BEFORE on_predict_start, so elapsed_ms
            // excludes the cold load. Derived from the trace, not hard-coded: which checkpoint
            // answers is the default's business, and this test is about the sequence.
            List<String> kinds = trace.events.stream().map(e -> e.split(":")[0]).toList();
            assertEquals(List.of("route", "load", "start", "end"), kinds,
                    trace.events.toString());
        }
    }

    @Test
    @DisplayName("on_load is dispatched with no router lock held, as seen from a SECOND thread")
    void dispatchHoldsNoRouterLock(@TempDir Path root) throws Exception {
        TinyCheckpoint.write(root, 64, 32);
        // A second thread, not the dispatching one: `lock` is a ReentrantLock, so a callback on the
        // dispatching thread re-enters it successfully whether or not it is held. Only another
        // thread can tell the difference.
        AtomicBoolean reached = new AtomicBoolean();
        try (Router router = Router.builder().agents(new StubAgents(root)).maxLoaded(2).build()) {
            router.hooks().addHook(new Hook() {
                @Override
                public void onLoad(PredictContext ctx) {
                    Thread other = new Thread(() -> {
                        router.loaded();
                        reached.set(true);
                    }, "router-lock-probe");
                    other.start();
                    try {
                        other.join(5_000);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
            });
            router.load("english");
            assertTrue(reached.get(),
                    "a second thread could not reach the router during on_load, so the dispatch "
                    + "held the lock");
        }
    }

    @Test
    @DisplayName("unloadAll dispatches on_evict for every checkpoint, which is what close() uses")
    void unloadAllDispatches(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(new StubAgents(root)).maxLoaded(3).build()) {
            router.load("english");
            router.load("multilingual");
            router.hooks().addHook(trace);
            assertEquals(2, router.unloadAll().size());
            assertEquals(List.of("evict:english", "evict:multilingual"), trace.events);
        }
    }

    @Test
    @DisplayName("preload dispatches on_load for what it built")
    void preloadDispatches(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(new StubAgents(root)).maxLoaded(3).build()) {
            router.hooks().addHook(trace);
            router.preload(List.of("english", "multilingual"));
            assertEquals(List.of("load:english", "load:multilingual"), trace.events);
        }
    }

    @Test
    @DisplayName("a throwing on_load hook does not strand the lease that lease() took")
    void aThrowingHookDoesNotLeakTheLease(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        try (Router router = Router.builder().agents(new StubAgents(root)).maxLoaded(2).build()) {
            router.hooks().addHook(new Hook() {
                @Override
                public void onLoad(PredictContext ctx) {
                    throw new IllegalStateException("the sink refused it");
                }
            });
            assertThrows(IllegalStateException.class, () -> router.lease("english"));
            // A stranded lease leaves the slot retired-but-never-closed: unload would report it
            // freed while the session stayed open.
            assertEquals(List.of(Checkpoint.ENGLISH), router.unload("english"));
        }
    }

    /** English's session throws on close, so evicting it fails after multilingual has published. */
    private static Router.AgentFactory angryEnglish(Path root) {
        return checkpoint -> TinyCheckpoint.agent(root, checkpoint == Checkpoint.ENGLISH
                ? new TinyCheckpoint.RecordingSession() {
                    @Override
                    public void close() {
                        super.close();
                        throw new IllegalStateException("a native close can fail");
                    }
                }
                : new TinyCheckpoint.RecordingSession());
    }

    @Test
    @DisplayName("lease still dispatches on_evict and on_load when closing the victim throws")
    void leaseDispatchesWhenTheEvictedCloseThrows(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(angryEnglish(root)).maxLoaded(1).build()) {
            router.load("english");
            router.hooks().addHook(trace);
            assertThrows(IllegalStateException.class, () -> router.lease("multilingual"));
            assertEquals(List.of("evict:english", "load:multilingual"), trace.events);
        }
    }

    @Test
    @DisplayName("a hook failing on that owed dispatch is attached to the close failure, not swapped")
    void anOwedHookFailureDoesNotMaskTheCloseFailure(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        try (Router router = Router.builder().agents(angryEnglish(root)).maxLoaded(1).build()) {
            router.load("english");
            AtomicBoolean once = new AtomicBoolean();   // only once: close() evicts again
            router.hooks().addHook(new Hook() {
                @Override
                public void onEvict(PredictContext ctx) {
                    if (once.compareAndSet(false, true)) {
                        throw new IllegalArgumentException("the sink refused it");
                    }
                }
            });
            IllegalStateException thrown =
                    assertThrows(IllegalStateException.class, () -> router.lease("multilingual"));
            assertEquals("a native close can fail", thrown.getMessage());
            assertEquals(1, thrown.getSuppressed().length);
            assertEquals("the sink refused it", thrown.getSuppressed()[0].getMessage());
        }
    }

    @Test
    @DisplayName("a load failure in predict reaches on_error and on_predict_end, with no start")
    void predictReportsALoadFailure(@TempDir Path root) throws IOException {
        Trace trace = new Trace();
        Router.AgentFactory broken = checkpoint -> {
            throw new IllegalStateException("no weights here");
        };
        try (Router router = Router.builder().agents(broken).build()) {
            router.hooks().addHook(trace);
            IllegalStateException thrown = assertThrows(IllegalStateException.class,
                    () -> router.predict("hello", questions()));
            assertEquals("no weights here", thrown.getMessage());
            // The load was attempted on a decided checkpoint, so error and end name it, as the
            // reference's ctx.model does once route() has returned.
            String model = trace.events.get(0).split(":")[1];
            assertEquals(List.of("route:" + model, "error:" + model, "end:" + model), trace.events);
        }
    }

    @Test
    @DisplayName("a routing failure in predict reaches on_error and on_predict_end, as in the reference")
    void predictReportsARouteFailure(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(new StubAgents(root)).build()) {
            router.hooks().addHook(trace);
            assertThrows(RuntimeException.class, () -> router.predict("hello", questions(),
                    Router.RouteOptions.none().model("no-such-checkpoint")));
            assertEquals(List.of("error:null", "end:null"), trace.events);
        }
    }

    @Test
    @DisplayName("load: a hook failing on the owed dispatch is attached to the close failure")
    void loadDoesNotMaskTheCloseFailure(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        try (Router router = Router.builder().agents(angryEnglish(root)).maxLoaded(1).build()) {
            router.load("english");
            router.hooks().addHook(throwsOnceOnEvict(new IllegalArgumentException("the sink refused it")));
            IllegalStateException thrown =
                    assertThrows(IllegalStateException.class, () -> router.load("multilingual"));
            assertEquals("a native close can fail", thrown.getMessage());
            assertEquals(1, thrown.getSuppressed().length);
        }
    }

    @Test
    @DisplayName("unload: a hook failing on the owed dispatch is attached to the close failure")
    void unloadDoesNotMaskTheCloseFailure(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        try (Router router = Router.builder().agents(angryEnglish(root)).maxLoaded(2).build()) {
            router.load("english");
            router.hooks().addHook(throwsOnceOnEvict(new IllegalArgumentException("the sink refused it")));
            IllegalStateException thrown =
                    assertThrows(IllegalStateException.class, () -> router.unload("english"));
            assertEquals("a native close can fail", thrown.getMessage());
            assertEquals(1, thrown.getSuppressed().length);
        }
    }

    @Test
    @DisplayName("an Error from a hook on the owed dispatch is attached too, not swapped in")
    void anOwedHookErrorDoesNotMaskTheCloseFailure(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        try (Router router = Router.builder().agents(angryEnglish(root)).maxLoaded(1).build()) {
            router.load("english");
            router.hooks().addHook(throwsOnceOnEvict(new AssertionError("an Error from a hook")));
            IllegalStateException thrown =
                    assertThrows(IllegalStateException.class, () -> router.lease("multilingual"));
            assertEquals("a native close can fail", thrown.getMessage());
            assertTrue(thrown.getSuppressed()[0] instanceof AssertionError);
        }
    }

    @Test
    @DisplayName("on_load still fires when an attach wins the race with the build, as in the reference")
    void loadDispatchesWhenAnAttachWinsTheRace(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        AtomicReference<Router> self = new AtomicReference<>();
        Router.AgentFactory racing = checkpoint -> {
            // Attached while this build is in progress, so the build loses and is discarded.
            self.get().attach("english",
                    TinyCheckpoint.agent(root, new TinyCheckpoint.RecordingSession()));
            return TinyCheckpoint.agent(root, new TinyCheckpoint.RecordingSession());
        };
        try (Router router = Router.builder().agents(racing).maxLoaded(2).build()) {
            self.set(router);
            router.hooks().addHook(trace);
            router.load("english");
            assertEquals(List.of("load:english"), trace.events);
        }
    }

    @Test
    @DisplayName("a hook rethrowing ctx.error() from any error or end event leaves the real failure")
    void rethrowingCtxErrorNeverReplacesTheFailure(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        // A start hook failing goes through Hooks.around; a load failing through failedBeforeStart.
        for (String rethrowIn : List.of("error", "end")) {
            Hook rethrow = new Hook() {
                @Override
                public void onPredictStart(PredictContext ctx) {
                    throw new IllegalStateException("start refused");
                }

                @Override
                public void onError(PredictContext ctx) {
                    if (rethrowIn.equals("error")) {
                        throw (RuntimeException) ctx.error();
                    }
                }

                @Override
                public void onPredictEnd(PredictContext ctx) {
                    if (rethrowIn.equals("end")) {
                        throw (RuntimeException) ctx.error();
                    }
                }
            };
            try (Router router = Router.builder().agents(new StubAgents(root)).build()) {
                router.hooks().addHook(rethrow);
                IllegalStateException thrown = assertThrows(IllegalStateException.class,
                        () -> router.predict("hello", questions()));
                assertEquals("start refused", thrown.getMessage(), "rethrown in " + rethrowIn);
            }
        }
        Router.AgentFactory broken = checkpoint -> {
            throw new IllegalStateException("no weights here");
        };
        try (Router router = Router.builder().agents(broken).build()) {
            router.hooks().addHook(new Hook() {
                @Override
                public void onPredictEnd(PredictContext ctx) {
                    throw (RuntimeException) ctx.error();
                }
            });
            IllegalStateException thrown = assertThrows(IllegalStateException.class,
                    () -> router.predict("hello", questions()));
            assertEquals("no weights here", thrown.getMessage());
        }
    }

    @Test
    @DisplayName("failedBeforeStart refuses a null failure and a context that already has an outcome")
    void failedBeforeStartRefusesMisuse() {
        PredictContext fresh = new PredictContext(List.of("s"), Map.of(), "english", null);
        assertThrows(NullPointerException.class, () -> Hooks.failedBeforeStart(List.of(), fresh,
                Hooks.Policy.raising(), null, System.nanoTime()));
        PredictContext reported = new PredictContext(List.of("s"), Map.of(), "english", null);
        Hooks.failedBeforeStart(List.of(), reported, Hooks.Policy.raising(),
                new IllegalStateException("first"), System.nanoTime());
        assertThrows(IllegalStateException.class, () -> Hooks.failedBeforeStart(List.of(),
                reported, Hooks.Policy.raising(), new IllegalStateException("second"),
                System.nanoTime()));
        assertEquals("first", reported.error().getMessage());
    }

    /** A factory that keeps every session it hands out, so a test can see which were closed. */
    private static Router.AgentFactory recording(Path root, List<TinyCheckpoint.RecordingSession> out) {
        return checkpoint -> {
            TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
            out.add(session);
            return TinyCheckpoint.agent(root, session);
        };
    }

    @Test
    @DisplayName("a hooked predict releases its lease, so unloading closes the agent")
    void hookedPredictReleasesItsLease(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        List<TinyCheckpoint.RecordingSession> sessions = new ArrayList<>();
        try (Router router = Router.builder().agents(recording(root, sessions)).build()) {
            router.hooks().addHook(new Trace());
            router.predict("hello", questions());
            router.unloadAll();
            assertEquals(1, sessions.size());
            assertTrue(sessions.get(0).closed, "a held lease keeps a retired agent open forever");
        }
    }

    @Test
    @DisplayName("a hooked predict whose start hook throws still releases its lease")
    void hookedPredictReleasesItsLeaseOnFailure(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        List<TinyCheckpoint.RecordingSession> sessions = new ArrayList<>();
        try (Router router = Router.builder().agents(recording(root, sessions)).build()) {
            router.hooks().addHook(new Hook() {
                @Override
                public void onPredictStart(PredictContext ctx) {
                    throw new IllegalStateException("start refused");
                }
            });
            assertThrows(IllegalStateException.class, () -> router.predict("hello", questions()));
            router.unloadAll();
            assertTrue(sessions.get(0).closed, "a held lease keeps a retired agent open forever");
        }
    }

    @Test
    @DisplayName("unloadAll: every eviction is dispatched and a hook failure is attached, not swapped")
    void unloadAllDoesNotMaskTheCloseFailure(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        Trace trace = new Trace();
        try (Router router = Router.builder().agents(angryEnglish(root)).maxLoaded(2).build()) {
            router.load("english");
            router.load("multilingual");
            router.hooks().addHook(trace);            // first, so it sees the event that throws
            router.hooks().addHook(throwsOnceOnEvict(new IllegalArgumentException("the sink refused it")));
            IllegalStateException thrown =
                    assertThrows(IllegalStateException.class, router::unloadAll);
            assertEquals("a native close can fail", thrown.getMessage());
            assertEquals(1, thrown.getSuppressed().length);
            assertEquals(List.of("evict:english", "evict:multilingual"), trace.events,
                    "a hook failure on the first eviction must not skip the second");
        }
    }

    @Test
    @DisplayName("null state or questions are refused before routing, with or without hooks")
    void nullsAreRefusedBeforeAnyLoad(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        for (boolean hooked : new boolean[] {true, false}) {
            StubAgents agents = new StubAgents(root);
            Trace trace = new Trace();
            try (Router router = Router.builder().agents(agents).build()) {
                if (hooked) {
                    router.hooks().addHook(trace);
                }
                assertThrows(IllegalArgumentException.class, () -> router.predict("hello", null));
                assertThrows(IllegalArgumentException.class,
                        () -> router.predict(null, questions()));
                assertEquals(0, agents.builds, "hooked=" + hooked + ": a refused call loaded");
                assertEquals(List.of(), trace.events);
            }
        }
    }

    @Test
    @DisplayName("a failure before start: elapsedMs covers the load, ctx.error is the failure")
    void aFailureBeforeStartIsTimedAndCarriesTheError(@TempDir Path root) throws IOException {
        IllegalStateException failure = new IllegalStateException("no weights here");
        Router.AgentFactory slow = checkpoint -> {
            try {
                Thread.sleep(60);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            throw failure;
        };
        AtomicReference<Double> elapsed = new AtomicReference<>();
        AtomicReference<Throwable> seen = new AtomicReference<>();
        try (Router router = Router.builder().agents(slow).build()) {
            router.hooks().addHook(new Hook() {
                @Override
                public void onPredictEnd(PredictContext ctx) {
                    elapsed.set(ctx.elapsedMs());
                    seen.set(ctx.error());
                }
            });
            long before = System.nanoTime();
            assertSame(failure, assertThrows(IllegalStateException.class,
                    () -> router.predict("hello", questions())));
            double wall = (System.nanoTime() - before) / 1_000_000.0;
            assertTrue(elapsed.get() >= 50.0, "elapsedMs " + elapsed.get() + " missed the load");
            assertTrue(elapsed.get() <= wall, "elapsedMs " + elapsed.get() + " exceeds the call");
            assertSame(failure, seen.get());
        }
    }

    @Test
    @DisplayName("an on_error hook that rethrows ctx.error() leaves the caller the real failure")
    void aRethrowingErrorHookDoesNotReplaceTheFailure(@TempDir Path root) throws IOException {
        Router.AgentFactory broken = checkpoint -> {
            throw new IllegalStateException("no weights here");
        };
        try (Router router = Router.builder().agents(broken).build()) {
            router.hooks().addHook(new Hook() {
                @Override
                public void onError(PredictContext ctx) {
                    throw (RuntimeException) ctx.error();     // log-and-rethrow
                }
            });
            IllegalStateException thrown = assertThrows(IllegalStateException.class,
                    () -> router.predict("hello", questions()));
            assertEquals("no weights here", thrown.getMessage());
        }
    }

    @Test
    @DisplayName("an Error while loading still reaches on_error and on_predict_end")
    void anErrorBeforeStartIsReported(@TempDir Path root) throws IOException {
        Trace trace = new Trace();
        Router.AgentFactory native_ = checkpoint -> {
            throw new UnsatisfiedLinkError("no onnxruntime here");
        };
        try (Router router = Router.builder().agents(native_).build()) {
            router.hooks().addHook(trace);
            assertThrows(UnsatisfiedLinkError.class, () -> router.predict("hello", questions()));
            List<String> kinds = trace.events.stream().map(e -> e.split(":")[0]).toList();
            assertEquals(List.of("route", "error", "end"), kinds, trace.events.toString());
        }
    }

    /** A hook whose first on_evict throws {@code failure}; later ones (from close()) do not. */
    private static Hook throwsOnceOnEvict(Throwable failure) {
        AtomicBoolean once = new AtomicBoolean();
        return new Hook() {
            @Override
            public void onEvict(PredictContext ctx) {
                if (once.compareAndSet(false, true)) {
                    if (failure instanceof Error error) {
                        throw error;
                    }
                    throw (RuntimeException) failure;
                }
            }
        };
    }

    @Test
    @DisplayName("a router with no hooks dispatches nothing and builds no context")
    void noHooksNoDispatch(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 64, 32);
        StubAgents agents = new StubAgents(root);
        try (Router router = Router.builder().agents(agents).build()) {
            assertFalse(router.hooks().hooks().iterator().hasNext());
            router.predict("hello", questions());      // the unhooked fast path still answers
            assertEquals(1, agents.builds);
        }
    }
}
