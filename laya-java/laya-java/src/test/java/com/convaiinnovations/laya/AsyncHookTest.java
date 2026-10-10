package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.hooks.AsyncHook;
import com.convaiinnovations.laya.hooks.Hook;
import com.convaiinnovations.laya.hooks.Hooks;
import com.convaiinnovations.laya.hooks.PredictContext;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** {@link AsyncHook} runs a hook elsewhere but still waits for it, as the reference does. */
class AsyncHookTest {

    private static PredictContext ctx() {
        return new PredictContext(List.of("state"), Map.of(), "english", null);
    }

    @Test
    @DisplayName("the callback runs on the executor, not the calling thread")
    void runsOnTheExecutor() {
        ExecutorService pool = Executors.newSingleThreadExecutor(r -> new Thread(r, "hook-pool"));
        try {
            AtomicReference<String> ran = new AtomicReference<>();
            Hook async = AsyncHook.of(new Hook() {
                @Override
                public void onPredictStart(PredictContext c) {
                    ran.set(Thread.currentThread().getName());
                }
            }, pool);
            async.onPredictStart(ctx());
            assertEquals("hook-pool", ran.get());
            assertNotEquals(Thread.currentThread().getName(), ran.get());
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    @DisplayName("the call waits: the callback has finished before the method returns")
    void waitsForTheCallback() {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            CountDownLatch done = new CountDownLatch(1);
            Hook async = AsyncHook.of(new Hook() {
                @Override
                public void onPredictStart(PredictContext c) {
                    try {
                        Thread.sleep(50);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    done.countDown();
                }
            }, pool);
            async.onPredictStart(ctx());
            assertEquals(0, done.getCount(), "returned before the callback finished");
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    @DisplayName("the delegate's exception reaches the caller unwrapped, so the policy sees it")
    void propagatesTheDelegateFailure() {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            IllegalStateException boom = new IllegalStateException("from the hook");
            Hook async = AsyncHook.of(new Hook() {
                @Override
                public void onPredictEnd(PredictContext c) {
                    throw boom;
                }
            }, pool);
            assertSame(boom, assertThrows(IllegalStateException.class,
                    () -> async.onPredictEnd(ctx())));
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    @DisplayName("a callback that outlasts the timeout fails the dispatch")
    void timesOut() {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            Hook async = AsyncHook.of(new Hook() {
                @Override
                public void onRoute(PredictContext c) {
                    try {
                        Thread.sleep(5_000);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
            }, pool, Duration.ofMillis(50));
            IllegalStateException thrown =
                    assertThrows(IllegalStateException.class, () -> async.onRoute(ctx()));
            assertTrue(thrown.getMessage().contains("did not finish"), thrown.getMessage());
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    @DisplayName("an overrunning callback is abandoned, so its late writes are refused")
    void anOverrunningCallbackIsAbandoned() throws Exception {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            CountDownLatch wrote = new CountDownLatch(1);
            Hook async = AsyncHook.of(new Hook() {
                @Override
                public void onPredictStart(PredictContext c) {
                    try {
                        Thread.sleep(400);          // overruns the 50ms deadline below
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    try {
                        c.states(List.of("written after the call moved on"));
                    } catch (RuntimeException refused) {
                        wrote.countDown();          // refused, which is the point
                    }
                }
            }, pool, Duration.ofMillis(50));
            PredictContext ctx = ctx();
            assertThrows(IllegalStateException.class, () -> async.onPredictStart(ctx));
            // cancel(true) cannot interrupt a task already on an executor, so it keeps running.
            // ctx.abandon is what stops it writing to a call that has finished.
            assertTrue(wrote.await(3, TimeUnit.SECONDS), "the late write was never attempted");
            assertEquals(List.of("state"), ctx.states(),
                    "an abandoned callback's write reached the context");
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    @DisplayName("after an overrun, the same call's next callback on that pooled thread is accepted")
    void aPooledThreadIsReleasedWhenTheOverrunEnds() {
        // One thread, so the later callback necessarily runs on the thread that overran. Before
        // the release, that thread stayed abandoned for this context and the write was refused.
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            assertReleasedAfterOverrun(pool, false);
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    @DisplayName("the release also happens when the overrunning callback throws")
    void aPooledThreadIsReleasedWhenTheOverrunThrows() {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            assertReleasedAfterOverrun(pool, true);
        } finally {
            pool.shutdownNow();
        }
    }

    private static void assertReleasedAfterOverrun(ExecutorService pool, boolean overrunThrows) {
        AtomicReference<Thread> overran = new AtomicReference<>();
        AtomicReference<Thread> next = new AtomicReference<>();
        Hook slow = AsyncHook.of(new Hook() {
            @Override
            public void onPredictStart(PredictContext c) {
                overran.set(Thread.currentThread());
                try {
                    Thread.sleep(300);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                if (overrunThrows) {
                    throw new IllegalStateException("the overrun failed as well");
                }
            }
        }, pool, Duration.ofMillis(50));
        Hook later = AsyncHook.of(new Hook() {
            @Override
            public void onPredictEnd(PredictContext c) {
                next.set(Thread.currentThread());
                c.states(List.of("written by the next callback"));
            }
        }, pool);
        PredictContext ctx = ctx();
        assertThrows(IllegalStateException.class, () -> slow.onPredictStart(ctx));
        later.onPredictEnd(ctx);                    // queued behind the overrun, then runs
        assertSame(overran.get(), next.get(), "the point is the SAME pooled thread");
        assertEquals(List.of("written by the next callback"), ctx.states());
    }

    @Test
    @DisplayName("an AsyncHook on an abandoned thread does not lift that abandonment")
    void anOuterAbandonmentSurvivesAnInnerAsyncHook() throws Exception {
        // Hooks abandons its timeout thread; that thread then runs an AsyncHook on a direct
        // executor -- the same thread -- which must not release what it did not abandon.
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<String> outcome = new AtomicReference<>();
        Hook inner = AsyncHook.of(new Hook() { }, Runnable::run);
        Hook outer = new Hook() {
            @Override
            public void onPredictStart(PredictContext c) {
                try {
                    Thread.sleep(200);              // overruns the 50ms policy deadline
                    inner.onPredictEnd(c);
                    c.states(List.of("late write"));
                    outcome.set("accepted");
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (RuntimeException refused) {
                    outcome.set("refused");
                } finally {
                    done.countDown();
                }
            }
        };
        PredictContext ctx = ctx();
        assertThrows(RuntimeException.class, () -> Hooks.dispatch(List.of(outer),
                Hooks.Event.PREDICT_START, ctx, Hooks.Policy.raising().timeout(Duration.ofMillis(50))));
        assertTrue(done.await(3, TimeUnit.SECONDS));
        assertEquals("refused", outcome.get());
        assertEquals(List.of("state"), ctx.states());
    }

    @Test
    @DisplayName("an interrupted wait cuts the callback off, as a deadline does")
    void anInterruptedWaitAbandonsTheCallback() throws Exception {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            CountDownLatch release = new CountDownLatch(1);
            CountDownLatch started = new CountDownLatch(1);
            CountDownLatch done = new CountDownLatch(1);
            Hook async = AsyncHook.of(new Hook() {
                @Override
                public void onPredictStart(PredictContext c) {
                    started.countDown();
                    awaitQuietly(release);
                    try {
                        c.states(List.of("late"));
                    } catch (RuntimeException refused) {
                        // refused, which is the point
                    } finally {
                        done.countDown();
                    }
                }
            }, pool, Duration.ofSeconds(5));       // timed: an untimed join() ignores interrupts
            assertLateWriteRefusedAfterInterrupt(ctx -> async.onPredictStart(ctx), started, release,
                    done);
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    @DisplayName("an interrupted wait on a Hooks timeout thread cuts that hook off too")
    void anInterruptedHooksWaitAbandonsTheHook() throws Exception {
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(1);
        Hook slow = new Hook() {
            @Override
            public void onPredictStart(PredictContext c) {
                started.countDown();
                awaitQuietly(release);
                try {
                    c.states(List.of("late"));
                } catch (RuntimeException refused) {
                    // refused, which is the point
                } finally {
                    done.countDown();
                }
            }
        };
        assertLateWriteRefusedAfterInterrupt(ctx -> Hooks.dispatch(List.of(slow),
                Hooks.Event.PREDICT_START, ctx, Hooks.Policy.raising().timeout(Duration.ofSeconds(5))),
                started, release, done);
    }

    private static void assertLateWriteRefusedAfterInterrupt(
            java.util.function.Consumer<PredictContext> waitOn, CountDownLatch started,
            CountDownLatch release, CountDownLatch done) throws Exception {
        PredictContext ctx = ctx();
        AtomicReference<Throwable> caught = new AtomicReference<>();
        Thread caller = new Thread(() -> {
            try {
                waitOn.accept(ctx);
            } catch (RuntimeException e) {
                caught.set(e);
            }
        });
        caller.start();
        assertTrue(started.await(3, TimeUnit.SECONDS));
        caller.interrupt();
        caller.join(3_000);
        assertTrue(caught.get() instanceof IllegalStateException, String.valueOf(caught.get()));
        release.countDown();                        // only now may the hook try its late write
        assertTrue(done.await(3, TimeUnit.SECONDS));
        assertEquals(List.of("state"), ctx.states(), "a write after the caller moved on landed");
    }

    /** Waits for the test's go-ahead; a latch, so no sleep decides the order of events. */
    private static void awaitQuietly(CountDownLatch release) {
        try {
            release.await(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    @Test
    @DisplayName("a non-positive timeout is refused rather than meaning 'never wait'")
    void refusesANonPositiveTimeout() {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            Hook plain = new Hook() { };
            assertThrows(IllegalArgumentException.class,
                    () -> AsyncHook.of(plain, pool, Duration.ZERO));
            assertThrows(IllegalArgumentException.class,
                    () -> AsyncHook.of(plain, pool, Duration.ofMillis(-1)));
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    @DisplayName("every event is forwarded, not just the predict pair")
    void forwardsEveryEvent() {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            List<String> seen = new ArrayList<>();
            Hook async = AsyncHook.of(new Hook() {
                @Override
                public void onPredictStart(PredictContext c) {
                    seen.add("start");
                }

                @Override
                public void onPredictEnd(PredictContext c) {
                    seen.add("end");
                }

                @Override
                public void onRoute(PredictContext c) {
                    seen.add("route");
                }

                @Override
                public void onLoad(PredictContext c) {
                    seen.add("load");
                }

                @Override
                public void onEvict(PredictContext c) {
                    seen.add("evict");
                }

                @Override
                public void onError(PredictContext c) {
                    seen.add("error");
                }
            }, pool);
            PredictContext ctx = ctx();
            async.onPredictStart(ctx);
            async.onRoute(ctx);
            async.onLoad(ctx);
            async.onEvict(ctx);
            async.onError(ctx);
            async.onPredictEnd(ctx);
            assertEquals(List.of("start", "route", "load", "evict", "error", "end"), seen);
        } finally {
            pool.shutdownNow();
        }
    }
}
