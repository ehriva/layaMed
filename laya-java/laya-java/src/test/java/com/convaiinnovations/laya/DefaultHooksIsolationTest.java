package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.convaiinnovations.laya.hooks.Hook;
import com.convaiinnovations.laya.hooks.Hooks;
import com.convaiinnovations.laya.hooks.PredictContext;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;

/**
 * That {@link DefaultHooksIsolation} is actually registered, from a class that opts into nothing.
 *
 * <p>This class deliberately has NO {@code @AfterEach} and no {@code @ExtendWith}: it is the
 * class that forgot, which is the only case the extension exists for. The first test leaves a
 * process-wide default hook installed on purpose; the second asserts it is gone. Ordered,
 * because "the next test" is the whole claim and JUnit's default method order is deterministic
 * but not specified.
 *
 * <p>Without the service-loader registration — or without the
 * {@code junit.jupiter.extensions.autodetection.enabled} property the build sets — the second
 * test fails, which is what makes the mechanism a mechanism rather than a comment.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
final class DefaultHooksIsolationTest {

    private static final Hook LEAK = new Hook() {
        @Override
        public void onPredictStart(PredictContext ctx) {
            // Never dispatched; this hook exists to be left behind.
        }
    };

    @Test
    @Order(1)
    @DisplayName("a test that installs a process-wide default and does not clean up after itself")
    void leavesADefaultInstalled() {
        Hooks.addDefaultHook(LEAK);
        assertEquals(List.of(LEAK), Hooks.defaultHooks(), "the default really was installed");
    }

    @Test
    @Order(2)
    @DisplayName("...does not take the next test with it")
    void theNextTestStartsClean() {
        assertEquals(List.of(), Hooks.defaultHooks(),
                "the previous test left a process-wide default hook installed and nothing in "
                + "this class clears it, so DefaultHooksIsolation is not registered -- every "
                + "test in the module after one that forgets is now running with a hook it "
                + "never asked for");
    }
}
