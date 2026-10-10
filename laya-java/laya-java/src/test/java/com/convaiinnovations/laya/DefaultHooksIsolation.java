package com.convaiinnovations.laya;

import com.convaiinnovations.laya.hooks.Hooks;
import org.junit.jupiter.api.extension.AfterEachCallback;
import org.junit.jupiter.api.extension.Extension;
import org.junit.jupiter.api.extension.ExtensionContext;

/**
 * Clears the process-wide default hooks after EVERY test in this module.
 *
 * <p>{@link Hooks#setDefaultHooks} is process-wide, mutable and has no automatic restore: what
 * one test sets applies to every agent in the JVM until something clears it, and the pollution
 * surfaces as a failure in whichever test happens to run next. The global itself is properly
 * guarded — {@code Hooks} holds a mutex over it — so this is not a race; it is the ordinary
 * hazard of global state, and nothing about it is visible from the test that pays for it.
 *
 * <p>Two test classes already clear it in their own {@code @AfterEach}, and they still do. That
 * was the whole protection, and it was DISCIPLINE: correct today, and correct only until the
 * third class that touches the defaults forgets. This makes it structural instead. The two local
 * teardowns are deliberately kept rather than deleted, so that a class keeps its isolation even
 * if the registration below is ever switched off.
 *
 * <p>Registered by SERVICE LOADER, through
 * {@code src/test/resources/META-INF/services/org.junit.jupiter.api.extension.Extension} and the
 * {@code junit.jupiter.extensions.autodetection.enabled} property the build sets — which is what
 * makes it apply to classes that have not opted in, and therefore to the class that forgets.
 * That property is global to this module's tests, so the service file is the one place to look
 * for what it turns on, and this is the only entry in it.
 */
public final class DefaultHooksIsolation implements Extension, AfterEachCallback {

    @Override
    public void afterEach(ExtensionContext context) {
        Hooks.clearDefaultHooks();
    }
}
