package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import com.convaiinnovations.laya.hooks.Hook;
import com.convaiinnovations.laya.hooks.HookCall;
import com.convaiinnovations.laya.hooks.HookRegistry;
import com.convaiinnovations.laya.hooks.Hooks;
import com.convaiinnovations.laya.hooks.PredictContext;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/**
 * The hook model against {@code fixtures/hooks.json}, recorded from {@code laya.hooks}.
 *
 * <p>Hooks are the one place a caller's own code runs inside a prediction, so almost everything
 * about them is observable and almost none of it is derivable from the answers. The fixture
 * records the chain as DATA — a name, the events it implements, and a behaviour string per event
 * — so the identical chain is rebuilt here rather than described twice; see {@link HookStubs}.
 *
 * <p>What a port gets wrong by default, and what each section therefore pins:
 *
 * <ul>
 *   <li>ORDER. Defaults, then installed, then per-call, with the two convenience callbacks after
 *       the hook objects. Every order runs every hook, so nothing fails — except that a tracer
 *       installed to watch what a per-call hook did now runs first and sees nothing.</li>
 *   <li>{@code skip} does not stop the chain. A port that returns early from dispatch passes
 *       every single-hook case and loses the contract that a later hook can overrule.</li>
 *   <li>A failing hook is ATTACHED to the failure it observed, never substituted for it. On the
 *       JVM that is {@code addSuppressed} where the reference uses {@code __context__}, and the
 *       thing that must not change is which exception the caller catches.</li>
 *   <li>Where a {@code finally} would be wrong. Java's replaces the pending exception, so the
 *       end-hook stage is written out longhand; the two "inference failed AND a hook failed"
 *       cases are what prove it.</li>
 * </ul>
 *
 * <p>Three of the reference's refusals are unrepresentable once {@link Hook} is a type — a class
 * rather than an instance, an object implementing none of the events, and an event attribute
 * that is not callable. They are asserted to still be refusals ON THE PYTHON SIDE rather than
 * skipped, so that a reference which stopped refusing them would be caught here instead of
 * leaving the port's javadoc quietly wrong.
 */
final class HooksTest {

    @SuppressWarnings("unchecked")
    private static Map<String, Object> family() {
        return (Map<String, Object>) (Map<?, ?>) Fixtures.load("hooks.json");
    }

    /**
     * One recorded section, which may never be EMPTY.
     *
     * <p>JUnit does not fail a {@code @TestFactory} that returns no tests, so a section that
     * emptied — a renamed key, a generator branch that stopped recording, a bad merge — produced
     * a factory with nothing in it and a green suite.
     *
     * <p>The count floor in CI is the other half of that guard and it is not enough on its own.
     * Measured with this check deleted and {@code as_sequence} — 5 cases, the smallest section —
     * emptied: Gradle still exits 0, 812 of 831 tests run, and the floor that was in place when
     * the section was last recorded, 794, does not notice at all. A floor only ever notices
     * because it sits within a few tests of the total, which makes WHICH lane it is in decide
     * whether it notices: the {@code build} lane catches that emptying by one test, and a floor
     * shared with {@code jdk-matrix}, which runs one test more because
     * {@code TestJvmVersionTest} is enabled only there, would have landed exactly ON it and
     * exited 0. The two lanes now carry floors one apart for that reason, and this check is the
     * per-section half that does not depend on the arithmetic at all. It names the section.
     */
    @SuppressWarnings("unchecked")
    private static List<Object> section(String name) {
        Object recorded = family().get(name);
        assertNotNull(recorded, "hooks.json has no section called " + name
                + "; this factory would otherwise register no tests at all");
        List<Object> cases = (List<Object>) recorded;
        assertFalse(cases.isEmpty(), "hooks.json section " + name + " is EMPTY, so the factory "
                + "reading it registers no tests -- which JUnit does not treat as a failure");
        return cases;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object value) {
        return (Map<String, Object>) value;
    }

    @SuppressWarnings("unchecked")
    private static List<Object> list(Object value) {
        return (List<Object>) value;
    }

    private static List<String> strings(Object value) {
        List<String> out = new ArrayList<>();
        if (value != null) {
            for (Object item : list(value)) {
                out.add((String) item);
            }
        }
        return out;
    }

    private static Integer integer(Object value) {
        return value == null ? null : ((Number) value).intValue();
    }

    @AfterEach
    void noDefaultsLeakIntoTheNextTest() {
        // Process-wide state, so a case that set it and failed must not take the next one with
        // it. Clearing in teardown rather than only in each case is the difference between one
        // red test and a cascade nobody can attribute.
        Hooks.clearDefaultHooks();
    }

    // -- the events ---------------------------------------------------------

    @Test
    @DisplayName("the six events, and their order, are the reference's")
    void eventsMatch() {
        List<String> recorded = strings(family().get("events"));
        List<String> ported = new ArrayList<>();
        for (Hooks.Event event : Hooks.Event.values()) {
            ported.add(event.wireName());
        }
        assertEquals(recorded, ported,
                "the enum's declaration order is the reference's HOOK_EVENTS order");
    }

    // -- composition --------------------------------------------------------

    /**
     * The label the fixture gives one entry of a composed list.
     *
     * <p>A hook object is its name. A convenience callback has been wrapped by
     * {@link Hooks#onPredictStart}, which is opaque by design, so it is identified by the
     * {@code toString} that method documents.
     */
    private static String label(Hook hook) {
        if (hook instanceof HookStubs.StubHook stub) {
            return stub.name();
        }
        String text = hook.toString();
        if (text.startsWith("onPredictStart(")) {
            return "start:" + text.substring("onPredictStart(".length(), text.length() - 1);
        }
        if (text.startsWith("onPredictEnd(")) {
            return "end:" + text.substring("onPredictEnd(".length(), text.length() - 1);
        }
        return text;
    }

    /** A callback whose {@code toString} is its name, so a composed adapter can be identified. */
    private static Consumer<PredictContext> callback(String name) {
        return new Consumer<>() {
            @Override
            public void accept(PredictContext ctx) {
                // Composition order is the subject here; what the callback does is not.
            }

            @Override
            public String toString() {
                return name;
            }
        };
    }

    @TestFactory
    @DisplayName("defaults, then installed, then per-call, with callbacks last")
    @SuppressWarnings("try")   // javac reports the unreferenced resource; see withoutDefaultHooks
    List<DynamicTest> compositionOrderMatches() {
        List<DynamicTest> tests = new ArrayList<>();
        for (Object raw : section("composition")) {
            Map<String, Object> one = map(raw);
            tests.add(dynamicTest((String) one.get("case"), () -> {
                List<Object> log = new ArrayList<>();
                try {
                    List<Hook> defaults = new ArrayList<>();
                    for (String name : strings(one.get("defaults"))) {
                        defaults.add(HookStubs.hook(log, name));
                    }
                    Hooks.setDefaultHooks(defaults);

                    List<Hook> installed = new ArrayList<>();
                    for (String name : strings(one.get("installed"))) {
                        installed.add(HookStubs.hook(log, name));
                    }
                    HookCall call = HookCall.none();
                    for (String name : strings(one.get("hooks"))) {
                        call = call.andThen(HookStubs.hook(log, name));
                    }
                    for (String name : strings(one.get("on_predict_start"))) {
                        call = call.onStart(callback(name));
                    }
                    for (String name : strings(one.get("on_predict_end"))) {
                        call = call.onEnd(callback(name));
                    }

                    List<Hook> composed;
                    if (Boolean.TRUE.equals(one.get("skip_defaults"))) {
                        try (Hooks.DefaultsScope ignored = Hooks.withoutDefaultHooks()) {
                            composed = Hooks.compose(installed, call);
                        }
                    } else {
                        composed = Hooks.compose(installed, call);
                    }
                    List<String> order = new ArrayList<>();
                    for (Hook hook : composed) {
                        order.add(label(hook));
                    }
                    assertEquals(strings(one.get("order")), order, "composition order");
                } finally {
                    Hooks.clearDefaultHooks();
                }
            }));
        }
        return tests;
    }

    @Test
    @DisplayName("the process-wide defaults are set, added to, read as a copy and cleared")
    void defaultHooksApiMatches() {
        Map<String, Object> recorded = map(family().get("default_hooks_api"));
        List<Object> log = new ArrayList<>();
        Hooks.setDefaultHooks(List.of(HookStubs.hook(log, "probe")));
        assertEquals(strings(recorded.get("after_set")), labels(Hooks.defaultHooks()));

        List<Hook> borrowed = Hooks.defaultHooks();
        borrowed.add(HookStubs.hook(log, "intruder"));
        assertEquals(strings(recorded.get("mutating_the_returned_list_changes_nothing")),
                labels(Hooks.defaultHooks()),
                "the read hands back a copy, so editing it edits a list of your own");

        Hooks.addDefaultHook(HookStubs.hook(log, "added"));
        assertEquals(strings(recorded.get("after_add")), labels(Hooks.defaultHooks()));

        Hooks.clearDefaultHooks();
        assertEquals(strings(recorded.get("after_clear")), labels(Hooks.defaultHooks()));
    }

    private static List<String> labels(List<Hook> hooks) {
        List<String> out = new ArrayList<>();
        for (Hook hook : hooks) {
            out.add(label(hook));
        }
        return out;
    }

    @Test
    @DisplayName("withoutDefaultHooks suppresses the defaults for a call without unsetting them")
    @SuppressWarnings("try")
    void withoutDefaultHooksIsAScopeNotAnEdit() {
        List<Object> log = new ArrayList<>();
        Hook probe = HookStubs.hook(log, "probe");
        Hooks.setDefaultHooks(List.of(probe));
        try (Hooks.DefaultsScope ignored = Hooks.withoutDefaultHooks()) {
            assertEquals(List.of(), Hooks.compose(List.of(), HookCall.none()),
                    "the scope suppresses them for a call");
            assertEquals(List.of(probe), Hooks.defaultHooks(),
                    "...and a read still says what is installed, or the scope cannot be debugged");
        }
        assertEquals(List.of(probe), Hooks.compose(List.of(), HookCall.none()),
                "and they are back afterwards");
    }

    // -- normalise ----------------------------------------------------------

    @TestFactory
    @DisplayName("what the reference refuses to accept as a hook, it still refuses")
    List<DynamicTest> normaliseRefusalsStillRefuse() {
        List<DynamicTest> tests = new ArrayList<>();
        for (Object raw : section("normalise_refusals")) {
            Map<String, Object> one = map(raw);
            tests.add(dynamicTest((String) one.get("input"), () ->
                    // Unrepresentable on this side -- a class is not a Hook, an object with no
                    // event methods is not a Hook, and an event that is not callable cannot
                    // exist once the events are methods. Asserting the reference still refuses
                    // them is what keeps that claim true: if it stopped, this port's javadoc
                    // would be wrong and nothing else here would notice.
                    assertNotNull(one.get("error"),
                            "the reference no longer refuses this, so the type-system argument "
                            + "in Hooks.normalise is out of date")));
        }
        return tests;
    }

    @Test
    @DisplayName("a null entry is representable, so it is refused here")
    void normaliseRefusesNull() {
        List<Hook> withNull = new ArrayList<>();
        withNull.add(null);
        assertThrows(IllegalArgumentException.class, () -> Hooks.normalise(withNull, null, null));
        assertThrows(IllegalArgumentException.class, () -> Hooks.onPredictStart(null));
        assertThrows(IllegalArgumentException.class, () -> Hooks.onPredictEnd(null));
    }

    @TestFactory
    @DisplayName("one hook, several, or none all flatten the same way")
    List<DynamicTest> flatteningMatches() {
        List<DynamicTest> tests = new ArrayList<>();
        List<Object> log = new ArrayList<>();
        Hook one = HookStubs.hook(log, "one");
        Map<String, List<Hook>> inputs = new LinkedHashMap<>();
        inputs.put("none", null);
        inputs.put("one", List.of(one));
        inputs.put("list-of-two", List.of(one, one));
        inputs.put("tuple-of-three", List.of(one, one, one));
        inputs.put("empty-list", List.of());
        for (Object raw : section("as_sequence")) {
            Map<String, Object> entry = map(raw);
            String name = (String) entry.get("input");
            tests.add(dynamicTest(name, () -> {
                assertTrue(inputs.containsKey(name), "unrecorded flattening case " + name);
                assertEquals(integer(entry.get("length")).intValue(),
                        Hooks.normalise(inputs.get(name), null, null).size());
            }));
        }
        return tests;
    }

    // -- dispatch -----------------------------------------------------------

    @TestFactory
    @DisplayName("every dispatch calls what the reference called, in order, and fails where it did")
    List<DynamicTest> dispatchMatches() {
        List<DynamicTest> tests = new ArrayList<>();
        for (Object raw : section("dispatch")) {
            Map<String, Object> one = map(raw);
            tests.add(dynamicTest((String) one.get("case"), () -> {
                List<Object> log = new ArrayList<>();
                List<Hook> chain = new ArrayList<>();
                for (Object link : list(one.get("chain"))) {
                    Map<String, Object> spec = map(link);
                    chain.add(HookStubs.hook(log, (String) spec.get("name"),
                            strings(spec.get("events")), behaviourOf(spec.get("behaviour")),
                            false));
                }
                PredictContext ctx = new PredictContext(List.of("s0"),
                        HookStubs.questions(List.of("q")), null, null);
                List<String> warnings = new ArrayList<>();
                Hooks.Policy policy = new Hooks.Policy((Boolean) one.get("raise_errors"), null,
                        null, warnings::add);
                Hooks.Event event = eventOf((String) one.get("event"));

                Map<String, Object> raised = one.get("raised") == null ? null
                        : map(one.get("raised"));
                if (raised == null) {
                    Hooks.dispatch(chain, event, ctx, policy);
                } else {
                    RuntimeException problem = assertThrows(RuntimeException.class,
                            () -> Hooks.dispatch(chain, event, ctx, policy));
                    assertEquals(raised.get("message"), problem.getMessage());
                    assertEquals(javaTypeFor((String) raised.get("type")),
                            problem.getClass().getSimpleName());
                }
                assertEquals(strings(one.get("calls")), log, "which hooks ran, in order");
                assertEquals(strings(one.get("warnings")), warnings,
                        "what a swallowed failure reported");
                assertEquals(integer(one.get("results")),
                        ctx.results() == null ? null : ctx.results().size(),
                        "a skip assigns results without stopping the chain");
            }));
        }
        return tests;
    }

    private static Map<String, String> behaviourOf(Object recorded) {
        Map<String, String> out = new LinkedHashMap<>();
        if (recorded instanceof String single) {
            for (Hooks.Event event : Hooks.Event.values()) {
                out.put(event.wireName(), single);
            }
        } else if (recorded instanceof Map<?, ?> perEvent) {
            perEvent.forEach((key, value) -> out.put((String) key, (String) value));
        }
        return out;
    }

    private static Hooks.Event eventOf(String wireName) {
        for (Hooks.Event event : Hooks.Event.values()) {
            if (event.wireName().equals(wireName)) {
                return event;
            }
        }
        throw new AssertionError("no event called " + wireName);
    }

    /** The exception the reference raises, as this port spells it. */
    private static String javaTypeFor(String pythonType) {
        return switch (pythonType) {
            case "ValueError" -> "IllegalArgumentException";
            case "RuntimeError" -> "RuntimeException";
            case "TypeError" -> "IllegalArgumentException";
            default -> throw new AssertionError("unmapped reference exception " + pythonType);
        };
    }

    // -- ctx.skip -----------------------------------------------------------

    @TestFactory
    @DisplayName("skip takes one result per state, or one for the call, and nothing else")
    List<DynamicTest> skipMatches() {
        List<DynamicTest> tests = new ArrayList<>();
        for (Object raw : section("skip")) {
            Map<String, Object> one = map(raw);
            tests.add(dynamicTest((String) one.get("case"), () -> {
                int states = integer(one.get("states"));
                int count = integer(one.get("results"));
                List<Object> given = new ArrayList<>();
                for (int i = 0; i < states; i++) {
                    given.add("s" + i);
                }
                List<Prediction> answers = new ArrayList<>();
                for (int i = 0; i < count; i++) {
                    answers.add(HookStubs.result(i));
                }
                PredictContext ctx = new PredictContext(given,
                        HookStubs.questions(List.of("q")), null, null);
                if (Boolean.TRUE.equals(one.get("accepted"))) {
                    ctx.skip(answers);
                    List<Object> assigned = new ArrayList<>();
                    for (Prediction prediction : ctx.results()) {
                        assigned.add(prediction.usage().inputTokens());
                    }
                    assertEquals(list(one.get("assigned")).stream().map(HooksTest::integer)
                            .toList(), assigned);
                } else {
                    IllegalArgumentException problem = assertThrows(
                            IllegalArgumentException.class, () -> ctx.skip(answers));
                    assertEquals(one.get("error"), problem.getMessage());
                    assertNull(ctx.results(), "a refused skip leaves the call unanswered");
                }
            }));
        }
        return tests;
    }

    // -- aggregate_usage ----------------------------------------------------

    @TestFactory
    @DisplayName("usage totals over the call, including the shapes a record cannot hold")
    List<DynamicTest> usageMatches() {
        List<DynamicTest> tests = new ArrayList<>();
        for (Object raw : section("usage")) {
            Map<String, Object> one = map(raw);
            tests.add(dynamicTest((String) one.get("case"), () -> {
                List<Prediction> results = new ArrayList<>();
                for (Object block : list(one.get("results"))) {
                    Map<String, Object> usage = map(block);
                    // A missing block, a null block, a missing key and a null value are all
                    // "no number" in the reference, which coerces them to zero because a result
                    // there is a plain dict a hook may have rewritten. A Usage record holds
                    // primitive ints, so the representable equivalent of every one of them is
                    // the same: zero. A float truncates, which `int()` and a Java cast agree on.
                    results.add(new Prediction(Prediction.MODEL, Map.of(),
                            new Usage(truncate(usage.get("input_tokens")),
                                    truncate(usage.get("output_tokens")), 0, 0, false,
                                    List.of(), Map.of())));
                }
                Map<String, Object> total = map(one.get("total"));
                Hooks.Totals got = Hooks.aggregateUsage(results);
                assertEquals(integer(total.get("input_tokens")).intValue(), got.inputTokens());
                assertEquals(integer(total.get("output_tokens")).intValue(), got.outputTokens());
            }));
        }
        return tests;
    }

    private static int truncate(Object value) {
        return value instanceof Number number ? (int) number.doubleValue() : 0;
    }

    // -- validate_timeout ---------------------------------------------------

    @TestFactory
    @DisplayName("a timeout is a positive duration, is refused when non-positive, and renders as the reference renders it")
    List<DynamicTest> timeoutsMatch() {
        List<DynamicTest> tests = new ArrayList<>();
        for (Object raw : section("timeouts")) {
            Map<String, Object> one = map(raw);
            String input = (String) one.get("input");
            if (!DURATIONS.containsKey(input)) {
                if (!UNREPRESENTABLE.contains(input)) {
                    // A value was added to the reference's HOOK_TIMEOUT_VALUES and nobody said
                    // which of the two it is. It MUST NOT fall through to the unrepresentable
                    // branch by default, because that branch asserts only about the Python side
                    // and never calls validateTimeout at all -- so a representable addition
                    // would quietly stop testing Java. Measured: appending a plain `2` to
                    // HOOK_TIMEOUT_VALUES survived the whole suite before this guard existed.
                    tests.add(dynamicTest(input + " (UNRECORDED)", () ->
                            fail("hooks.json records the timeout value " + input + ", which this "
                                 + "factory maps to neither a Duration nor the unrepresentable "
                                 + "set. Add it to DURATIONS, or to UNREPRESENTABLE with the "
                                 + "reason -- leaving it out stops the Java side being tested "
                                 + "for it.")));
                    continue;
                }
                // nan, inf and -inf have no Duration, and '0.5' is a string the reference
                // coerces where Java has no such value to be handed. Unreachable, not skipped:
                // the refusal is still asserted on the reference's side.
                tests.add(dynamicTest(input + " (unrepresentable as a Duration)", () -> {
                    if (Boolean.FALSE.equals(one.get("accepted"))) {
                        assertNotNull(one.get("error"));
                    } else {
                        assertNotNull(one.get("value"));
                    }
                }));
                continue;
            }
            Duration candidate = DURATIONS.get(input);
            tests.add(dynamicTest(input, () -> {
                if (Boolean.TRUE.equals(one.get("accepted"))) {
                    assertEquals(candidate, Hooks.validateTimeout(candidate));
                    if (candidate != null) {
                        // The deadline as the overrun line spells it. `%g`, recorded from
                        // CPython, is `1` for a one-second deadline where `%s` on a double is
                        // `1.0` -- and a whole number of seconds is the commonest deadline there
                        // is, so it is the commonest form of that line.
                        assertEquals(one.get("seconds"), Hooks.seconds(candidate),
                                "the deadline in the overrun message, as the reference formats "
                                + "it with %g");
                    }
                } else {
                    assertThrows(IllegalArgumentException.class,
                            () -> Hooks.validateTimeout(candidate));
                }
            }));
        }
        return tests;
    }

    /**
     * Every recorded timeout value that HAS a {@link Duration}, including null for "no limit".
     *
     * <p>A map with an explicit null value rather than a {@code switch} returning null on its
     * default arm: the default arm made "unmapped" and "representable as no limit" the same
     * answer, which is how a representable addition could slip into the branch that tests
     * nothing. {@code containsKey} tells them apart.
     */
    private static final Map<String, Duration> DURATIONS = durations();

    private static Map<String, Duration> durations() {
        Map<String, Duration> out = new LinkedHashMap<>();
        out.put("None", null);
        out.put("1", Duration.ofSeconds(1));
        out.put("2", Duration.ofSeconds(2));
        out.put("10", Duration.ofSeconds(10));
        out.put("60", Duration.ofSeconds(60));
        out.put("1.5", Duration.ofMillis(1500));
        out.put("0.05", Duration.ofMillis(50));
        out.put("1e-07", Duration.ofNanos(100));
        out.put("0", Duration.ZERO);
        out.put("0.0", Duration.ZERO);
        out.put("-1", Duration.ofSeconds(-1));
        out.put("-0.5", Duration.ofMillis(-500));
        return out;
    }

    /** The recorded values a {@link Duration} cannot hold at all, each for a stated reason. */
    private static final List<String> UNREPRESENTABLE =
            List.of("'0.5'", "nan", "inf", "-inf");

    // -- the registry -------------------------------------------------------

    @TestFactory
    @DisplayName("installing, removing and scoping hooks leaves the list the reference leaves")
    List<DynamicTest> registryMatches() {
        List<DynamicTest> tests = new ArrayList<>();
        for (Object raw : section("registry")) {
            Map<String, Object> one = map(raw);
            tests.add(dynamicTest((String) one.get("case"), () -> {
                HookRegistry registry = new HookRegistry();
                List<Object> log = new ArrayList<>();
                Map<String, Hook> pool = new LinkedHashMap<>();
                Map<String, HookRegistry.Scope> blocks = new LinkedHashMap<>();
                for (Object rawStep : list(one.get("steps"))) {
                    Map<String, Object> step = map(rawStep);
                    String op = (String) step.get("op");
                    int colon = op.indexOf(':');
                    String verb = colon < 0 ? op : op.substring(0, colon);
                    String rest = colon < 0 ? "" : op.substring(colon + 1);
                    Object returned = null;
                    switch (verb) {
                        case "add" -> {
                            if (rest.startsWith("[")) {
                                registry.addHooks(named(pool, log, rest));
                            } else {
                                registry.addHook(named(pool, log, rest).get(0));
                            }
                        }
                        case "remove" ->
                                returned = registry.removeHook(named(pool, log, rest).get(0));
                        case "enter" -> {
                            int second = rest.indexOf(':');
                            String block = rest.substring(0, second);
                            // The reference takes several arguments, each a hook or a sequence,
                            // and flattens them in order into one list -- so one list here is
                            // the same thing, which is what the several-argument case asserts.
                            blocks.put(block, registry.hooksInstalled(
                                    named(pool, log, rest.substring(second + 1))));
                        }
                        case "exit" -> blocks.remove(rest).close();
                        default -> throw new AssertionError("unknown registry op " + op);
                    }
                    assertEquals(step.get("returned"), returned, "what " + op + " reported");
                    assertEquals(strings(step.get("hooks")), labels(registry.hooks()),
                            "the installed list after " + op);
                }
                assertEquals(strings(one.get("final")), labels(registry.hooks()));
            }));
        }
        return tests;
    }

    /** The hooks a scripted op names, one instance per name so identity is stable. */
    private static List<Hook> named(Map<String, Hook> pool, List<Object> log, String spec) {
        List<Hook> out = new ArrayList<>();
        for (String name : spec.replace("[", "").replace("]", "").split(",")) {
            if (!name.isEmpty()) {
                out.add(pool.computeIfAbsent(name, key -> HookStubs.hook(log, key)));
            }
        }
        return out;
    }

    @Test
    @DisplayName("a scope closed twice does not take someone else's copy")
    void closingAScopeTwiceIsIdempotent() {
        HookRegistry registry = new HookRegistry();
        List<Object> log = new ArrayList<>();
        Hook tracer = HookStubs.hook(log, "tracer");
        registry.addHook(tracer);
        HookRegistry.Scope scope = registry.hooksInstalled(tracer);
        assertEquals(List.of("tracer", "tracer"), labels(registry.hooks()));
        scope.close();
        scope.close();
        assertEquals(List.of("tracer"), labels(registry.hooks()),
                "the second close must not remove the copy the application installed");
    }

    // -- the whole call -----------------------------------------------------

    @TestFactory
    @DisplayName("a hooked call does what the reference's predict_batch did, end to end")
    List<DynamicTest> aroundMatches() {
        List<DynamicTest> tests = new ArrayList<>();
        for (Object raw : section("predict_batch")) {
            Map<String, Object> one = map(raw);
            tests.add(dynamicTest((String) one.get("case"), () -> runPredictCase(one)));
        }
        return tests;
    }

    private void runPredictCase(Map<String, Object> one) {
        List<Object> log = new ArrayList<>();
        try {
            Hooks.setDefaultHooks(stubs(log, one.get("defaults")));
            List<Hook> installed = stubs(log, one.get("installed"));
            HookCall call = HookCall.of(stubs(log, one.get("hooks")));

            List<Object> states = new ArrayList<>(strings(one.get("states")));
            Map<String, Question> questions = HookStubs.questions(strings(one.get("questions")));
            boolean inferenceRaises = Boolean.TRUE.equals(one.get("inference_raises"));
            Map<String, Object>[] inferenceSaw = asArray();

            Hooks.Inference inference = (hookedStates, asked, maxLen, headMaxLen) -> {
                Map<String, Object> seen = new LinkedHashMap<>();
                seen.put("states", new ArrayList<>(hookedStates));
                seen.put("questions", new ArrayList<>(asked.keySet()));
                Map<String, Object> overrides = new LinkedHashMap<>();
                if (maxLen != null) {
                    overrides.put("max_len", maxLen);
                }
                if (headMaxLen != null) {
                    overrides.put("head_max_len", headMaxLen);
                }
                seen.put("overrides", overrides);
                inferenceSaw[0] = seen;
                if (inferenceRaises) {
                    throw new RuntimeException("the graph refused this batch");
                }
                List<Prediction> out = new ArrayList<>();
                for (int i = 0; i < hookedStates.size(); i++) {
                    out.add(HookStubs.result(100 + i));
                }
                return out;
            };

            List<String> warnings = new ArrayList<>();
            Hooks.Policy policy = new Hooks.Policy((Boolean) one.get("hooks_raise"), null, null,
                    warnings::add);
            PredictContext ctx = new PredictContext(states, questions, "stub-checkpoint", null);
            List<Hook> active = Hooks.compose(installed, call);

            Map<String, Object> raised = one.get("raised") == null ? null : map(one.get("raised"));
            List<Prediction> results = null;
            if (raised == null) {
                results = Hooks.around(active, ctx, policy, inference);
            } else {
                RuntimeException problem = assertThrows(RuntimeException.class,
                        () -> Hooks.around(active, ctx, policy, inference));
                assertEquals(raised.get("message"), problem.getMessage(), "what the caller sees");
                assertEquals(javaTypeFor((String) raised.get("type")),
                        problem.getClass().getSimpleName());
                Object context = raised.get("context");
                if (context == null) {
                    assertEquals(0, problem.getSuppressed().length,
                            "nothing else failed, so nothing is attached");
                } else {
                    // The reference chains a failing error/end hook onto the real failure with
                    // `__context__`; the JVM's equivalent is a suppressed exception. What must
                    // not change is which one the caller catches.
                    assertEquals(1, problem.getSuppressed().length,
                            "a failing hook is attached to the failure it observed");
                    assertEquals(javaTypeFor((String) context),
                            problem.getSuppressed()[0].getClass().getSimpleName());
                }
            }

            assertEquals(describeInference(one.get("inference")), describe(inferenceSaw[0]),
                    "what inference was handed, or that it never ran");
            assertEquals(strings(one.get("warnings")), warnings);
            assertEquals(tokensOf(one.get("result")),
                    results == null ? null : tokens(results), "what the caller receives");
            assertObservations(list(one.get("calls")), log);
        } finally {
            Hooks.clearDefaultHooks();
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object>[] asArray() {
        // A one-slot holder so the inference lambda can report what it was handed.
        return (Map<String, Object>[]) new Map<?, ?>[1];
    }

    private static List<Hook> stubs(List<Object> log, Object recorded) {
        List<Hook> out = new ArrayList<>();
        if (recorded != null) {
            for (Object raw : list(recorded)) {
                Map<String, Object> spec = map(raw);
                out.add(HookStubs.hook(log, (String) spec.get("name"), HookStubs.Hooks0.EVENTS,
                        behaviourOf(spec.get("behaviour")), true));
            }
        }
        return out;
    }

    private static List<Object> tokens(List<Prediction> results) {
        List<Object> out = new ArrayList<>();
        for (Prediction prediction : results) {
            out.add(prediction.usage().inputTokens());
        }
        return out;
    }

    private static List<Object> tokensOf(Object recorded) {
        if (recorded == null) {
            return null;
        }
        List<Object> out = new ArrayList<>();
        for (Object item : list(recorded)) {
            out.add(integer(item));
        }
        return out;
    }

    private static String describeInference(Object recorded) {
        return recorded == null ? "never ran" : describeSeen(map(recorded));
    }

    private static String describe(Map<String, Object> seen) {
        return seen == null ? "never ran" : describeSeen(seen);
    }

    private static String describeSeen(Map<String, Object> seen) {
        Map<String, Object> overrides = map(seen.get("overrides"));
        Map<String, Object> normalised = new LinkedHashMap<>();
        overrides.forEach((key, value) -> normalised.put(key, integer(value)));
        return seen.get("states") + " " + seen.get("questions") + " " + normalised;
    }

    /** Each hook invocation, field by field, so a mismatch names the field rather than the blob. */
    private void assertObservations(List<Object> recorded, List<Object> got) {
        assertEquals(recorded.size(), got.size(), "how many hook invocations happened");
        for (int at = 0; at < recorded.size(); at++) {
            Map<String, Object> want = map(recorded.get(at));
            Map<String, Object> have = map(got.get(at));
            String where = "invocation " + at + " (" + want.get("call") + ")";
            assertEquals(want.get("call"), have.get("call"), where);
            assertEquals(strings(want.get("states")), have.get("states"), where + " states");
            assertEquals(strings(want.get("questions")), have.get("questions"),
                    where + " questions");
            assertEquals(want.get("model"), have.get("model"), where + " model");
            assertEquals(integer(want.get("max_len")), have.get("max_len"), where + " max_len");
            assertEquals(integer(want.get("head_max_len")), have.get("head_max_len"),
                    where + " head_max_len");
            assertEquals(tokensOf(want.get("results")), have.get("results"), where + " results");
            assertEquals(usageOf(want.get("usage")), have.get("usage"), where + " usage");
            assertEquals(want.get("error"), have.get("error"), where + " error");
            assertEquals(want.get("elapsed_ms_set"), have.get("elapsed_ms_set"),
                    where + " elapsed");
            assertEquals(32, ((String) have.get("run_id")).length(), where + " run id length");
            assertEquals(map(got.get(0)).get("run_id"), have.get("run_id"),
                    where + ": one run id per call, or a tracer cannot pair a start with its end");
        }
    }

    private static Map<String, Object> usageOf(Object recorded) {
        if (recorded == null) {
            return null;
        }
        Map<String, Object> want = map(recorded);
        // Long, because Hooks.Totals counts in longs -- see that record for the measured int
        // overflow. A recorded Integer would never equal what a hook actually read.
        return Map.of("input_tokens", longOf(want.get("input_tokens")),
                "output_tokens", longOf(want.get("output_tokens")));
    }

    private static Long longOf(Object value) {
        return value == null ? null : ((Number) value).longValue();
    }

    // -- the parts the reference cannot record ------------------------------

    @Test
    @DisplayName("a hook that overruns its deadline fails the call rather than holding it")
    void aSlowHookTimesOut() {
        List<Object> log = new ArrayList<>();
        // One second, not five. The thread this test abandons cannot be joined -- that is the
        // behaviour being asserted -- so the only thing that bounds it is how long the hook
        // sleeps, and the suite should not carry a thread for five seconds to prove a 50 ms
        // deadline. A 20x margin over the deadline is still not a race: `join(50)` returns at
        // 50 ms whatever the hook is doing. `noAbandonedHookThreadOutlivesTheSuite` below is
        // what makes the bound an assertion rather than an intention.
        Hook slow = HookStubs.hook(log, "slow", HookStubs.Hooks0.EVENTS,
                Map.of("on_predict_start", "sleep:1"), false);
        PredictContext ctx = new PredictContext(List.of("s0"),
                HookStubs.questions(List.of("q")), null, null);
        Hooks.Policy policy = Hooks.Policy.raising().timeout(Duration.ofMillis(50));
        long began = System.nanoTime();
        Hooks.HookTimeoutException problem = assertThrows(Hooks.HookTimeoutException.class,
                () -> Hooks.dispatch(List.of(slow), Hooks.Event.PREDICT_START, ctx, policy));
        double waited = (System.nanoTime() - began) / 1_000_000.0;
        assertTrue(waited < 4000.0, "the deadline bounded the call, not the hook: waited " + waited);
        // The hook is named by its CLASS and event, as the reference's `__qualname__` names it,
        // and the deadline is in seconds -- so the line a caller greps for is the same text on
        // both runtimes.
        assertEquals("laya: hook StubHook.on_predict_start exceeded 0.05s", problem.getMessage());
    }

    @Test
    @DisplayName("a hook under a timeout still reports its own failure, not a timeout")
    void aFailingHookUnderATimeoutStillFails() {
        List<Object> log = new ArrayList<>();
        Hook angry = HookStubs.hook(log, "angry", HookStubs.Hooks0.EVENTS,
                Map.of("on_predict_start", "raise"), false);
        PredictContext ctx = new PredictContext(List.of("s0"),
                HookStubs.questions(List.of("q")), null, null);
        IllegalArgumentException problem = assertThrows(IllegalArgumentException.class,
                () -> Hooks.dispatch(List.of(angry), Hooks.Event.PREDICT_START, ctx,
                        Hooks.Policy.raising().timeout(Duration.ofSeconds(5))));
        assertEquals("boom from angry", problem.getMessage(),
                "the hook's failure crosses back from the thread it ran on");
    }

    @Test
    @DisplayName("a non-concurrent registry serialises its hooks behind one lock")
    void serialisedHooksDoNotOverlap() throws Exception {
        HookRegistry registry = new HookRegistry().concurrent(false);
        AtomicInteger inside = new AtomicInteger();
        AtomicInteger overlaps = new AtomicInteger();
        Hook counting = new Hook() {
            @Override
            public void onPredictStart(PredictContext ctx) {
                if (inside.incrementAndGet() > 1) {
                    overlaps.incrementAndGet();
                }
                try {
                    Thread.sleep(2);
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                } finally {
                    inside.decrementAndGet();
                }
            }
        };
        registry.addHook(counting);
        Hooks.Policy policy = registry.policyFor(HookCall.none());
        assertNotNull(policy.lock(), "concurrent(false) is what installs the lock");
        List<Thread> workers = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            Thread worker = new Thread(() -> Hooks.dispatch(registry.hooks(),
                    Hooks.Event.PREDICT_START,
                    new PredictContext(List.of("s"), HookStubs.questions(List.of("q")), null,
                            null),
                    policy));
            workers.add(worker);
            worker.start();
        }
        for (Thread worker : workers) {
            worker.join();
        }
        assertEquals(0, overlaps.get(), "two calls ran the same hook at once");
    }

    @Test
    @DisplayName("the per-call policy overrides the agent's, and only where it says so")
    void perCallPolicyOverrides() {
        HookRegistry registry = new HookRegistry().raiseErrors(false)
                .timeout(Duration.ofSeconds(3));
        assertFalse(registry.policyFor(HookCall.none()).raiseErrors());
        assertEquals(Duration.ofSeconds(3), registry.policyFor(HookCall.none()).timeout());

        HookCall call = HookCall.none().raiseErrors(true);
        assertTrue(registry.policyFor(call).raiseErrors());
        assertEquals(Duration.ofSeconds(3), registry.policyFor(call).timeout(),
                "a call that overrides one knob must not reset the other");
        assertEquals(Duration.ofMillis(10),
                registry.policyFor(HookCall.none().timeout(Duration.ofMillis(10))).timeout());
        assertThrows(IllegalArgumentException.class,
                () -> HookCall.none().timeout(Duration.ZERO));
        assertSame(HookCall.none(), HookCall.none(), "the empty call is a constant");
        assertTrue(HookCall.none().isEmpty());
        assertFalse(HookCall.none().raiseErrors(true).isEmpty());
    }

    // -- the defects an adversarial review reproduced -----------------------

    @Test
    @DisplayName("a scope removes the MOST RECENT copy, which is an order and not just a count")
    @SuppressWarnings("try")
    void aScopeRemovesTheMostRecentCopy() {
        HookRegistry registry = new HookRegistry();
        List<Object> log = new ArrayList<>();
        Hook own = HookStubs.hook(log, "own");
        Hook other = HookStubs.hook(log, "other");
        registry.addHook(own);
        registry.addHook(other);
        try (HookRegistry.Scope scope = registry.hooksInstalled(own)) {
            assertEquals(List.of("own", "other", "own"), labels(registry.hooks()),
                    "the scope's copy is appended, so the caller's own copy is the EARLIER one");
        }
        // The direction of the scan is observable after all. Scanning forward takes the copy at
        // index 0 and leaves [other, own]; scanning backward takes index 2 and leaves
        // [own, other]. Same multiset, different dispatch order -- and order is the one thing
        // about a hook list this port calls a contract, because it decides whether `own` sees
        // what `other` did or the other way round.
        assertEquals(List.of("own", "other"), labels(registry.hooks()),
                "a forward scan would leave [other, own] and silently reorder two hooks the "
                + "caller installed itself");
    }

    @Test
    @DisplayName("an Error is never swallowed, whatever raiseErrors says")
    void anErrorIsNeverSwallowed() {
        List<Object> log = new ArrayList<>();
        Hook fatal = new Hook() {
            @Override
            public void onPredictStart(PredictContext ctx) {
                // Not a RuntimeException. The reference catches `Exception` and deliberately not
                // `BaseException`, so a KeyboardInterrupt or a SystemExit is never swallowed by
                // a telemetry hook; on the JVM the same rule is "RuntimeException, not Error".
                throw new StackOverflowError("the hook recursed");
            }
        };
        Hook after = HookStubs.hook(log, "after");
        PredictContext ctx = new PredictContext(List.of("s0"),
                HookStubs.questions(List.of("q")), null, null);
        List<String> reported = new ArrayList<>();
        Hooks.Policy swallowing =
                Hooks.Policy.raising().raiseErrors(false).onFailure(reported::add);

        StackOverflowError problem = assertThrows(StackOverflowError.class,
                () -> Hooks.dispatch(List.of(fatal, after), Hooks.Event.PREDICT_START, ctx,
                        swallowing),
                "raiseErrors(false) governs a RuntimeException and nothing else; an "
                + "OutOfMemoryError reported as a line of text and carried on from is the same "
                + "mistake");
        assertEquals("the hook recursed", problem.getMessage(),
                "the Error itself reaches the caller, not something wrapping it");
        assertEquals(List.of(), reported,
                "...and the failure sink never saw it, so nothing logged it as handled");
        assertEquals(List.of(), log, "...and the rest of the chain did not run");
    }

    @Test
    @DisplayName("withoutDefaultHooks nests: an inner scope closing does not cancel the outer")
    @SuppressWarnings("try")
    void withoutDefaultHooksNests() {
        List<Object> log = new ArrayList<>();
        Hook probe = HookStubs.hook(log, "probe");
        Hooks.setDefaultHooks(List.of(probe));
        try (Hooks.DefaultsScope outer = Hooks.withoutDefaultHooks()) {
            try (Hooks.DefaultsScope inner = Hooks.withoutDefaultHooks()) {
                assertEquals(List.of(), Hooks.compose(List.of(), HookCall.none()));
            }
            // The close restores the value it FOUND, which was already "suppressed". Restoring a
            // literal false here instead would read as "the scope is over" and switch the
            // defaults back on inside a scope that is still open -- which is the whole reason the
            // reference uses `_SKIP_DEFAULTS.reset(token)` rather than `set(False)`.
            assertEquals(List.of(), Hooks.compose(List.of(), HookCall.none()),
                    "the inner scope closing must not cancel the outer suppression");
        }
        assertEquals(List.of(probe), Hooks.compose(List.of(), HookCall.none()),
                "and the outer scope closing does restore them");
    }

    @Test
    @DisplayName("a hook that outran its deadline keeps running, on a daemon thread")
    void anAbandonedHookRunsOnADaemonThread() throws Exception {
        Parked parked = new Parked();
        PredictContext ctx = new PredictContext(List.of("s0"),
                HookStubs.questions(List.of("q")), null, null);
        Hooks.Policy policy = Hooks.Policy.raising().timeout(Duration.ofMillis(50));
        try {
            assertThrows(Hooks.HookTimeoutException.class, () ->
                    Hooks.dispatch(List.of(parked), Hooks.Event.PREDICT_START, ctx, policy));
            Thread runner = parked.ranOn.get();
            assertNotNull(runner, "the hook never started, so this proves nothing");
            assertEquals("laya-hook-timeout", runner.getName(),
                    "the name is what makes an abandoned hook identifiable in a thread dump");
            assertTrue(runner.isAlive(),
                    "the hook is still running -- which is the point: the deadline bounded the "
                    + "WAIT, not the hook");
            assertTrue(runner.isDaemon(),
                    "a non-daemon thread would hold the JVM open after the caller is finished, "
                    + "so a timed-out hook could stop a process from exiting at all");
        } finally {
            parked.release();
        }
    }

    @Test
    @DisplayName("a 50ms deadline cuts the wait off near 50ms, not near a second")
    void theDeadlineIsActuallyTheDeadline() throws Exception {
        Parked parked = new Parked();
        PredictContext ctx = new PredictContext(List.of("s0"),
                HookStubs.questions(List.of("q")), null, null);
        Hooks.Policy policy = Hooks.Policy.raising().timeout(Duration.ofMillis(50));
        try {
            long began = System.nanoTime();
            assertThrows(Hooks.HookTimeoutException.class, () ->
                    Hooks.dispatch(List.of(parked), Hooks.Event.PREDICT_START, ctx, policy));
            double waited = (System.nanoTime() - began) / 1_000_000.0;
            // Two bounds, both loose on purpose. The lower one catches a deadline that is
            // IGNORED -- a wait that returns before the hook could possibly have finished makes
            // the outcome of a fast hook a race, which is the reason validateTimeout refuses a
            // zero. The upper one catches a deadline that is INFLATED: nothing else in the suite
            // noticed when every deadline was multiplied by twenty, because the parked hook
            // never finishes either way and the only difference is how long the call waited.
            // 15x the deadline, not 1.2x, so a loaded machine cannot turn this red.
            assertTrue(waited >= 45.0,
                    "the wait returned after " + waited + " ms on a 50 ms deadline, which is "
                    + "before the hook could have finished");
            assertTrue(waited < 750.0,
                    "a 50 ms deadline held the call for " + waited + " ms");
        } finally {
            parked.release();
        }
    }

    @Test
    @DisplayName("being interrupted while waiting for a hook restores the flag before wrapping it")
    void interruptWhileWaitingRestoresTheFlag() throws Exception {
        Parked parked = new Parked();
        PredictContext ctx = new PredictContext(List.of("s0"),
                HookStubs.questions(List.of("q")), null, null);
        // Long enough that the deadline cannot be what ends the wait.
        Hooks.Policy policy = Hooks.Policy.raising().timeout(Duration.ofSeconds(30));
        AtomicReference<Throwable> thrown = new AtomicReference<>();
        AtomicReference<Boolean> flagAfterwards = new AtomicReference<>();
        Thread waiter = new Thread(() -> {
            try {
                Hooks.dispatch(List.of(parked), Hooks.Event.PREDICT_START, ctx, policy);
            } catch (RuntimeException problem) {
                thrown.set(problem);
                flagAfterwards.set(Thread.currentThread().isInterrupted());
            }
        }, "interrupt-while-waiting");
        try {
            waiter.start();
            assertTrue(parked.entered.await(10, TimeUnit.SECONDS),
                    "the hook must be inside its event before its waiter is interrupted");
            waiter.interrupt();
            waiter.join(10_000);
            assertNotNull(thrown.get(), "the waiter was interrupted and did not report it");
            assertEquals(IllegalStateException.class, thrown.get().getClass());
            assertTrue(thrown.get().getMessage()
                            .startsWith("interrupted while waiting for hook "),
                    "the message named: " + thrown.get().getMessage());
            // Catching InterruptedException CLEARS the flag. Re-asserting it is the only thing
            // that keeps a shutdown request alive: a caller up the stack polling
            // Thread.interrupted() would otherwise be told it was never asked to stop.
            assertTrue(Boolean.TRUE.equals(flagAfterwards.get()),
                    "the interrupt flag was swallowed along with the exception");
        } finally {
            parked.release();
        }
    }

    @Test
    @DisplayName("a hook cut off by its deadline can no longer write to the call")
    void anAbandonedHookCannotWriteToTheCall() throws Exception {
        Parked parked = new Parked(ctx -> ctx.results(List.of(HookStubs.result(999))));
        PredictContext ctx = new PredictContext(List.of("s0"),
                HookStubs.questions(List.of("q")), null, null);
        Hooks.Policy policy = Hooks.Policy.raising().timeout(Duration.ofMillis(50));
        assertThrows(Hooks.HookTimeoutException.class, () ->
                Hooks.dispatch(List.of(parked), Hooks.Event.PREDICT_START, ctx, policy));
        // The call carries on without it, which is what the deadline is for.
        ctx.results(List.of(HookStubs.result(111)));
        parked.release();

        assertNotNull(parked.refused.get(),
                "the abandoned hook's write was ACCEPTED, so a hook the call stopped waiting for "
                + "can still rewrite the answer the caller already has");
        assertEquals(IllegalStateException.class, parked.refused.get().getClass());
        // `HookStubs.result(i)` reports `10 + i` input tokens, so this is the 111 above and not
        // the 999 the abandoned hook tried to put there.
        assertEquals(tokens(List.of(HookStubs.result(111))), tokens(ctx.results()),
                "...and the call kept the answer it had moved on with");
    }

    @Test
    @DisplayName("the overrun line names the hook and spells the deadline as the reference does")
    void theOverrunLineIsGreppableAcrossBothRuntimes() throws Exception {
        Parked parked = new Parked();
        // Through the convenience wrapper, which is the shape most likely to be named in one of
        // these lines and the shape that used to report no name at all.
        Hook wrapped = Hooks.onPredictStart(parked::onPredictStart);
        PredictContext ctx = new PredictContext(List.of("s0"),
                HookStubs.questions(List.of("q")), null, null);
        try {
            Hooks.HookTimeoutException problem = assertThrows(Hooks.HookTimeoutException.class,
                    () -> Hooks.dispatch(List.of(wrapped), Hooks.Event.PREDICT_START, ctx,
                            Hooks.Policy.raising().timeout(Duration.ofSeconds(1))));
            // Two divergences in one line, both measured against CPython:
            //   "exceeded 1.0s" -- %s on a double, where the reference's %g gives "1s", and a
            //                      whole number of seconds is the common case for a deadline;
            //   "hook .on_"     -- getSimpleName() of the anonymous class this wrapper used to
            //                      return is the EMPTY string, where the reference's
            //                      __qualname__ is "_StartAdapter.on_predict_start".
            assertEquals("laya: hook _StartAdapter.on_predict_start exceeded 1s",
                    problem.getMessage());
        } finally {
            parked.release();
        }
    }

    @Test
    @DisplayName("the convenience callbacks name themselves when their failure is swallowed")
    void convenienceHooksAreNamedInTheirOwnFailureLine() {
        PredictContext ctx = new PredictContext(List.of("s0"),
                HookStubs.questions(List.of("q")), null, null);
        List<String> reported = new ArrayList<>();
        Hooks.Policy swallowing =
                Hooks.Policy.raising().raiseErrors(false).onFailure(reported::add);

        Hooks.dispatch(List.of(Hooks.onPredictStart(seen -> {
            throw new IllegalArgumentException("boom");
        })), Hooks.Event.PREDICT_START, ctx, swallowing);
        Hooks.dispatch(List.of(Hooks.onPredictEnd(seen -> {
            throw new IllegalArgumentException("bang");
        })), Hooks.Event.PREDICT_END, ctx, swallowing);

        // raiseErrors(false) exists for a telemetry hook, and this line is the ONLY trace such a
        // hook's failure leaves. An empty name makes it untraceable to the hook that produced it.
        assertEquals(List.of("laya: hook _StartAdapter.on_predict_start failed: boom",
                        "laya: hook _EndAdapter.on_predict_end failed: bang"), reported,
                "the reference names these _StartAdapter/_EndAdapter, and an anonymous Java "
                + "class names nothing");
    }

    @Test
    @DisplayName("a caller's own anonymous hook still has a name, which the reference cannot lack")
    void anAnonymousHookStillReportsAName() {
        PredictContext ctx = new PredictContext(List.of("s0"),
                HookStubs.questions(List.of("q")), null, null);
        List<String> reported = new ArrayList<>();
        Hooks.dispatch(List.of(new Hook() {
            @Override
            public void onPredictStart(PredictContext seen) {
                throw new IllegalArgumentException("boom");
            }
        }), Hooks.Event.PREDICT_START, ctx,
                Hooks.Policy.raising().raiseErrors(false).onFailure(reported::add));

        assertEquals(1, reported.size());
        // Python has no anonymous class and `__qualname__` always names something, so there is
        // no reference behaviour to match here -- only a line that must not be blank.
        assertTrue(reported.get(0).startsWith("laya: hook HooksTest$"),
                "an anonymous hook reported: " + reported.get(0));
        assertFalse(reported.get(0).contains("hook ."),
                "getSimpleName() of an anonymous class is the empty string, which left the hook "
                + "unnamed in the one line its swallowed failure produces");
    }

    @Test
    @DisplayName("usage totals do not wrap where the reference cannot wrap")
    void usageTotalsDoNotOverflow() {
        List<Prediction> huge = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            huge.add(new Prediction(Prediction.MODEL, Map.of(),
                    new Usage(1_000_000_000, 1_000_000_000, 0, 0, false, List.of(), Map.of())));
        }
        Hooks.Totals got = Hooks.aggregateUsage(huge);
        // An int accumulator gives -1294967296, which no Python int can produce. `usage` is what
        // a metering hook bills from, so a wrap there is a wrong invoice rather than a wrong log.
        assertEquals(3_000_000_000L, got.inputTokens(),
                "three billion input tokens wrapped into an int accumulator");
        assertEquals(3_000_000_000L, got.outputTokens());
    }

    @Test
    @DisplayName("a null hook is refused by name from every entry point that takes one")
    void aNullHookIsRefusedByNameEverywhere() {
        String expected = "a hooks entry must not be null";
        // Each of these wrapped the hook in List.of(...) first, so the caller got a bare NPE
        // from inside the JDK and the message below was unreachable from all four.
        assertEquals(expected, assertThrows(IllegalArgumentException.class,
                () -> Hooks.addDefaultHook(null)).getMessage());
        assertEquals(expected, assertThrows(IllegalArgumentException.class,
                () -> new HookRegistry().addHook(null)).getMessage());
        assertEquals(expected, assertThrows(IllegalArgumentException.class,
                () -> HookCall.of((Hook) null)).getMessage());
        assertEquals(expected, assertThrows(IllegalArgumentException.class,
                () -> HookCall.none().andThen(null)).getMessage());
        List<Hook> withNull = new ArrayList<>();
        withNull.add(null);
        assertEquals(expected, assertThrows(IllegalArgumentException.class,
                () -> HookCall.of(withNull)).getMessage());
    }

    @Test
    @DisplayName("a context refuses a null states or questions by name, rather than NPEing")
    void aContextRefusesNullsByName() {
        Map<String, Question> questions = HookStubs.questions(List.of("q"));
        assertEquals("states must not be null", assertThrows(IllegalArgumentException.class,
                () -> new PredictContext(null, questions, null, null)).getMessage());
        // Before the hooks were wired in, `predictBatch(List.of(), null)` returned an empty list:
        // the empty-state short-circuit ran before anything looked at the questions. It now
        // refuses, which is what the reference does for an empty batch as much as any other --
        // its own check is `questions must be a dict of question id -> definition`, reached even
        // when `states` is empty. Tolerating the null instead would answer a real batch with
        // empty answers and zero usage.
        assertEquals("questions must not be null", assertThrows(IllegalArgumentException.class,
                () -> new PredictContext(List.of(), null, null, null)).getMessage());
        assertEquals("questions must not be null", assertThrows(IllegalArgumentException.class,
                () -> new PredictContext(List.of("s0"), null, null, null)).getMessage());
    }

    /**
     * No test may leave a hook thread running past this class.
     *
     * <p>Every test here that abandons one releases it, except {@link #aSlowHookTimesOut}, which
     * cannot — not being able to is the behaviour it asserts — and which therefore sleeps for a
     * bounded second rather than the five it used to. This is what turns that bound into an
     * assertion instead of an intention: the review that found the rest of this commit observed
     * two abandoned {@code laya-hook-timeout} threads alive in the middle of a run of the
     * shipped suite, and a leak nobody counts is a leak that grows.
     */
    @AfterAll
    static void noAbandonedHookThreadOutlivesTheSuite() throws InterruptedException {
        List<Thread> lingering = new ArrayList<>();
        for (Thread thread : Thread.getAllStackTraces().keySet()) {
            if ("laya-hook-timeout".equals(thread.getName())) {
                lingering.add(thread);
            }
        }
        for (Thread thread : lingering) {
            thread.join(10_000);
        }
        List<String> stuck = new ArrayList<>();
        for (Thread thread : lingering) {
            if (thread.isAlive()) {
                stuck.add(thread.toString());
            }
        }
        assertEquals(List.of(), stuck,
                "a hook thread outlived the suite: a test abandoned it and nothing released it");
    }

    /**
     * A hook that parks inside its event until the test releases it.
     *
     * <p>A sleep would do the same job less well in both directions: it is a race against the
     * deadline at one end — the thing under test — and at the other it leaves a thread running
     * for however long it was given. This one is deterministic (the hook is provably still inside
     * {@code on_predict_start} when the deadline expires) and {@link #release} joins the thread,
     * so a test asserting that a hook CANNOT be stopped does not have to leak one to prove it.
     */
    private static final class Parked implements Hook {

        private final CountDownLatch entered = new CountDownLatch(1);
        private final CountDownLatch released = new CountDownLatch(1);
        private final AtomicReference<Thread> ranOn = new AtomicReference<>();
        private final AtomicReference<Throwable> refused = new AtomicReference<>();
        private final Consumer<PredictContext> afterRelease;

        Parked() {
            this(ctx -> {
                // Park and nothing more: the deadline is the subject, not what the hook does.
            });
        }

        Parked(Consumer<PredictContext> afterRelease) {
            this.afterRelease = afterRelease;
        }

        @Override
        public void onPredictStart(PredictContext ctx) {
            ranOn.set(Thread.currentThread());
            entered.countDown();
            try {
                released.await();
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                return;
            }
            try {
                afterRelease.accept(ctx);
            } catch (RuntimeException problem) {
                refused.set(problem);
            }
        }

        /** Lets the parked hook finish, and waits for its thread to be gone. */
        void release() throws InterruptedException {
            released.countDown();
            Thread thread = ranOn.get();
            if (thread != null && thread != Thread.currentThread()) {
                thread.join(10_000);
                if (thread.isAlive()) {
                    fail("the released hook thread did not finish");
                }
            }
        }
    }
}
