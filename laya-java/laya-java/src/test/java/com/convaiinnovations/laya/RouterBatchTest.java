package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.Router.Checkpoint;
import com.convaiinnovations.laya.Router.Request;
import com.convaiinnovations.laya.Router.RouteOptions;
import com.convaiinnovations.laya.hooks.Hook;
import com.convaiinnovations.laya.hooks.Hooks;
import com.convaiinnovations.laya.hooks.PredictContext;
import com.convaiinnovations.laya.infer.InferenceSession;
import com.convaiinnovations.laya.sequence.Collator;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * {@link Router#predictBatch(List)} with a stub agent per checkpoint, each on its own recording
 * session, so which checkpoint answered and how many graph calls it took are observable.
 */
class RouterBatchTest {

    private static final RouteOptions ENGLISH = RouteOptions.none().model("english");
    private static final RouteOptions MULTILINGUAL = RouteOptions.none().model("multilingual");

    @TempDir
    Path root;

    private final Map<Checkpoint, TinyCheckpoint.RecordingSession> sessions =
            new EnumMap<>(Checkpoint.class);

    @BeforeEach
    void writeCheckpoint() throws IOException {
        TinyCheckpoint.write(root, 64, 32);
    }

    /** A router whose every checkpoint is a stub agent on a session of its own. */
    private Router router() throws IOException {
        Router router = Router.builder().maxLoaded(3).build();
        for (Checkpoint checkpoint : Checkpoint.values()) {
            TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
            sessions.put(checkpoint, session);
            router.attach(checkpoint.wireName(), TinyCheckpoint.agent(root, session));
        }
        return router;
    }

    private int calls(Checkpoint checkpoint) {
        return sessions.get(checkpoint).batches.size();
    }

    private static Map<String, Question> urgent() {
        return Map.of("urgent", Question.noul("Needs a human."));
    }

    private static Map<String, Question> topic() {
        Map<String, Question> questions = new LinkedHashMap<>();
        questions.put("topic", Question.choice("Topic.", TinyCheckpoint.ordered(
                "billing", "money", "login", "access", "other", "anything else")));
        return questions;
    }

    private static List<Request> interleaved() {
        return List.of(
                Request.of("refund my card", urgent()).options(ENGLISH),
                Request.of("mi tarjeta", urgent()).options(MULTILINGUAL),
                Request.of("cannot log in", topic()).options(ENGLISH),
                Request.of("बिल गलत है", topic()).options(MULTILINGUAL),
                Request.of("charged twice", urgent()).options(ENGLISH),
                Request.of("doppelt belastet", urgent()).options(MULTILINGUAL));
    }

    @Test
    @DisplayName("results come back in request order and equal per-request predict")
    void orderPreservedAndEqualToPredict() throws IOException {
        List<Request> requests = interleaved();
        List<Prediction> batched;
        try (Router router = router()) {
            batched = router.predictBatch(requests);
        }
        assertEquals(requests.size(), batched.size());
        try (Router router = router()) {
            for (int i = 0; i < requests.size(); i++) {
                Request request = requests.get(i);
                Prediction single = router.predict(request.state(), request.questions(),
                        request.options());
                assertEquals(single, batched.get(i), "request " + i);
            }
        }
        // Distinct per state, so a swap of two results could not pass the comparison above.
        assertNotEquals(batched.get(0), batched.get(4));
    }

    @Test
    @DisplayName("an interleaved workload costs one forward pass per checkpoint and schema")
    void oneCallPerCheckpoint() throws IOException {
        try (Router router = router()) {
            router.predictBatch(List.of(
                    Request.of("a", urgent()).options(ENGLISH),
                    Request.of("b", urgent()).options(MULTILINGUAL),
                    Request.of("c", urgent()).options(ENGLISH),
                    Request.of("d", urgent()).options(MULTILINGUAL),
                    Request.of("e", urgent()).options(ENGLISH)));
            assertEquals(1, calls(Checkpoint.ENGLISH));
            assertEquals(1, calls(Checkpoint.MULTILINGUAL));
            assertEquals(0, calls(Checkpoint.TYPED_DECISIONS));
            assertEquals(3, sessions.get(Checkpoint.ENGLISH).batches.get(0).rows());
            assertEquals(2, sessions.get(Checkpoint.MULTILINGUAL).batches.get(0).rows());
        }
    }

    @Test
    @DisplayName("requests with different schemas or budgets on one checkpoint split the pass")
    void schemaAndBudgetSplit() throws IOException {
        try (Router router = router()) {
            router.predictBatch(List.of(
                    Request.of("a", urgent()).options(ENGLISH),
                    Request.of("b", topic()).options(ENGLISH),
                    Request.of("c", urgent()).options(ENGLISH).maxLen(48),
                    Request.of("d", urgent()).options(ENGLISH)));
            List<Collator.Batch> batches = sessions.get(Checkpoint.ENGLISH).batches;
            assertEquals(3, batches.size(), "urgent, topic, urgent at maxLen 48");
            assertEquals(List.of(2, 1, 1), batches.stream().map(Collator.Batch::rows).toList());
        }
    }

    @Test
    @DisplayName("a per-request budget reaches the agent")
    void budgetForwarded() throws IOException {
        String state = "x".repeat(200);
        try (Router router = router()) {
            Prediction full = router.predictBatch(List.of(
                    Request.of(state, urgent()).options(ENGLISH))).get(0);
            Prediction capped = router.predictBatch(List.of(
                    Request.of(state, urgent()).options(ENGLISH).maxLen(40))).get(0);
            assertTrue(capped.usage().inputTokens() < full.usage().inputTokens(),
                    full.usage() + " vs " + capped.usage());
        }
    }

    @Test
    @DisplayName("model pins each request, overriding detection")
    void pinForwarded() throws IOException {
        try (Router router = router()) {
            // English text, which detection would send to the english checkpoint.
            router.predictBatch(List.of("my card was charged twice", "cannot log in"), urgent(),
                    MULTILINGUAL);
            assertEquals(0, calls(Checkpoint.ENGLISH));
            assertEquals(1, calls(Checkpoint.MULTILINGUAL));
            router.predictBatch(List.of(
                    Request.of("my card was charged twice", urgent()).options(
                            RouteOptions.none().model("typed")),
                    Request.of("my card was charged twice", urgent())));
            assertEquals(1, calls(Checkpoint.TYPED_DECISIONS), "alias pin");
            assertEquals(1, calls(Checkpoint.ENGLISH), "the unpinned request routes itself");
        }
    }

    @Test
    @DisplayName("the BatchPredictor form routes each state on its own")
    void statesRouteApart() throws IOException {
        try (Router router = router()) {
            List<Prediction> out = router.predictBatch(
                    List.of("my card was charged twice", "मेरा कार्ड दो बार चार्ज हुआ"), urgent());
            assertEquals(2, out.size());
            assertEquals(1, calls(Checkpoint.ENGLISH));
            assertEquals(1, calls(Checkpoint.MULTILINGUAL));
        }
    }

    @Test
    @DisplayName("Decisions.decideBatch runs on a router and equals decide per state")
    void decideBatchOnRouter() throws IOException {
        Map<String, Object> schema = Map.of("type", "object", "properties",
                new LinkedHashMap<>(Map.of("urgent", Map.of("type", "boolean"))));
        List<String> states = List.of("my card was charged twice", "मेरा कार्ड दो बार चार्ज हुआ",
                "login broken");
        try (Router router = router()) {
            List<Decisions.Decision> batched = Decisions.decideBatch(router, states, schema);
            assertEquals(states.size(), batched.size());
            for (int i = 0; i < states.size(); i++) {
                assertEquals(Decisions.decide(router, states.get(i), schema).values(),
                        batched.get(i).values(), "state " + i);
            }
        }
        try (Router router = router()) {
            BatchPredictor pinned = (s, asked) -> router.predictBatch(s, asked, MULTILINGUAL);
            Decisions.decideBatch(pinned, states, schema);
            assertEquals(0, calls(Checkpoint.ENGLISH));
            assertEquals(1, calls(Checkpoint.MULTILINGUAL));
        }
    }

    @Test
    @DisplayName("an empty batch routes nothing and returns nothing")
    void empty() throws IOException {
        List<String> events = new ArrayList<>();
        try (Router router = router()) {
            router.hooks().addHook(new Hook() {
                @Override
                public void onRoute(PredictContext ctx) {
                    events.add("route");
                }
            });
            assertEquals(List.of(), router.predictBatch(List.<Request>of()));
            assertEquals(List.of(), events);
        }
    }

    @Test
    @DisplayName("a null request, state or questions, or a negative batch size, is refused")
    void refusals() throws IOException {
        try (Router router = router()) {
            List<Request> withNull = new ArrayList<>();
            withNull.add(Request.of("a", urgent()));
            withNull.add(null);
            IllegalArgumentException missing = assertThrows(IllegalArgumentException.class,
                    () -> router.predictBatch(withNull));
            assertTrue(missing.getMessage().contains("request 1"), missing.getMessage());
            assertThrows(IllegalArgumentException.class, () -> Request.of(null, urgent()));
            assertThrows(IllegalArgumentException.class, () -> Request.of("a", null));
            assertThrows(IllegalArgumentException.class,
                    () -> router.predictBatch(List.of(Request.of("a", urgent())), -1, false));
            assertEquals(0, calls(Checkpoint.ENGLISH));
        }
    }

    /** Records each predict event with the state it carried. */
    private static final class Trace implements Hook {

        final List<String> events = new ArrayList<>();

        @Override
        public void onPredictStart(PredictContext ctx) {
            events.add("start:" + ctx.states().get(0));
        }

        @Override
        public void onError(PredictContext ctx) {
            events.add("error:" + ctx.states().get(0));
        }

        @Override
        public void onPredictEnd(PredictContext ctx) {
            events.add("end:" + ctx.states().get(0));
        }
    }

    @Test
    @DisplayName("router hooks run per request: starts in order, ends reversed, per checkpoint")
    void hooksPerRequest() throws IOException {
        Trace trace = new Trace();
        try (Router router = router()) {
            router.hooks().addHook(trace);
            router.predictBatch(List.of(
                    Request.of("a", urgent()).options(ENGLISH),
                    Request.of("b", urgent()).options(MULTILINGUAL),
                    Request.of("c", urgent()).options(ENGLISH)));
        }
        assertEquals(List.of("start:a", "start:c", "end:c", "end:a", "start:b", "end:b"),
                trace.events);
    }

    @Test
    @DisplayName("a start hook's skip keeps its request out of the forward pass")
    void skipLeavesThePass() throws IOException {
        Prediction canned = new Prediction(Prediction.MODEL, Map.of(),
                new Usage(0, 0, 0, 0, false, List.of(), Map.of()));
        try (Router router = router()) {
            router.hooks().addHook(new Hook() {
                @Override
                public void onPredictStart(PredictContext ctx) {
                    if ("cached".equals(ctx.states().get(0))) {
                        ctx.skip(List.of(canned));
                    }
                }
            });
            List<Prediction> out = router.predictBatch(List.of(
                    Request.of("a", urgent()).options(ENGLISH),
                    Request.of("cached", urgent()).options(ENGLISH),
                    Request.of("c", urgent()).options(ENGLISH)));
            assertSame(canned, out.get(1));
            assertEquals(1, calls(Checkpoint.ENGLISH));
            assertEquals(2, sessions.get(Checkpoint.ENGLISH).batches.get(0).rows());
        }
    }

    @Test
    @DisplayName("a start hook's budget applies to its own request only")
    void hookBudgetPerRequest() throws IOException {
        try (Router router = router()) {
            router.hooks().addHook(new Hook() {
                @Override
                public void onPredictStart(PredictContext ctx) {
                    if ("b".equals(ctx.states().get(0))) {
                        ctx.maxLen(48);
                    }
                }
            });
            router.predictBatch(List.of(
                    Request.of("a", urgent()).options(ENGLISH),
                    Request.of("b", urgent()).options(ENGLISH),
                    Request.of("c", urgent()).options(ENGLISH)));
            assertEquals(List.of(2, 1), sessions.get(Checkpoint.ENGLISH).batches.stream()
                    .map(Collator.Batch::rows).toList());
        }
    }

    @Test
    @DisplayName("a failed checkpoint fails each of its requests, after earlier ones ended")
    void failureEndsEveryStartedRequest() throws IOException {
        Trace trace = new Trace();
        try (Router router = Router.builder().maxLoaded(3).build()) {
            router.attach("english", TinyCheckpoint.agent(root,
                    new TinyCheckpoint.RecordingSession()));
            router.attach("multilingual", TinyCheckpoint.agent(root, new InferenceSession() {
                @Override
                public Output run(Collator.Batch batch) {
                    throw new IllegalStateException("graph failed");
                }

                @Override
                public void close() {
                }
            }));
            router.hooks().addHook(trace);
            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> router.predictBatch(List.of(
                            Request.of("a", urgent()).options(ENGLISH),
                            Request.of("b", urgent()).options(MULTILINGUAL),
                            Request.of("c", urgent()).options(MULTILINGUAL))));
            assertEquals("graph failed", failure.getMessage());
        }
        assertEquals(List.of("start:a", "end:a", "start:b", "start:c",
                "error:c", "end:c", "error:b", "end:b"), trace.events);
    }

    @Test
    @DisplayName("a hook that leaves no result is refused, naming the request")
    void emptyResultRefused() throws IOException {
        try (Router router = router()) {
            router.hooks().addHook(new Hook() {
                @Override
                public void onPredictEnd(PredictContext ctx) {
                    ctx.results(List.of());
                }
            });
            IllegalStateException refused = assertThrows(IllegalStateException.class,
                    () -> router.predictBatch(List.of(Request.of("a", urgent()))));
            assertTrue(refused.getMessage().contains("request 0"), refused.getMessage());
        }
    }

    @Test
    @DisplayName("per-language temperatures split the pass by language and still equal predict")
    void languageSplitsOnlyWhenTempered(@TempDir Path tempered) throws IOException {
        TinyCheckpoint.write(tempered, 64, 32);
        Files.writeString(tempered.resolve("rl_agent_config.json"),
                "{\"max_len\": 64, \"head_max_len\": 32, \"temperature\": [1.0, 1.0, 1.0],"
                + " \"temperature_by_options\": {},"
                + " \"lang_temperatures\": {\"hi\": {\"temperature\": [3.0, 3.0, 3.0]}}}");
        List<Request> requests = List.of(
                Request.of("a", urgent()).options(MULTILINGUAL.lang("hi")),
                Request.of("b", urgent()).options(MULTILINGUAL.lang("en")),
                Request.of("c", urgent()).options(MULTILINGUAL.lang("hi")));
        TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
        try (Router router = Router.builder().build()) {
            router.attach("multilingual", TinyCheckpoint.agent(tempered, session));
            List<Prediction> batched = router.predictBatch(requests);
            assertEquals(2, session.batches.size(), "hi and en apart");
            for (int i = 0; i < requests.size(); i++) {
                Request request = requests.get(i);
                assertEquals(router.predict(request.state(), request.questions(),
                        request.options()), batched.get(i), "request " + i);
            }
            assertNotEquals(router.predict("a", urgent(), MULTILINGUAL.lang("en")),
                    batched.get(0), "the temperature took effect");
        }
        // Untempered, the same three share one pass.
        try (Router router = router()) {
            router.predictBatch(requests);
            assertEquals(1, calls(Checkpoint.MULTILINGUAL));
        }
    }

    /** Lifecycle events by checkpoint, predict events by state. */
    private static final class Lifecycle implements Hook {

        final List<String> events = new ArrayList<>();

        @Override
        public void onRoute(PredictContext ctx) {
            events.add("route:" + ctx.model());
        }

        @Override
        public void onLoad(PredictContext ctx) {
            events.add("load:" + ctx.model());
        }

        @Override
        public void onEvict(PredictContext ctx) {
            events.add("evict:" + ctx.model());
        }

        @Override
        public void onPredictStart(PredictContext ctx) {
            events.add("start:" + ctx.states().get(0));
        }

        @Override
        public void onError(PredictContext ctx) {
            events.add("error:" + ctx.states().get(0) + "@" + ctx.model());
        }

        @Override
        public void onPredictEnd(PredictContext ctx) {
            events.add("end:" + ctx.states().get(0));
        }
    }

    /** Builds stub agents on demand, failing for the checkpoints named. */
    private Router.AgentFactory building(Checkpoint... failing) {
        List<Checkpoint> refused = List.of(failing);
        return checkpoint -> {
            if (refused.contains(checkpoint)) {
                throw new IOException("cannot build " + checkpoint.wireName());
            }
            return TinyCheckpoint.agent(root, new TinyCheckpoint.RecordingSession());
        };
    }

    @Test
    @DisplayName("every route, then per checkpoint evict/load before its starts, as the reference")
    void lifecycleOrder() throws IOException {
        Lifecycle trace = new Lifecycle();
        try (Router router = Router.builder().agents(building()).maxLoaded(1).build()) {
            router.hooks().addHook(trace);
            router.predictBatch(List.of(
                    Request.of("a", urgent()).options(ENGLISH),
                    Request.of("b", urgent()).options(MULTILINGUAL),
                    Request.of("c", urgent()).options(ENGLISH)));
            assertEquals(List.of("route:english", "route:multilingual", "route:english",
                    "load:english", "start:a", "start:c", "end:c", "end:a",
                    "evict:english", "load:multilingual", "start:b", "end:b"), trace.events);
        }
    }

    @Test
    @DisplayName("a load failure reaches every request of its checkpoint, with no start")
    void loadFailureBeforeStart() throws IOException {
        Lifecycle trace = new Lifecycle();
        try (Router router = Router.builder().agents(building(Checkpoint.MULTILINGUAL))
                .build()) {
            router.hooks().addHook(trace);
            RuntimeException failure = assertThrows(RuntimeException.class,
                    () -> router.predictBatch(List.of(
                            Request.of("a", urgent()).options(ENGLISH),
                            Request.of("b", urgent()).options(MULTILINGUAL),
                            Request.of("c", urgent()).options(MULTILINGUAL))));
            assertTrue(String.valueOf(failure.getCause()).contains("cannot build multilingual")
                    || failure.getMessage().contains("cannot build multilingual"),
                    failure.toString());
            assertEquals(List.of("route:english", "route:multilingual", "route:multilingual",
                    "load:english", "start:a", "end:a",
                    "error:b@multilingual", "end:b", "error:c@multilingual", "end:c"),
                    trace.events);
        }
    }

    @Test
    @DisplayName("a routing failure reaches its own request before anything loads")
    void routeFailureBeforeStart() throws IOException {
        Lifecycle trace = new Lifecycle();
        try (Router router = Router.builder().agents(building()).build()) {
            router.hooks().addHook(trace);
            assertThrows(IllegalArgumentException.class, () -> router.predictBatch(List.of(
                    Request.of("a", urgent()).options(ENGLISH),
                    Request.of("bad", urgent()).options(RouteOptions.none().model("nope")),
                    Request.of("c", urgent()).options(ENGLISH))));
            assertEquals(List.of(), router.loaded());
            assertEquals(List.of("route:english", "error:bad@null", "end:bad"), trace.events);
        }
    }

    @Test
    @DisplayName("a start hook's replacement questions are the ones its request is asked")
    void hookQuestionsHonoured() throws IOException {
        try (Router router = router()) {
            router.hooks().addHook(new Hook() {
                @Override
                public void onPredictStart(PredictContext ctx) {
                    if ("b".equals(ctx.states().get(0))) {
                        ctx.questions(topic());
                    }
                }
            });
            List<Prediction> out = router.predictBatch(List.of(
                    Request.of("a", urgent()).options(ENGLISH),
                    Request.of("b", urgent()).options(ENGLISH),
                    Request.of("c", urgent()).options(ENGLISH)));
            assertEquals(List.of("urgent"), List.copyOf(out.get(0).answers().keySet()));
            assertEquals(List.of("topic"), List.copyOf(out.get(1).answers().keySet()));
            assertEquals(List.of("urgent"), List.copyOf(out.get(2).answers().keySet()));
            assertEquals(2, calls(Checkpoint.ENGLISH), "the rewritten request has its own pass");
        }
    }

    @Test
    @DisplayName("a throwing start hook fails every started request once, itself included")
    void startHookFailure() throws IOException {
        Lifecycle trace = new Lifecycle();
        try (Router router = router()) {
            router.hooks().addHook(trace);
            router.hooks().addHook(new Hook() {
                @Override
                public void onPredictStart(PredictContext ctx) {
                    if ("b".equals(ctx.states().get(0))) {
                        throw new IllegalStateException("start refused");
                    }
                }
            });
            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> router.predictBatch(List.of(
                            Request.of("a", urgent()).options(ENGLISH),
                            Request.of("b", urgent()).options(ENGLISH),
                            Request.of("c", urgent()).options(ENGLISH))));
            assertEquals("start refused", failure.getMessage());
            assertEquals(List.of("route:english", "route:english", "route:english",
                    "start:a", "start:b", "error:b@english", "end:b", "error:a@english", "end:a"),
                    trace.events);
            assertEquals(0, calls(Checkpoint.ENGLISH), "nothing reached the model");
        }
    }

    /** Counts start events, for the process-wide defaults. */
    private static final class Starts implements Hook {

        int count;

        @Override
        public void onPredictStart(PredictContext ctx) {
            count++;
        }
    }

    @Test
    @DisplayName("a process-wide default hook starts once per request on predictBatch")
    void defaultsOncePerRequestBatch() throws IOException {
        Starts starts = new Starts();
        Hooks.setDefaultHooks(List.of(starts));
        try (Router router = router()) {
            router.predictBatch(List.of(
                    Request.of("a", urgent()).options(ENGLISH),
                    Request.of("b", urgent()).options(ENGLISH),
                    Request.of("c", urgent()).options(ENGLISH)));
            assertEquals(3, starts.count, "the forward pass runs without the defaults");
            assertEquals(1, calls(Checkpoint.ENGLISH));
        } finally {
            Hooks.clearDefaultHooks();
        }
    }

    @Test
    @DisplayName("a process-wide default hook starts once on Router.predict")
    void defaultsOncePerPredict() throws IOException {
        Starts starts = new Starts();
        Starts installed = new Starts();
        Hooks.setDefaultHooks(List.of(starts));
        try (Router router = Router.builder().build()) {
            Agent agent = TinyCheckpoint.agent(root, new TinyCheckpoint.RecordingSession());
            agent.hooks().addHook(installed);
            router.attach("english", agent);
            router.predict("a", urgent(), ENGLISH);
            assertEquals(1, starts.count, "the forward pass runs without the defaults");
            assertEquals(1, installed.count, "the agent's own hooks still run");
        } finally {
            Hooks.clearDefaultHooks();
        }
    }
}
