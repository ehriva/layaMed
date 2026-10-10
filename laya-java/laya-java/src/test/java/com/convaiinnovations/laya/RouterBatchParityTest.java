package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.Router.Checkpoint;
import com.convaiinnovations.laya.Router.Request;
import com.convaiinnovations.laya.Router.RouteOptions;
import com.convaiinnovations.laya.hooks.HookCall;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * {@link Router#predictBatch(List)} on real checkpoints equals one {@link Router#predict} per
 * request. Skipped unless the multilingual and typed-decisions checkpoints and both graphs are
 * configured; see {@link Fixtures}.
 */
final class RouterBatchParityTest {

    /** Batched rows are padded to their group's longest, so the logits may move in the last bits. */
    private static final double TOLERANCE = 1e-4;

    private static Path graph(String env) {
        String graph = System.getenv(env);
        Assumptions.assumeTrue(graph != null && !graph.isBlank(), "set " + env);
        Path path = Paths.get(graph);
        Path directory = Files.isDirectory(path) ? path : path.getParent();
        Assumptions.assumeTrue(directory != null && Files.isDirectory(directory),
                env + " does not name a graph: " + graph);
        return directory;
    }

    private static Path checkpoint(String name) {
        Path model = Fixtures.checkpoint(name);
        Assumptions.assumeTrue(model != null, Fixtures.missingCheckpoint(name));
        return model;
    }

    private static Map<String, Question> triage() {
        Map<String, Question> questions = new LinkedHashMap<>();
        questions.put("urgent", Question.noul("Does this need a human today?"));
        questions.put("topic", Question.choice("What is this about?", TinyCheckpoint.ordered(
                "billing", "charges, refunds, invoices", "access", "login or password",
                "other", "anything else")));
        return questions;
    }

    private static Map<String, Question> sentiment() {
        return Map.of("angry", Question.noul("Is the writer angry?"));
    }

    @Test
    @DisplayName("predictBatch equals per-request predict on real checkpoints")
    void batchEqualsPerRequest() throws IOException {
        Path multilingual = checkpoint("multilingual");
        Path typed = checkpoint("typed-decisions");
        Path multilingualGraph = graph(Fixtures.GRAPH_ENV);
        Path typedGraph = graph(Fixtures.TYPED_GRAPH_ENV);
        RouteOptions toMultilingual = RouteOptions.none().model("multilingual");
        RouteOptions toTyped = RouteOptions.none().model("typed-decisions");
        List<Request> requests = List.of(
                Request.of("मेरे कार्ड से दो बार पैसे कटे, कृपया जल्दी मदद करें", triage()),
                Request.of("I was charged twice for one order", triage()).options(toMultilingual),
                Request.of("No puedo entrar a mi cuenta y estoy harto", sentiment())
                        .options(toMultilingual),
                Request.of("Please refund the duplicate charge on invoice 4411", triage())
                        .options(toTyped),
                Request.of("我的账户被锁定了，无法登录", triage()),
                Request.of("The app logs me out every time, fix it now", sentiment())
                        .options(toTyped),
                Request.of("I was charged twice for one order and nobody has answered my three "
                        + "emails about it, so I am asking again", sentiment())
                        .options(toMultilingual).maxLen(32));
        Router.AgentFactory agents = checkpointName -> switch (checkpointName) {
            case MULTILINGUAL -> Agent.open(multilingual, multilingualGraph);
            case TYPED_DECISIONS -> Agent.open(typed, typedGraph);
            default -> throw new IOException("no graph for " + checkpointName);
        };
        try (Router router = Router.builder().agents(agents).maxLoaded(2).build()) {
            List<Prediction> batched = router.predictBatch(requests);
            assertEquals(requests.size(), batched.size());
            for (int i = 0; i < requests.size(); i++) {
                Request request = requests.get(i);
                Prediction single;
                if (request.maxLen() == null) {
                    single = router.predict(request.state(), request.questions(),
                            request.options());
                } else {
                    // Router.predict takes no budget, so the reference is the agent with one.
                    Checkpoint target = router.route(request.state(), request.questions(),
                            request.options()).model();
                    try (Router.Lease lease = router.lease(target)) {
                        single = lease.agent().predictBatch(List.of(request.state()),
                                request.questions(), null, 0, false,
                                HookCall.none().onStart(ctx -> ctx.maxLen(request.maxLen())))
                                .get(0);
                    }
                    assertTrue(single.usage().stateTokensDropped() > 0,
                            "the budget truncated: " + single.usage());
                }
                assertSame("request " + i, single, batched.get(i));
            }
            assertEquals(List.of(Checkpoint.MULTILINGUAL, Checkpoint.TYPED_DECISIONS),
                    new ArrayList<>(router.loaded()).stream().sorted().toList());
        }
    }

    private static void assertSame(String at, Prediction want, Prediction got) {
        assertEquals(want.model(), got.model(), at + ".model");
        assertEquals(want.usage(), got.usage(), at + ".usage");
        assertIterableEquals(want.answers().keySet(), got.answers().keySet(), at + " ids");
        for (Map.Entry<String, Answer> entry : want.answers().entrySet()) {
            String id = at + "." + entry.getKey();
            Answer w = entry.getValue();
            Answer g = got.answers().get(entry.getKey());
            assertEquals(w.type(), g.type(), id + ".type");
            assertEquals(w.confidence(), g.confidence(), TOLERANCE, id + ".confidence");
            assertEquals(w.answerConfidence(), g.answerConfidence(), TOLERANCE,
                    id + ".answer_confidence");
            assertEquals(w.actProbability(), g.actProbability(), TOLERANCE, id + ".act");
            if (w instanceof Answer.Choice wc && g instanceof Answer.Choice gc) {
                assertIterableEquals(wc.probabilities().keySet(), gc.probabilities().keySet());
                for (String option : wc.probabilities().keySet()) {
                    assertEquals(wc.probabilities().get(option), gc.probabilities().get(option),
                            TOLERANCE, id + ".p[" + option + "]");
                }
                assertEquals(wc.choice(), gc.choice(), id + ".choice");
            } else if (w instanceof Answer.Noul wn && g instanceof Answer.Noul gn) {
                assertEquals(wn.noul(), gn.noul(), TOLERANCE, id + ".noul");
            }
        }
    }
}
