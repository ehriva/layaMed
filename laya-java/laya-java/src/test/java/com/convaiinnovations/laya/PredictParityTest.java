package com.convaiinnovations.laya;

import static com.convaiinnovations.laya.PredictionGolden.assertAnswers;
import static com.convaiinnovations.laya.PredictionGolden.assertUsage;
import static com.convaiinnovations.laya.PredictionGolden.questionsOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

/**
 * End to end on a real graph: the Java runtime must answer what Python's {@code ONNXAgent} answered.
 *
 * <p>The only check that covers the whole stack at once -- tokenizer, sequence budgets, collation,
 * graph I/O, decode, rounding and usage -- which the fixture families cannot do individually.
 *
 * <p>This one covers the multilingual checkpoint; {@link TypedDecisionsParityTest} covers
 * {@code typed-decisions}. The comparison rules they share live in {@link PredictionGolden}.
 */
final class PredictParityTest {

    /**
     * One agent for the whole class, closed when the class is done.
     *
     * <p>Opening it per factory loaded the 1.2 GB graph twice and doubled the parity lane's
     * wall-clock for no coverage. {@code @AutoClose} ties it to the class lifecycle rather than
     * leaking the native session for the rest of the JVM.
     */
    private static Agent shared;

    @AfterAll
    static void closeAgent() {
        if (shared != null) {
            shared.close();
            shared = null;
        }
    }

    @TestFactory
    @DisplayName("single-state predictions match Python")
    @SuppressWarnings("unchecked")
    List<DynamicTest> singleMatches() throws IOException {
        Map<String, Object> golden = requireGolden();
        Map<String, Object> singles = (Map<String, Object>) golden.get("single");
        List<DynamicTest> tests = new ArrayList<>();
        Agent agent = agent(golden);
        for (Map.Entry<String, Object> entry : singles.entrySet()) {
            Map<String, Object> c = (Map<String, Object>) entry.getValue();
            tests.add(DynamicTest.dynamicTest(entry.getKey(), () -> {
                Prediction got = agent.predict(c.get("state"),
                        questionsOf((Map<String, Object>) c.get("questions")),
                        (String) c.get("lang"));
                assertEquals(c.get("model"), got.model());
                assertAnswers((Map<String, Object>) c.get("answers"), got.answers());
                assertUsage((Map<String, Object>) c.get("usage"), got.usage());
            }));
        }
        assertTrue(tests.size() >= 5, "expected the golden's cases, got " + tests.size());
        return tests;
    }

    @TestFactory
    @DisplayName("batched predictions match Python at every batch size and sort order")
    @SuppressWarnings("unchecked")
    List<DynamicTest> batchMatches() throws IOException {
        Map<String, Object> golden = requireGolden();
        Agent agent = agent(golden);
        List<DynamicTest> tests = new ArrayList<>();
        for (Object configObject : (List<Object>) golden.get("batch")) {
            Map<String, Object> config = (Map<String, Object>) configObject;
            Object size = config.get("batch_size");
            boolean sort = Boolean.TRUE.equals(config.get("sort_by_length"));
            tests.add(DynamicTest.dynamicTest(
                    "batch_size=" + (size == null ? "all" : size) + " sort=" + sort, () -> {
                        List<Prediction> got = agent.predictBatch(
                                (List<Object>) config.get("states"),
                                questionsOf((Map<String, Object>) config.get("questions")),
                                null, size == null ? 0 : ((Number) size).intValue(), sort);
                        List<Object> want = (List<Object>) config.get("results");
                        assertEquals(want.size(), got.size(), "one prediction per state");
                        for (int i = 0; i < want.size(); i++) {
                            Map<String, Object> w = (Map<String, Object>) want.get(i);
                            // Order matters: grouping is an implementation choice and the results
                            // must come back in the caller's order whatever the grouping was.
                            assertAnswers((Map<String, Object>) w.get("answers"),
                                    got.get(i).answers());
                            assertUsage((Map<String, Object>) w.get("usage"), got.get(i).usage());
                        }
                    }));
        }
        return tests;
    }

    private static Map<String, Object> requireGolden() {
        Map<String, Object> golden = Fixtures.load("predict.json");
        Assumptions.assumeFalse(golden.containsKey("skipped"),
                "predict.json was recorded without a graph: set " + Fixtures.GRAPH_ENV
                + " and re-run scripts/gen_fixtures.py");
        return golden;
    }

    /** The shared agent, opened on first use so the assumptions still skip cleanly. */
    private static synchronized Agent agent(Map<String, Object> golden) throws IOException {
        if (shared == null) {
            shared = openAgent(golden);
        }
        return shared;
    }

    private static Agent openAgent(Map<String, Object> golden) throws IOException {
        String checkpoint = (String) golden.getOrDefault("checkpoint", "multilingual");
        Path model = Fixtures.checkpoint(checkpoint);
        Assumptions.assumeTrue(model != null, Fixtures.missingCheckpoint(checkpoint));
        String graph = System.getenv(Fixtures.GRAPH_ENV);
        Assumptions.assumeTrue(graph != null && !graph.isBlank(),
                "set " + Fixtures.GRAPH_ENV + " to the exported laya.onnx this golden was recorded from");
        Path graphPath = Paths.get(graph);
        Path graphDirectory = Files.isDirectory(graphPath) ? graphPath : graphPath.getParent();
        Assumptions.assumeTrue(graphDirectory != null && Files.isDirectory(graphDirectory),
                Fixtures.GRAPH_ENV + " does not name a graph: " + graph);
        return Agent.open(model, graphDirectory);
    }
}
