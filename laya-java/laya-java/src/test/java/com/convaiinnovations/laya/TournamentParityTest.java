package com.convaiinnovations.laya;

import static com.convaiinnovations.laya.PredictionGolden.assertAnswers;
import static com.convaiinnovations.laya.PredictionGolden.assertUsage;
import static com.convaiinnovations.laya.PredictionGolden.questionsOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

/**
 * {@link Shortlist#predictTournament} on the real graph against {@code tournament_golden.json}.
 *
 * <p>Every call is compared: the groups asked (exactly, since the generator refuses a near-tie in
 * any round), each call's answers within {@link PredictionGolden}'s tolerance, and the usage. Skips
 * when the golden was recorded without a graph or the checkpoint and graph are not present.
 */
final class TournamentParityTest {

    private static Agent shared;

    @AfterAll
    static void closeAgent() {
        if (shared != null) {
            shared.close();
            shared = null;
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object value) {
        return (Map<String, Object>) value;
    }

    @SuppressWarnings("unchecked")
    private static List<Object> list(Object value) {
        return (List<Object>) value;
    }

    @TestFactory
    @DisplayName("every recorded tournament replays call for call on the real graph")
    List<DynamicTest> replays() {
        Map<String, Object> golden = Fixtures.load("tournament_golden.json");
        Assumptions.assumeFalse(golden.containsKey("skipped"),
                "tournament_golden.json was recorded without a graph: set " + Fixtures.GRAPH_ENV
                + " and re-run scripts/gen_fixtures.py");
        List<DynamicTest> tests = new ArrayList<>();
        for (Object entry : list(golden.get("cases"))) {
            Map<String, Object> row = map(entry);
            String name = (String) row.get("name");
            tests.add(DynamicTest.dynamicTest(name, () -> {
                Agent agent = agent(golden);
                List<Map<String, Question>> asked = new ArrayList<>();
                List<Prediction> returned = new ArrayList<>();
                Predictor recording = (state, questions) -> {
                    asked.add(new LinkedHashMap<>(questions));
                    Prediction out = agent.predict(state, questions);
                    returned.add(out);
                    return out;
                };
                Object groupSize = row.get("group_size");
                Map<String, Question> questions = questionsOf(choicesAsMaps(map(row.get("questions"))));
                Shortlist.Tournament result = groupSize == null
                        ? Shortlist.predictTournament(recording, row.get("state"), questions)
                        : Shortlist.predictTournament(recording, row.get("state"), questions,
                                ((Number) groupSize).intValue());

                List<Object> calls = list(row.get("calls"));
                assertEquals(calls.size(), asked.size(), name + ": calls");
                for (int c = 0; c < calls.size(); c++) {
                    Map<String, Object> call = map(calls.get(c));
                    Map<String, Object> expected = map(call.get("asked"));
                    assertEquals(new ArrayList<>(expected.keySet()),
                            new ArrayList<>(asked.get(c).keySet()), name + " call " + c + ": ids");
                    for (Map.Entry<String, Object> q : expected.entrySet()) {
                        Object labels = map(q.getValue()).get("labels");
                        if (labels != null) {
                            assertEquals(labels, asked.get(c).get(q.getKey()).labels(),
                                    name + " call " + c + "." + q.getKey() + ": labels");
                        }
                    }
                    assertAnswers(map(call.get("answers")), returned.get(c).answers());
                    assertUsage(map(call.get("usage")), returned.get(c).usage());
                }
                assertSame(returned.get(returned.size() - 1), result.prediction(),
                        name + ": answers and usage are the final call's");
                Map<String, Object> meta = map(row.get("tournament"));
                assertEquals(new ArrayList<>(meta.keySet()),
                        new ArrayList<>(result.tournament().keySet()));
                for (Map.Entry<String, Object> bracket : meta.entrySet()) {
                    Map<String, Object> want = map(bracket.getValue());
                    Shortlist.Bracket got = result.tournament().get(bracket.getKey());
                    assertEquals(want.get("labels"), got.labels(), name + ": finalists");
                    assertEquals(((Number) want.get("n")).intValue(), got.total());
                    assertEquals(((Number) want.get("rounds")).intValue(), got.rounds());
                }
            }));
        }
        assertTrue(tests.size() >= 3, "only " + tests.size() + " recorded tournaments");
        return tests;
    }

    /** List criteria on a choice become a map of null descriptions, as the reference reads them. */
    private static Map<String, Object> choicesAsMaps(Map<String, Object> defs) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : defs.entrySet()) {
            Map<String, Object> q = new LinkedHashMap<>(map(entry.getValue()));
            if ("choice".equals(q.get("type")) && q.get("criteria") instanceof List) {
                Map<String, Object> options = new LinkedHashMap<>();
                for (Object label : list(q.get("criteria"))) {
                    options.put((String) label, null);
                }
                q.put("criteria", options);
            }
            out.put(entry.getKey(), q);
        }
        return out;
    }

    private static synchronized Agent agent(Map<String, Object> golden) throws IOException {
        if (shared == null) {
            String checkpoint = (String) golden.getOrDefault("checkpoint", "multilingual");
            Path model = Fixtures.checkpoint(checkpoint);
            Assumptions.assumeTrue(model != null, Fixtures.missingCheckpoint(checkpoint));
            String graph = System.getenv(Fixtures.GRAPH_ENV);
            Assumptions.assumeTrue(graph != null && !graph.isBlank(),
                    "set " + Fixtures.GRAPH_ENV + " to the exported laya.onnx this golden was"
                    + " recorded from");
            Path graphPath = Paths.get(graph);
            Path directory = Files.isDirectory(graphPath) ? graphPath : graphPath.getParent();
            Assumptions.assumeTrue(directory != null && Files.isDirectory(directory),
                    Fixtures.GRAPH_ENV + " does not name a graph: " + graph);
            shared = Agent.open(model, directory);
        }
        return shared;
    }
}
