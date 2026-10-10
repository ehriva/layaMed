package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * One-option questions on a real graph. The exported head takes {@code topk(2)} over the option
 * slots, so a run whose questions all have one option failed in ONNX Runtime until the collator
 * padded a masked second slot.
 */
final class OneOptionGraphTest {

    private static final String STATE = "I was charged twice for the same order, please refund me.";

    private static Agent shared;

    @AfterAll
    static void closeAgent() {
        if (shared != null) {
            shared.close();
            shared = null;
        }
    }

    @Test
    @DisplayName("a one-option choice is answered with that option at probability 1")
    void oneOptionChoice() throws IOException {
        Prediction got = agent().predict(STATE, Map.of("topic",
                Question.choiceOf("What is this about?", Map.entry("billing", "a billing issue"))));
        Answer.Choice choice = assertInstanceOf(Answer.Choice.class, got.answers().get("topic"));
        assertEquals("billing", choice.choice());
        assertEquals(Map.of("billing", 1.0), choice.probabilities());
        // What laya's eager Agent and ONNXAgent give for this question on this checkpoint.
        assertEquals(1.0, choice.actProbability(), 1e-4);
    }

    @Test
    @DisplayName("a one-level score is answered at level 0")
    void oneLevelScore() throws IOException {
        Prediction got = agent().predict(STATE, Map.of("urgency",
                Question.score("How urgent is this?", List.of("urgent"))));
        Answer.Score score = assertInstanceOf(Answer.Score.class, got.answers().get("urgency"));
        assertEquals(0.0, score.score());
        assertEquals(1, score.probabilities().size());
        assertEquals(1.0, score.probabilities().values().iterator().next());
    }

    @Test
    @DisplayName("one-option questions batched together are each answered")
    void oneOptionBatch() throws IOException {
        Map<String, Question> questions = new LinkedHashMap<>();
        questions.put("topic",
                Question.choiceOf("What is this about?", Map.entry("billing", "a billing issue")));
        questions.put("urgency", Question.score("How urgent is this?", List.of("urgent")));
        List<Prediction> got = agent().predictBatch(List.of(STATE, "Where is my parcel?"),
                questions, null, 0, false);
        assertEquals(2, got.size());
        for (Prediction prediction : got) {
            assertEquals("billing",
                    ((Answer.Choice) prediction.answers().get("topic")).choice());
            assertEquals(0.0, ((Answer.Score) prediction.answers().get("urgency")).score());
        }
    }

    private static synchronized Agent agent() throws IOException {
        if (shared == null) {
            Path model = Fixtures.checkpoint("multilingual");
            Assumptions.assumeTrue(model != null, Fixtures.missingCheckpoint("multilingual"));
            String graph = System.getenv(Fixtures.GRAPH_ENV);
            Assumptions.assumeTrue(graph != null && !graph.isBlank(),
                    "set " + Fixtures.GRAPH_ENV + " to an exported laya.onnx");
            Path graphPath = Paths.get(graph);
            Path directory = Files.isDirectory(graphPath) ? graphPath : graphPath.getParent();
            Assumptions.assumeTrue(directory != null && Files.isDirectory(directory),
                    Fixtures.GRAPH_ENV + " does not name a graph: " + graph);
            shared = Agent.open(model, directory);
        }
        return shared;
    }
}
