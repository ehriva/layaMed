package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Comparing a recorded {@code predict} result against what this port answers.
 *
 * <p>Shared by {@link PredictParityTest} and {@link TypedDecisionsParityTest} rather than copied
 * into each. The copy is the thing worth avoiding: the tolerance rules below are the gate those
 * suites are, so two copies drifting apart means one checkpoint is held to a standard the other
 * is not, and nothing reports it.
 *
 * <p><b>Tolerance.</b> Structure is compared exactly: the answer type, the chosen label, the
 * question ids and their order, the legend, and every integer in {@link Usage}. Probabilities are
 * compared within {@value #PROBABILITY_TOLERANCE}, two units in the last reported decimal, because
 * the encoder's own arithmetic is platform-sensitive and the golden is recorded on one machine. A
 * chosen label is only asserted when the top two probabilities are clear of that tolerance; when
 * they are not, the argmax is genuinely a coin toss and asserting it would be asserting noise.
 */
final class PredictionGolden {

    /** Two units in the last decimal an answer reports. */
    static final double PROBABILITY_TOLERANCE = 2e-4;

    private PredictionGolden() {
    }

    @SuppressWarnings("unchecked")
    static void assertAnswers(Map<String, Object> want, Map<String, Answer> got) {
        assertIterableEquals(want.keySet(), got.keySet(), "question ids and their order");
        for (Map.Entry<String, Object> entry : want.entrySet()) {
            Map<String, Object> w = (Map<String, Object>) entry.getValue();
            Answer g = got.get(entry.getKey());
            String at = entry.getKey();
            assertEquals(w.get("type"), g.type(), at + ".type");
            assertEquals(number(w.get("answer_confidence")), g.answerConfidence(),
                    PROBABILITY_TOLERANCE, at + ".answer_confidence");
            assertEquals(number(w.get("confidence")), g.confidence(),
                    PROBABILITY_TOLERANCE, at + ".confidence");
            assertEquals(number(((Map<String, Object>) w.get("action")).get("act_probability")),
                    g.actProbability(), PROBABILITY_TOLERANCE, at + ".act_probability");
            if (g instanceof Answer.Choice choice) {
                Map<String, Object> probabilities = (Map<String, Object>) w.get("probabilities");
                assertProbabilities(at, probabilities, choice.probabilities());
                if (decisive(probabilities)) {
                    assertEquals(w.get("choice"), choice.choice(), at + ".choice");
                }
            } else if (g instanceof Answer.Score score) {
                assertEquals(number(w.get("score")), score.score(), PROBABILITY_TOLERANCE * 4,
                        at + ".score");
                assertProbabilities(at, (Map<String, Object>) w.get("probabilities"),
                        score.probabilities());
                Map<String, Object> legend = (Map<String, Object>) w.get("legend");
                assertIterableEquals(legend.keySet(), score.legend().keySet(), at + ".legend order");
                for (Map.Entry<String, Object> level : legend.entrySet()) {
                    assertEquals(level.getValue(), score.legend().get(level.getKey()));
                }
            } else if (g instanceof Answer.Noul noul) {
                assertEquals(number(w.get("noul")), noul.noul(), PROBABILITY_TOLERANCE, at + ".noul");
            }
        }
    }

    /** Whether the top two probabilities are far enough apart for the argmax to be meaningful. */
    static boolean decisive(Map<String, Object> probabilities) {
        double best = -1.0;
        double second = -1.0;
        for (Object value : probabilities.values()) {
            double p = number(value);
            if (p > best) {
                second = best;
                best = p;
            } else if (p > second) {
                second = p;
            }
        }
        return best - second > PROBABILITY_TOLERANCE * 5;
    }

    static void assertProbabilities(String at, Map<String, Object> want, Map<String, Double> got) {
        assertIterableEquals(want.keySet(), got.keySet(), at + ".probability key order");
        for (Map.Entry<String, Object> entry : want.entrySet()) {
            assertEquals(number(entry.getValue()), got.get(entry.getKey()),
                    PROBABILITY_TOLERANCE, at + ".p[" + entry.getKey() + "]");
        }
    }

    @SuppressWarnings("unchecked")
    static void assertUsage(Map<String, Object> want, Usage got) {
        // Exact: these are counts, and a count that drifts is a bug, not float noise.
        assertEquals(((Number) want.get("input_tokens")).intValue(), got.inputTokens(),
                "usage.input_tokens");
        assertEquals(((Number) want.get("output_tokens")).intValue(), got.outputTokens());
        assertEquals(((Number) want.get("state_tokens")).intValue(), got.stateTokens(),
                "usage.state_tokens");
        assertEquals(((Number) want.get("state_tokens_dropped")).intValue(),
                got.stateTokensDropped(), "usage.state_tokens_dropped");
        assertEquals(want.get("truncated"), got.truncated(), "usage.truncated");
        assertIterableEquals((List<Object>) want.get("truncated_questions"),
                got.truncatedQuestions(), "usage.truncated_questions");
        Map<String, Object> options = (Map<String, Object>) want.get("options");
        if (options == null) {
            assertTrue(got.collapsedOptions().isEmpty(),
                    "no collapse expected, got " + got.collapsedOptions().keySet());
            return;
        }
        assertEquals(options.keySet(), got.collapsedOptions().keySet(), "usage.options");
        for (Map.Entry<String, Object> entry : options.entrySet()) {
            Map<String, Object> w = (Map<String, Object>) entry.getValue();
            Usage.CollapsedOptions g = got.collapsedOptions().get(entry.getKey());
            assertEquals(((Number) w.get("total")).intValue(), g.total());
            assertEquals(((Number) w.get("distinct")).intValue(), g.distinct());
        }
    }

    static double number(Object value) {
        return ((Number) value).doubleValue();
    }

    /** A recorded question mapping, rebuilt as the {@link Question} objects the port takes. */
    @SuppressWarnings("unchecked")
    static Map<String, Question> questionsOf(Map<String, Object> defs) {
        Map<String, Question> out = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : defs.entrySet()) {
            Map<String, Object> q = (Map<String, Object>) entry.getValue();
            String type = (String) q.get("type");
            String instructions = String.valueOf(q.get("instructions"));
            Object criteria = q.get("criteria");
            if ("choice".equals(type)) {
                out.put(entry.getKey(), Question.choice(instructions, (Map<String, Object>) criteria));
            } else if ("score".equals(type)) {
                out.put(entry.getKey(), Question.score(instructions, (List<Object>) criteria));
            } else {
                Map<String, Object> sides = criteria instanceof Map
                        ? (Map<String, Object>) criteria : Map.of();
                out.put(entry.getKey(), Question.noul(instructions, sides.get("false"),
                        sides.get("true"), null));
            }
        }
        return out;
    }
}
