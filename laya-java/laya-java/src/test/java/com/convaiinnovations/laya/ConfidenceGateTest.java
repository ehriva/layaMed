package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/**
 * {@link ConfidenceGate} against {@code fixtures/confidence_gate.json}, recorded from
 * {@code laya.confidence}.
 *
 * <p>The gate's whole contract is what it writes, so the fixture records each answer before and
 * after and this asserts the difference. Three things it pins that a port gets wrong by default:
 *
 * <ul>
 *   <li>WHICH NUMBER is read. {@code answerConfidence} first, {@code confidence} only as a
 *       fallback — different quantities on different scales, so a port reading the wrong one
 *       gates at a number that looks plausible and is not the one the calibration fitted.</li>
 *   <li>That an UNGATED call writes NOTHING. The presence of the verdict is how a caller tells
 *       "no gate ran" from "everything passed", so a port that reports a fourth "unconfigured"
 *       state breaks that distinction for every caller.</li>
 *   <li>That a scalar {@code 0.0} IS a configured gate. Nothing can fall below it, so every
 *       answer with a usable number reads {@code passed} and any stale flag is cleared.</li>
 * </ul>
 *
 * <p>Two of the reference's "unusable confidence" paths — a missing field and a bool — are
 * unrepresentable here, because {@link Answer#answerConfidence()} is a primitive double. Those
 * cases are asserted to reach {@code UNEVALUATED} through the one route that remains, NaN,
 * rather than being skipped.
 */
final class ConfidenceGateTest {

    @SuppressWarnings("unchecked")
    private static Map<String, Object> family() {
        return (Map<String, Object>) (Map<?, ?>) Fixtures.load("confidence_gate.json");
    }

    /** One recorded answer, as the typed record. {@code "__nan__"} is how the fixture spells NaN. */
    @SuppressWarnings("unchecked")
    private static Answer answerOf(Map<String, Object> recorded) {
        double confidence = number(recorded.get("confidence"));
        double answerConfidence = number(recorded.get("answer_confidence"));
        double act = number(((Map<String, Object>) recorded.get("action")).get("act_probability"));
        Map<String, Double> probabilities = new LinkedHashMap<>();
        ((Map<String, Object>) recorded.get("probabilities"))
                .forEach((k, v) -> probabilities.put(k, number(v)));
        switch ((String) recorded.get("type")) {
            case "noul":
                return new Answer.Noul(number(recorded.get("noul")), confidence,
                        answerConfidence, act);
            case "score": {
                Map<String, String> legend = new LinkedHashMap<>();
                ((Map<String, Object>) recorded.get("legend"))
                        .forEach((k, v) -> legend.put(k, (String) v));
                return new Answer.Score(number(recorded.get("score")), legend, probabilities,
                        confidence, answerConfidence, act);
            }
            default:
                return new Answer.Choice((String) recorded.get("choice"), probabilities,
                        confidence, answerConfidence, act);
        }
    }

    /**
     * A recorded number.
     *
     * <p>An ABSENT field becomes NaN, not zero. The reference treats "no usable confidence" and
     * "a confidence of 0.0" as different outcomes — one is {@code unevaluated} and the other is
     * {@code abstained} — and a port that defaulted the absent case to 0.0 would turn every
     * answer the gate could not evaluate into one it rejected.
     */
    private static double number(Object value) {
        if (value == null || "__nan__".equals(value) || value instanceof Boolean) {
            // Absent, NaN and a BOOL all mean "no usable confidence". The reference rejects a
            // bool explicitly, because `isinstance(True, int)` is true in Python and `True`
            // would otherwise gate as 1.0 -- an answer that always clears every threshold. A
            // primitive double cannot hold one, so NaN is the nearest thing Java can express
            // and the outcome it must produce is the same: UNEVALUATED.
            //
            // Absent is NOT zero. "No usable confidence" and "a confidence of 0.0" are
            // different outcomes -- unevaluated against abstained -- and defaulting the first
            // to zero would turn every answer the gate could not read into one it rejected.
            return Double.NaN;
        }
        return ((Number) value).doubleValue();
    }

    @TestFactory
    @DisplayName("every recorded gate outcome matches, including the no-op")
    @SuppressWarnings("unchecked")
    List<DynamicTest> recordedOutcomesMatch() {
        List<DynamicTest> tests = new ArrayList<>();
        int ungated = 0;
        int unevaluated = 0;
        for (Object raw : (List<Object>) family().get("cases")) {
            Map<String, Object> one = (Map<String, Object>) raw;
            Object threshold = one.get("min_confidence");
            if (threshold == null) {
                ungated++;
            }
            Map<String, Object> after = (Map<String, Object>) one.get("after");
            for (Object a : after.values()) {
                if ("unevaluated".equals(((Map<String, Object>) a).get("abstention"))) {
                    unevaluated++;
                }
            }
            tests.add(dynamicTest((String) one.get("case"), () -> {
                Map<String, Answer> answers = new LinkedHashMap<>();
                ((Map<String, Object>) one.get("before")).forEach(
                        (qid, value) -> answers.put(qid, answerOf((Map<String, Object>) value)));

                Optional<Map<String, ConfidenceGate.Gated>> report;
                if (threshold == null) {
                    report = ConfidenceGate.apply(answers, (Double) null);
                } else if (threshold instanceof Map) {
                    report = ConfidenceGate.apply(answers,
                            (Map<String, ? extends Number>) threshold);
                } else {
                    report = ConfidenceGate.apply(answers, ((Number) threshold).doubleValue());
                }

                if (threshold == null) {
                    // The whole no-op contract: not an empty report, NO report. The presence of
                    // the verdict is what tells a caller the gate ran.
                    assertTrue(report.isEmpty(),
                            "an ungated call must produce no verdict at all");
                    for (Object value : after.values()) {
                        Map<String, Object> a = (Map<String, Object>) value;
                        assertFalse(a.containsKey("abstention"),
                                "the reference wrote nothing, so nothing may be reported");
                        assertFalse(a.containsKey("abstention_threshold"), "nor a threshold");
                    }
                    return;
                }

                Map<String, ConfidenceGate.Gated> got = report.orElseThrow();
                assertEquals(after.keySet(), got.keySet(), "one verdict per answer");
                for (Map.Entry<String, Object> expected : after.entrySet()) {
                    Map<String, Object> a = (Map<String, Object>) expected.getValue();
                    ConfidenceGate.Gated verdict = got.get(expected.getKey());
                    String id = one.get("case") + "/" + expected.getKey();

                    assertEquals(a.get("abstention"), verdict.abstention().wireName(),
                            id + " abstention state");
                    assertEquals(number(a.get("abstention_threshold")), verdict.threshold(),
                            1e-12, id + " echoed threshold");
                    // `low_confidence` is ABSENT in the reference when it does not apply, which
                    // is false here -- and must not be true merely because the answer abstained
                    // for want of a number.
                    assertEquals(Boolean.TRUE.equals(a.get("low_confidence")),
                            verdict.lowConfidence(), id + " low_confidence flag");
                }
            }));
        }
        final int noop = ungated;
        final int unknown = unevaluated;
        assertTrue(noop >= 2, () -> "the no-op contract needs cases and has " + noop);
        assertTrue(unknown >= 3,
                () -> "UNEVALUATED is the state a boolean cannot express and only " + unknown
                        + " case(s) reach it");
        return tests;
    }

    @TestFactory
    @DisplayName("every threshold the reference refuses is refused here")
    @SuppressWarnings("unchecked")
    List<DynamicTest> refusalsMatch() {
        List<DynamicTest> tests = new ArrayList<>();
        for (Object raw : (List<Object>) family().get("refusals")) {
            Map<String, Object> one = (Map<String, Object>) raw;
            String input = (String) one.get("input");
            boolean refused = one.get("error") != null;
            // Only the cases Java can express: a string threshold and a non-string map key are
            // compile errors here, not runtime ones, which is a stronger guarantee than the
            // reference's and is why they are skipped rather than asserted.
            Double scalar = switch (input) {
                case "1.5" -> 1.5;
                case "-0.1" -> -0.1;
                case "nan" -> Double.NaN;
                default -> null;
            };
            if (scalar == null) {
                continue;
            }
            tests.add(dynamicTest("scalar " + input, () -> {
                assertTrue(refused, "the reference refuses " + input);
                assertThrows(IllegalArgumentException.class,
                        () -> ConfidenceGate.checkMinConfidence(scalar),
                        input + " must be refused");
            }));
        }
        tests.add(dynamicTest("an empty map", () ->
                assertThrows(IllegalArgumentException.class,
                        () -> ConfidenceGate.checkMinConfidenceMap(Map.of()),
                        "an empty map is not a gate")));
        tests.add(dynamicTest("a map with an out-of-range value", () ->
                assertThrows(IllegalArgumentException.class,
                        () -> ConfidenceGate.checkMinConfidenceMap(Map.of("choice:2", 2.0)),
                        "a value outside [0,1] is refused wherever it appears")));
        return tests;
    }

    @Test
    @DisplayName("the option bucket is the reference's, including the noul fallback")
    void buckets() {
        assertEquals(Optional.of("choice:2"), ConfidenceGate.optionBucket(
                choice(2)), "two options");
        assertEquals(Optional.of("choice:3-5"), ConfidenceGate.optionBucket(choice(4)));
        assertEquals(Optional.of("choice:3-5"), ConfidenceGate.optionBucket(choice(5)),
                "5 is the top of the 3-5 band, not the bottom of the next");
        assertEquals(Optional.of("choice:6-10"), ConfidenceGate.optionBucket(choice(6)));
        assertEquals(Optional.of("choice:11+"), ConfidenceGate.optionBucket(choice(12)));
        // A noul carries no probability map at all, and the reference buckets it as 2 rather
        // than refusing -- so a per-bucket map can name `noul:2` and have it apply.
        assertEquals(Optional.of("noul:2"),
                ConfidenceGate.optionBucket(new Answer.Noul(0.6, 0.6, 0.6, 0.5)));
    }

    @Test
    @DisplayName("an unnamed bucket with no default gates at zero, so it never abstains")
    void unnamedBucketNeverAbstainsBySurprise() {
        Answer twelve = choice(12);
        assertEquals(0.0, ConfidenceGate.resolve(twelve, Map.of("choice:2", 0.9)), 1e-12,
                "an unconfigured bucket must not inherit another bucket's threshold");
        assertEquals(0.95, ConfidenceGate.resolve(twelve,
                Map.of("choice:2", 0.9, "default", 0.95)), 1e-12, "...but a default applies");
    }

    @Test
    @DisplayName("gateConfidence prefers the calibrated number and falls back to entropy")
    void whichNumberIsRead() {
        Answer both = new Answer.Choice("a", Map.of("a", 0.9, "b", 0.1), 0.1, 0.9, 0.5);
        assertEquals(0.9, ConfidenceGate.gateConfidence(both).orElseThrow(), 1e-12,
                "answerConfidence wins; the two differ here so a port reading the wrong one "
                + "gates at 0.1 instead of 0.9");
        Answer onlyEntropy = new Answer.Choice("a", Map.of("a", 0.9, "b", 0.1),
                0.3, Double.NaN, 0.5);
        assertEquals(0.3, ConfidenceGate.gateConfidence(onlyEntropy).orElseThrow(), 1e-12,
                "an answer carrying only the older field is still gated, not silently passed");
        Answer neither = new Answer.Choice("a", Map.of("a", 0.9, "b", 0.1),
                Double.NaN, Double.NaN, 0.5);
        assertTrue(ConfidenceGate.gateConfidence(neither).isEmpty(),
                "no usable number is empty, never zero -- zero would read as a real confidence "
                + "that fails every threshold");
    }

    private static Answer choice(int options) {
        Map<String, Double> probabilities = new LinkedHashMap<>();
        for (int i = 0; i < options; i++) {
            probabilities.put(Integer.toString(i), 1.0 / options);
        }
        return new Answer.Choice("0", probabilities, 0.5, 0.7, 0.5);
    }
}
