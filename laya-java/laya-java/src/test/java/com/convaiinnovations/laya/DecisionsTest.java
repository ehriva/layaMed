package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/**
 * {@link Decisions} against {@code fixtures/structured.json}, recorded from
 * {@code laya.structured}.
 *
 * <p>Four sections, because the module has four contracts and they fail independently: the
 * schema-to-questions mapping, the projection back onto the schema's values, every refusal
 * message, and the end-to-end wiring of {@code decide} / {@code decideBatch} to the confidence
 * gate.
 *
 * <p>The refusals assert the <b>message</b>, not the exception type. The message names the path
 * — {@code properties.urgency: ...} — and the path is the only part of it a caller can act on,
 * so a port that refuses the right schemas with the wrong text has not ported the contract.
 *
 * <p>Two of the reference's argument refusals are not asserted, because they are compile errors
 * here rather than runtime ones: {@code decide_batch} over a string, and {@code decide_batch}
 * with a runner that cannot batch. {@link Decisions#decideBatch} takes a {@link List} and a
 * {@link BatchPredictor}, which is a stronger guarantee than the reference's and is why those
 * two are skipped rather than reproduced.
 */
final class DecisionsTest {

    @SuppressWarnings("unchecked")
    private static Map<String, Object> family() {
        return Fixtures.load("structured.json");
    }

    @SuppressWarnings("unchecked")
    private static List<Object> section(String name) {
        return (List<Object>) family().get(name);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object value) {
        return (Map<String, Object>) value;
    }

    // ------------------------------------------------------------------ schema -> questions

    @TestFactory
    @DisplayName("every recorded schema maps to the reference's questions, in its own order")
    List<DynamicTest> schemasMap() {
        List<DynamicTest> tests = new ArrayList<>();
        int choices = 0;
        int scores = 0;
        int nouls = 0;
        for (Object raw : section("schemas")) {
            Map<String, Object> one = map(raw);
            for (Object planned : (List<?>) one.get("plan")) {
                switch ((String) map(planned).get("kind")) {
                    case "choice": choices++; break;
                    case "score": scores++; break;
                    default: nouls++; break;
                }
            }
            tests.add(dynamicTest((String) one.get("case"), () -> {
                Map<String, Object> schema = map(one.get("schema"));
                Map<String, Object> expected = map(one.get("questions"));
                Decisions.Plan plan = Decisions.plan(schema);
                Map<String, Question> questions = plan.questions();

                // Property ORDER, not just the set: the questions are asked in this order and a
                // state is re-encoded per question, so a reordering is a different request.
                assertEquals(new ArrayList<>(expected.keySet()),
                        new ArrayList<>(questions.keySet()), "question order");
                expected.forEach((name, value) -> {
                    Map<String, Object> want = map(value);
                    Map<String, Object> got = questions.get(name).spec();
                    assertEquals(want, got, name + " question");
                    // A choice's options are rendered in the criteria map's INSERTION order, so
                    // two orders are two different questions -- and Map.equals above cannot see
                    // the difference.
                    if (want.get("criteria") instanceof Map) {
                        assertEquals(new ArrayList<>(map(want.get("criteria")).keySet()),
                                new ArrayList<>(map(got.get("criteria")).keySet()),
                                name + " criteria order");
                    }
                });

                List<?> expectedPlan = (List<?>) one.get("plan");
                assertEquals(expectedPlan.size(), plan.fields().size(), "one field per property");
                for (int i = 0; i < expectedPlan.size(); i++) {
                    Map<String, Object> want = map(expectedPlan.get(i));
                    Decisions.Field got = plan.fields().get(i);
                    String id = one.get("case") + "/" + want.get("name");
                    assertEquals(want.get("name"), got.name(), id + " name");
                    assertEquals(want.get("kind"), got.kind().wireName(), id + " kind");
                    assertEquals(want.get("minimum"), got.minimum(), id + " minimum");
                    // The option VALUES never reach a question and are exactly what the
                    // projection needs, so they are asserted here rather than left to whichever
                    // projection case happens to cover them.
                    List<?> wantOptions = (List<?>) want.get("options");
                    assertEquals(wantOptions.size(), got.options().size(), id + " option count");
                    for (int o = 0; o < wantOptions.size(); o++) {
                        List<?> pair = (List<?>) wantOptions.get(o);
                        assertEquals(pair.get(0), got.options().get(o).label(), id + " label");
                        assertEquals(pair.get(1), got.options().get(o).value(), id + " value");
                    }
                }
            }));
        }
        final int c = choices;
        final int s = scores;
        final int n = nouls;
        assertTrue(c >= 10 && s >= 8 && n >= 5,
                () -> "the three question kinds are the whole mapping and this reaches "
                        + c + " choice, " + s + " score, " + n + " noul");
        return tests;
    }

    // ------------------------------------------------------------------ projection

    @TestFactory
    @DisplayName("every recorded answer projects onto the schema's own values")
    List<DynamicTest> projectionsMatch() {
        List<DynamicTest> tests = new ArrayList<>();
        int nulled = 0;
        int omitted = 0;
        for (Object raw : section("projections")) {
            Map<String, Object> one = map(raw);
            Map<String, Object> recorded = map(one.get("answers"));
            Map<String, Object> expected = map(one.get("values"));
            for (Map.Entry<String, Object> entry : expected.entrySet()) {
                if (entry.getValue() == null
                        && Boolean.TRUE.equals(map(recorded.get(entry.getKey()))
                                .get("low_confidence"))) {
                    nulled++;
                }
            }
            omitted += map(map(one.get("schema")).get("properties")).size() - expected.size();
            tests.add(dynamicTest((String) one.get("case"), () -> {
                Map<String, Answer> answers = answers(recorded);
                Decisions.Plan plan = Decisions.plan(map(one.get("schema")));
                // The reference reads `low_confidence` off the answer dict. On this side the
                // flag lives in the gate's report -- a record cannot be written into -- so the
                // recorded flags are handed over as a report. That IS the wiring under test:
                // the projection must read the gate rather than re-deriving the rule.
                Map<String, ConfidenceGate.Gated> gate = gateOf(recorded, answers);
                Map<String, Object> values = plan.project(answers, gate);

                assertEquals(expected, values, "projected values");
                // A field with no answer is ABSENT, and a flagged one is present and null. Map
                // equality above cannot tell those apart, so the key set is asserted too.
                assertEquals(new ArrayList<>(expected.keySet()), new ArrayList<>(values.keySet()),
                        "which fields are present, and in which order");
                if (gate == null) {
                    // No flags: the ungated spelling must agree, so `answersToJson` is not a
                    // second implementation of the same projection.
                    assertEquals(expected, Decisions.answersToJson(answers,
                            map(one.get("schema"))), "answersToJson agrees");
                }
            }));
        }
        final int flagged = nulled;
        final int missing = omitted;
        assertTrue(flagged >= 3, () -> "a low-confidence answer projects to null and only "
                + flagged + " case(s) reach that");
        assertTrue(missing >= 1, () -> "an unanswered field is omitted and no case reaches that");
        return tests;
    }

    // ------------------------------------------------------------------ refusals

    @TestFactory
    @DisplayName("every schema the reference refuses is refused here, with the same message")
    List<DynamicTest> refusalsMatch() {
        List<DynamicTest> tests = new ArrayList<>();
        for (Object raw : section("refusals")) {
            Map<String, Object> one = map(raw);
            tests.add(dynamicTest((String) one.get("case"), () -> {
                SchemaException refused = assertThrows(SchemaException.class,
                        () -> Decisions.questions(map(one.get("schema"))),
                        one.get("case") + " must be refused");
                assertEquals(one.get("error"), refused.getMessage(),
                        "the message names the path, which is the part a caller acts on");
            }));
        }
        assertTrue(tests.size() >= 20,
                () -> "only " + tests.size() + " refusal(s); this is the half of the mapping "
                        + "that tells a caller what to fix");
        return tests;
    }

    @TestFactory
    @DisplayName("decide takes exactly one of schema and questions")
    List<DynamicTest> argumentRefusalsMatch() {
        Map<String, Question> questions = Map.of("ok", Question.noul("Is it?"));
        Map<String, Object> schema = Map.of("type", "object");
        Predictor never = (state, asked) -> {
            throw new AssertionError("the runner must not be reached");
        };
        BatchPredictor neverBatch = (states, asked) -> {
            throw new AssertionError("the runner must not be reached");
        };
        // The ungated FOUR-argument form. Spelled with the five-argument one these four read
        // `(Double) null`, because a bare null there matches both the `Double` and the `Map`
        // threshold overload and does not compile -- and a cast is not an argument.
        Map<String, Runnable> calls = new LinkedHashMap<>();
        calls.put("decide-with-both", () -> Decisions.decide(never, "s", schema, questions));
        calls.put("decide-with-neither", () -> Decisions.decide(never, "s", null, null));
        calls.put("decide-batch-with-both", () -> Decisions.decideBatch(neverBatch, List.of("s"),
                schema, questions));
        calls.put("decide-batch-with-neither", () -> Decisions.decideBatch(neverBatch,
                List.of("s"), null, null));

        List<DynamicTest> tests = new ArrayList<>();
        int matched = 0;
        for (Object raw : section("arguments")) {
            Map<String, Object> one = map(raw);
            Runnable call = calls.get((String) one.get("case"));
            if (call == null) {
                // `decide_batch` over a string and `decide_batch` without a batching runner.
                // Both are compile errors here; see the class note.
                continue;
            }
            matched++;
            tests.add(dynamicTest((String) one.get("case"), () -> {
                IllegalArgumentException refused =
                        assertThrows(IllegalArgumentException.class, call::run);
                assertEquals(one.get("error"), refused.getMessage());
            }));
        }
        final int reached = matched;
        assertEquals(4, reached, "all four representable argument refusals must be reached");
        return tests;
    }

    // ------------------------------------------------------------------ end to end

    @TestFactory
    @DisplayName("decide and decideBatch reproduce the recorded decisions")
    List<DynamicTest> callsMatch() {
        List<DynamicTest> tests = new ArrayList<>();
        int gatedNulls = 0;
        int batched = 0;
        for (Object raw : section("calls")) {
            Map<String, Object> one = map(raw);
            boolean batch = Boolean.TRUE.equals(one.get("batch"));
            if (batch) {
                batched++;
            }
            for (Object decision : (List<?>) one.get("decisions")) {
                for (Object value : map(map(decision).get("values")).values()) {
                    if (value == null) {
                        gatedNulls++;
                    }
                }
            }
            tests.add(dynamicTest((String) one.get("case"), () -> {
                Map<String, Object> schema = map(one.get("schema"));
                Double threshold = one.get("min_confidence") == null ? null
                        : ((Number) one.get("min_confidence")).doubleValue();
                List<Prediction> scripted = new ArrayList<>();
                for (Object predicted : (List<?>) one.get("predictions")) {
                    scripted.add(new Prediction(Prediction.MODEL, answers(map(predicted)),
                            new Usage(12, 0, 9, 0, false, List.of(), Map.of())));
                }
                List<String> states = new ArrayList<>();
                for (int i = 0; i < scripted.size(); i++) {
                    states.add("state " + i);
                }

                List<Decisions.Decision> got;
                if (batch) {
                    BatchPredictor runner = (asked, questions) -> {
                        assertEquals(states, asked, "the states reach the runner in order");
                        return scripted;
                    };
                    got = Decisions.decideBatch(runner, states, schema, null, threshold);
                } else {
                    Predictor runner = (state, questions) -> {
                        assertEquals(states.get(0), state);
                        return scripted.get(0);
                    };
                    got = List.of(Decisions.decide(runner, states.get(0), schema, null,
                            threshold));
                }

                List<?> expected = (List<?>) one.get("decisions");
                assertEquals(expected.size(), got.size(), "one decision per state, in order");
                for (int i = 0; i < expected.size(); i++) {
                    Map<String, Object> want = map(expected.get(i));
                    Decisions.Decision decision = got.get(i);
                    String id = one.get("case") + "[" + i + "]";

                    assertEquals(map(want.get("values")), decision.values(), id + " values");
                    assertEquals(new ArrayList<>(map(want.get("values")).keySet()),
                            new ArrayList<>(decision.values().keySet()), id + " value order");
                    map(want.get("confidence")).forEach((name, value) ->
                            assertEquals(((Number) value).doubleValue(),
                                    decision.confidence().get(name), 1e-12, id + " confidence"));
                    map(want.get("answer_confidence")).forEach((name, value) ->
                            assertEquals(value == null ? OptionalDouble.empty()
                                            : OptionalDouble.of(((Number) value).doubleValue()),
                                    decision.answerConfidence().get(name),
                                    id + " answer_confidence"));
                    map(want.get("probabilities")).forEach((name, value) -> {
                        Map<String, Object> sides = map(value);
                        Map<String, Double> reported = decision.probabilities().get(name);
                        assertEquals(sides.keySet(), reported.keySet(), id + " probability keys");
                        sides.forEach((label, p) -> assertEquals(((Number) p).doubleValue(),
                                reported.get(label), 1e-12, id + " p(" + label + ")"));
                    });

                    // The gate's own contract, carried through: no threshold means NO report,
                    // which is how a caller tells "no gate ran" from "everything passed".
                    if (threshold == null) {
                        assertTrue(decision.gate().isEmpty(), id + " ungated writes no report");
                    } else {
                        assertEquals(decision.answers().keySet(),
                                decision.gate().orElseThrow().keySet(),
                                id + " one verdict per answer");
                    }
                }
            }));
        }
        final int nulls = gatedNulls;
        final int batches = batched;
        assertTrue(nulls >= 3, () -> "the gate nulls a field and only " + nulls
                + " recorded value(s) reach that");
        assertTrue(batches >= 2, () -> "only " + batches + " batch case(s)");
        return tests;
    }

    @Test
    @DisplayName("the limits are the reference's")
    void limitsMatch() {
        Map<String, Object> limits = map(family().get("limits"));
        assertEquals(((Number) limits.get("max_properties")).intValue(), Decisions.MAX_PROPERTIES);
        assertEquals(((Number) limits.get("max_options")).intValue(), Decisions.MAX_OPTIONS);
        assertEquals(((Number) limits.get("max_score_levels")).intValue(),
                Decisions.MAX_SCORE_LEVELS);
    }

    @Test
    @DisplayName("the explicit-questions path returns the raw answers")
    void questionsPathReturnsRawAnswers() {
        Map<String, Question> questions = new LinkedHashMap<>();
        questions.put("ok", Question.noul("Is it?"));
        Answer.Noul answer = new Answer.Noul(0.8, 0.8, 0.8, 0.2);
        Predictor runner = (state, asked) -> {
            assertEquals(questions, asked, "the questions reach the runner unchanged");
            return new Prediction(Prediction.MODEL, Map.of("ok", answer),
                    new Usage(1, 0, 1, 0, false, List.of(), Map.of()));
        };
        // Ungated explicit questions, with no threshold type to pick: the entry point that did
        // not exist, and the reason the four calls above had to be cast.
        Decisions.Decision decision = Decisions.decide(runner, "s", null, questions);
        // No schema, so nothing is projected: the reference returns `dict(answers)` and so does
        // this. The values are the answers themselves, not a schema-shaped map.
        assertEquals(Map.of("ok", answer), decision.values());
        assertTrue(decision.gate().isEmpty());
    }

    @Test
    @DisplayName("a per-bucket threshold gates each answer at its own bucket")
    void perBucketThresholds() {
        Map<String, Object> schema = Map.of("type", "object", "properties",
                new LinkedHashMap<>(Map.of("ok", Map.of("type", "boolean"))));
        Answer.Noul answer = new Answer.Noul(0.9, 0.6, 0.6, 0.2);
        Predictor runner = (state, asked) -> new Prediction(Prediction.MODEL,
                Map.of("ok", answer), new Usage(1, 0, 1, 0, false, List.of(), Map.of()));

        Decisions.Decision abstained = Decisions.decide(runner, "s", schema, null,
                Map.of("noul:2", 0.8));
        assertTrue(abstained.values().containsKey("ok"), "the field is present...");
        assertNull(abstained.values().get("ok"), "...and null, because its bucket abstained");

        Decisions.Decision passed = Decisions.decide(runner, "s", schema, null,
                Map.of("choice:2", 0.8));
        // An unnamed bucket with no default gates at zero, so it never abstains by surprise.
        assertEquals(Boolean.TRUE, passed.values().get("ok"));
    }

    @Test
    @DisplayName("an answer of the wrong shape is refused rather than defaulted")
    void answerShapeIsChecked() {
        Map<String, Object> schema = Map.of("type", "object", "properties",
                new LinkedHashMap<>(Map.of("n",
                        Map.of("type", "integer", "minimum", 1L, "maximum", 3L))));
        Map<String, Answer> answers = Map.of("n", new Answer.Noul(0.9, 0.6, 0.6, 0.2));
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> Decisions.answersToJson(answers, schema));
        assertTrue(refused.getMessage().contains("score"), refused.getMessage());
        assertFalse(refused instanceof SchemaException, "the schema is fine; the answer is not");
    }

    @Test
    @DisplayName("the reference forwards min_confidence from decide only, and nothing is "
            + "forwarded here")
    void forwardingIsPinned() {
        // `forwarded` is the ONLY recorded datum for the documented forwarding divergence: the
        // reference passes `min_confidence` down to `runner.predict` and falls back to gating
        // the result itself when the runner does not take the keyword, while `decide_batch`
        // always gates itself and forwards nothing. Here there is no such keyword on
        // `Predictor`, so the fallback is the only path -- and that choice is only defensible
        // while the reference's own behaviour is what the class note says it is. Asserted
        // rather than merely recorded, because recorded-and-unasserted data reads as coverage
        // and is not.
        int gated = 0;
        int batched = 0;
        for (Object raw : section("calls")) {
            Map<String, Object> one = map(raw);
            String id = (String) one.get("case");
            boolean batch = Boolean.TRUE.equals(one.get("batch"));
            Object recorded = one.get("forwarded");
            assertTrue(recorded instanceof List, id + " records what the reference forwarded");
            List<?> forwarded = (List<?>) recorded;
            if (!batch && one.get("min_confidence") != null) {
                gated++;
                assertEquals(List.of("min_confidence"), forwarded,
                        id + ": the reference forwards the threshold from decide");
            } else {
                if (batch) {
                    batched++;
                }
                assertEquals(List.of(), forwarded,
                        id + ": nothing to forward, so nothing is forwarded on either side");
            }
        }
        final int forwards = gated;
        final int batches = batched;
        assertTrue(forwards >= 3,
                () -> "only " + forwards + " gated decide case(s) record a forward");
        assertTrue(batches >= 2, () -> "only " + batches + " batch case(s) record no forward");

        // And the structural half: the keyword the reference forwards has nowhere to go here.
        // A `Predictor` takes the state and the questions and nothing else, so "forward
        // nothing" is not a choice made at the call site that a later edit could quietly
        // reverse -- it is the only thing the interface can express.
        long shaped = java.util.Arrays.stream(Predictor.class.getMethods())
                .filter(method -> "predict".equals(method.getName()))
                .filter(method -> method.getParameterCount() == 2)
                .count();
        assertEquals(1, shaped,
                "Predictor.predict(state, questions) is the whole interface; a third parameter "
                        + "would be somewhere for min_confidence to go and the class note would "
                        + "be out of date");
    }

    @Test
    @DisplayName("a score answer that is not finite is refused, not narrowed into the range")
    void nonFiniteScoreIsRefused() {
        Map<String, Object> schema = Map.of("type", "object", "properties",
                new LinkedHashMap<>(Map.of("n",
                        Map.of("type", "integer", "minimum", 1L, "maximum", 3L))));
        // `(long)` on a non-finite double is a silent narrowing that -Werror -Xlint:all cannot
        // see: NaN becomes 0 and +Infinity becomes Long.MAX_VALUE, so `minimum + level`
        // projected 1 for a NaN and -9223372036854775808 for an infinity onto a field declared
        // 1..3 -- measured, and with no exception and nothing in the Decision saying so.
        // CPython refuses both: `ValueError: cannot convert float NaN to integer` and
        // `OverflowError: cannot convert float infinity to integer`. Reached only when
        // `probabilities` is empty; otherwise the argmax path runs and the score is unused.
        for (Map.Entry<String, Double> bad : new LinkedHashMap<>(Map.of(
                "nan", Double.NaN,
                "inf", Double.POSITIVE_INFINITY,
                "-inf", Double.NEGATIVE_INFINITY)).entrySet()) {
            Map<String, Answer> answers = Map.of("n",
                    new Answer.Score(bad.getValue(), Map.of(), Map.of(), 0.9, 0.9, 0.2));
            IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                    () -> Decisions.answersToJson(answers, schema), bad.getKey());
            assertEquals("n is a score field, so its score must be finite and is " + bad.getKey(),
                    refused.getMessage());
            assertFalse(refused instanceof SchemaException,
                    "the schema is fine; the answer is not");
        }
        // The finite path is untouched, including Python's half-to-even at 2.5 -> level 2.
        assertEquals(Map.of("n", 3L), Decisions.answersToJson(
                Map.of("n", new Answer.Score(2.5, Map.of(), Map.of(), 0.9, 0.9, 0.2)), schema));
        // And a non-finite score is ignored outright when probabilities decide the level.
        assertEquals(Map.of("n", 2L), Decisions.answersToJson(
                Map.of("n", new Answer.Score(Double.NaN, Map.of(),
                        new LinkedHashMap<>(Map.of("0", 0.1, "1", 0.9)), 0.9, 0.9, 0.2)), schema));
    }

    @Test
    @DisplayName("a non-finite float is spelled CPython's way, in a label and in a refusal")
    void nonFiniteFloatsUseCPythonSpelling() {
        // Not reachable through `Json.parse`, which rejects a bare NaN, and fully reachable from
        // the hand-built Map the public `Decisions.questions(Map)` signature takes -- so it is
        // not in `fixtures/structured.json`, whose schemas are JSON. The expectations below are
        // CPython's, measured on the reference:
        //
        //   repr(float("nan")) == "nan";  repr(float("inf")) == "inf"
        //   questions_from_json_schema({"type": "object",
        //                               "properties": {"x": {"enum": [float("nan"), 1]}}})
        //     -> {"x": {..., "criteria": {"nan": None, "1": None}}}
        //   ... {"properties": {"x": {"a": float("inf")}}}
        //     -> SchemaError("properties.x: unsupported schema {'a': inf}")
        //
        // `PythonJson.repr(double)` gives "NaN"/"Infinity" on purpose: that is `json.dumps`'s
        // spelling, and `dumps` is what it serves. This is the `%r` and `str` path.
        Map<String, Object> labelled = Map.of("type", "object", "properties",
                new LinkedHashMap<>(Map.of("x",
                        Map.of("enum", List.of(Double.NaN, Double.POSITIVE_INFINITY,
                                Double.NEGATIVE_INFINITY, 1L)))));
        Map<String, Object> spec = Decisions.questions(labelled).get("x").spec();
        // The LABEL is what the model is shown and what the answer comes back keyed by, so a
        // port spelling it "NaN" hands a cross-language caller a different answer key.
        assertEquals(List.of("nan", "inf", "-inf", "1"),
                new ArrayList<>(map(spec.get("criteria")).keySet()));

        for (Map.Entry<String, Double> bad : new LinkedHashMap<>(Map.of(
                "nan", Double.NaN,
                "inf", Double.POSITIVE_INFINITY,
                "-inf", Double.NEGATIVE_INFINITY)).entrySet()) {
            Map<String, Object> schema = Map.of("type", "object", "properties",
                    new LinkedHashMap<>(Map.of("x", Map.of("a", bad.getValue()))));
            SchemaException refused = assertThrows(SchemaException.class,
                    () -> Decisions.questions(schema), bad.getKey());
            assertEquals("properties.x: unsupported schema {'a': " + bad.getKey() + "}",
                    refused.getMessage());
        }
    }

    // ------------------------------------------------------------------ fixture decoding

    /** Every recorded answer as the typed record. */
    private static Map<String, Answer> answers(Map<String, Object> recorded) {
        Map<String, Answer> out = new LinkedHashMap<>();
        recorded.forEach((name, value) -> out.put(name, answerOf(map(value))));
        return out;
    }

    @SuppressWarnings("unchecked")
    private static Answer answerOf(Map<String, Object> recorded) {
        double confidence = number(recorded.get("confidence"));
        double answerConfidence = number(recorded.get("answer_confidence"));
        double act = number(map(recorded.get("action")).get("act_probability"));
        Map<String, Double> probabilities = new LinkedHashMap<>();
        if (recorded.get("probabilities") != null) {
            map(recorded.get("probabilities"))
                    .forEach((k, v) -> probabilities.put(k, number(v)));
        }
        switch ((String) recorded.get("type")) {
            case "noul":
                return new Answer.Noul(number(recorded.get("noul")), confidence, answerConfidence,
                        act);
            case "score": {
                Map<String, String> legend = new LinkedHashMap<>();
                map(recorded.get("legend")).forEach((k, v) -> legend.put(k, (String) v));
                return new Answer.Score(number(recorded.get("score")), legend, probabilities,
                        confidence, answerConfidence, act);
            }
            default:
                return new Answer.Choice((String) recorded.get("choice"), probabilities,
                        confidence, answerConfidence, act);
        }
    }

    /** An absent number is NaN, never zero: "no usable confidence" is not "a confidence of 0". */
    private static double number(Object value) {
        return value == null ? Double.NaN : ((Number) value).doubleValue();
    }

    /**
     * The gate report the recorded {@code low_confidence} flags imply, or null when none is set.
     *
     * <p>Null and not an empty map: an ungated call has no report at all, and handing the
     * projection an empty one would make "no gate ran" indistinguishable from "a gate ran and
     * flagged nothing".
     */
    private static Map<String, ConfidenceGate.Gated> gateOf(Map<String, Object> recorded,
                                                            Map<String, Answer> answers) {
        boolean any = false;
        for (Object value : recorded.values()) {
            any |= Boolean.TRUE.equals(map(value).get("low_confidence"));
        }
        if (!any) {
            return null;
        }
        Map<String, ConfidenceGate.Gated> gate = new LinkedHashMap<>();
        recorded.forEach((name, value) -> {
            boolean low = Boolean.TRUE.equals(map(value).get("low_confidence"));
            gate.put(name, new ConfidenceGate.Gated(answers.get(name), low,
                    low ? ConfidenceGate.Abstention.ABSTAINED : ConfidenceGate.Abstention.PASSED,
                    0.5));
        });
        return gate;
    }
}
