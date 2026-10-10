package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/**
 * {@link Shortlist#predictTournament} against the calls recorded from
 * {@code laya.shortlist.predict_tournament}.
 *
 * <p>Both sides answer with the same stand-in model: every choice goes to the label with the
 * lowest unsigned FNV-1a of {@code rank:<label>}. That makes every round reproducible, so the
 * fixture pins the groups asked, their order, the winners carried forward and the final cut.
 */
class TournamentTest {

    private static final long FNV_OFFSET = 0xCBF29CE484222325L;
    private static final long FNV_PRIME = 0x100000001B3L;

    private static Map<String, Object> fixture() {
        return Fixtures.load("tournament.json");
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
        for (Object item : list(value)) {
            out.add((String) item);
        }
        return out;
    }

    private static long rank(String label) {
        long hash = FNV_OFFSET;
        for (byte value : ("rank:" + label).getBytes(StandardCharsets.UTF_8)) {
            hash = (hash ^ (value & 0xFFL)) * FNV_PRIME;
        }
        return hash;
    }

    private static Usage noUsage() {
        return new Usage(0, 0, 0, 0, false, List.of(), Map.of());
    }

    /** The stand-in model, recording each call: choices only, like the generator's. */
    private static final class Ranker implements Predictor {

        final List<Map<String, Question>> calls = new ArrayList<>();
        final List<Prediction> returned = new ArrayList<>();
        final List<Object> states = new ArrayList<>();

        @Override
        public Prediction predict(Object state, Map<String, Question> questions) {
            calls.add(new LinkedHashMap<>(questions));
            states.add(state);
            Map<String, Answer> answers = new LinkedHashMap<>();
            for (Map.Entry<String, Question> entry : questions.entrySet()) {
                if (entry.getValue().type() != Question.Type.CHOICE) {
                    continue;
                }
                String best = null;
                for (String label : entry.getValue().labels()) {
                    if (best == null || Long.compareUnsigned(rank(label), rank(best)) < 0) {
                        best = label;
                    }
                }
                answers.put(entry.getKey(), new Answer.Choice(best, Map.of(best, 1.0), 1, 1, 0));
            }
            Prediction out = new Prediction("stub-" + calls.size(), answers, noUsage());
            returned.add(out);
            return out;
        }
    }

    /** A recorded question; list criteria become a map of null descriptions, as the reference does. */
    private static Question question(Map<String, Object> spec) {
        String type = (String) spec.get("type");
        String instructions = (String) spec.get("instructions");
        Object criteria = spec.get("criteria");
        if ("choice".equals(type)) {
            Map<String, Object> options = new LinkedHashMap<>();
            if (criteria instanceof List) {
                for (Object label : list(criteria)) {
                    options.put((String) label, null);
                }
            } else {
                options.putAll(map(criteria));
            }
            return Question.choice(instructions, options);
        }
        if ("score".equals(type)) {
            return Question.score(instructions, list(criteria));
        }
        return Question.noul(instructions);
    }

    private static Map<String, Question> questions(Map<String, Object> specs) {
        Map<String, Question> out = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : specs.entrySet()) {
            out.put(entry.getKey(), question(map(entry.getValue())));
        }
        return out;
    }

    private static Map<String, Question> oneChoice(int n) {
        Map<String, Object> criteria = new LinkedHashMap<>();
        for (int i = 0; i < n; i++) {
            criteria.put(String.format("l%04d", i), null);
        }
        return Map.of("q", Question.choice("pick", criteria));
    }

    // ------------------------------------------------------------------ golden parity

    @Test
    @DisplayName("the stand-in model ranks labels as the generator did")
    void rankMatchesTheGenerator() {
        Map<String, Object> probe = map(fixture().get("rank_probe"));
        assertTrue(probe.size() >= 3, "only " + probe.size() + " probe labels");
        for (Map.Entry<String, Object> entry : probe.entrySet()) {
            assertEquals(entry.getValue(), Long.toUnsignedString(rank(entry.getKey())),
                    "rank of " + entry.getKey());
        }
        assertEquals(((Number) fixture().get("default_group_size")).intValue(),
                Shortlist.DEFAULT_TOURNAMENT_GROUP);
    }

    @TestFactory
    @DisplayName("every recorded split: group sizes per round, finalists and the final choice")
    List<DynamicTest> splits() {
        List<DynamicTest> tests = new ArrayList<>();
        for (Object entry : list(fixture().get("splits"))) {
            Map<String, Object> row = map(entry);
            int groupSize = ((Number) row.get("group_size")).intValue();
            int n = ((Number) row.get("n")).intValue();
            tests.add(DynamicTest.dynamicTest("n=" + n + " group_size=" + groupSize, () -> {
                Ranker ranker = new Ranker();
                Shortlist.Tournament result = Shortlist.predictTournament(ranker, "s",
                        oneChoice(n), groupSize);
                List<Object> rounds = list(row.get("rounds"));
                assertEquals(rounds.size() + 1, ranker.calls.size(), "calls");
                for (int r = 0; r < rounds.size(); r++) {
                    List<Integer> sizes = new ArrayList<>();
                    for (Question asked : ranker.calls.get(r).values()) {
                        sizes.add(asked.labels().size());
                    }
                    List<Integer> expected = new ArrayList<>();
                    for (Object size : list(rounds.get(r))) {
                        expected.add(((Number) size).intValue());
                    }
                    assertEquals(expected, sizes, "group sizes of round " + r);
                }
                Shortlist.Bracket bracket = result.tournament().get("q");
                assertEquals(strings(row.get("finalists")), bracket.labels(), "finalists");
                assertEquals(rounds.size(), bracket.rounds(), "rounds");
                assertEquals(n, bracket.total(), "total");
                assertEquals(row.get("choice"),
                        ((Answer.Choice) result.prediction().answer("q")).choice(), "choice");
            }));
        }
        assertTrue(tests.size() >= 100, "only " + tests.size() + " splits recorded");
        return tests;
    }

    @TestFactory
    @DisplayName("every recorded tournament asks the same calls and reports the same result")
    List<DynamicTest> cases() {
        List<DynamicTest> tests = new ArrayList<>();
        String state = (String) fixture().get("state");
        for (Object entry : list(fixture().get("cases"))) {
            Map<String, Object> row = map(entry);
            String name = (String) row.get("name");
            tests.add(DynamicTest.dynamicTest(name, () -> {
                Map<String, Question> questions = questions(map(row.get("questions")));
                Map<String, Question> original = new LinkedHashMap<>(questions);
                Ranker ranker = new Ranker();
                Object groupSize = row.get("group_size");
                Shortlist.Tournament result = groupSize == null
                        ? Shortlist.predictTournament(ranker, state, questions)
                        : Shortlist.predictTournament(ranker, state, questions,
                                ((Number) groupSize).intValue());

                List<Object> calls = list(row.get("calls"));
                assertEquals(calls.size(), ranker.calls.size(), name + ": calls");
                for (int c = 0; c < calls.size(); c++) {
                    assertAsked(name + " call " + c, map(calls.get(c)), ranker.calls.get(c));
                    assertSame(state, ranker.states.get(c), name + ": the state is passed through");
                }
                assertEquals(row.get("model"), result.prediction().model(),
                        name + ": the result is the final call's");
                assertSame(ranker.returned.get(ranker.returned.size() - 1), result.prediction());
                Map<String, Object> answers = map(row.get("answers"));
                assertEquals(new ArrayList<>(answers.keySet()),
                        new ArrayList<>(result.prediction().answers().keySet()));
                for (Map.Entry<String, Object> answer : answers.entrySet()) {
                    assertEquals(answer.getValue(), ((Answer.Choice) result.prediction()
                            .answer(answer.getKey())).choice(), name + "." + answer.getKey());
                }
                Map<String, Object> meta = map(row.get("tournament"));
                assertEquals(new ArrayList<>(meta.keySet()),
                        new ArrayList<>(result.tournament().keySet()), name + ": brackets");
                for (Map.Entry<String, Object> expected : meta.entrySet()) {
                    Map<String, Object> want = map(expected.getValue());
                    Shortlist.Bracket got = result.tournament().get(expected.getKey());
                    String at = name + "." + expected.getKey();
                    assertEquals(strings(want.get("labels")), got.labels(), at + ".labels");
                    assertEquals(((Number) want.get("n")).intValue(), got.total(), at + ".n");
                    assertEquals(((Number) want.get("rounds")).intValue(), got.rounds(),
                            at + ".rounds");
                }
                assertEquals(Boolean.TRUE, row.get("caller_questions_unmutated"));
                assertEquals(original, questions, name + ": the caller's map was modified");
            }));
        }
        // Floors: a multi-round case and a case where questions play different numbers of rounds.
        int multiRound = 0;
        int mixedRounds = 0;
        for (Object entry : list(fixture().get("cases"))) {
            List<Integer> played = new ArrayList<>();
            for (Object bracket : map(map(entry).get("tournament")).values()) {
                played.add(((Number) map(bracket).get("rounds")).intValue());
            }
            multiRound += played.stream().anyMatch(r -> r > 1) ? 1 : 0;
            mixedRounds += played.stream().distinct().count() > 1 ? 1 : 0;
        }
        assertTrue(multiRound >= 1, "no case plays more than one round");
        assertTrue(mixedRounds >= 1, "no case mixes round counts");
        return tests;
    }

    private static void assertAsked(String at, Map<String, Object> expected,
            Map<String, Question> seen) {
        assertEquals(new ArrayList<>(expected.keySet()), new ArrayList<>(seen.keySet()),
                at + ": question ids and order");
        for (Map.Entry<String, Object> entry : expected.entrySet()) {
            Map<String, Object> want = map(entry.getValue());
            Question got = seen.get(entry.getKey());
            String where = at + "." + entry.getKey();
            assertEquals(want.get("type"), got.type().wireName(), where + ".type");
            assertEquals(want.get("instructions"), got.instructions(), where + ".instructions");
            if (want.containsKey("labels")) {
                assertEquals(strings(want.get("labels")), got.labels(), where + ".labels");
            }
            if (want.containsKey("criteria")) {
                assertEquals(map(want.get("criteria")), got.spec().get("criteria"),
                        where + ".criteria");
            }
        }
    }

    @TestFactory
    @DisplayName("a group size below two is refused with the reference's message")
    List<DynamicTest> refusals() {
        List<DynamicTest> tests = new ArrayList<>();
        for (Object entry : list(fixture().get("refusals"))) {
            Map<String, Object> row = map(entry);
            int bad = ((Number) row.get("group_size")).intValue();
            tests.add(DynamicTest.dynamicTest("group_size=" + bad, () -> {
                assertEquals("ValueError", row.get("error"));
                IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                        () -> Shortlist.predictTournament(new Ranker(), "s", Map.of(), bad));
                assertEquals(row.get("message"), refused.getMessage());
            }));
        }
        assertTrue(tests.size() >= 3);
        return tests;
    }

    // ------------------------------------------------------------------ model-free behaviour

    @Test
    @DisplayName("77 labels at the default group go out as five balanced groups in criteria order")
    void balancedGroups() {
        List<String> labels = new ArrayList<>(oneChoice(77).get("q").labels());
        List<List<String>> groups = Shortlist.groups(labels, 16);
        List<Integer> sizes = new ArrayList<>();
        List<String> joined = new ArrayList<>();
        for (List<String> group : groups) {
            sizes.add(group.size());
            joined.addAll(group);
        }
        assertEquals(List.of(15, 15, 16, 15, 16), sizes);
        assertEquals(labels, joined, "groups are contiguous slices in criteria order");
        assertEquals(1, Shortlist.groups(labels.subList(0, 16), 16).size());
    }

    @Test
    @DisplayName("a consistent ranker's best label wins from any position")
    void bestLabelWinsFromAnyPosition() {
        for (int[] shape : new int[][] {{77, 2}, {257, 3}}) {
            int n = shape[0];
            List<String> labels = new ArrayList<>(oneChoice(n).get("q").labels());
            for (String best : List.of(labels.get(0), labels.get(n / 2), labels.get(n - 1))) {
                List<Map<String, Question>> calls = new ArrayList<>();
                Predictor favours = (s, qs) -> {
                    calls.add(qs);
                    Map<String, Answer> answers = new LinkedHashMap<>();
                    for (Map.Entry<String, Question> q : qs.entrySet()) {
                        List<String> options = q.getValue().labels();
                        String pick = options.contains(best) ? best : options.get(0);
                        answers.put(q.getKey(), new Answer.Choice(pick, Map.of(), 1, 1, 0));
                    }
                    return new Prediction("stub", answers, noUsage());
                };
                Shortlist.Tournament out = Shortlist.predictTournament(favours, "s", oneChoice(n));
                assertEquals(best, ((Answer.Choice) out.prediction().answer("q")).choice(),
                        n + " labels, best " + best);
                assertEquals(shape[1], calls.size(), n + " labels, best " + best + ": calls");
            }
        }
    }

    @Test
    @DisplayName("nothing to narrow: one call, with the caller's questions as given")
    void nothingToNarrowIsOneCall() {
        Map<String, Question> asked = new LinkedHashMap<>();
        asked.put("small", Question.choice("pick", Map.of("x", "")));
        asked.put("level", Question.score("rate", List.of("low", "high")));
        Ranker ranker = new Ranker();
        Shortlist.Tournament out = Shortlist.predictTournament(ranker, "s", asked, 2);
        assertEquals(1, ranker.calls.size());
        assertEquals(asked, ranker.calls.get(0));
        assertSame(asked.get("small"), ranker.calls.get(0).get("small"));
        assertEquals(0, out.tournament().get("small").rounds());
        assertEquals(List.of("small"), new ArrayList<>(out.tournament().keySet()));
    }

    @TestFactory
    @DisplayName("irregular round answers advance or fail exactly where the reference does")
    List<DynamicTest> irregularAnswers() {
        List<DynamicTest> tests = new ArrayList<>();
        String state = (String) fixture().get("state");
        for (Object entry : list(fixture().get("irregular"))) {
            Map<String, Object> row = map(entry);
            String name = (String) row.get("name");
            tests.add(DynamicTest.dynamicTest(name, () -> assertIrregular(name, state, row)));
        }
        assertTrue(tests.size() >= 9, "only " + tests.size() + " irregular cases");
        return tests;
    }

    /**
     * One irregular case. A Java choice always has map criteria, so a list-criteria case is held
     * to the dict outcome: duplicates collapse in what is asked, and a non-label fails at its cut.
     */
    private static void assertIrregular(String name, String state, Map<String, Object> row) {
        Map<String, Object> spec = map(map(row.get("questions")).get("q"));
        boolean listCriteria = spec.get("criteria") instanceof List;
        String mode = (String) row.get("mode");
        String label = (String) row.get("label");
        List<Map<String, Question>> calls = new ArrayList<>();
        Predictor fixed = (s, qs) -> {
            calls.add(new LinkedHashMap<>(qs));
            Map<String, Answer> answers = new LinkedHashMap<>();
            for (String id : qs.keySet()) {
                if ("label".equals(mode)) {
                    answers.put(id, new Answer.Choice(label, Map.of(), 1, 1, 0));
                } else if ("no-choice".equals(mode)) {
                    answers.put(id, new Answer.Noul(0.5, 1, 1, 0));
                }
            }
            return new Prediction("fixed-" + calls.size(), answers, noUsage());
        };
        Map<String, Question> questions = questions(map(row.get("questions")));
        boolean nonLabel = "label".equals(mode) && !questions.get("q").labels().contains(label);
        Object error = row.get("error");
        if (listCriteria && nonLabel) {
            assertEquals(null, error, name + ": the reference accepts a non-label in a list");
            error = "KeyError";
        }
        List<Object> recorded = list(row.get("calls"));
        if (error != null) {
            assertEquals("KeyError", error, name);
            IllegalArgumentException failed = assertThrows(IllegalArgumentException.class,
                    () -> Shortlist.predictTournament(fixed, state, questions, 16));
            assertTrue(failed.getMessage() != null && !failed.getMessage().isEmpty());
            if (!(listCriteria && nonLabel)) {
                assertEquals(recorded.size(), calls.size(), name + ": calls before the failure");
            }
            return;
        }
        Shortlist.Tournament result = Shortlist.predictTournament(fixed, state, questions, 16);
        assertEquals(recorded.size(), calls.size(), name + ": calls");
        for (int c = 0; c < recorded.size(); c++) {
            Map<String, Object> asked = map(recorded.get(c));
            assertEquals(new ArrayList<>(asked.keySet()), new ArrayList<>(calls.get(c).keySet()));
            for (Map.Entry<String, Object> q : asked.entrySet()) {
                List<String> labels = new ArrayList<>(new LinkedHashSet<>(
                        strings(map(q.getValue()).get("labels"))));
                assertEquals(labels, calls.get(c).get(q.getKey()).labels(),
                        name + " call " + c + "." + q.getKey());
            }
        }
        Map<String, Object> want = map(map(row.get("tournament")).get("q"));
        Shortlist.Bracket got = result.tournament().get("q");
        assertEquals(strings(want.get("labels")), got.labels(), name + ": bracket labels");
        assertEquals(((Number) want.get("rounds")).intValue(), got.rounds());
        assertEquals(map(row.get("answers")).get("q"),
                ((Answer.Choice) result.prediction().answer("q")).choice());
    }

    @Test
    @DisplayName("a null round prediction is refused like a missing answer")
    void nullRoundPredictionIsRefused() {
        Predictor nothing = (s, qs) -> null;
        assertThrows(IllegalArgumentException.class,
                () -> Shortlist.predictTournament(nothing, "s", oneChoice(40), 16));
        assertThrows(IllegalArgumentException.class,
                () -> Shortlist.predictTournament(nothing, "s", oneChoice(3), 16),
                "a null final prediction");
    }

    @Test
    @DisplayName("null arguments and contradictory brackets are refused")
    void refusesNullsAndBadBrackets() {
        assertThrows(IllegalArgumentException.class,
                () -> Shortlist.predictTournament(null, "s", Map.of()));
        assertThrows(IllegalArgumentException.class,
                () -> Shortlist.predictTournament(new Ranker(), "s", null));
        assertThrows(IllegalArgumentException.class,
                () -> new Shortlist.Bracket(List.of(), 0, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new Shortlist.Bracket(List.of("a", "b"), 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new Shortlist.Bracket(List.of("a"), 2, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new Shortlist.Bracket(List.of("a"), 2, -1));
        assertEquals(1, new Shortlist.Bracket(List.of("a"), 2, 1).rounds());
    }
}
