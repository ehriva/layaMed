package com.convaiinnovations.laya.sequence;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import com.convaiinnovations.laya.Fixtures;
import com.convaiinnovations.laya.Question;
import com.convaiinnovations.laya.tokenizer.Tokenizer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/**
 * {@link WindowPlan} against {@code fixtures/window_plan.json}, recorded from {@code laya.common}.
 *
 * <p>The numbers are the small part. Every clamp this class performs is invisible to the caller --
 * their window shrinks, their stride shrinks with it, and the forward-pass count can triple -- so
 * the reference's {@code RuntimeWarning} text is the only thing that explains a scan that got
 * slower. The warnings are therefore compared VERBATIM: a port that clamped to exactly the right
 * numbers and said nothing would pass a test that only checked the numbers, and would leave a
 * caller with no way to find out why.
 */
final class WindowPlanTest {

    @SuppressWarnings("unchecked")
    private static Map<String, Object> family() {
        return (Map<String, Object>) (Map<?, ?>) Fixtures.load("window_plan.json");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> forCheckpoint(String name) {
        Map<String, Object> by = (Map<String, Object>) family().get("by");
        Map<String, Object> entry = (Map<String, Object>) by.get(name);
        Assumptions.assumeTrue(entry != null && !entry.containsKey("skipped"),
                "the " + name + " checkpoint was not present when the fixture was recorded");
        return entry;
    }

    private static Tokenizer open(String name) {
        Path model = Fixtures.checkpoint(name);
        Assumptions.assumeTrue(model != null, Fixtures.missingCheckpoint(name));
        try {
            return Tokenizer.fromModelDirectory(model);
        } catch (java.io.IOException problem) {
            throw new IllegalStateException("cannot open the " + name + " tokenizer", problem);
        }
    }

    /**
     * The generator's {@code _window_question}, rebuilt here.
     *
     * <p>Rebuilt rather than recorded, because a 160-option question is a lot of fixture for a
     * recipe this short. The risk in rebuilding is that the two drift and the test then compares
     * the right answer to the wrong question -- which the recorded {@code room} catches: it is
     * derived from the head this question tokenizes to, so a question that differs by one token
     * fails every case for that checkpoint rather than passing quietly.
     */
    private static Question windowQuestion(int options) {
        Map<String, Object> criteria = new LinkedHashMap<>();
        for (int i = 0; i < options; i++) {
            StringBuilder value = new StringBuilder();
            for (int word = 0; word < 4; word++) {
                if (word > 0) {
                    value.append(' ');
                }
                value.append("criterion").append(i).append(" word").append(word);
            }
            criteria.put(String.format("opt%03d", i), value.toString());
        }
        return Question.choice("Pick one.", criteria);
    }

    @TestFactory
    @DisplayName("stateRoom matches the reference for every option count, on both checkpoints")
    @SuppressWarnings("unchecked")
    List<DynamicTest> stateRoomMatches() {
        List<DynamicTest> tests = new ArrayList<>();
        for (String name : List.of("english", "multilingual")) {
            Map<String, Object> entry = forCheckpoint(name);
            int maxLen = number(entry.get("default_max_len"));
            int headMaxLen = number(entry.get("default_head_max_len"));
            List<Object> rooms = (List<Object>) entry.get("state_room");
            assertTrue(rooms.size() >= 8, () -> "the state_room table shrank: " + rooms.size());
            Tokenizer tok = open(name);
            for (Object raw : rooms) {
                Map<String, Object> row = (Map<String, Object>) raw;
                int options = number(row.get("options"));
                tests.add(dynamicTest(name + " " + options + " options", () -> {
                    Question question = windowQuestion(options);
                    assertEquals(number(row.get("room")),
                            WindowPlan.stateRoom(tok, question, maxLen, headMaxLen),
                            "room at the checkpoint's own budget");
                    // A second budget, because `maxLen - headLen - 1` is right for the wrong
                    // reason if the head is independently truncated: this one truncates it.
                    assertEquals(number(row.get("room_tight")),
                            WindowPlan.stateRoom(tok, question, 256, 128),
                            "room at max_len=256 head_max_len=128");
                }));
            }
        }
        return tests;
    }

    @TestFactory
    @DisplayName("every window_budget branch matches, warnings and refusals included")
    @SuppressWarnings("unchecked")
    List<DynamicTest> budgetCases() {
        List<DynamicTest> tests = new ArrayList<>();
        int warned = 0;
        int refused = 0;
        for (String name : List.of("english", "multilingual")) {
            Map<String, Object> entry = forCheckpoint(name);
            List<Object> cases = (List<Object>) entry.get("cases");
            assertTrue(cases.size() >= 24, () -> "the case list shrank: " + cases.size());
            Tokenizer tok = open(name);
            for (Object raw : cases) {
                Map<String, Object> one = (Map<String, Object>) raw;
                String label = name + " " + one.get("case");
                List<Object> warningList = (List<Object>) one.get("warnings");
                if (!warningList.isEmpty()) {
                    warned++;
                }
                if (one.containsKey("error")) {
                    refused++;
                }
                tests.add(dynamicTest(label, () -> {
                    List<Question> questions = new ArrayList<>();
                    for (Object count : (List<Object>) one.get("options")) {
                        questions.add(windowQuestion(number(count)));
                    }
                    int maxLen = number(one.get("max_len"));
                    int headMaxLen = number(one.get("head_max_len"));
                    Integer window = optional(one.get("window"));
                    Integer stride = optional(one.get("stride"));

                    if (one.containsKey("error")) {
                        IllegalArgumentException refusal = assertThrows(
                                IllegalArgumentException.class,
                                () -> WindowPlan.budget(tok, questions, maxLen, headMaxLen,
                                        window, stride),
                                label + " must be refused");
                        // The message, not just the type. Both refusals are
                        // IllegalArgumentException and they say entirely different things -- one
                        // is "your label set is too big for any window", the other is "your
                        // stride leaves tokens no window reads" -- so a port that raised the
                        // wrong one would be indistinguishable from a correct one.
                        assertEquals(one.get("error"), refusal.getMessage(),
                                "the refusal must say what the reference says");
                        return;
                    }

                    WindowPlan.Budget got = WindowPlan.budget(tok, questions, maxLen, headMaxLen,
                            window, stride);
                    Map<String, Object> want = (Map<String, Object>) one.get("result");
                    assertEquals(number(want.get("window")), got.window(), label + " window");
                    assertEquals(number(want.get("stride")), got.stride(), label + " stride");
                    assertEquals(number(want.get("room")), got.room(), label + " room");

                    assertEquals(warningList.size(), got.warnings().size(),
                            () -> label + " warning count; got " + got.warnings());
                    for (int i = 0; i < warningList.size(); i++) {
                        assertEquals(warningList.get(i), got.warnings().get(i),
                                label + " warning " + i);
                    }
                    assertEquals(!warningList.isEmpty(), got.clamped(), label + " clamped()");
                }));
            }
        }
        // The branches have to be REACHED, not merely compared. A fixture whose cases stopped
        // clamping would still pass case by case while testing none of the hard parts -- which is
        // exactly what happened when the option counts were literals: they covered every branch on
        // english and zero of the warning branches on multilingual.
        final int warnings = warned;
        final int refusals = refused;
        assertTrue(warnings >= 8,
                () -> "window_budget has three warning branches on two checkpoints and the cases "
                        + "reach only " + warnings);
        assertTrue(refusals >= 4,
                () -> "window_budget has two refusals on two checkpoints and the cases reach only "
                        + refusals);
        return tests;
    }

    @Test
    @DisplayName("batchCap matches the reference across the whole recorded grid")
    @SuppressWarnings("unchecked")
    void batchCapMatchesTheGrid() {
        List<Object> grid = (List<Object>) family().get("batch_cap");
        assertTrue(grid.size() >= 800, () -> "the grid shrank: " + grid.size());
        int capped = 0;
        for (Object raw : grid) {
            Map<String, Object> one = (Map<String, Object>) raw;
            int nWindows = number(one.get("n_windows"));
            int window = number(one.get("window"));
            int configBudget = number(one.get("config_budget"));
            Integer batchSize = optional(one.get("batch_size"));
            Integer got = WindowPlan.batchCap(nWindows, window, configBudget, batchSize);
            Object want = one.get("cap");
            String label = String.format("n=%d window=%d budget=%d batch=%s",
                    nWindows, window, configBudget, batchSize);
            if (want == null) {
                assertNull(got, label + " must be one shared pass");
            } else {
                assertEquals(number(want), got, label);
                if (batchSize == null || batchSize <= 0) {
                    capped++;
                }
            }
        }
        // Without this the grid could be all nulls and a port that never capped would pass.
        final int reached = capped;
        assertTrue(reached >= 100,
                () -> "only " + reached + " grid points reach the capping branch, so this cannot "
                        + "tell a port that caps from one that returns null every time");
    }

    @Test
    @DisplayName("null arguments are caller errors, not silent defaults")
    void nullArguments() {
        Tokenizer tok = open("english");
        assertThrows(IllegalArgumentException.class,
                () -> WindowPlan.budget(null, List.of(), 512, 192, null, null),
                "a null tokenizer cannot be defaulted");
        assertThrows(IllegalArgumentException.class,
                () -> WindowPlan.budget(tok, null, 512, 192, null, null),
                "a null question list is not the same as no questions; an empty list says that");
    }

    private static int number(Object value) {
        return ((Number) value).intValue();
    }

    /** A recorded {@code null} stays null; the reference treats it, 0 and negatives alike. */
    private static Integer optional(Object value) {
        return value == null ? null : ((Number) value).intValue();
    }
}
