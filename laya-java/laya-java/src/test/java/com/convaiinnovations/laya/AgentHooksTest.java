package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.hooks.Hook;
import com.convaiinnovations.laya.hooks.HookCall;
import com.convaiinnovations.laya.hooks.HookRegistry;
import com.convaiinnovations.laya.hooks.Hooks;
import com.convaiinnovations.laya.hooks.PredictContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Hooks as an {@link Agent} actually runs them, over a synthetic checkpoint.
 *
 * <p>{@link HooksTest} pins the hook model itself against the reference, driving
 * {@link Hooks#around} with a stub inference — which is what lets "the model did not run" be an
 * observable. This covers the other half: that {@code Agent} is wired to that model at all, and
 * that a hook's edits reach the tokenizer rather than being recorded and dropped. None of it can
 * come from the fixture, because the reference's inference is a different implementation.
 *
 * <p>The sharpest test here is the last one. {@link Agent#predictLong} deliberately runs NO
 * hooks — not per-call, not installed, not process-wide — because the reference's
 * {@code predict_long} only survives contact with them by way of a start probe, a post-chain
 * budget check and two separate "a hook answered the document" paths, none of which this port
 * has. A silent regression there would be a hook firing once per WINDOW instead of once per
 * document, with a scan whose windows were sized before the hook could change anything.
 */
final class AgentHooksTest {

    private static Map<String, Question> twoQuestions() {
        Map<String, Question> questions = new LinkedHashMap<>();
        questions.put("intent", Question.choice("What next?",
                TinyCheckpoint.ordered("refund", "money back", "escalate", "a human")));
        questions.put("urgent", Question.noul("Needs a human."));
        return questions;
    }

    @AfterEach
    void noDefaultsLeakIntoTheNextTest() {
        Hooks.clearDefaultHooks();
    }

    /** Records every context it is handed, so a test can read what a hook actually saw. */
    private static final class Watcher implements Hook {

        final List<String> events = new ArrayList<>();
        PredictContext start;
        PredictContext end;

        @Override
        public void onPredictStart(PredictContext ctx) {
            events.add("start");
            start = ctx;
        }

        @Override
        public void onPredictEnd(PredictContext ctx) {
            events.add("end");
            end = ctx;
        }
    }

    @Test
    @DisplayName("a start hook sees the agent's own states, questions and checkpoint name")
    void aStartHookSeesTheCall(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 128, 48);
        try (TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
             Agent agent = Agent.using(
                     com.convaiinnovations.laya.tokenizer.Tokenizer.fromModelDirectory(root),
                     com.convaiinnovations.laya.config.AgentConfig.fromModelDirectory(root),
                     session, "tiny")) {
            Watcher watcher = new Watcher();
            agent.hooks().addHook(watcher);
            agent.predictBatch(List.of("alpha", "beta"), twoQuestions());

            assertEquals(List.of("start", "end"), watcher.events);
            assertEquals(List.of("alpha", "beta"), watcher.start.states());
            assertEquals(List.of("intent", "urgent"),
                    new ArrayList<>(watcher.start.questions().keySet()));
            assertEquals("tiny", watcher.start.model());
            assertSame(agent, watcher.start.predictor());
            assertSame(watcher.start, watcher.end, "one context per call, not one per event");
            assertNotNull(watcher.end.usage());
            assertEquals(2, watcher.end.results().size());
            assertTrue(watcher.end.elapsedMs() >= 0.0);
        }
    }

    @Test
    @DisplayName("ctx.skip answers the call without the graph running at all")
    void skipShortCircuitsInference(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 128, 48);
        Prediction cached = new Prediction(Prediction.MODEL, Map.of(),
                new Usage(7, 0, 7, 0, false, List.of(), Map.of()));
        try (TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
             Agent agent = TinyCheckpoint.agent(root, session)) {
            List<Prediction> got = agent.predictBatch(List.of("alpha"), twoQuestions(), null, 0,
                    false, HookCall.of(Hooks.onPredictStart(ctx -> ctx.skip(List.of(cached)))));

            assertEquals(List.of(cached), got, "the hook's payload, not the model's");
            assertTrue(session.batches.isEmpty(),
                    "the graph ran anyway, so skip is recorded rather than honoured");
        }
    }

    @Test
    @DisplayName("a start hook's token budget reaches the tokenizer, not just the context")
    void aBudgetOverrideReachesTheSequenceBuilder(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 256, 48);
        String state = "x".repeat(400);
        Map<String, Question> questions = Map.of("urgent", Question.noul("Needs a human."));
        try (TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
             Agent agent = TinyCheckpoint.agent(root, session)) {
            Usage wide = agent.predict(state, questions).usage();
            Usage narrow = agent.predictBatch(List.of(state), questions, null, 0, false,
                    HookCall.of(Hooks.onPredictStart(ctx -> ctx.maxLen(96))))
                    .get(0).usage();

            assertTrue(narrow.stateTokensDropped() > wide.stateTokensDropped(),
                    "a smaller budget must drop more of the state: " + wide + " vs " + narrow);
            assertEquals(wide.stateTokens(), narrow.stateTokens(),
                    "the state itself did not change, only the room for it");

            // The head budget is the OTHER half of the same arithmetic: spending more of
            // max_len on the question leaves less for the state. Asserted separately because a
            // port that threads one override and drops the other passes the check above.
            Usage wideHead = agent.predictBatch(List.of(state), questions, null, 0, false,
                    HookCall.of(Hooks.onPredictStart(ctx -> ctx.headMaxLen(200))))
                    .get(0).usage();
            assertTrue(wideHead.stateTokensDropped() > wide.stateTokensDropped(),
                    "a bigger head budget must leave less room for the state: " + wide + " vs "
                    + wideHead);
        }
    }

    @Test
    @DisplayName("an end hook replaces what the caller receives")
    void anEndHookRewritesTheResults(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 128, 48);
        Prediction replacement = new Prediction(Prediction.MODEL, Map.of(),
                new Usage(1, 0, 1, 0, false, List.of(), Map.of()));
        try (TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
             Agent agent = TinyCheckpoint.agent(root, session)) {
            List<Prediction> got = agent.predictBatch(List.of("alpha"), twoQuestions(), null, 0,
                    false,
                    HookCall.of(Hooks.onPredictEnd(ctx -> ctx.results(List.of(replacement)))));

            assertEquals(List.of(replacement), got);
            assertFalse(session.batches.isEmpty(), "inference still ran; only the answer changed");
        }
    }

    @Test
    @DisplayName("predict runs the chain too, and a failing hook fails the call")
    void predictIsHookedAndRespectsThePolicy(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 128, 48);
        try (TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
             Agent agent = TinyCheckpoint.agent(root, session)) {
            Watcher watcher = new Watcher();
            agent.hooks().addHook(watcher);
            agent.predict("alpha", twoQuestions());
            assertEquals(List.of("start", "end"), watcher.events,
                    "a single-state predict is a batch of one, hooks and all");

            agent.hooks().addHook(new Hook() {
                @Override
                public void onPredictStart(PredictContext ctx) {
                    throw new IllegalStateException("no");
                }
            });
            assertThrows(IllegalStateException.class, () -> agent.predict("beta", twoQuestions()));

            agent.hooks().raiseErrors(false);
            assertNotNull(agent.predict("gamma", twoQuestions()),
                    "with raiseErrors off a broken hook must not be able to fail a request");
        }
    }

    @Test
    @DisplayName("a scoped hook is installed for the block and gone after it")
    @SuppressWarnings("try")   // javac reports the unreferenced scope; see hooksInstalled
    void scopedHooks(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 128, 48);
        try (TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
             Agent agent = TinyCheckpoint.agent(root, session)) {
            Watcher watcher = new Watcher();
            try (HookRegistry.Scope ignored = agent.hooks().hooksInstalled(watcher)) {
                agent.predict("alpha", twoQuestions());
            }
            agent.predict("beta", twoQuestions());
            assertEquals(List.of("start", "end"), watcher.events,
                    "the hook saw the call inside the block and nothing after it");
        }
    }

    @Test
    @DisplayName("a closed agent refuses before any hook can watch a call that will not happen")
    void aClosedAgentDispatchesNothing(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 128, 48);
        TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
        Agent agent = TinyCheckpoint.agent(root, session);
        Watcher watcher = new Watcher();
        agent.hooks().addHook(watcher);
        agent.close();
        assertThrows(IllegalStateException.class, () -> agent.predict("alpha", twoQuestions()));
        assertEquals(List.of(), watcher.events);
    }

    @Test
    @DisplayName("predictLong runs no hook: not per-call, not installed, not process-wide")
    void predictLongIsDeliberatelyUnhooked(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 128, 48);
        Map<String, Question> questions = Map.of("urgent", Question.noul("Needs a human."));
        try (TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
             Agent agent = TinyCheckpoint.agent(root, session)) {
            Watcher installed = new Watcher();
            Watcher processWide = new Watcher();
            agent.hooks().addHook(installed);
            Hooks.setDefaultHooks(List.of(processWide));

            // Long enough to need several windows, which is the case where a hook firing per
            // call rather than per document would be visible.
            LongPrediction scanned = agent.predictLong("y".repeat(600), questions);
            assertTrue(scanned.usage().windows() > 1, "the state has to actually be scanned");
            // ...and short enough to be answered in one call, which takes the other branch.
            LongPrediction single = agent.predictLong("short", questions);
            assertEquals(1, single.usage().windows());

            assertEquals(List.of(), installed.events,
                    "a scan dispatched the installed hooks; see predictLong's javadoc for why "
                    + "it must not until the reference's start probe and budget check are ported");
            assertEquals(List.of(), processWide.events,
                    "a scan dispatched the process-wide defaults, so a tracer installed for the "
                    + "whole process would see one event per window and call it a prediction");

            // The ordinary path still dispatches, so the assertion above is about predictLong
            // and not about the hooks having been switched off entirely.
            agent.predict("alpha", questions);
            assertEquals(List.of("start", "end"), installed.events);
        }
    }
}
