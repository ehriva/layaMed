package com.convaiinnovations.laya;

import static com.convaiinnovations.laya.PredictionGolden.assertAnswers;
import static com.convaiinnovations.laya.PredictionGolden.assertUsage;
import static com.convaiinnovations.laya.PredictionGolden.questionsOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.config.AgentConfig;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;
import org.junit.jupiter.api.TestMethodOrder;

/**
 * End to end on the {@code typed-decisions} checkpoint, which nothing in this port used to run.
 *
 * <p>{@code Router} has always resolved the name and its four aliases, and {@link RouterTest}
 * asserts that it does. What no test did was take the checkpoint those decisions name and push a
 * state through it, so every number this port would produce from those weights was unmeasured --
 * the routing decision was gated and its destination was not. This class closes that: the same
 * comparison {@link PredictParityTest} applies to the multilingual checkpoint, applied to the
 * weights {@code typed}, {@code typed_decisions}, {@code decisions} and
 * {@code laya-typed-decisions} all resolve to.
 *
 * <p>What this checkpoint has that the multilingual one does not, and why each case is here:
 *
 * <ul>
 *   <li>Four fine-tuned workflow signatures. Each recorded case names the workflow its question
 *       ids are, and the workflow is re-derived here through
 *       {@link Router#matchTypedDecisionsWorkflow} -- so the routing decision and the forward
 *       pass are asserted on one object instead of in two suites that never meet.
 *   <li>A populated {@code temperature_by_options}, where the multilingual checkpoint ships an
 *       empty one. The {@code many-options} case has 14 options, which lands in the
 *       {@code choice:11+} bucket this checkpoint ships at 0.1006 -- outside the accepted
 *       [0.5, 5] band, so it is clamped to 0.5. Skipping the clamp sharpens the recorded top
 *       probability of 0.3035 by roughly 5x and still returns the same argmax, so the
 *       probabilities are the only thing that separates the two implementations.
 *   <li>{@code max_len} 1024 against the english checkpoint's 512. The {@code over-budget} case
 *       drops 1211 state tokens at that budget and marks all five questions truncated.
 *   <li>A question whose option count lands in a bucket NEITHER checkpoint ships.
 *       {@code unbucketed-score} is a six-level score question, so {@code temp_bucket} returns
 *       {@code score:6-10}, the lookup misses, and the fitted {@code temperature[QTYPE_SCORE]}
 *       is the scale applied: 1.0374 here against 1.2514 on english. Every other recorded
 *       question resolves to a bucket both checkpoints ship -- their
 *       {@code temperature_by_options} tables are identical, every key and every digit -- so
 *       without this case the fitted triple was asserted and never applied.
 * </ul>
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
final class TypedDecisionsParityTest {

    /** The checkpoint this golden was recorded from; also the directory it lives in. */
    private static final String CHECKPOINT = "typed-decisions";

    /** Shared for the class: the graph is 1.7 GB of weights and opening it twice buys nothing. */
    private static Agent shared;

    /**
     * The identity failure, if the checkpoint opened is not the one this golden was recorded from.
     *
     * <p>Recorded rather than thrown once, because {@link #agent} rethrows it to EVERY consumer.
     * Caching it is what keeps that from costing a second 1.7 GB open per test.
     */
    private static AssertionError mispaired;

    @AfterAll
    static void closeAgent() {
        if (shared != null) {
            shared.close();
            shared = null;
        }
        mispaired = null;
    }

    /**
     * The loaded config is the typed-decisions one.
     *
     * <p>Not ceremony. This checkpoint is fine-tuned FROM the english one and ships its
     * {@code tokenizer.json} byte for byte and its {@code answerdotai/ModernBERT-large} encoder,
     * so the english checkpoint opened against this graph raises nothing and reproduces the
     * recorded probabilities -- {@code off-workflow}'s {@code {billing: 0.8967, other: 0.1033}}
     * among them, digit for digit, because the weights are the graph's.
     * {@link Prediction#model()} does not separate them either: the ONNX path reports
     * {@code laya-rl-agent-onnx} whichever checkpoint it opened. What separates them is the
     * budgets and the fitted temperatures, so those are what is asserted. NOT
     * {@code config.model_name}, which the golden records and which would name the checkpoint
     * outright: {@link AgentConfig} does not parse that key, so there is nothing to compare it
     * against without adding a field to it.
     *
     * <p>This test is only the NAMED place that failure is reported. The assertion itself runs
     * inside {@link #agent}, on the shared agent as it is opened, so it gates every case in this
     * class instead of being one more test among them -- which is what it was: JUnit 5 applies no
     * ordering of its own, and measured against the english checkpoint symlinked in as
     * {@code typed-decisions} this check ran NINTH of thirteen, after all eight probability
     * comparisons. {@link Order} pins it first as well, so the report reads in the order the
     * gating actually happens.
     */
    @Test
    @Order(1)
    @DisplayName("the agent under test is the typed-decisions checkpoint, not the english one")
    void configIsTheTypedCheckpoint() throws IOException {
        Map<String, Object> golden = requireGolden();
        assertTypedCheckpoint(golden, agent(golden).config());
    }

    /** The identity comparison itself, so {@link #agent} can apply it to every consumer. */
    @SuppressWarnings("unchecked")
    private static void assertTypedCheckpoint(Map<String, Object> golden, AgentConfig config) {
        Map<String, Object> want = (Map<String, Object>) golden.get("config");
        assertEquals(want.get("encoder"), config.encoder(), "encoder");
        assertEquals(((Number) want.get("max_len")).intValue(), config.maxLen(), "max_len");
        assertEquals(((Number) want.get("head_max_len")).intValue(), config.headMaxLen(),
                "head_max_len");
        List<Object> temperature = (List<Object>) want.get("temperature");
        assertEquals(temperature.size(), config.temperature().length, "temperature length");
        for (int i = 0; i < temperature.size(); i++) {
            // Exact: these are read out of a JSON file, not computed, so any difference is a
            // parse bug rather than float noise.
            assertEquals(((Number) temperature.get(i)).doubleValue(), config.temperature()[i],
                    "temperature[" + i + "]");
        }
        Map<String, Object> buckets = (Map<String, Object>) want.get("temperature_by_options");
        assertEquals(buckets.keySet(), config.temperatureByOptions().keySet(),
                "temperature_by_options buckets");
        for (Map.Entry<String, Object> entry : buckets.entrySet()) {
            assertEquals(((Number) entry.getValue()).doubleValue(),
                    config.temperatureByOptions().get(entry.getKey()),
                    "temperature_by_options[" + entry.getKey() + "]");
        }
        // The clamp, asserted on the one bucket that needs it rather than inferred from the map
        // above. 0.1006 in `rl_agent_config.json`, 0.5 in use: a port that read the file value
        // straight through would sharpen the many-options distribution by ~5x, and a lone 0.5 in
        // the golden could not be told from a bucket that was fitted at 0.5.
        //
        // This catches a PORT bug, and nothing else in this method does less. It has no power
        // over the wrong-checkpoint pairing: measured, the english and typed-decisions
        // `temperature_by_options` tables are identical, every key and every digit, 0.1006
        // included, so the keyset assert, the per-bucket loop, the raw-value assert and this
        // clamp all pass unchanged under the wrong checkpoint. What discriminates above is
        // `max_len` (1024 vs 512), `head_max_len` (256 vs 192) and the fitted `temperature`
        // triple -- those three and nothing more.
        Map<String, Object> raw = (Map<String, Object>) want.get("temperature_by_options_raw");
        assertEquals(0.10058280825614929, ((Number) raw.get("choice:11+")).doubleValue(),
                "the checkpoint still ships the sharpening bucket this case depends on");
        assertEquals(AgentConfig.TEMP_MIN, config.temperatureByOptions().get("choice:11+"),
                "choice:11+ must be clamped to TEMP_MIN, not applied as fitted");
    }

    /**
     * Every recorded workflow id set still routes to the workflow it was recorded under.
     *
     * <p>The link MODELS.md said was missing. A signature that stops matching would otherwise
     * leave the forward-pass cases below passing under a workflow name the router no longer
     * returns, which reads as coverage of a path that no longer exists.
     */
    @Test
    @Order(2)
    @DisplayName("each case's question ids still match the workflow it was recorded under")
    @SuppressWarnings("unchecked")
    void workflowsStillMatch() {
        Map<String, Object> golden = requireGolden();
        Map<String, Object> workflows = (Map<String, Object>) golden.get("workflows");
        assertEquals(4, workflows.size(), "the checkpoint is fine-tuned on four workflows");
        // DISTINCT, not a count of cases: two cases share the invoice_processing signature, so
        // counting cases made four workflows look like five and the floor failed on arithmetic
        // instead of on coverage.
        Set<String> covered = new TreeSet<>();
        for (Map.Entry<String, Object> entry
                : ((Map<String, Object>) golden.get("single")).entrySet()) {
            Map<String, Object> c = (Map<String, Object>) entry.getValue();
            Map<String, Object> questions = (Map<String, Object>) c.get("questions");
            String want = (String) c.get("workflow");
            assertEquals(want, Router.matchTypedDecisionsWorkflow(questions),
                    entry.getKey() + ": workflow for ids " + questions.keySet());
            if (want != null) {
                assertTrue(workflows.containsKey(want), want + " is not one of the four");
                covered.add(want);
            }
        }
        assertEquals(workflows.keySet(), covered,
                "every workflow needs a recorded forward pass");
    }

    @TestFactory
    @Order(3)
    @DisplayName("single-state typed-decisions predictions match Python")
    @SuppressWarnings("unchecked")
    List<DynamicTest> singleMatches() throws IOException {
        Map<String, Object> golden = requireGolden();
        Map<String, Object> singles = (Map<String, Object>) golden.get("single");
        Agent agent = agent(golden);
        List<DynamicTest> tests = new ArrayList<>();
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
        // Four workflows plus the four cases that are here for the checkpoint's own budgets and
        // temperature buckets. A factory whose fixture key emptied used to generate zero
        // assertions and still exit 0, which is what this floor is for.
        assertTrue(tests.size() >= 9, "expected the golden's cases, got " + tests.size());
        return tests;
    }

    @TestFactory
    @Order(4)
    @DisplayName("batched typed-decisions predictions match Python at every batch size")
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
                            // Recorded on every batch result by the generator and, until now,
                            // compared only on the single-state path -- a field the fixture
                            // carries that nothing read.
                            assertEquals(w.get("model"), got.get(i).model(), "model");
                            // The caller's order, whatever order the batcher grouped them in.
                            assertAnswers((Map<String, Object>) w.get("answers"),
                                    got.get(i).answers());
                            assertUsage((Map<String, Object>) w.get("usage"), got.get(i).usage());
                        }
                    }));
        }
        assertTrue(tests.size() >= 3, "expected the golden's batch configs, got " + tests.size());
        return tests;
    }

    private static Map<String, Object> requireGolden() {
        Map<String, Object> golden = Fixtures.load("typed_decisions.json");
        Assumptions.assumeFalse(golden.containsKey("skipped"),
                "typed_decisions.json was recorded without a graph: set " + Fixtures.TYPED_GRAPH_ENV
                + " and re-run scripts/gen_fixtures.py");
        return golden;
    }

    /**
     * The shared agent, opened on first use so the assumptions still skip cleanly.
     *
     * <p>Where the identity check lives, so it gates every consumer rather than racing them.
     * JUnit 5 orders nothing by default and this class shares one agent, so whichever test ran
     * first opened the weights and the config comparison was just another test in the list --
     * measured ninth of thirteen, after all eight probability comparisons had already passed
     * against the wrong checkpoint. Asserted here, a mispairing fails every case that touches
     * the weights, with the reason named. The failure is cached because rethrowing it is cheap
     * and reopening 1.7 GB of graph per test is not; {@code shared} is set either way so the
     * open happens once.
     */
    private static synchronized Agent agent(Map<String, Object> golden) throws IOException {
        if (shared == null) {
            // Deliberately NOT guarded by a separate `opened` flag: `openAgent` aborts by
            // assumption when the checkpoint or the graph is missing, and that abort has to be
            // rethrown to every caller, not swallowed into a null `shared`.
            Agent agent = openAgent(golden);
            try {
                assertTypedCheckpoint(golden, agent.config());
            } catch (AssertionError failure) {
                mispaired = failure;
            }
            shared = agent;
        }
        if (mispaired != null) {
            throw mispaired;
        }
        return shared;
    }

    private static Agent openAgent(Map<String, Object> golden) throws IOException {
        String checkpoint = (String) golden.getOrDefault("checkpoint", CHECKPOINT);
        Path model = Fixtures.checkpoint(checkpoint);
        Assumptions.assumeTrue(model != null, Fixtures.missingCheckpoint(checkpoint));
        String graph = System.getenv(Fixtures.TYPED_GRAPH_ENV);
        Assumptions.assumeTrue(graph != null && !graph.isBlank(),
                "set " + Fixtures.TYPED_GRAPH_ENV + " to the laya.onnx exported from the "
                + checkpoint + " checkpoint");
        Path graphPath = Paths.get(graph);
        Path graphDirectory = Files.isDirectory(graphPath) ? graphPath : graphPath.getParent();
        Assumptions.assumeTrue(graphDirectory != null && Files.isDirectory(graphDirectory),
                Fixtures.TYPED_GRAPH_ENV + " does not name a graph: " + graph);
        return Agent.open(model, graphDirectory);
    }
}
