package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * A per-bucket threshold key must name a bucket an answer can produce.
 *
 * <p>This exists because {@code fixtures/confidence_gate.json} cannot hold it. The reference's
 * recorded refusal uses the key {@code 1} — an int — and this port's signature is
 * {@code Map<String, ? extends Number>}, so that case is unrepresentable here and the fixture
 * comparison skips it. A BAD STRING key is representable, the reference refuses it since
 * {@code cb85656}, and nothing in the recorded set reaches that path.
 *
 * <p>Why it matters more than a message: a threshold filed under {@code "choice:99"} is never
 * applied to anything, because no answer ever resolves to that bucket. The caller believes they
 * gated a decision and nothing is gated. Accepting the key the reference refuses converts a loud
 * error into exactly that silence, which is the failure this gate exists to prevent.
 */
final class ConfidenceGateBucketKeyTest {

    @Test
    @DisplayName("a key that names no reachable bucket is refused, with the reference's message")
    void unreachableBucketKeyIsRefused() {
        for (String key : new String[]{"choice:99", "nonsense", "Choice:3-5", "choice:3_5", ""}) {
            IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                    () -> ConfidenceGate.checkMinConfidenceMap(Map.of(key, 0.5)),
                    "expected " + key + " to be refused");
            assertEquals("min_confidence map keys must be a bucket like 'choice:3-5' or "
                    + "'default', got '" + key + "'", refused.getMessage(), "message for " + key);
        }
    }

    @Test
    @DisplayName("every bucket an answer can produce is accepted, and so is default")
    void everyReachableBucketIsAccepted() {
        for (String type : new String[]{"choice", "score", "noul"}) {
            for (String size : new String[]{"2", "3-5", "6-10", "11+"}) {
                String key = type + ":" + size;
                assertDoesNotThrow(() -> ConfidenceGate.checkMinConfidenceMap(Map.of(key, 0.5)),
                        key + " is a bucket optionBucket emits, so it must be accepted");
            }
        }
        assertDoesNotThrow(() -> ConfidenceGate.checkMinConfidenceMap(Map.of("default", 0.5)));
    }

    /**
     * The allowed set and {@link ConfidenceGate#optionBucket} must not drift apart.
     *
     * <p>Asserted through the public API rather than by reading the private set: every bucket the
     * bucketing function can emit is fed back in as a key, so a change to one that is not mirrored
     * in the other fails here rather than in a caller's config.
     */
    @Test
    @DisplayName("optionBucket cannot emit a bucket checkMinConfidenceMap would reject")
    void bucketingAndValidationAgree() {
        for (int options = 0; options <= 40; options++) {
            Map<String, Double> probabilities = new java.util.LinkedHashMap<>();
            for (int i = 0; i < options; i++) {
                probabilities.put("option" + i, 1.0 / Math.max(1, options));
            }
            Answer.Choice choice =
                    new Answer.Choice("option0", probabilities, 0.5, 0.5, 0.5);
            ConfidenceGate.optionBucket(choice).ifPresent(bucket ->
                    assertDoesNotThrow(() -> ConfidenceGate.checkMinConfidenceMap(Map.of(bucket, 0.5)),
                            "optionBucket emitted " + bucket + " but the key check rejects it"));
        }
    }
}
