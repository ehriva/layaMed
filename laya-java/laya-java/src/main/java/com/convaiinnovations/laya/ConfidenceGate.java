package com.convaiinnovations.laya;

import com.convaiinnovations.laya.json.PythonJson;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * The opt-in abstention gate: which answers a caller should not act on without review.
 *
 * <p>A gate is a policy, and a policy whose application cannot be observed is not one. So this
 * reports three states rather than a boolean — an answer that carried no usable confidence is
 * {@link Abstention#UNEVALUATED}, which a boolean cannot express, and reporting it as a pass is
 * the same lie as reporting it as a flag.
 *
 * <p><b>This does not mutate.</b> The reference writes {@code abstention},
 * {@code abstention_threshold} and {@code low_confidence} into the answer dicts in place, which
 * a record cannot do. {@link #apply} returns a report keyed the same way as the answers instead,
 * and an UNGATED call returns {@link Optional#empty()} rather than a report full of nulls —
 * because the reference's contract is that an ungated call writes <em>nothing at all</em>, and
 * the presence of the field is what tells a caller the gate ran. A sentinel for the unconfigured
 * case would put that information back into every caller's payload, which is what the contract
 * avoids.
 *
 * <p>The gate reads {@code answerConfidence} — the probability mass on the answer being
 * reported, which is what temperature scaling fits — and falls back to {@code confidence} only
 * when the first is unusable. Those are different quantities on different scales, and
 * {@code confidence} is normalised entropy that does not transfer across option counts, so the
 * order matters and is pinned by the fixtures.
 *
 * <p>A threshold is <em>not</em> a claim that the number is calibrated. "About c of the answers
 * returned at c are correct" holds only after temperatures have been fitted and validated for
 * that checkpoint and question shape.
 */
public final class ConfidenceGate {

    private ConfidenceGate() {
    }

    /** What the gate concluded about one answer. The three states, and only these three. */
    public enum Abstention {
        /** A gate ran and this answer's confidence cleared it. */
        PASSED("passed"),
        /** A gate ran and this answer's confidence fell below it. */
        ABSTAINED("abstained"),
        /** A gate ran and this answer carried no usable confidence, so it could not decide. */
        UNEVALUATED("unevaluated");

        private final String wireName;

        Abstention(String wireName) {
            this.wireName = wireName;
        }

        /** The name the reference writes, which is what a cross-language caller compares. */
        public String wireName() {
            return wireName;
        }
    }

    /**
     * One answer's verdict.
     *
     * @param lowConfidence the reference's {@code low_confidence} flag. Redundant with
     *     {@link Abstention#ABSTAINED} and kept because it is what the wire format carries and
     *     what an existing caller filters on
     * @param threshold     the threshold THIS answer was gated at, which differs per answer
     *     under a per-bucket map. Echoed because the flag consumes the threshold and drops it,
     *     so without this a batch gated per bucket cannot be re-split
     */
    public record Gated(Answer answer, boolean lowConfidence, Abstention abstention,
                        double threshold) {
    }

    /**
     * The calibrated confidence, or empty when the answer did not report a usable one.
     *
     * <p>In the reference this also rejects a bool and a missing field. Neither is representable
     * here — {@link Answer#answerConfidence()} is a primitive {@code double} that every answer
     * has — so NaN and infinity are the only ways to be unusable on this side. The reference's
     * other two paths are unreachable rather than unimplemented.
     */
    public static OptionalDouble answerConfidence(Answer answer) {
        double value = answer.answerConfidence();
        return Double.isFinite(value) ? OptionalDouble.of(value) : OptionalDouble.empty();
    }

    /**
     * The number the gate compares against the threshold, or empty when there is none.
     *
     * <p>{@code answerConfidence} first, falling back to {@code confidence} so an answer
     * carrying only the older field is still gated rather than silently passed. Empty means "no
     * usable number", never an unusable one: a caller that reads the value must be able to tell
     * "nothing to gate on" from "a gate ran on a NaN", and {@link #apply} reports those
     * differently.
     */
    public static OptionalDouble gateConfidence(Answer answer) {
        OptionalDouble calibrated = answerConfidence(answer);
        if (calibrated.isPresent()) {
            return calibrated;
        }
        double fallback = answer.confidence();
        return Double.isFinite(fallback) ? OptionalDouble.of(fallback) : OptionalDouble.empty();
    }

    /**
     * This answer's option-count bucket, in the reference's spelling, or empty when it has none.
     *
     * <p>{@code "choice:2"}, {@code "choice:3-5"}, {@code "score:6-10"}, {@code "noul:2"} and so
     * on. One confidence threshold does not transfer across option counts, which is why a map
     * can gate each bucket at the level its calibration actually earns.
     */
    public static Optional<String> optionBucket(Answer answer) {
        int options;
        if (answer instanceof Answer.Choice choice) {
            options = choice.probabilities().size();
        } else if (answer instanceof Answer.Score score) {
            options = score.probabilities().size();
        } else if (answer instanceof Answer.Noul) {
            // A noul carries no probability map, and the reference falls back to 2 for it rather
            // than refusing to bucket it.
            options = 2;
        } else {
            return Optional.empty();
        }
        if (options == 0) {
            return Optional.empty();
        }
        String size = options <= 2 ? "2" : options <= 5 ? "3-5" : options <= 10 ? "6-10" : "11+";
        return Optional.of(answer.type() + ":" + size);
    }

    /**
     * The threshold this answer's bucket is gated at under a per-bucket map.
     *
     * <p>Falls back to the map's {@code "default"} entry, then to 0.0 — gate nothing — so a
     * bucket the map does not name never abstains by surprise.
     */
    public static double resolve(Answer answer, Map<String, Double> thresholds) {
        Optional<String> bucket = optionBucket(answer);
        if (bucket.isPresent()) {
            Double exact = thresholds.get(bucket.get());
            if (exact != null) {
                return exact;
            }
        }
        return thresholds.getOrDefault("default", 0.0);
    }

    /**
     * Every bucket key an answer can actually produce, plus {@code "default"}.
     *
     * <p>Derived from the same spelling {@link #optionBucket} emits, rather than written out, so
     * the two cannot drift: three question types crossed with the four size bands.
     *
     * <p>This set is the whole point of the check below. The reference validated only that a key
     * was a string until {@code cb85656}; it now requires the key to name a bucket an answer can
     * produce, because a threshold under {@code "choice:99"} is silently never applied -- the
     * caller believes they gated something and nothing is gated. A port that accepts the key the
     * reference refuses turns a loud refusal into exactly that silence.
     */
    private static final java.util.Set<String> ALLOWED_BUCKETS;

    static {
        java.util.Set<String> allowed = new java.util.LinkedHashSet<>();
        for (String type : new String[]{"choice", "score", "noul"}) {
            for (String size : new String[]{"2", "3-5", "6-10", "11+"}) {
                allowed.add(type + ":" + size);
            }
        }
        allowed.add("default");
        ALLOWED_BUCKETS = java.util.Set.copyOf(allowed);
    }

    /** Validates a scalar threshold, returning it. */
    public static double checkMinConfidence(double value) {
        if (!Double.isFinite(value) || value < 0.0 || value > 1.0) {
            throw new IllegalArgumentException(String.format(
                    "min_confidence must be a float in [0.0, 1.0], got %s", value));
        }
        return value;
    }

    /** Validates a per-bucket threshold map, returning it. */
    public static Map<String, Double> checkMinConfidenceMap(Map<String, ? extends Number> map) {
        if (map == null || map.isEmpty()) {
            throw new IllegalArgumentException(
                    "a min_confidence map must be a non-empty map of bucket -> float, got " + map);
        }
        Map<String, Double> out = new LinkedHashMap<>();
        for (Map.Entry<String, ? extends Number> entry : map.entrySet()) {
            if (entry.getKey() == null || !ALLOWED_BUCKETS.contains(entry.getKey())) {
                // Message matched to `laya/confidence.py` character for character, including the
                // `%r` spelling of the offending key, because a caller greps for this line.
                throw new IllegalArgumentException(String.format(
                        "min_confidence map keys must be a bucket like 'choice:3-5' or "
                        + "'default', got %s",
                        entry.getKey() == null ? "None" : PythonJson.repr(entry.getKey())));
            }
            out.put(entry.getKey(), checkMinConfidence(entry.getValue().doubleValue()));
        }
        return Map.copyOf(out);
    }

    /**
     * Gates every answer at one threshold.
     *
     * @param minConfidence the threshold, or null for an UNGATED call
     * @return the report, or empty when no gate was configured — which is how a caller tells
     *     "this run had no gate" from "every answer cleared the gate"
     */
    public static Optional<Map<String, Gated>> apply(Map<String, Answer> answers,
                                                     Double minConfidence) {
        if (minConfidence == null) {
            return Optional.empty();
        }
        double threshold = checkMinConfidence(minConfidence);
        return Optional.of(gate(answers, answer -> threshold, threshold == 0.0));
    }

    /**
     * Gates each answer at its own bucket's threshold.
     *
     * @param minConfidence the map, or null for an UNGATED call
     */
    public static Optional<Map<String, Gated>> apply(Map<String, Answer> answers,
                                                     Map<String, ? extends Number> minConfidence) {
        if (minConfidence == null) {
            return Optional.empty();
        }
        Map<String, Double> thresholds = checkMinConfidenceMap(minConfidence);
        // A map is never the "0.0 clears the flag" case: the reference applies that only to a
        // scalar 0.0, because under a map each answer has its own threshold and one of them
        // being zero says nothing about the others.
        return Optional.of(gate(answers, answer -> resolve(answer, thresholds), false));
    }

    private interface ThresholdFor {
        double of(Answer answer);
    }

    private static Map<String, Gated> gate(Map<String, Answer> answers, ThresholdFor thresholds,
                                           boolean scalarZero) {
        Map<String, Gated> out = new LinkedHashMap<>();
        for (Map.Entry<String, Answer> entry : answers.entrySet()) {
            Answer answer = entry.getValue();
            double threshold = thresholds.of(answer);
            OptionalDouble confidence = gateConfidence(answer);

            boolean low = confidence.isPresent() && confidence.getAsDouble() < threshold;
            if (scalarZero) {
                // A scalar 0.0 was set, so states ARE reported -- but nothing can fall below it,
                // and the reference clears any stale flag rather than leaving one from an
                // earlier, stricter run.
                low = false;
            }
            Abstention state = low ? Abstention.ABSTAINED
                    : confidence.isEmpty() ? Abstention.UNEVALUATED
                    : Abstention.PASSED;
            out.put(entry.getKey(), new Gated(answer, low, state, threshold));
        }
        return Map.copyOf(out);
    }
}
