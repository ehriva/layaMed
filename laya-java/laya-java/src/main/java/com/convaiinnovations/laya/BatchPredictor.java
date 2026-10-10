package com.convaiinnovations.laya;

import java.util.List;
import java.util.Map;

/**
 * Anything that can answer one question set over many states in a shared forward pass.
 *
 * <p>Separate from {@link Predictor} rather than folded into it, because not every runtime
 * batches. The reference draws the same line at runtime, raising {@code TypeError} when a runner
 * has no {@code predict_batch} rather than looping {@code decide} N times behind the caller's
 * back; here the line is drawn by the type, so the same mistake does not compile.
 *
 * <p>{@link Agent} shares one graph call across the states. {@link Router} routes each state and
 * shares one call per checkpoint, so states that route apart are still answered together where
 * they can be.
 */
public interface BatchPredictor {

    /**
     * Ask every question about every state.
     *
     * @return one prediction per state, <b>in the order the states were given</b>, which is the
     *     property a caller zipping the results back onto its own inputs depends on
     */
    List<Prediction> predictBatch(List<?> states, Map<String, Question> questions);
}
