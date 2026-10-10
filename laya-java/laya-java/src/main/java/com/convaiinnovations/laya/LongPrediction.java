package com.convaiinnovations.laya;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * The result of scanning a state longer than one sequence.
 *
 * <p>Separate from {@link Prediction} rather than bolted onto it, because every answer here
 * carries something an ordinary answer cannot: the window that decided it. Putting that on
 * {@link Answer} would give every answer from {@link Agent#predict} a field that is meaningless
 * for it, and {@code Usage} would need a window count and a truncation COUNT where the single-call
 * shape has a truncation flag.
 *
 * <p>The probability and confidence on each answer are the DECIDING WINDOW'S, not a calibrated
 * number for the whole document. A long document is mostly neutral text, so averaging would drown
 * a localized signal; the reference takes the strongest window for {@code noul} and the most
 * confident one for a choice or a score, and reports that window's own numbers so the fields stay
 * mutually consistent. {@link Window} is what makes that readable rather than mysterious.
 */
public record LongPrediction(String model, Map<String, Windowed> answers, LongUsage usage) {

    public LongPrediction {
        answers = Map.copyOf(answers);
    }

    /**
     * An answer and the window it came from.
     *
     * @param window the deciding window, or null when no window decided -- a state short enough
     *     to answer in one call has no window attribution, which is the reference's rule too
     */
    public record Windowed(Answer answer, Window window) {
    }

    /**
     * Where in the tokenized state the deciding window sat.
     *
     * @param index      which window, counting from zero
     * @param tokenStart first token of the window, into the tokenized state
     * @param tokenEnd   one past its last token, CLAMPED to the end of the state -- the final
     *                   window is shorter than the others whenever the state does not divide
     *                   evenly, and reporting {@code start + window} there would name tokens that
     *                   do not exist
     * @param count      how many windows the scan ran, so a caller can tell "the strongest of 40"
     *                   from "the only one"
     */
    public record Window(int index, int tokenStart, int tokenEnd, int count) {

        public Window {
            if (index < 0 || tokenStart < 0 || tokenEnd < tokenStart || count <= 0) {
                throw new IllegalArgumentException(String.format(
                        "a window must be a non-empty span inside a non-empty scan, got"
                        + " index=%d [%d,%d) of %d", index, tokenStart, tokenEnd, count));
            }
        }

        /** How many state tokens this window covered. */
        public int tokens() {
            return tokenEnd - tokenStart;
        }
    }

    /**
     * Usage for a scan.
     *
     * <p>Not {@link Usage}: across windows the reference SUMS the numeric fields, so
     * {@code truncated} becomes a count of windows that truncated rather than a flag, and
     * {@code windows} has no counterpart at all on the single-call shape. Keeping the two types
     * apart is what stops {@code truncated} meaning two different things depending on which call
     * produced it.
     *
     * @param truncatedWindows  how many windows truncated, summed. For a state answered in one
     *     call this is 1 or 0, so the field means the same thing either way
     * @param truncatedQuestions the LAST window's list, carried as the reference carries it --
     *     which is why {@code truncatedWindows} can be above zero while this is empty
     * @param collapsedOptions  merged across windows, not replaced. It is keyed by question and
     *     set only on windows where option spans actually collapsed, so replacing it would leave
     *     a caller holding whichever collapsing window came last -- and the deciding window is
     *     the most confident one, not the last one
     * @param windows           how many windows were scored. Zero when a state was answered
     *     without the model reading a window
     */
    public record LongUsage(int inputTokens, int outputTokens, int stateTokens,
                            int stateTokensDropped, int truncatedWindows,
                            List<String> truncatedQuestions,
                            Map<String, Usage.CollapsedOptions> collapsedOptions, int windows) {

        public LongUsage {
            truncatedQuestions = List.copyOf(truncatedQuestions);
            collapsedOptions = Map.copyOf(collapsedOptions);
        }
    }

    /** The answer to one question, or null when it was not asked. */
    public Answer answer(String questionId) {
        Windowed windowed = answers.get(questionId);
        return windowed == null ? null : windowed.answer();
    }

    /** The window that decided one question, empty when no window did. */
    public Optional<Window> window(String questionId) {
        Windowed windowed = answers.get(questionId);
        return windowed == null ? Optional.empty() : Optional.ofNullable(windowed.window());
    }
}
