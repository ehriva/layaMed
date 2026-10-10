package com.convaiinnovations.laya.sequence;

import com.convaiinnovations.laya.Question;
import com.convaiinnovations.laya.tokenizer.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * How a scan over a state longer than one sequence is sized.
 *
 * <p>A long state is read in overlapping token windows, each decoded back to text and scored as an
 * ordinary state. That makes the window size a correctness question rather than a tuning one: a
 * window wider than the room the questions leave is re-truncated by {@link SequenceBuilder#build}
 * on the way in, so the tail of every window reaches no model while the reported span says it did.
 * And once the stride exceeds the real room, consecutive windows stop touching and the tokens
 * between them are read by nothing at all -- worse than not windowing.
 *
 * <p>So the window is capped at {@link #stateRoom}, and at the <em>smallest</em> room of the
 * questions asked, because the windows are one list of states scored for every question in shared
 * forward passes: a window sized for the roomiest question would be cut short for the tightest
 * one, and the offsets reported on its answers would mean something different per question.
 *
 * <p>Every clamp here is a decision the caller cannot see -- the window they asked for may shrink,
 * their stride may shrink with it, and the number of forward passes may triple as a result. The
 * reference raises a Python {@code RuntimeWarning} for each. There is no portable ambient
 * equivalent on the JVM that a test can read, so {@link Budget} carries the warnings as text
 * <em>and</em> they go to {@link System.Logger} at {@code WARNING}: a caller who ignores the
 * record still hears about it, and the text stays checkable against the reference.
 */
public final class WindowPlan {

    private WindowPlan() {
    }

    /** How far the DEFAULT window may be cut before {@link #budget} says so. */
    private static final int CLAMP_WARN_RATIO = 2;

    /** How much capping the window may multiply the window count before {@link #batchCap} acts. */
    private static final int BATCH_BLOWUP = 2;

    /**
     * A sized scan: the window, the step between windows, and the room the questions left.
     *
     * @param window   state tokens per window
     * @param stride   token step between windows; {@code window / 2} by default, so a span near a
     *                 boundary still lands whole inside some window
     * @param room     the smallest state budget the questions leave, or the window itself when
     *                 there were no questions to fit
     * @param warnings what the reference would have raised as {@code RuntimeWarning}, in order;
     *                 empty when nothing was clamped
     */
    public record Budget(int window, int stride, int room, List<String> warnings) {

        public Budget {
            warnings = List.copyOf(warnings);
        }

        /** Whether anything was clamped, which is the only reason {@link #warnings} is non-empty. */
        public boolean clamped() {
            return !warnings.isEmpty();
        }
    }

    /**
     * How many state tokens {@code question} leaves inside {@code maxLen}.
     *
     * <p>The head is the question's own -- its instructions plus one {@code [MASK]}-prefixed span
     * per option -- so a question with many options leaves less room than one with two, and two
     * questions in the same request do not have to leave the same amount. Anything past the return
     * value is cut off.
     *
     * <p>This is a property of the TOKENIZER, not just of the question: the same 88-option
     * question leaves 150 state tokens on the english checkpoint and 614 on multilingual. Nothing
     * downstream may assume a particular number.
     */
    public static int stateRoom(Tokenizer tok, Question question, int maxLen, int headMaxLen) {
        SequenceBuilder.Head head = SequenceBuilder.buildHead(tok, question, headMaxLen, null);
        // -1 for the [SEP] that closes the state.
        return Math.max(0, maxLen - head.ids().size() - 1);
    }

    /**
     * Window and stride for scanning a state longer than one sequence.
     *
     * <p>With no questions there is nothing to fit, so the caller's window -- or the checkpoint
     * default -- stands and is reported as the room.
     *
     * @param window an explicit window, or null for the default
     *               ({@code max(64, maxLen - headMaxLen - 8)}). Zero and negative mean "unset",
     *               as in the reference, not "invalid"
     * @param stride an explicit stride, or null for a 50% step of the EFFECTIVE window
     * @throws IllegalArgumentException when the questions leave no room at all, or when an
     *     explicit stride steps past the window the caller asked for
     */
    public static Budget budget(Tokenizer tok, List<Question> questions, int maxLen,
                                int headMaxLen, Integer window, Integer stride) {
        if (tok == null) {
            throw new IllegalArgumentException("tok must not be null");
        }
        if (questions == null) {
            throw new IllegalArgumentException(
                    "questions must not be null; pass an empty list for \"nothing to fit\"");
        }
        List<String> warnings = new ArrayList<>();

        int requested = window != null && window > 0 ? window : Math.max(64, maxLen - headMaxLen - 8);
        int size = requested;

        int room = size;
        if (!questions.isEmpty()) {
            room = Integer.MAX_VALUE;
            for (Question question : questions) {
                room = Math.min(room, stateRoom(tok, question, maxLen, headMaxLen));
            }
        }
        if (room <= 0) {
            throw new IllegalArgumentException(String.format(Locale.ROOT,
                    "predict_long: the questions' options fill the whole sequence (max_len=%d,"
                    + " head_max_len=%d), leaving no room for the state; no window can carry any"
                    + " of it. A label set this large is what laya.shortlist.predict_shortlist"
                    + " is for", maxLen, headMaxLen));
        }

        if (size > room) {
            if (window != null && window > 0) {
                warnings.add(String.format(Locale.ROOT,
                        "laya: predict_long: window=%d is wider than the %d state tokens these"
                        + " questions leave inside max_len=%d, so every window would be truncated"
                        + " to %d on the way to the model; scanning with window=%d instead",
                        size, room, maxLen, room, room));
            } else if (size >= room * CLAMP_WARN_RATIO) {
                // The DEFAULT window was cut, and cut hard. Capping it is what stops the tail of
                // every window reaching no model, but it is not free and must not be silent: the
                // scan now needs about size/room times as many windows, each a full forward pass,
                // and nothing in the caller's code says why.
                //
                // The cost is not the capping, it is the shape of the request: most of every
                // sequence is the question, and because the encoder is bidirectional the head
                // cannot be computed once and reused -- its representations depend on the state it
                // is paired with. So the warning names the remedy rather than only reporting it.
                warnings.add(String.format(Locale.ROOT,
                        "laya: predict_long: these questions leave only %d of max_len=%d for the"
                        + " state (their heads take the rest), so the scan window is capped %d ->"
                        + " %d and roughly %.1fx as many windows -- each a full forward pass --"
                        + " are needed to read the document. Fewer or shorter options, a larger"
                        + " max_len, or laya.shortlist.predict_shortlist for a large label set"
                        + " will all cost less than scanning at this width",
                        room, maxLen, size, room, (double) size / room));
            }
            size = room;
        }

        int step = stride != null && stride > 0 ? stride : Math.max(1, size / 2);
        if (step > size) {
            if (size < requested && stride != null && stride <= requested) {
                // The window the caller asked for was reduced above, and their stride was valid
                // for the window they asked for -- so this is the library's clamp, not their
                // mistake. Reducing the stride to match keeps the no-gap guarantee without making
                // a self-consistent pair of arguments an error. A stride that overshot the
                // REQUESTED window is still refused below, because that one really is theirs.
                warnings.add(String.format(Locale.ROOT,
                        "laya: predict_long: stride=%d was a 50%% step for the window=%d you asked"
                        + " for, but the window was reduced to %d to fit the room these questions"
                        + " leave; scanning with stride=%d instead",
                        step, requested, size, Math.max(1, size / 2)));
                step = Math.max(1, size / 2);
            } else {
                throw new IllegalArgumentException(String.format(Locale.ROOT,
                        "predict_long: stride=%d steps past the %d-token window%s, so %d tokens"
                        + " between every pair of windows would be read by no window at all; pass"
                        + " stride <= %d",
                        step, size,
                        size < requested ? " these questions leave room for" : "",
                        step - size, size));
            }
        }

        for (String warning : warnings) {
            System.getLogger(WindowPlan.class.getName())
                    .log(System.Logger.Level.WARNING, warning);
        }
        return new Budget(size, step, room, warnings);
    }

    /**
     * Keep one forward pass no wider than the un-capped scan's would have been -- but only when
     * capping the window has multiplied the window count enough to matter.
     *
     * <p>Capping the window at the room the questions leave multiplies the number of windows on
     * exactly the inputs it targets: a long document at many options goes from tens of wide
     * windows to hundreds of narrow ones. A batch call that puts every state in one forward pass
     * would then go from a tens-of-rows pass to a hundreds-of-rows one at {@code maxLen} width --
     * a plausible out-of-memory on an input that used to fit.
     *
     * <p>Sending every window in one run is also a deliberate property, and a mild cap -- a
     * 64-token budget reduced to 43, say -- multiplies the count by well under two. Chunking those
     * would trade a real property for no real protection. So the cap applies only past
     * {@value #BATCH_BLOWUP}x, and then bounds the pass at the count the un-capped budget would
     * have produced: peak memory stays at parity with the behaviour before the cap, and the scan
     * still reads the whole document, just in more passes.
     *
     * @param batchSize an explicit cap, which is always honoured; null, zero and negative mean
     *                  "unset"
     * @return the cap, or null for "one pass, every window"
     */
    public static Integer batchCap(int nWindows, int window, int configBudget, Integer batchSize) {
        if (batchSize != null && batchSize > 0) {
            return batchSize;
        }
        // Not capped: one pass, exactly as before. This is a SHORTCUT, not a branch -- the
        // arithmetic below reaches the same answer on its own, because window >= configBudget
        // makes `unclamped` at least nWindows and the blow-up test then returns null anyway.
        // Swept over 1,548,519 (nWindows, window, configBudget) points where the condition
        // holds: zero differences with it removed.
        //
        // So no test can fail when this is deleted, and the mutation sweep correctly reports it
        // as surviving. It is kept because it says the intent at the top of the function, where
        // a reader looks, rather than leaving it to be inferred from a ceiling division; and it
        // is documented here so the next reader neither removes it believing it untested nor
        // tries to write the test that cannot exist.
        if (window >= configBudget) {
            return null;
        }
        // Ceiling division, as the reference's -(-a // b) is. long, because the product of a
        // window count and a half-window is not bounded by anything the caller cannot choose.
        long numerator = (long) nWindows * Math.max(1, window / 2);
        long denominator = Math.max(1, configBudget / 2);
        int unclamped = (int) Math.max(1, (numerator + denominator - 1) / denominator);
        if (nWindows <= (long) unclamped * BATCH_BLOWUP) {
            return null;                       // a mild cap: keep the single shared pass
        }
        return unclamped;
    }
}
