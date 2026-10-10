package com.convaiinnovations.laya.lang;

import com.convaiinnovations.laya.decode.Rounding;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Script and language detection, used to route a state between the English and multilingual
 * checkpoints. A port of {@code laya/lang.py}.
 *
 * <p>Routing needs one decision: <em>is this English Latin-script text, or is it something the
 * English checkpoint cannot read?</em> Script detection is exact. The Latin-script language guess
 * is a stopword and diacritic heuristic and is explicitly best-effort -- pass an explicit model or
 * language when you already know it.
 *
 * <p><b>Why the regexes are hand-written scanners.</b> The reference uses four patterns, and
 * {@link java.util.regex} is not a drop-in for any of them. Java's {@code \\w} is a different set
 * from Python's (it admits combining marks and join controls), its {@code \\d} and {@code \\p{L}}
 * follow the JDK's Unicode version rather than CPython's, and {@code String.split} drops trailing
 * empty fields where Python's keeps them. Each pattern is therefore written out as a scan over
 * {@link UnicodeTables}, whose predicates are recorded from CPython. The identifier scan also
 * reproduces the reference's lookbehind, which is what keeps it linear: without it the pattern
 * retries at every offset inside a run of word characters, and a state is user input.
 */
public final class LanguageDetection {

    /** How much of a state detection reads, matching the reference's {@code max_chars}. */
    public static final int MAX_CHARS = 4000;

    /** Nesting depth beyond which a state is not descended into. */
    private static final int MAX_DEPTH = 6;

    /** Below this code point every letter is Latin, so script lookup can be skipped. */
    private static final int LATIN_BELOW_COUNTS = 0x02B0;

    /**
     * The same cut for {@code scriptOf}, which the reference sets lower than the one above.
     *
     * <p>The two differ on U+0250..U+02AF, the IPA extensions: a letter there is counted as Latin
     * but names no script, so it can neither start nor extend a non-Latin word run. Kept as two
     * constants rather than unified, because unifying them would change which runs
     * {@code nonLatinWords} reports.
     */
    private static final int LATIN_BELOW_SCRIPT_OF = 0x0250;

    /** The tally slot every letter no named script claims is counted under. */
    private static final int OTHER_SLOT = LanguageTables.SCRIPT_COUNT;

    /** Non-English Latin letters, sorted for binary search; the set itself is a generated table. */
    private static final int[] DIACRITICS = sortedCodePoints(LanguageTables.NON_EN_DIACRITICS);

    /** The Swedish phrase openers {@code latinProfile} accepts a short login request on. */
    private static final Set<String> SWEDISH_PHRASE_OPENERS = Set.of("kan", "jag", "vi");

    /**
     * How many letters a matched stopword needs before an all-caps line is believed.
     *
     * <p>{@code LOS}, {@code LAS} and {@code EL} are three, three and two letters, so any bar of
     * four or more rejects them. The reference's sweep over 500,000 acronym runs and 50,000
     * upper-cased US address lines reports that a bar of three still reads 3,751 acronym runs and
     * 109 address lines as prose, while four closes both columns outright and five buys nothing
     * measurable at the cost of 1,860 more real lines. Non-English diacritics are accepted in
     * place of a long stopword, because the module already treats them as non-English evidence
     * ({@link LanguageTables#NON_EN_DIACRITIC_RATE}) and neither an acronym nor a US place name
     * carries any.
     */
    private static final int SHOUTED_MIN_STOPWORD = 4;

    /**
     * How many letters make a token a word rather than an acronym.
     *
     * <p>Used twice -- on an all-caps line, by {@link #shoutedEvidence}, and in
     * {@link #analyseText} to decide whether an all-caps run inside mixed-case text is an acronym
     * worth blanking. The reference states plainly that the sweep above does not justify this bar
     * on the committed corpora -- at a stopword bar of four both false-positive columns are
     * already zero -- and keeps it because the larger uncommitted corpora it was first chosen
     * against did report acronym false positives surviving a stopword bar of four. It is costly:
     * 2,198 of 10,960 shouted-prose lines are given up, 82.62% of them six tokens or longer.
     */
    private static final int SHOUTED_MIN_WORD = 5;

    private LanguageDetection() {
    }

    /** The evidence behind the Latin-script language guess. {@code language} is null when undecided. */
    public record LatinProfile(String language, int englishHits, double diacriticRate,
                               boolean looksNonEnglish) {
    }

    /**
     * A full detection result.
     *
     * @param script           dominant script: {@code latin}, {@code han}, ... or {@code unknown}
     * @param scriptProfile    fraction of alphabetic characters per script, Latin first
     * @param language         best-effort language code, or null when undecided
     * @param english          whether the English checkpoint can be expected to read this
     * @param languageUndecided whether the language could not be named
     * @param diacriticRate    share of characters that are non-English Latin letters
     * @param nonLatinFraction share of letters outside Latin
     * @param mixedSegment     the line or field that made a mostly English state non-English
     */
    public record Analysis(String script, Map<String, Double> scriptProfile, String language,
                           boolean english, boolean languageUndecided, double diacriticRate,
                           double nonLatinFraction, String mixedSegment) {
    }

    // ------------------------------------------------------------------ state flattening

    /**
     * The string leaves of a state, which is what detection reads.
     *
     * <p>Keys are ignored: they are usually English field names. {@code byte[]} stands for the
     * reference's {@code bytes} and is decoded strictly, so an undecodable leaf contributes
     * nothing rather than mojibake. A {@code Set} is deliberately not descended into even though
     * it is a {@code Collection}: its iteration order is unspecified, and detection must not
     * depend on it.
     */
    private static List<String> iterText(Object state) {
        List<String> out = new ArrayList<>();
        collectText(state, 0, out);
        return out;
    }

    private static void collectText(Object state, int depth, List<String> out) {
        if (depth > MAX_DEPTH || state == null) {
            return;
        }
        if (state instanceof String) {
            out.add((String) state);
        } else if (state instanceof byte[]) {
            String decoded = decodeUtf8((byte[]) state);
            if (decoded != null) {
                out.add(decoded);
            }
        } else if (state instanceof Map) {
            for (Object value : ((Map<?, ?>) state).values()) {
                collectText(value, depth + 1, out);
            }
        } else if (state instanceof List) {
            for (Object value : (List<?>) state) {
                collectText(value, depth + 1, out);
            }
        }
    }

    /** Strict UTF-8, or null for what the reference catches as {@code UnicodeDecodeError}. */
    private static String decodeUtf8(byte[] bytes) {
        CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT);
        try {
            return decoder.decode(ByteBuffer.wrap(bytes)).toString();
        } catch (CharacterCodingException e) {
            return null;
        }
    }

    /** Flatten a state into the text used for detection. */
    public static String stateText(Object state) {
        return stateText(state, MAX_CHARS);
    }

    /** Flatten a state into at most {@code maxChars} code points of text. */
    public static String stateText(Object state, int maxChars) {
        List<String> parts = new ArrayList<>();
        int budget = maxChars;
        for (String leaf : iterText(state)) {
            if (budget <= 0) {
                break;
            }
            int length = codePointLength(leaf);
            if (length > budget) {
                parts.add(head(leaf, budget));
                break;
            }
            parts.add(leaf);
            // Account for the joining space without materialising the full text first.
            budget -= length + 1;
        }
        return head(String.join(" ", parts), maxChars);
    }

    // ------------------------------------------------------------------ scripts

    /**
     * Count the alphabetic characters of {@code text} by script, in one pass.
     *
     * <p>Latin is inserted last so that {@code scriptFromCounts} keeps the tie-break: a named
     * script wins a tie against Latin, because the maximum is the first of equal values and Latin
     * is the last key. Both the dominant script and the per-script fractions are read off one
     * pass.
     */
    static Map<String, Integer> scriptCounts(String text) {
        // Tallied into an int[] and named once at the end, rather than a map update per letter:
        // the reference hashes a string per non-Latin character, and so does laya-ts.
        int[] tally = new int[OTHER_SLOT + 1];
        int[] firstSeen = new int[OTHER_SLOT + 1];
        int distinct = 0;
        int latin = 0;
        int i = 0;
        while (i < text.length()) {
            int cp = text.codePointAt(i);
            i += Character.charCount(cp);
            if (!UnicodeTables.isAlpha(cp)) {
                continue;
            }
            if (isLatinRange(cp, LATIN_BELOW_COUNTS)) {
                latin++;
                continue;
            }
            int index = LanguageTables.lookupScriptIndex(cp);
            // An alphabetic character no range claims is counted under "other" rather than
            // nowhere: text written only in an unlisted script would otherwise total zero, read
            // as "unknown", and be handed to the English checkpoint that has no tokens for it.
            int slot = index < 0 ? OTHER_SLOT : index;
            if (tally[slot]++ == 0) {
                firstSeen[distinct++] = slot;
            }
        }
        // Insertion order is first-encounter order in the text, which is what the reference's
        // dict gives and what `profileFromCounts` carries through to the public profile.
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (int n = 0; n < distinct; n++) {
            int slot = firstSeen[n];
            counts.put(slot == OTHER_SLOT ? "other" : LanguageTables.scriptName(slot), tally[slot]);
        }
        counts.put("latin", latin);
        return counts;
    }

    /** Latin, IPA extensions, Latin Extended-Additional and the fullwidth Latin letters. */
    private static boolean isLatinRange(int cp, int below) {
        return cp < below
                || (cp >= 0x1E00 && cp <= 0x1EFF)
                || (cp >= 0xFF21 && cp <= 0xFF3A)
                || (cp >= 0xFF41 && cp <= 0xFF5A);
    }

    static String scriptFromCounts(Map<String, Integer> counts) {
        String best = null;
        int bestCount = 0;
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            if (best == null || entry.getValue() > bestCount) {
                best = entry.getKey();
                bestCount = entry.getValue();
            }
        }
        return bestCount == 0 ? "unknown" : best;
    }

    static Map<String, Double> profileFromCounts(Map<String, Integer> counts) {
        int total = 0;
        for (int value : counts.values()) {
            total += value;
        }
        if (total == 0) {
            return Map.of();
        }
        // Latin first in the public mapping, the order callers saw before this was refactored.
        Map<String, Double> profile = new LinkedHashMap<>();
        Integer latin = counts.get("latin");
        if (latin != null && latin != 0) {
            profile.put("latin", latin / (double) total);
        }
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            if (!"latin".equals(entry.getKey()) && entry.getValue() != 0) {
                profile.put(entry.getKey(), entry.getValue() / (double) total);
            }
        }
        return Collections.unmodifiableMap(profile);
    }

    /** Dominant script of {@code text}, or {@code unknown} when it holds no letters. */
    public static String detectScript(String text) {
        return scriptFromCounts(scriptCounts(text));
    }

    /** Fraction of alphabetic characters belonging to each detected script. */
    public static Map<String, Double> scriptProfile(String text) {
        return profileFromCounts(scriptCounts(text));
    }

    /** The named non-Latin script of one letter, or null for Latin and for unclaimed letters. */
    static String scriptOf(int cp) {
        int index = scriptIndexOf(cp);
        return index < 0 ? null : LanguageTables.scriptName(index);
    }

    /**
     * {@link #scriptOf} as an index, which is what the run scan compares.
     *
     * <p>{@code scriptOf} delegates here rather than the other way round, so there is one
     * implementation of the rule and the two cannot drift.
     */
    private static int scriptIndexOf(int cp) {
        if (isLatinRange(cp, LATIN_BELOW_SCRIPT_OF)) {
            return -1;
        }
        return LanguageTables.lookupScriptIndex(cp);
    }

    /**
     * Whether the non-Latin letters of a Latin-plurality text are enough to carry the state.
     *
     * <p>Named here rather than left inline, because the caps bars of {@link #analyseText} ask the
     * same question for the opposite reason: a state this far from Latin is kept out of them.
     *
     * @param nonLatin the non-Latin share of the alphabetic characters
     * @param nonLatinLetters how many letters that share works out to
     */
    static boolean nonLatinCarries(double nonLatin, int nonLatinLetters) {
        return nonLatin >= LanguageTables.NON_LATIN_FRACTION
                || (nonLatin >= LanguageTables.NON_LATIN_MIN_FRACTION
                    && nonLatinLetters >= LanguageTables.NON_LATIN_MIN_LETTERS);
    }

    /**
     * Non-Latin runs that read as words rather than as annotation inside English prose.
     *
     * <p>English prose carries three kinds of non-Latin letter that are not a request written in
     * another script, and each is excluded here: a symbol ({@code Set alpha to 0.05}, one letter),
     * a proper name (capitalised), and a pronunciation in IPA, which no script range claims. A
     * combining mark belongs to the letter before it and never splits a word, so an accented
     * Cyrillic name stays one capitalised word rather than becoming two.
     */
    static List<String> nonLatinWords(String text) {
        List<String> runs = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        int script = -1;
        int i = 0;
        while (i < text.length()) {
            int cp = text.codePointAt(i);
            i += Character.charCount(cp);
            if (UnicodeTables.isCombining(cp)) {
                continue;
            }
            int here = scriptIndexOf(cp);
            if (here >= 0 && here == script) {
                current.appendCodePoint(cp);
                continue;
            }
            if (current.length() > 0) {
                runs.add(current.toString());
            }
            current = new StringBuilder();
            if (here >= 0) {
                current.appendCodePoint(cp);
            }
            script = here;
        }
        if (current.length() > 0) {
            runs.add(current.toString());
        }
        List<String> out = new ArrayList<>();
        for (String run : runs) {
            if (codePointLength(run) >= 2 && !UnicodeTables.isUpper(run.codePointAt(0))) {
                out.add(run);
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ the four patterns

    /** The reference's {@code [^\\W\\d_]+}: maximal runs of word characters that are not digits. */
    static List<String> words(String text) {
        List<String> out = new ArrayList<>();
        int i = 0;
        int length = text.length();
        while (i < length) {
            int cp = text.codePointAt(i);
            int width = Character.charCount(cp);
            if (!UnicodeTables.isWordChar(cp)) {
                i += width;
                continue;
            }
            int start = i;
            while (i < length) {
                int inner = text.codePointAt(i);
                if (!UnicodeTables.isWordChar(inner)) {
                    break;
                }
                i += Character.charCount(inner);
            }
            out.add(text.substring(start, i));
        }
        return out;
    }

    /** A character of the reference's {@code [\\w-]}. */
    private static boolean isIdentifierChar(int cp) {
        return cp == '-' || UnicodeTables.isPythonWordChar(cp);
    }

    /**
     * The reference's {@code (?<![\\w-])[\\w-]*(?:[.@][\\w-]+)+} replaced by a space.
     *
     * <p>A token whose dot or at-sign joins word characters is an identifier, not prose:
     * {@code github.com}, {@code user@acme.com}, {@code v1.2.3}, {@code U.S.A.}. The word scan
     * splits them into pieces that collide with real function words -- {@code com} is Portuguese
     * for "with", {@code o} is its article -- so a state that was mostly links scored a language
     * it does not contain. A sentence-final period keeps its word: there must be word characters
     * on both sides of the dot.
     *
     * <p>No backtracking is needed, which is what makes this linear. {@code [.@]} and
     * {@code [\\w-]} are disjoint, so once a maximal run of identifier characters is consumed the
     * next character either is a joiner or is not, and giving a character back cannot help. The
     * lookbehind means a match can only begin where a run begins.
     */
    static String substituteIdentifiers(String text) {
        StringBuilder out = null;
        int i = 0;
        int length = text.length();
        int copied = 0;
        while (i < length) {
            int cp = text.codePointAt(i);
            int width = Character.charCount(cp);
            boolean atRunStart = i == 0 || !isIdentifierChar(text.codePointBefore(i));
            if (!atRunStart) {
                i += width;
                continue;
            }
            int end = matchIdentifier(text, i);
            if (end < 0) {
                i += width;
                continue;
            }
            if (out == null) {
                out = new StringBuilder(length);
            }
            out.append(text, copied, i).append(' ');
            i = end;
            copied = end;
        }
        if (out == null) {
            return text;
        }
        return out.append(text, copied, length).toString();
    }

    /** End offset of an identifier match starting at {@code from}, or -1 when there is none. */
    private static int matchIdentifier(String text, int from) {
        int i = runEnd(text, from);
        int groups = 0;
        while (i < text.length()) {
            char joiner = text.charAt(i);
            if (joiner != '.' && joiner != '@') {
                break;
            }
            int after = runEnd(text, i + 1);
            if (after == i + 1) {
                break;                      // the joiner is not followed by [\w-]+
            }
            i = after;
            groups++;
        }
        return groups > 0 ? i : -1;
    }

    /** End of the maximal run of identifier characters starting at {@code from}. */
    private static int runEnd(String text, int from) {
        int i = from;
        while (i < text.length()) {
            int cp = text.codePointAt(i);
            if (!isIdentifierChar(cp)) {
                break;
            }
            i += Character.charCount(cp);
        }
        return i;
    }

    /**
     * The reference's {@code [=;{}\\[\\]]|\\w\\(}: does this line carry code syntax?
     *
     * <p>Code is not prose in any language, but split into words it reads as one: {@code os.path}
     * is Portuguese, {@code round(el, 2)} Spanish. Prose keeps "Deu erro (500)": the parenthesis
     * follows a space rather than a word character.
     */
    static boolean hasCodeLine(String text) {
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (ch == '=' || ch == ';' || ch == '{' || ch == '}' || ch == '[' || ch == ']') {
                return true;
            }
            if (ch == '(' && i > 0 && UnicodeTables.isPythonWordChar(text.codePointBefore(i))) {
                return true;
            }
        }
        return false;
    }

    /**
     * The reference's {@code [^\\W_][._/\\\\][^\\W_]}: is this token a compound name?
     *
     * <p>{@code Nav/Com} and {@code OS/2} read as Portuguese, {@code C:\\DOS\\mode} too. A token
     * holding a letter or digit, a joiner, and another letter or digit is dropped whole.
     */
    static boolean hasJoined(String token) {
        int i = 0;
        int length = token.length();
        while (i < length) {
            int cp = token.codePointAt(i);
            int width = Character.charCount(cp);
            if ((cp == '.' || cp == '_' || cp == '/' || cp == '\\')
                    && i > 0 && i + width < length
                    && UnicodeTables.isWordOrDigit(token.codePointBefore(i))
                    && UnicodeTables.isWordOrDigit(token.codePointAt(i + width))) {
                return true;
            }
            i += width;
        }
        return false;
    }

    /**
     * The reference's {@code [^\\W\\d_]{2,}} substitution that blanks all-caps runs.
     *
     * <p>An all-caps token inside mixed-case text is an acronym or a code: {@code MON}, {@code LA},
     * {@code EST}, {@code COM}, {@code DES} are hockey teams, states, time zones and radio bands,
     * not French or Portuguese. A segment written entirely in capitals keeps its words -- a
     * customer shouting in Portuguese is still Portuguese -- which is why the callers apply this
     * only to text that is not {@link #shouted}, and hold the shouted case to
     * {@link #shoutedEvidence} instead.
     *
     * <p>This blanks <em>every</em> all-caps run, which is what the reference records as a helper
     * and what the fixture pins. {@link #analyseText} needs the narrower rule -- only runs short
     * enough to be acronyms -- and filters before calling, rather than this taking a flag.
     */
    static String blankUpperRuns(String text) {
        StringBuilder out = new StringBuilder(text.length());
        int i = 0;
        int length = text.length();
        while (i < length) {
            int cp = text.codePointAt(i);
            int width = Character.charCount(cp);
            if (!UnicodeTables.isWordChar(cp)) {
                out.appendCodePoint(cp);
                i += width;
                continue;
            }
            int start = i;
            while (i < length) {
                int inner = text.codePointAt(i);
                if (!UnicodeTables.isWordChar(inner)) {
                    break;
                }
                i += Character.charCount(inner);
            }
            String run = text.substring(start, i);
            if (codePointLength(run) >= 2 && UnicodeTables.isUpperString(run)) {
                out.append(' ');
            } else {
                out.append(run);
            }
        }
        return out.toString();
    }

    /**
     * The all-caps runs {@link #blankUpperRuns} would blank, in order.
     *
     * <p>The reference reads them off the same {@code finditer} it substitutes with, so the two
     * cannot disagree about what a run is. Java has no equivalent of handing one scan to both a
     * substitution and a match list without either a regex -- whose {@code \\w} is a different
     * set from Python's -- or a callback, so the scan is written twice and the agreement is
     * stated here instead: both build maximal runs of {@link UnicodeTables#isWordChar} and keep
     * the ones of at least two code points that {@link UnicodeTables#isUpperString} accepts.
     */
    private static List<String> upperLetterRuns(String text) {
        List<String> runs = new ArrayList<>();
        int i = 0;
        int length = text.length();
        while (i < length) {
            int cp = text.codePointAt(i);
            if (!UnicodeTables.isWordChar(cp)) {
                i += Character.charCount(cp);
                continue;
            }
            int start = i;
            while (i < length) {
                int inner = text.codePointAt(i);
                if (!UnicodeTables.isWordChar(inner)) {
                    break;
                }
                i += Character.charCount(inner);
            }
            String run = text.substring(start, i);
            if (codePointLength(run) >= 2 && UnicodeTables.isUpperString(run)) {
                runs.add(run);
            }
        }
        return runs;
    }

    // ------------------------------------------------------------------ the Latin guess

    /**
     * Whether plain-English function words outvote a marginal diacritic rate.
     *
     * <p>The rate is measured over every character, so one accented loanword or proper noun in a
     * short sentence clears the floor outright. English still wins when it shows at least two
     * distinct function words no other list holds and at most one word carrying a non-English
     * letter: one loanword is not a non-English vocabulary, while the odd word a Danish or Swedish
     * sentence picks up is not an English sentence either. A rate well above the rescue rate
     * vetoes regardless.
     */
    static boolean englishRescuedByWords(List<String> words, double diacriticRate) {
        if (diacriticRate >= LanguageTables.ENGLISH_RESCUE_DIACRITIC_RATE) {
            return false;
        }
        Set<String> unique = new LinkedHashSet<>(words);
        int englishOnly = 0;
        for (String word : unique) {
            if (LanguageTables.EN_ONLY_WORDS.contains(word)) {
                englishOnly++;
            }
        }
        if (englishOnly < 2) {
            return false;
        }
        int withDiacritic = 0;
        for (String word : unique) {
            if (hasDiacritic(word)) {
                withDiacritic++;
            }
        }
        return withDiacritic <= 1;
    }

    /**
     * The evidence behind the Latin-script language guess.
     *
     * <p>A non-English language is only named when it matched at least one word that no other list
     * claims: shared function words alone identify no particular language. "Undecided" and
     * "English" are different answers and only one of them is safe to send to the English
     * checkpoint, which is why the evidence is returned and not just the verdict.
     */
    public static LatinProfile latinProfile(String text) {
        // U+0130 lowercases to i plus a combining dot, which matches no word list.
        List<String> words = words(pythonLower(substituteIdentifiers(text).replace("\u0130", "i")));
        String lowered = pythonLower(text);
        int diacritics = 0;
        int i = 0;
        while (i < lowered.length()) {
            int cp = lowered.codePointAt(i);
            i += Character.charCount(cp);
            if (isDiacritic(cp)) {
                diacritics++;
            }
        }
        double diacriticRate = diacritics / (double) Math.max(1, codePointLength(lowered));
        boolean nonEnglish = diacriticRate >= LanguageTables.NON_EN_DIACRITIC_RATE;
        Set<String> unique = new LinkedHashSet<>(words);
        boolean nordicOverlap = intersects(unique, LanguageTables.NORDIC_OVERLAP_WORDS)
                && !intersects(unique, LanguageTables.EN_ONLY_WORDS);

        if (words.size() > 1 && words.size() < 4
                && intersects(unique, LanguageTables.SHORT_SWEDISH_WORDS)) {
            return new LatinProfile("sv", 0, diacriticRate, nonEnglish);
        }
        if (words.size() < 4) {
            return new LatinProfile(null, 0, diacriticRate, nonEnglish || nordicOverlap);
        }

        // A collision word counts once however often it repeats; every other word counts its hits.
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (String word : words) {
            counts.merge(word, 1, Integer::sum);
        }
        Map<String, Integer> scores = new LinkedHashMap<>();
        for (Map.Entry<String, Set<String>> entry : LanguageTables.STOP_WORDS.entrySet()) {
            Set<String> stop = entry.getValue();
            int score = 0;
            for (Map.Entry<String, Integer> seen : counts.entrySet()) {
                if (stop.contains(seen.getKey())) {
                    score += LanguageTables.EN_COLLISION_WORDS.contains(seen.getKey())
                            ? 1 : seen.getValue();
                }
            }
            scores.put(entry.getKey(), score);
        }
        int english = scores.getOrDefault("en", 0);

        // Only a language that matched at least one word no other list claims may be named.
        // Without that condition the top score can be pure overlap -- `la` and `e` in Romanian
        // text made Italian the winner -- which is a guess dressed as a detection. Such a
        // language is dropped from the running rather than merely losing the tie, so a lesser
        // score with real evidence still gets named, and the text stays undecided when no list
        // has any.
        String bestLanguage = null;
        int best = 0;
        boolean seenAny = false;
        for (Map.Entry<String, Integer> entry : scores.entrySet()) {
            String language = entry.getKey();
            if ("en".equals(language) || !hasOwnEvidence(unique, language)) {
                continue;
            }
            if (!seenAny || entry.getValue() > best) {
                bestLanguage = language;
                best = entry.getValue();
                seenAny = true;
            }
        }

        String language = null;
        if (bestLanguage != null && best >= Math.max(2, english + 2)) {
            // a non-English language needs a clear margin over English function words
            language = bestLanguage;
        } else if ("sv".equals(bestLanguage) && words.contains("inte") && words.contains("kan")
                && SWEDISH_PHRASE_OPENERS.contains(words.get(0)) && english <= 1) {
            // Short login requests such as "kan inte logga in" carry a distinctive Swedish phrase
            // but also one English-shaped token (`in`). Do not treat `kan` alone as Swedish: it is
            // common in Danish and Norwegian too.
            language = bestLanguage;
        } else if (bestLanguage != null && nonEnglish && best >= Math.max(2, english)) {
            // Needs two hits here too. One shared function word ("para" in Turkish text) named
            // Spanish on the strength of the diacritics alone, which is a guess dressed as a
            // detection.
            language = bestLanguage;
        } else if (english != 0
                && (!nonEnglish || englishRescuedByWords(words, diacriticRate))) {
            language = "en";
        }
        boolean looksNonEnglish = nonEnglish || (language == null && nordicOverlap);
        return new LatinProfile(language, english, diacriticRate, looksNonEnglish);
    }

    /** Whether {@code language} matched a word of its own, rather than only shared ones. */
    private static boolean hasOwnEvidence(Set<String> words, String language) {
        Set<String> stop = LanguageTables.STOP_WORDS.get(language);
        if (stop == null) {
            return false;
        }
        for (String word : words) {
            if (stop.contains(word) && !LanguageTables.SHARED_WORDS.contains(word)) {
                return true;
            }
        }
        return false;
    }

    /** Best-effort language code for Latin-script text, or null when undecided. */
    public static String guessLatinLanguage(String text) {
        return latinProfile(text).language();
    }

    /**
     * True for a <em>cased</em> segment written entirely in capitals.
     *
     * <p>The uppercase half is not redundant. A caseless script -- Devanagari, Bengali, CJK -- has
     * no lowercase either, so testing only for the absence of lowercase would call every such
     * segment shouted and hold it to a bar it cannot clear: the word scan splits at every
     * combining mark, so the tokens are short by construction. Nothing is routed on that path
     * today (the Bengali stopwords are romanised, so a Bengali-script line is named by script and
     * never reaches here), but the bar belongs to cased text and saying so here keeps it that way.
     *
     * <p>That argument covers <em>pure</em> caseless text, which never reaches a caps bar at all.
     * Text that merely contains some has no lowercase either and does reach one;
     * {@link #analyseText} keeps both bars off it, because a bar that works by discarding Latin
     * evidence has nothing to say about the half of the state it cannot read.
     *
     * <p>Both halves go through the recorded case tables rather than {@code Character.isUpperCase}
     * and {@code Character.isLowerCase}: Python's {@code isupper} and {@code islower} are the
     * general categories plus the {@code Other_Uppercase} and {@code Other_Lowercase} properties,
     * which the JDK's two methods do not agree with on every code point.
     */
    static boolean shouted(String text) {
        boolean sawUpper = false;
        int i = 0;
        while (i < text.length()) {
            int codePoint = text.codePointAt(i);
            i += Character.charCount(codePoint);
            if (UnicodeTables.isLower(codePoint)) {
                return false;
            }
            if (UnicodeTables.isUpper(codePoint)) {
                sawUpper = true;
            }
        }
        return sawUpper;
    }

    /**
     * Whether an all-caps line's evidence for {@code language} is more than acronym-shaped tokens.
     *
     * <p>A word of {@link #SHOUTED_MIN_WORD} letters is required in every case, and then either
     * non-English diacritics or a matched stopword of {@link #SHOUTED_MIN_STOPWORD} letters.
     * Shouting is not a foreign language: {@code MON}, {@code LA}, {@code EST}, {@code COM} and
     * {@code DES} are hockey teams, states, time zones and radio bands that collide with French
     * stopwords, and nothing but their length tells them apart from words.
     *
     * @param tokens the word-scan tokens of the line
     * @param language the language the Latin guess named
     * @param diacriticRate the rate measured on the line
     */
    static boolean shoutedEvidence(List<String> tokens, String language, double diacriticRate) {
        boolean hasWord = false;
        for (String token : tokens) {
            if (codePointLength(token) >= SHOUTED_MIN_WORD) {
                hasWord = true;
                break;
            }
        }
        if (!hasWord) {
            return false;
        }
        if (diacriticRate >= LanguageTables.NON_EN_DIACRITIC_RATE) {
            return true;
        }
        Set<String> stop = LanguageTables.STOP_WORDS.get(language);
        if (stop == null) {
            // The reference's `_STOP.get(lang, set())` on a language it holds no list for: the
            // intersection is empty, so no stopword can clear the bar.
            return false;
        }
        for (String token : tokens) {
            String lowered = pythonLower(token);
            if (codePointLength(lowered) >= SHOUTED_MIN_STOPWORD && stop.contains(lowered)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Language code for one non-code line, or null when it does not name a foreign language.
     *
     * <p>Same evidence bar as the segment scan: four words, a language the Latin guess will name,
     * and two <em>different</em> words of that language. Acronyms and slash compounds are not
     * words. A segment in all capitals keeps its acronym-shaped tokens -- shouting is not a
     * foreign language -- so it must also clear {@link #shoutedEvidence}.
     */
    static String namedProseLanguage(String segment) {
        if (UnicodeTables.isBlank(segment) || hasCodeLine(segment)) {
            return null;
        }
        List<String> kept = new ArrayList<>();
        for (String token : UnicodeTables.splitOnWhitespace(segment)) {
            if (!hasJoined(token)) {
                kept.add(token);
            }
        }
        String prose = String.join(" ", kept);
        boolean allCaps = shouted(prose);
        if (!allCaps) {
            prose = blankUpperRuns(prose);
        }
        List<String> tokens = words(prose);
        if (tokens.size() < 4) {
            return null;
        }
        LatinProfile profile = latinProfile(prose);
        String language = profile.language();
        if (language == null || "en".equals(language)) {
            return null;
        }
        if (allCaps && !shoutedEvidence(tokens, language, profile.diacriticRate())) {
            return null;
        }
        Set<String> stop = LanguageTables.STOP_WORDS.get(language);
        if (stop == null) {
            return null;
        }
        Set<String> distinct = new LinkedHashSet<>();
        for (String token : tokens) {
            String lowered = pythonLower(token);
            if (stop.contains(lowered)) {
                distinct.add(lowered);
            }
        }
        return distinct.size() < 2 ? null : language;
    }

    /**
     * The first line or field that, read on its own, is named a non-English language.
     *
     * <p>A segment needs the evidence a whole state needs -- at least four words, and a language
     * the Latin guess names -- and, because one line carries far less text than a state, two
     * things more: the words that name the language must be two different ones ("COM ... COM" in
     * an English radio listing is one word seen twice), and acronyms and slash compounds are not
     * words. This adds no new way to call English text foreign; it only stops a longer English
     * part from outvoting a foreign one. Reads at most {@code maxChars} characters in all.
     *
     * @return the language and the segment, or null
     */
    static String[] nonEnglishSegment(Object state, int maxChars) {
        int seen = 0;
        for (String leaf : iterText(state)) {
            for (String line : leaf.split("\n", -1)) {
                if (seen >= maxChars) {
                    return null;
                }
                String segment = head(line, maxChars - seen);
                seen += codePointLength(segment);
                String language = namedProseLanguage(segment);
                if (language != null) {
                    return new String[] {language, UnicodeTables.strip(segment)};
                }
            }
        }
        return null;
    }

    // ------------------------------------------------------------------ the verdict

    /**
     * Detection result for one already-flattened string.
     *
     * <p>Does not look for a foreign line inside mostly-English text; {@link #analyse} does that,
     * because it needs the original state and not only the joined window.
     */
    static Analysis analyseText(String text) {
        Map<String, Integer> counts = scriptCounts(text);
        Map<String, Double> profile = profileFromCounts(counts);
        String script = scriptFromCounts(counts);
        double nonLatin = profile.isEmpty()
                ? 0.0
                : Rounding.round4(1.0 - profile.getOrDefault("latin", 0.0));
        int nonLatinLetters = roundHalfEven(nonLatin * countAlpha(text));
        if ("latin".equals(script) && !nonLatinWords(text).isEmpty()
                && nonLatinCarries(nonLatin, nonLatinLetters)) {
            // Non-Latin text is not for the English checkpoint even when Latin letters are the
            // plurality: a brand name or order code outvotes the CJK request around it letter for
            // letter, though one CJK character carries far more than a letter.
            script = dominantNonLatin(profile);
        }
        if ("unknown".equals(script)) {
            return new Analysis("unknown", profile, null, true, true, 0.0, 0.0, null);
        }
        if (!"latin".equals(script)) {
            return new Analysis(script, profile, null, false, true, 0.0, nonLatin, null);
        }
        LatinProfile latin = latinProfile(text);
        String language = latin.language();
        // The acronym bar belongs here too, not only in the segment scan. The scan runs only once
        // a state already reads English overall, so a state that is *nothing but* an acronym line
        // -- or one diluted by fewer English lines than it takes to tip this verdict -- never
        // reached it and was named foreign outright: "MON DES EST LA" routed multilingual on its
        // own, and so did the same line under one or two lines of English. Vetoing here leaves
        // the text undecided, so looksNonEnglish still decides it: a shouted line with
        // non-English diacritics is kept.
        //
        // Neither bar applies when the state is carried by a caseless script. Both of them answer
        // the verdict by taking evidence *away* from the Latin letters, and looksNonEnglish, which
        // decides what is left, reads Latin diacritics only -- it cannot see the caseless half.
        // Nothing else catches the fall, either: the script promotion above needs nonLatinWords,
        // which drops any run starting with a capital, so a shouted mixed-script line has no
        // promotable word by construction. "PRIVET MON DES EST LA" written in Cyrillic is 35%
        // Cyrillic letters and was being sent to the English checkpoint on the strength of
        // disbelieving "MON DES EST LA". Whatever the Latin part is worth, text this far from
        // Latin is not English, so there is nothing for either bar to buy here and a wrong
        // language name costs nothing: the checkpoint is chosen on is_english.
        if (language != null && !"en".equals(language)
                && !nonLatinCarries(nonLatin, nonLatinLetters)) {
            if (shouted(text)) {
                if (!shoutedEvidence(words(text), language, latin.diacriticRate())) {
                    language = null;
                }
            } else {
                // Mixed-case text: the segment scan has always held that an acronym is not a
                // word, but the whole-state verdict never applied that rule, so the acronyms
                // voted in it. That is what let a short English state be outvoted -- one or two
                // lines of English above "MON DES EST LA" still read as French overall, and only
                // at three did English win the margin. Re-take the verdict without the all-caps
                // runs. looksNonEnglish and the reported diacritic rate stay measured on the
                // original text, so a foreign word that happens to be shouted cannot be blanked
                // out of the diacritic safety net.
                //
                // Only *acronym-shaped* runs are blanked. Blanking every all-caps run deleted
                // emphasis capitals, which are ordinary in real prose and are usually on the
                // words that carry the sentence, so when the shouted words were the ones that
                // named the language the evidence simply went: "sag mir das HEUTIGE DATUM" and
                // "quiero cancelar mi PEDIDO POR FAVOR" both lost their verdict and routed
                // english. It also broke monotonicity -- the fully upper-cased form held the
                // shouted bar, but adding one lowercase English token moved the text to this
                // branch, where it had no bar at all. So a run is blanked only while every one of
                // them is short enough to be an acronym, by shoutedEvidence's own first test: a
                // run of SHOUTED_MIN_WORD letters is a word, and a word is not blanked out of its
                // own sentence. Measured over the 20,708 Latin-locale MASSIVE test rows,
                // emphasis-capitalised two ways: regressions 1,753 -> 293 and 3,835 -> 752.
                List<String> caps = upperLetterRuns(text);
                boolean allAcronyms = !caps.isEmpty();
                for (String run : caps) {
                    if (codePointLength(run) >= SHOUTED_MIN_WORD) {
                        allAcronyms = false;
                        break;
                    }
                }
                if (allAcronyms) {
                    String blanked = blankUpperRuns(text);
                    if (!blanked.equals(text)) {
                        language = latinProfile(blanked).language();
                    }
                }
            }
        }
        // Undecided is not English. Treating it as English sent every Latin-script language we
        // hold no stopwords for to the checkpoint that cannot read it, silently. When nothing
        // identifies the language, non-English letters or a shared Swedish-Danish marker can
        // still prefer the multilingual checkpoint; text with neither signal still goes to the
        // English one.
        boolean undecided = language == null;
        boolean english = "en".equals(language) || (undecided && !latin.looksNonEnglish());
        return new Analysis("latin", profile, language, english, undecided,
                Rounding.round4(latin.diacriticRate()), nonLatin, null);
    }

    /**
     * The non-Latin script with the largest share.
     *
     * <p>Only reached when the non-Latin fraction is at least
     * {@link LanguageTables#NON_LATIN_MIN_FRACTION}, which cannot hold unless the profile has a
     * non-Latin key -- so the fallback is unreachable, and the reference raises there too.
     */
    private static String dominantNonLatin(Map<String, Double> profile) {
        String best = null;
        double bestShare = 0.0;
        for (Map.Entry<String, Double> entry : profile.entrySet()) {
            if ("latin".equals(entry.getKey())) {
                continue;
            }
            if (best == null || entry.getValue() > bestShare) {
                best = entry.getKey();
                bestShare = entry.getValue();
            }
        }
        if (best == null) {
            throw new IllegalStateException("no non-Latin script in a profile that is not all Latin");
        }
        return best;
    }

    /**
     * A string value that is itself not safe for the English checkpoint, else null.
     *
     * <p>A one-word name and a capitalised non-Latin name stay out: the same rules the Latin
     * guess and the non-Latin word scan already use, so a name field cannot pull an English
     * ticket onto the multilingual checkpoint. Code lines, acronyms and slash compounds stay out
     * too, so a pasted traceback is not a message. Each line is capped on its own, so a long
     * earlier field does not consume the budget of the next one.
     */
    static Analysis leafNonEnglish(String leaf) {
        int bestLetters = -1;
        Analysis best = null;
        for (String line : leaf.split("\n", -1)) {
            // A line this short cannot be selected, so skip it before the slice, the blank check,
            // the code-line scan and a full pass. Each branch below needs four word tokens or
            // NON_LATIN_MIN_LETTERS letters. The word scan matches maximal runs of letters and
            // letter-like numerals, so four tokens need three separators between them: seven
            // characters, and ten letters need ten. What makes seven safe rather than six is that
            // every one of those counts is taken on the raw line -- U+0130 lowers to two code
            // points, so counting on the lowered line would let a four-character line reach four
            // tokens, which is why the Latin guess replaces U+0130 before lowering.
            if (codePointLength(line) < 7) {
                continue;
            }
            String sample = head(line, MAX_CHARS);
            if (UnicodeTables.isBlank(sample) || hasCodeLine(sample)) {
                continue;
            }
            Analysis detected = analyseText(sample);
            if (detected.english()) {
                continue;
            }
            // A named language still has to survive the prose scan: acronyms and slash compounds
            // are not words. But a veto is not a verdict of English. Dropping the line here
            // skipped the diacritic branch below, which exists for exactly this -- text that
            // carries non-English letters and that no stopword list can name. An upper-cased
            // German question with umlauts was vetoed for having no long stopword and then thrown
            // away, so a German field routed english; now it falls through and its umlauts carry
            // it. That is also why languageUndecided is no longer required below: a language named
            // and then disbelieved is in the same evidential position as one never named.
            //
            // The ASCII spelling of that same question is a different case and is not fixed here.
            // It has no diacritics to fall through to, and its longest matched German stopword is
            // three letters, so SHOUTED_MIN_STOPWORD vetoes it and nothing recovers it: genuine
            // German on the English checkpoint, one of the upper-cased casualties the reference
            // counts and not an example of what this branch buys.
            String language = detected.language();
            boolean named = language != null && !"en".equals(language)
                    && namedProseLanguage(sample) != null;
            if (!named) {
                if (!"latin".equals(detected.script()) && !"unknown".equals(detected.script())) {
                    if (nonLatinWords(sample).isEmpty()
                            || countAlpha(sample) < LanguageTables.NON_LATIN_MIN_LETTERS) {
                        continue;
                    }
                } else if (!(detected.diacriticRate() >= LanguageTables.NON_EN_DIACRITIC_RATE
                        && words(sample).size() >= 4)) {
                    continue;
                }
            }
            int letters = countAlpha(sample);
            if (letters > bestLetters) {
                bestLetters = letters;
                best = detected;
            }
        }
        return best;
    }

    /**
     * Full detection result for a state.
     *
     * <p>String values are what get read. When a state has several of them, one non-English value
     * is enough: joining every value into one window let a long English note fill the window, or
     * outvote a short German message, and that message was then sent to the English checkpoint.
     * The segment scan stops at {@link #MAX_CHARS}, which is what keeps a huge field cheap; a
     * value it did not reach is read on its own afterwards.
     */
    public static Analysis analyse(Object state) {
        Analysis result = analyseText(stateText(state));
        if ("latin".equals(result.script()) && result.english()) {
            // A Portuguese ticket with an English stack trace, error payload or form template
            // reads as English as a whole, because the English part is longer -- yet the part a
            // question is about is the customer's, and the English checkpoint cannot read it. The
            // cost is lopsided: English sent to multilingual loses a few points, the reverse
            // loses calibration. So a state that would go to English is checked line by line and
            // field by field.
            List<String> leaves = iterText(state);
            boolean multiline = false;
            for (String leaf : leaves) {
                if (leaf.indexOf('\n') >= 0) {
                    multiline = true;
                    break;
                }
            }
            // a single line has no other part to be outvoted by, and was just read whole
            if (leaves.size() > 1 || multiline) {
                String[] found = nonEnglishSegment(state, MAX_CHARS);
                if (found != null) {
                    result = new Analysis(result.script(), result.scriptProfile(), found[0],
                            false, false, result.diacriticRate(), result.nonLatinFraction(),
                            found[1]);
                }
            }
        }
        // A plain string was just read whole. A structured state can still hide a message past
        // the segment cap, or in a script the Latin guess does not name.
        if (state == null || state instanceof String || state instanceof byte[]
                || !result.english()) {
            return result;
        }
        int bestLetters = -1;
        Analysis best = null;
        for (String leaf : iterText(state)) {
            Analysis detected = leafNonEnglish(leaf);
            if (detected == null) {
                continue;
            }
            int letters = countAlpha(head(leaf, MAX_CHARS));
            if (letters > bestLetters) {
                bestLetters = letters;
                best = detected;
            }
        }
        if (best == null) {
            return result;
        }
        return new Analysis(result.script(), result.scriptProfile(), best.language(), false,
                best.languageUndecided(), result.diacriticRate(), result.nonLatinFraction(),
                result.mixedSegment());
    }

    /** Whether the English checkpoint can be expected to read this state. */
    public static boolean isEnglish(Object state) {
        return analyse(state).english();
    }

    // ------------------------------------------------------------------ small helpers

    private static String pythonLower(String text) {
        return UnicodeTables.pythonLower(text);
    }

    private static boolean intersects(Set<String> words, Set<String> other) {
        for (String word : words) {
            if (other.contains(word)) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasDiacritic(String word) {
        int i = 0;
        while (i < word.length()) {
            int cp = word.codePointAt(i);
            i += Character.charCount(cp);
            if (isDiacritic(cp)) {
                return true;
            }
        }
        return false;
    }

    private static boolean isDiacritic(int codePoint) {
        return Arrays.binarySearch(DIACRITICS, codePoint) >= 0;
    }

    /** The generated set as a sorted array, so membership needs no boxing per character. */
    private static int[] sortedCodePoints(Set<String> characters) {
        int[] out = new int[characters.size()];
        int at = 0;
        for (String character : characters) {
            out[at++] = character.codePointAt(0);
        }
        Arrays.sort(out);
        return out;
    }

    static int countAlpha(String text) {
        int count = 0;
        int i = 0;
        while (i < text.length()) {
            int cp = text.codePointAt(i);
            i += Character.charCount(cp);
            if (UnicodeTables.isAlpha(cp)) {
                count++;
            }
        }
        return count;
    }

    private static int codePointLength(String text) {
        return text.codePointCount(0, text.length());
    }

    /** The first {@code count} code points of {@code text}, which is Python's slice. */
    private static String head(String text, int count) {
        if (count <= 0) {
            return "";
        }
        if (codePointLength(text) <= count) {
            return text;
        }
        return text.substring(0, text.offsetByCodePoints(0, count));
    }

    /**
     * Python's one-argument {@code round}: half to <b>even</b> on the exact binary value.
     *
     * <p>{@link Math#rint} is exactly that rule -- IEEE round-to-nearest, ties-to-even -- so this
     * needs neither {@link java.math.BigDecimal} nor an allocation. {@code Math.round} is not:
     * it rounds half <b>up</b>.
     *
     * <p>Honest note on how far that matters. The rounded value feeds one comparison,
     * {@code nonLatinLetters >= NON_LATIN_MIN_LETTERS}, and the two modes differ only at an exact
     * k + 0.5 with k even. At an EVEN threshold both land on the same side of it, so with
     * {@code NON_LATIN_MIN_LETTERS} at 10 the mode is unobservable through {@link #analyse} --
     * measured, by a mutant that swapped it and passed the whole corpus. It becomes observable
     * the moment the threshold is odd: at 9, a product of 8.5 is 8 half-to-even and 9 half-up.
     * Python's rule is matched here so that a change to that table cannot introduce a divergence,
     * and the rule itself is pinned by a fixture rather than by the corpus.
     */
    static int roundHalfEven(double value) {
        return (int) Math.rint(value);
    }
}
