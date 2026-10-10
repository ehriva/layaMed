package com.convaiinnovations.laya.lang;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The caps bars, asserted directly rather than through the recorded fixture.
 *
 * <p>{@link LanguageDetectionParityTest} already covers every case in {@code lang_detect.json},
 * but it covers them as data: a rule can be removed and the fixture re-recorded from a Python
 * reference that lost the same rule, and nothing here would notice. These assertions name the
 * behaviour instead, so the rule has to be deleted from two places to disappear quietly.
 *
 * <p>The three cases the rule exists to balance are all here. An all-caps line of English
 * abbreviations that collide with French stopwords must not be named French; emphasis capitals in
 * genuine prose must not be blanked out of their own sentence; and a state carried by a caseless
 * script must be held to neither bar, because both work by discarding Latin evidence.
 */
class ShoutedAcronymBarTest {

    /** English abbreviations -- hockey teams, states, time zones, radio bands -- that read as French. */
    private static final String ACRONYMS = "MON LA EST COM DES";

    /** A genuinely Portuguese line, written without its accents as a customer would type it. */
    private static final String PORTUGUESE =
            "Nao consigo entrar na minha conta e a senha nao funciona de jeito nenhum";

    /**
     * The same line shouted, spelled out rather than upper-cased.
     *
     * <p>{@code String.toUpperCase()} without a locale is locale-sensitive -- under a Turkish
     * default it maps {@code i} to U+0130 -- and this line has four of them, so computing it
     * would make the assertion depend on the machine.
     */
    private static final String PORTUGUESE_SHOUTED =
            "NAO CONSIGO ENTRAR NA MINHA CONTA E A SENHA NAO FUNCIONA DE JEITO NENHUM";

    private static final String ENGLISH_OPENER =
            "The customer opened this ticket yesterday and we have asked for a screenshot.";

    private static final String ENGLISH_CLOSER =
            "We will escalate this to the platform team if it is not resolved today.";

    @Test
    @DisplayName("an all-caps line of English abbreviations does not name French")
    void acronymLineIsNotProse() {
        assertNull(LanguageDetection.namedProseLanguage(ACRONYMS),
                "bare abbreviations are acronym-shaped and name no language");
        assertEquals("pt", LanguageDetection.namedProseLanguage(PORTUGUESE),
                "the Portuguese line is still named, so the veto is not simply vetoing everything");
    }

    @Test
    @DisplayName("the genuinely foreign line decides a state the acronym line used to win")
    void foreignLineOutvotesTheAcronymLine() {
        String state = ENGLISH_OPENER + "\n" + ACRONYMS + "\n" + PORTUGUESE + "\n" + ENGLISH_CLOSER;
        LanguageDetection.Analysis analysis = LanguageDetection.analyse(state);
        assertEquals("pt", analysis.language(), "the Portuguese line decides, not the acronyms");
        assertFalse(analysis.english(), "a Portuguese message is not for the English checkpoint");
        assertFalse(analysis.languageUndecided(), "the language is named");
        assertEquals(PORTUGUESE, analysis.mixedSegment(), "and the line that named it is reported");
    }

    @Test
    @DisplayName("a state that is nothing but an acronym line is undecided, not French")
    void acronymOnlyStateIsUndecided() {
        LanguageDetection.Analysis analysis = LanguageDetection.analyse(ACRONYMS);
        assertNull(analysis.language(), "the whole-state verdict applies the bar too");
        assertTrue(analysis.languageUndecided(), "vetoing leaves the text undecided");
        // Undecided with no non-English letter is English: the veto takes the French verdict
        // away, it does not replace it with one of its own.
        assertTrue(analysis.english(), "nothing here is non-English, so the English checkpoint reads it");
    }

    @Test
    @DisplayName("one English line above the acronyms no longer routes multilingual")
    void shortEnglishStateIsNotOutvoted() {
        LanguageDetection.Analysis analysis =
                LanguageDetection.analyse("We have asked for a screenshot.\n" + ACRONYMS);
        assertEquals("en", analysis.language(), "the acronyms no longer vote in the verdict");
        assertTrue(analysis.english());
    }

    @Test
    @DisplayName("emphasis capitals keep the verdict they name")
    void emphasisCapitalsAreNotBlanked() {
        // Only acronym-shaped runs are blanked. Blanking every all-caps run deleted the words that
        // carried these two sentences, and both of them routed english with no language at all.
        LanguageDetection.Analysis german = LanguageDetection.analyse("sag mir das HEUTIGE DATUM");
        assertEquals("de", german.language(), "HEUTIGE and DATUM are words, not acronyms");
        assertFalse(german.english());

        LanguageDetection.Analysis spanish =
                LanguageDetection.analyse("quiero cancelar mi PEDIDO POR FAVOR");
        assertEquals("es", spanish.language(), "PEDIDO is a word, so no run is blanked");
        assertFalse(spanish.english());
    }

    @Test
    @DisplayName("an all-caps line with non-English letters clears the bar on its diacritics")
    void diacriticsStandInForALongStopword() {
        // Its longest matched German stopword is three letters, so the stopword half of the bar
        // vetoes it; the umlauts carry it, and the veto leaves it undecided rather than English.
        LanguageDetection.Analysis analysis = LanguageDetection.analyse("WIE SPÄT IST ES IN KÖLN");
        assertNull(analysis.language(), "no stopword list can name it");
        assertTrue(analysis.languageUndecided());
        assertFalse(analysis.english(), "a veto is not a verdict of English");
    }

    /**
     * A German field the whole-state verdict names but the prose scan disbelieves.
     *
     * <p>{@code wie} is the only German stopword it matches, and the prose scan wants two
     * <em>different</em> ones, so it is vetoed -- while the whole-state verdict, which has no such
     * rule, names it {@code de}. That disagreement is the only thing the restructure below
     * changes, so a field that merely clears both bars, or fails both, pins nothing.
     */
    private static final String VETOED_GERMAN = "wie Überprüfung wie Anmeldung";

    private static final String LONG_ENGLISH_NOTE =
            "The agent replied to the customer and closed the ticket today, and we have also "
            + "asked the platform team to confirm that the account is in good standing before "
            + "we escalate.";

    @Test
    @DisplayName("a vetoed field still reaches the multilingual checkpoint on its diacritics")
    void vetoIsNotAVerdictOfEnglishInAField() {
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("note", LONG_ENGLISH_NOTE);
        state.put("msg", VETOED_GERMAN);

        // Three anchors, so a change that makes the assertion below pass for some other reason is
        // not mistaken for this rule holding. The joined window has to read English -- the note is
        // long enough to dilute the umlauts below the diacritic floor -- the prose scan has to
        // veto the field, and the whole-state verdict has to name it anyway. Only then is the
        // per-field pass the last thing that can catch it, which is the position the fix is about.
        assertTrue(LanguageDetection.analyseText(LanguageDetection.stateText(state)).english(),
                "the joined window has to read English, or this case tests nothing");
        assertNull(LanguageDetection.namedProseLanguage(VETOED_GERMAN),
                "the prose scan has to veto the field, or this case tests nothing");
        assertEquals("de", LanguageDetection.analyseText(VETOED_GERMAN).language(),
                "and the whole-state verdict has to name it, or this case tests nothing");
        assertNull(LanguageDetection.nonEnglishSegment(state, LanguageDetection.MAX_CHARS),
                "so the segment scan finds nothing");

        // Before the fix the field was dropped here: a named-then-disbelieved language took the
        // first branch and never reached the diacritic branch, which exists for exactly this.
        LanguageDetection.Analysis leaf = LanguageDetection.leafNonEnglish(VETOED_GERMAN);
        assertNotNull(leaf, "a veto is not a verdict of English");
        assertFalse(leaf.english());

        LanguageDetection.Analysis analysis = LanguageDetection.analyse(state);
        assertFalse(analysis.english(),
                "so the German field pulls the state off the English checkpoint");
        assertEquals("de", analysis.language());
    }

    @Test
    @DisplayName("a shouted field with umlauts is carried by the diacritic branch")
    void shoutedFieldFallsThroughOnItsDiacritics() {
        // The same position reached the other way: here the whole-state verdict vetoes the field
        // too, on the shouted bar, so it arrives undecided rather than named.
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("note", LONG_ENGLISH_NOTE);
        state.put("msg", "WIE SPÄT IST ES IN KÖLN");
        assertTrue(LanguageDetection.analyseText(LanguageDetection.stateText(state)).english(),
                "the joined window has to read English, or this case tests nothing");
        assertFalse(LanguageDetection.analyse(state).english());
        assertTrue(LanguageDetection.analyse(state).languageUndecided());
    }

    @Test
    @DisplayName("a caseless script is held to neither caps bar")
    void caselessScriptKeepsTheBarsOff() {
        // 35% Cyrillic letters. Both bars answer the verdict by taking evidence away from the
        // Latin letters, and what decides what is left reads Latin diacritics only -- it cannot
        // see the caseless half. Text this far from Latin is not English whatever the Latin part
        // is worth, so the language name is allowed to stand.
        LanguageDetection.Analysis analysis =
                LanguageDetection.analyse("ПРИВЕТ MON DES EST LA");
        assertEquals("fr", analysis.language(), "the bar is kept off a mostly-non-Latin state");
        assertFalse(analysis.english(), "which is the only thing that decides the checkpoint");
    }

    @Test
    @DisplayName("shouted needs an uppercase character, so caseless text is not shouted")
    void shoutedNeedsAnUppercaseCharacter() {
        assertTrue(LanguageDetection.shouted(ACRONYMS));
        assertFalse(LanguageDetection.shouted("sag mir das HEUTIGE DATUM"), "one lowercase letter is enough");
        assertFalse(LanguageDetection.shouted("hola"));
        assertFalse(LanguageDetection.shouted("1234"), "no cased character at all");
        // Devanagari has no lowercase either; testing only for the absence of lowercase would call
        // this shouted and hold it to a bar its tokens cannot clear.
        assertFalse(LanguageDetection.shouted("हिन्दी बोलता"));
    }

    @Test
    @DisplayName("shouted evidence needs a long word, plus diacritics or a long stopword")
    void shoutedEvidenceNeedsMoreThanAcronyms() {
        assertFalse(LanguageDetection.shoutedEvidence(
                        LanguageDetection.words(ACRONYMS), "fr", 0.0),
                "every token is acronym-shaped");
        assertTrue(LanguageDetection.shoutedEvidence(
                        LanguageDetection.words(PORTUGUESE_SHOUTED), "pt", 0.0),
                "shouted Portuguese is still Portuguese");
        assertFalse(LanguageDetection.shoutedEvidence(List.of("PARA", "NAO"), "pt", 0.0),
                "no token reaches the word bar");
        assertTrue(LanguageDetection.shoutedEvidence(List.of("KOELN", "IST"), "de", 0.05),
                "non-English letters stand in for a long stopword");
        assertFalse(LanguageDetection.shoutedEvidence(List.of("SENHA", "NAO"), "zz", 0.0),
                "a language with no stopword list can match nothing");
    }

    @Test
    @DisplayName("both bars are at the value the reference swept to, not one letter lower")
    void barsSitWhereTheSweepPutThem() {
        // The acronym line above never reaches the stopword bar -- no token of it is long enough
        // to clear the word bar first -- so without these two the bars could each be loosened by
        // a letter with nothing failing. Each input clears one bar and is stopped by the other.

        // PARA and ESTA are four-letter Spanish stopwords: the stopword bar is cleared, and the
        // word bar is what stops it. At four it would pass.
        assertFalse(LanguageDetection.shoutedEvidence(
                        LanguageDetection.words("PARA ESTA PARA ESTO"), "es", 0.0),
                "a four-letter stopword is not a five-letter word");

        // SPAET and KOELN clear the word bar, and the longest German stopword matched is `wie`,
        // three letters: the stopword bar is what stops it. At three it would pass.
        assertFalse(LanguageDetection.shoutedEvidence(
                        LanguageDetection.words("WIE SPAET IST ES IN KOELN"), "de", 0.0),
                "three letters of stopword is not evidence");
    }

    @Test
    @DisplayName("the ASCII spelling of that German question is a casualty, and stays one")
    void asciiGermanIsAKnownCasualty() {
        // Not an endorsement: the reference counts this among the roughly 3,727 upper-cased rows
        // the pair of bars gives up, and says plainly that nothing recovers it -- it has no
        // diacritics to fall through to and no stopword long enough. It is asserted because it is
        // the cheapest behavioural pin on the stopword bar: lower the bar to three and genuine
        // German starts routing multilingual here, which is better, but it is not what the merged
        // reference does, and the Java port is not the place to change the rule.
        assertTrue(LanguageDetection.analyse("WIE SPAET IST ES IN KOELN").english(),
                "the bar vetoes it and no diacritic carries it");
    }
}
