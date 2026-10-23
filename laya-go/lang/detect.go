package lang

import (
	"math"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/NandhaKishorM/laya/laya-go/internal/unicodedata"
)

const (
	shoutedMinStopword = 4
	shoutedMinWord     = 5
)

// LatinProfile is what the Latin-script word lists say about a text.
type LatinProfile struct {
	// Language is the detected language code ("en", "fr", ...), or "" when undecided.
	Language string
	// EnglishHits is the English function-word score.
	EnglishHits int
	// DiacriticRate is the share of characters that are letters English does not use.
	DiacriticRate float64
	// LooksNonEnglish reports evidence against English even when no language is named.
	LooksNonEnglish bool
}

// Analysis is the full verdict on a state: lang.py's analyse().
type Analysis struct {
	// Script is the dominant script, or ScriptUnknown when the text has no letters.
	Script Script
	// ScriptProfile is the share of letters per script.
	ScriptProfile Profile
	// Language is the detected ISO 639-1 language code ("en", "fr", ...), or "" when no language
	// could be named. Non-Latin scripts have a Script but no Language here.
	Language string
	// English reports whether the text belongs on the English checkpoint.
	English bool
	// DiacriticRate is the rounded diacritic rate of Latin text, zero otherwise.
	DiacriticRate float64
	// NonLatinFraction is the rounded share of letters outside Latin.
	NonLatinFraction float64
	// MixedSegment is the foreign-language line found inside otherwise English text, or "".
	MixedSegment string
}

func isDiacritic(r rune) bool {
	i := sort.Search(len(nonEnDiacritics), func(i int) bool { return nonEnDiacritics[i] >= r })
	return i < len(nonEnDiacritics) && nonEnDiacritics[i] == r
}

func countAlpha(text string) int {
	n := 0
	for _, r := range text {
		if unicodedata.IsAlpha(r) {
			n++
		}
	}
	return n
}

func hasDiacritic(word string) bool {
	for _, r := range word {
		if isDiacritic(r) {
			return true
		}
	}
	return false
}

func intersects(words, other map[string]struct{}) bool {
	for w := range words {
		if _, ok := other[w]; ok {
			return true
		}
	}
	return false
}

func unique(words []string) map[string]struct{} {
	set := make(map[string]struct{}, len(words))
	for _, w := range words {
		set[w] = struct{}{}
	}
	return set
}

// words are the runs of letters (and Nl/No numerics) in text, as lang.py's [^\W\d_]+.
func words(text string) []string {
	var out []string
	start := -1
	for i, r := range text {
		if unicodedata.IsWord(r) {
			if start < 0 {
				start = i
			}
		} else if start >= 0 {
			out = append(out, text[start:i])
			start = -1
		}
	}
	if start >= 0 {
		out = append(out, text[start:])
	}
	return out
}

func isIdentifierChar(r rune) bool { return r == '-' || unicodedata.IsPythonWord(r) }

// runEnd is the offset just past the run of [\w-] characters starting at from.
func runEnd(text string, from int) int {
	i := from
	for i < len(text) {
		r, w := utf8.DecodeRuneInString(text[i:])
		if !isIdentifierChar(r) {
			break
		}
		i += w
	}
	return i
}

// matchIdentifier matches [\w-]*(?:[.@][\w-]+)+ at from and returns its end, or -1.
func matchIdentifier(text string, from int) int {
	i := runEnd(text, from)
	groups := 0
	for i < len(text) && (text[i] == '.' || text[i] == '@') {
		after := runEnd(text, i+1)
		if after == i+1 {
			break
		}
		i = after
		groups++
	}
	if groups == 0 {
		return -1
	}
	return i
}

// substituteIdentifiers replaces each dotted or @-joined identifier (file names, e-mail
// addresses, hosts) with a space, so its parts are not read as words. It is linear: a run is
// only tried where it starts.
func substituteIdentifiers(text string) string {
	var out strings.Builder
	copied := 0
	replaced := false
	for i := 0; i < len(text); {
		_, w := utf8.DecodeRuneInString(text[i:])
		if i > 0 {
			if before, _ := utf8.DecodeLastRuneInString(text[:i]); isIdentifierChar(before) {
				i += w
				continue
			}
		}
		end := matchIdentifier(text, i)
		if end < 0 {
			i += w
			continue
		}
		if !replaced {
			out.Grow(len(text))
			replaced = true
		}
		out.WriteString(text[copied:i])
		out.WriteByte(' ')
		i, copied = end, end
	}
	if !replaced {
		return text
	}
	out.WriteString(text[copied:])
	return out.String()
}

// hasCodeLine reports code punctuation: one of = ; { } [ ], or a word character before (.
func hasCodeLine(text string) bool {
	for i := 0; i < len(text); i++ {
		switch text[i] {
		case '=', ';', '{', '}', '[', ']':
			return true
		case '(':
			if i > 0 {
				if r, _ := utf8.DecodeLastRuneInString(text[:i]); unicodedata.IsPythonWord(r) {
					return true
				}
			}
		}
	}
	return false
}

// hasJoined reports a word character, one of . _ / \, and another word character in a row.
func hasJoined(token string) bool {
	for i, r := range token {
		if r != '.' && r != '_' && r != '/' && r != '\\' {
			continue
		}
		if i == 0 || i+1 >= len(token) {
			continue
		}
		before, _ := utf8.DecodeLastRuneInString(token[:i])
		after, _ := utf8.DecodeRuneInString(token[i+1:])
		if unicodedata.IsWordOrDigit(before) && unicodedata.IsWordOrDigit(after) {
			return true
		}
	}
	return false
}

// eachWordRun calls fn for every maximal run of word characters, and keep for the text between.
func eachWordRun(text string, keep func(string), fn func(string)) {
	start := -1
	last := 0
	for i, r := range text {
		if unicodedata.IsWord(r) {
			if start < 0 {
				if keep != nil && i > last {
					keep(text[last:i])
				}
				start = i
			}
		} else if start >= 0 {
			fn(text[start:i])
			start, last = -1, i
		}
	}
	if start >= 0 {
		fn(text[start:])
		return
	}
	if keep != nil && last < len(text) {
		keep(text[last:])
	}
}

// blankUpperRuns replaces every all-uppercase run of two or more letters with a space.
func blankUpperRuns(text string) string {
	var out strings.Builder
	out.Grow(len(text))
	eachWordRun(text, func(gap string) { out.WriteString(gap) }, func(run string) {
		if utf8.RuneCountInString(run) >= 2 && unicodedata.IsUpperString(run) {
			out.WriteByte(' ')
		} else {
			out.WriteString(run)
		}
	})
	return out.String()
}

func upperLetterRuns(text string) []string {
	var runs []string
	eachWordRun(text, nil, func(run string) {
		if utf8.RuneCountInString(run) >= 2 && unicodedata.IsUpperString(run) {
			runs = append(runs, run)
		}
	})
	return runs
}

// englishRescuedByWords reports that two or more English-only function words and at most one
// accented word outweigh a diacritic rate that would otherwise route the text away.
func englishRescuedByWords(ws []string, diacriticRate float64) bool {
	if diacriticRate >= englishRescueDiacriticRate {
		return false
	}
	uniq := unique(ws)
	englishOnly, withDiacritic := 0, 0
	for w := range uniq {
		if _, ok := enOnlyWords[w]; ok {
			englishOnly++
		}
		if hasDiacritic(w) {
			withDiacritic++
		}
	}
	return englishOnly >= 2 && withDiacritic <= 1
}

func hasOwnEvidence(ws map[string]struct{}, language string) bool {
	stop, ok := stopWords(language)
	if !ok {
		return false
	}
	for w := range ws {
		if _, in := stop[w]; in {
			if _, shared := sharedWords[w]; !shared {
				return true
			}
		}
	}
	return false
}

var swedishPhraseOpeners = newSet("kan", "jag", "vi")

func contains(ws []string, target string) bool {
	for _, w := range ws {
		if w == target {
			return true
		}
	}
	return false
}

// LatinProfileOf scores text against the per-language function-word lists.
func LatinProfileOf(text string) LatinProfile {
	text = unicodedata.ToValid(text)
	ws := words(unicodedata.Lower(strings.ReplaceAll(substituteIdentifiers(text), "İ", "i")))
	lowered := unicodedata.Lower(text)
	diacritics := 0
	for _, r := range lowered {
		if isDiacritic(r) {
			diacritics++
		}
	}
	n := utf8.RuneCountInString(lowered)
	if n < 1 {
		n = 1
	}
	rate := float64(diacritics) / float64(n)
	nonEnglish := rate >= nonEnDiacriticRate
	uniq := unique(ws)
	nordicOverlap := intersects(uniq, nordicOverlapWords) && !intersects(uniq, enOnlyWords)
	if len(ws) > 1 && len(ws) < 4 && intersects(uniq, shortSwedishWords) {
		return LatinProfile{"sv", 0, rate, nonEnglish}
	}
	if len(ws) < 4 {
		return LatinProfile{"", 0, rate, nonEnglish || nordicOverlap}
	}
	counts := make(map[string]int, len(ws))
	for _, w := range ws {
		counts[w]++
	}
	scores := make([]int, len(stopLists))
	english := 0
	for i := range stopLists {
		score := 0
		for w, c := range counts {
			if _, ok := stopLists[i].words[w]; ok {
				if _, collides := enCollisionWords[w]; collides {
					score++
				} else {
					score += c
				}
			}
		}
		scores[i] = score
		if stopLists[i].code == "en" {
			english = score
		}
	}
	bestLanguage, best, seen := "", 0, false
	for i := range stopLists {
		code := stopLists[i].code
		if code == "en" || !hasOwnEvidence(uniq, code) {
			continue
		}
		if !seen || scores[i] > best {
			bestLanguage, best, seen = code, scores[i], true
		}
	}
	language := ""
	switch {
	case seen && best >= max(2, english+2):
		language = bestLanguage
	case bestLanguage == "sv" && contains(ws, "inte") && contains(ws, "kan") &&
		hasKey(swedishPhraseOpeners, ws[0]) && english <= 1:
		language = bestLanguage
	case seen && nonEnglish && best >= max(2, english):
		language = bestLanguage
	case english != 0 && (!nonEnglish || englishRescuedByWords(ws, rate)):
		language = "en"
	}
	looksNonEnglish := nonEnglish || (language == "" && nordicOverlap)
	return LatinProfile{language, english, rate, looksNonEnglish}
}

func hasKey(set map[string]struct{}, key string) bool {
	_, ok := set[key]
	return ok
}

// GuessLatinLanguage is the language LatinProfileOf names for text, or "" when undecided.
func GuessLatinLanguage(text string) string {
	return LatinProfileOf(text).Language
}

// shouted reports a text with uppercase letters and no lowercase ones.
func shouted(text string) bool {
	sawUpper := false
	for _, r := range text {
		if unicodedata.IsLower(r) {
			return false
		}
		if unicodedata.IsUpper(r) {
			sawUpper = true
		}
	}
	return sawUpper
}

// shoutedEvidence reports that an all-caps text really is in language: it has a word long
// enough to judge by, and a diacritic or a stopword of that language to show for it.
func shoutedEvidence(tokens []string, language string, diacriticRate float64) bool {
	hasWord := false
	for _, t := range tokens {
		if utf8.RuneCountInString(t) >= shoutedMinWord {
			hasWord = true
			break
		}
	}
	if !hasWord {
		return false
	}
	if diacriticRate >= nonEnDiacriticRate {
		return true
	}
	stop, ok := stopWords(language)
	if !ok {
		return false
	}
	for _, t := range tokens {
		lowered := unicodedata.Lower(t)
		if utf8.RuneCountInString(lowered) >= shoutedMinStopword && hasKey(stop, lowered) {
			return true
		}
	}
	return false
}

// namedProseLanguage is the non-English language a prose segment is written in, or "".
func namedProseLanguage(segment string) string {
	if unicodedata.IsBlank(segment) || hasCodeLine(segment) {
		return ""
	}
	var kept []string
	for _, token := range unicodedata.Fields(segment) {
		if !hasJoined(token) {
			kept = append(kept, token)
		}
	}
	prose := strings.Join(kept, " ")
	allCaps := shouted(prose)
	if !allCaps {
		prose = blankUpperRuns(prose)
	}
	tokens := words(prose)
	if len(tokens) < 4 {
		return ""
	}
	profile := LatinProfileOf(prose)
	language := profile.Language
	if language == "" || language == "en" {
		return ""
	}
	if allCaps && !shoutedEvidence(tokens, language, profile.DiacriticRate) {
		return ""
	}
	stop, ok := stopWords(language)
	if !ok {
		return ""
	}
	distinct := make(map[string]struct{})
	for _, t := range tokens {
		if lowered := unicodedata.Lower(t); hasKey(stop, lowered) {
			distinct[lowered] = struct{}{}
		}
	}
	if len(distinct) < 2 {
		return ""
	}
	return language
}

// nonEnglishSegment finds the first line of state that is prose in a named non-English
// language, returning its code and the stripped line.
func nonEnglishSegment(leaves []string, maxChars int) (language, segment string, found bool) {
	seen := 0
	for _, leaf := range leaves {
		for _, line := range strings.Split(leaf, "\n") {
			if seen >= maxChars {
				return "", "", false
			}
			seg := head(line, maxChars-seen)
			seen += utf8.RuneCountInString(seg)
			if lang := namedProseLanguage(seg); lang != "" {
				return lang, unicodedata.Strip(seg), true
			}
		}
	}
	return "", "", false
}

func dominantNonLatin(profile Profile) Script {
	best, share := Script(""), 0.0
	for _, s := range profile {
		if s.Script == ScriptLatin {
			continue
		}
		if best == "" || s.Fraction > share {
			best, share = s.Script, s.Fraction
		}
	}
	return best
}

func analyzeText(text string) Analysis {
	counts := scriptCounts(text)
	profile := profileFromCounts(counts)
	script := scriptFromCounts(counts)
	nonLatin := 0.0
	if len(profile) > 0 {
		nonLatin = round4(1.0 - profile.Get(ScriptLatin))
	}
	nonLatinLetters := int(math.RoundToEven(nonLatin * float64(countAlpha(text))))
	if script == ScriptLatin && len(nonLatinWords(text)) > 0 && nonLatinCarries(nonLatin, nonLatinLetters) {
		script = dominantNonLatin(profile)
	}
	if script == ScriptUnknown {
		return Analysis{Script: ScriptUnknown, ScriptProfile: profile, English: true}
	}
	if script != ScriptLatin {
		return Analysis{Script: script, ScriptProfile: profile, NonLatinFraction: nonLatin}
	}
	latin := LatinProfileOf(text)
	language := latin.Language
	if language != "" && language != "en" && !nonLatinCarries(nonLatin, nonLatinLetters) {
		if shouted(text) {
			if !shoutedEvidence(words(text), language, latin.DiacriticRate) {
				language = ""
			}
		} else {
			caps := upperLetterRuns(text)
			allAcronyms := len(caps) > 0
			for _, run := range caps {
				if utf8.RuneCountInString(run) >= shoutedMinWord {
					allAcronyms = false
					break
				}
			}
			if allAcronyms {
				if blanked := blankUpperRuns(text); blanked != text {
					language = LatinProfileOf(blanked).Language
				}
			}
		}
	}
	english := language == "en" || (language == "" && !latin.LooksNonEnglish)
	return Analysis{
		Script: ScriptLatin, ScriptProfile: profile, Language: language, English: english,
		DiacriticRate:    round4(latin.DiacriticRate),
		NonLatinFraction: nonLatin,
	}
}

// leafNonEnglish is the most letter-rich line of one leaf that is clearly not English, or false.
func leafNonEnglish(leaf string) (Analysis, bool) {
	bestLetters := -1
	var best Analysis
	for _, line := range strings.Split(leaf, "\n") {
		if utf8.RuneCountInString(line) < 7 {
			continue
		}
		sample := head(line, MaxChars)
		if unicodedata.IsBlank(sample) || hasCodeLine(sample) {
			continue
		}
		detected := analyzeText(sample)
		if detected.English {
			continue
		}
		named := detected.Language != "" && detected.Language != "en" && namedProseLanguage(sample) != ""
		if !named {
			if detected.Script != ScriptLatin && detected.Script != ScriptUnknown {
				if len(nonLatinWords(sample)) == 0 || countAlpha(sample) < nonLatinMinLetters {
					continue
				}
			} else if !(detected.DiacriticRate >= nonEnDiacriticRate && len(words(sample)) >= 4) {
				continue
			}
		}
		if letters := countAlpha(sample); letters > bestLetters {
			bestLetters, best = letters, detected
		}
	}
	return best, bestLetters >= 0
}

// Analyze is lang.py's analyse(): the script, language and English verdict for a state.
//
// A state that is English as a whole is still scanned for a foreign-language line or leaf when it
// holds several leaves or lines, so one Portuguese sentence in an English ticket is not lost. The
// state is walked once, so the cost is O(number of leaves), plus the analysis of the text.
//
// A text with no letters is ScriptUnknown and English: Analyze("") has English true and Script
// ScriptUnknown, which routes it to the English checkpoint as lang.py does.
func Analyze(state any) Analysis {
	leaves := iterText(state)
	result := analyzeText(joinLeaves(leaves, MaxChars))
	if result.Script == ScriptLatin && result.English {
		multiline := false
		for _, leaf := range leaves {
			if strings.Contains(leaf, "\n") {
				multiline = true
				break
			}
		}
		if len(leaves) > 1 || multiline {
			if language, segment, ok := nonEnglishSegment(leaves, MaxChars); ok {
				result.Language, result.English = language, false
				result.MixedSegment = segment
			}
		}
	}
	if isScalar(state) || !result.English {
		return result
	}
	bestLetters := -1
	var best Analysis
	for _, leaf := range leaves {
		detected, ok := leafNonEnglish(leaf)
		if !ok {
			continue
		}
		if letters := countAlpha(head(leaf, MaxChars)); letters > bestLetters {
			bestLetters, best = letters, detected
		}
	}
	if bestLetters < 0 {
		return result
	}
	result.Language, result.English = best.Language, false
	return result
}

// DetectLanguage is the language Analyze names for state ("en", "fr", "de", ...), or "" when it
// cannot name one, for instance for non-Latin text, which has a script but no language here.
func DetectLanguage(state any) string {
	return Analyze(state).Language
}

// IsEnglish reports whether state belongs on the English checkpoint.
func IsEnglish(state any) bool {
	return Analyze(state).English
}

// LanguageUndecided reports that no language could be named: Language is "". lang.py records this
// as its own field, but it is always exactly that, so it is a method here.
func (a Analysis) LanguageUndecided() bool { return a.Language == "" }
