package lang

import (
	"math"
	"strconv"

	"github.com/NandhaKishorM/laya/laya-go/internal/unicodedata"
)

// stopList is one language's function words.
type stopList struct {
	code  string
	words map[string]struct{}
}

func newSet(words ...string) map[string]struct{} {
	set := make(map[string]struct{}, len(words))
	for _, w := range words {
		set[w] = struct{}{}
	}
	return set
}

func stopWords(code string) (map[string]struct{}, bool) {
	for i := range stopLists {
		if stopLists[i].code == code {
			return stopLists[i].words, true
		}
	}
	return nil, false
}

// Script names a writing system: one of the Script constants. The values are lang.py's script
// names, so they compare equal to the strings Python returns.
type Script string

// The scripts that are not in lang.py's table of named non-Latin scripts.
const (
	// ScriptLatin is Latin, including its extensions and the fullwidth forms.
	ScriptLatin Script = "latin"
	// ScriptOther counts letters no named script claims (the CJK extensions, kana supplements and
	// dozens of smaller scripts), so they stay visible and non-Latin.
	ScriptOther Script = "other"
	// ScriptUnknown is the verdict for a text with no letters.
	ScriptUnknown Script = "unknown"
)

// Share is the fraction of a text's letters written in one script.
type Share struct {
	Script   Script
	Fraction float64
}

// Profile lists the scripts a text uses with their shares, Latin first and the rest in order of
// first appearance, as lang.py's script_profile does. It is empty for a text with no letters.
type Profile []Share

// Get is the share of script, or zero when the text has none of it.
func (p Profile) Get(script Script) float64 {
	for _, s := range p {
		if s.Script == script {
			return s.Fraction
		}
	}
	return 0
}

type scriptCount struct {
	name Script
	n    int
}

const (
	latinBelowCounts   = 0x02B0 // letters below this count as Latin when tallying
	latinBelowScriptOf = 0x0250 // and below this when asking which script one letter is
)

func isLatinRange(r rune, below rune) bool {
	return r < below ||
		(r >= 0x1E00 && r <= 0x1EFF) ||
		(r >= 0xFF21 && r <= 0xFF3A) ||
		(r >= 0xFF41 && r <= 0xFF5A)
}

// lookupScript is the index into scriptNames of the script claiming r, or -1.
func lookupScript(r rune) int {
	if r > scriptMaxCodePoint {
		return -1
	}
	lo, hi := 0, len(scriptLookupIndex)-1
	for lo <= hi {
		mid := int(uint(lo+hi) >> 1)
		switch {
		case r < scriptLookup[mid*2]:
			hi = mid - 1
		case r > scriptLookup[mid*2+1]:
			lo = mid + 1
		default:
			return int(scriptLookupIndex[mid])
		}
	}
	return -1
}

// scriptCounts tallies the letters of text by script in one pass. Latin comes last, so a named
// script wins a tie against it, and an unlisted script counts under "other".
func scriptCounts(text string) []scriptCount {
	const other = len(scriptNames)
	var tally [other + 1]int
	var order [other + 1]int
	distinct, latin := 0, 0
	for _, r := range text {
		if !unicodedata.IsAlpha(r) {
			continue
		}
		if isLatinRange(r, latinBelowCounts) {
			latin++
			continue
		}
		slot := lookupScript(r)
		if slot < 0 {
			slot = other
		}
		if tally[slot] == 0 {
			order[distinct] = slot
			distinct++
		}
		tally[slot]++
	}
	counts := make([]scriptCount, 0, distinct+1)
	for _, slot := range order[:distinct] {
		name := ScriptOther
		if slot != other {
			name = scriptNames[slot]
		}
		counts = append(counts, scriptCount{name, tally[slot]})
	}
	return append(counts, scriptCount{ScriptLatin, latin})
}

func scriptFromCounts(counts []scriptCount) Script {
	best, bestN := Script(""), 0
	for _, c := range counts {
		if c.n > bestN {
			best, bestN = c.name, c.n
		}
	}
	if bestN == 0 {
		return ScriptUnknown
	}
	return best
}

func profileFromCounts(counts []scriptCount) Profile {
	total := 0
	for _, c := range counts {
		total += c.n
	}
	if total == 0 {
		return nil
	}
	profile := make(Profile, 0, len(counts))
	for _, c := range counts {
		if c.name == ScriptLatin && c.n != 0 {
			profile = append(profile, Share{c.name, float64(c.n) / float64(total)})
		}
	}
	for _, c := range counts {
		if c.name != ScriptLatin && c.n != 0 {
			profile = append(profile, Share{c.name, float64(c.n) / float64(total)})
		}
	}
	return profile
}

// DetectScript is the dominant script of text (ScriptLatin, ScriptHan, ScriptDevanagari, ...), or
// ScriptUnknown when it has no letters. A named script wins a tie against Latin.
func DetectScript(text string) Script {
	text = unicodedata.ToValid(text)
	return scriptFromCounts(scriptCounts(text))
}

// ScriptProfile is the share of text's letters written in each script.
func ScriptProfile(text string) Profile {
	text = unicodedata.ToValid(text)
	return profileFromCounts(scriptCounts(text))
}

// scriptOf is the named script of one letter, or "" when none claims it.
func scriptOf(r rune) Script {
	if i := scriptIndexOf(r); i >= 0 {
		return scriptNames[i]
	}
	return ""
}

func scriptIndexOf(r rune) int {
	if isLatinRange(r, latinBelowScriptOf) {
		return -1
	}
	return lookupScript(r)
}

func nonLatinCarries(nonLatin float64, nonLatinLetters int) bool {
	return nonLatin >= nonLatinFraction ||
		(nonLatin >= nonLatinMinFraction && nonLatinLetters >= nonLatinMinLetters)
}

// nonLatinWords are the runs of a single named script, at least two characters long and not
// starting with an uppercase letter. Combining marks are dropped and do not break a run.
func nonLatinWords(text string) []string {
	var runs []string
	var current []rune
	script := -1
	emit := func() {
		if len(current) >= 2 && !unicodedata.IsUpper(current[0]) {
			runs = append(runs, string(current))
		}
		current = current[:0]
	}
	for _, r := range text {
		if unicodedata.IsCombining(r) {
			continue
		}
		here := scriptIndexOf(r)
		if here >= 0 && here == script {
			current = append(current, r)
			continue
		}
		emit()
		if here >= 0 {
			current = append(current, r)
		}
		script = here
	}
	emit()
	return runs
}

// round4 is Python's round(x, 4): correctly rounded, ties to even on the exact binary value.
func round4(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return v
	}
	rounded, err := strconv.ParseFloat(strconv.FormatFloat(v, 'f', 4, 64), 64)
	if err != nil {
		return v
	}
	return rounded
}
