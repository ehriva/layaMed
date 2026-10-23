package lang

import (
	"crypto/sha256"
	"encoding/hex"
	"sort"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
)

// The compiled tables against the fixture recorded from laya/lang.py.

func TestScriptRangesMatchPython(t *testing.T) {
	golden := fixtures.List(t, fixtures.Load(t, "lang_tables.json").Must("script_ranges"))
	if len(golden) != len(scriptNames) {
		t.Fatalf("%d scripts compiled in, %d in lang.py", len(scriptNames), len(golden))
	}
	// lang.py scans its ranges in order and the first hit wins; the compiled table is sorted and
	// searched. They agree only if no two ranges overlap, so replay the ordered scan over every
	// code point and compare.
	type span struct{ lo, hi rune }
	ordered := make([][]span, len(golden))
	for i, raw := range golden {
		name := fixtures.Str(t, fixtures.Field(t, raw, "name"))
		if string(scriptNames[i]) != name {
			t.Fatalf("script %d is %q, lang.py has %q", i, scriptNames[i], name)
		}
		for _, pair := range fixtures.List(t, fixtures.Field(t, raw, "ranges")) {
			p := fixtures.List(t, pair)
			ordered[i] = append(ordered[i], span{rune(fixtures.Int(t, p[0])), rune(fixtures.Int(t, p[1]))})
		}
	}
	for r := rune(0); r < 0x110000; r++ {
		want := -1
	scan:
		for i, spans := range ordered {
			for _, s := range spans {
				if s.lo <= r && r <= s.hi {
					want = i
					break scan
				}
			}
		}
		if got := lookupScript(r); got != want {
			t.Fatalf("lookupScript(U+%04X) = %d, ordered scan gives %d", r, got, want)
		}
	}
}

func TestStopWordsMatchPython(t *testing.T) {
	fx := fixtures.Load(t, "lang_tables.json")
	order := fixtures.Strings(t, fx.Must("stop_word_order"))
	if len(order) != len(stopLists) {
		t.Fatalf("%d languages compiled in, %d in lang.py", len(stopLists), len(order))
	}
	stops := fixtures.Obj(t, fx.Must("stop_words"))
	for i, code := range order {
		if stopLists[i].code != code {
			t.Errorf("language %d is %q, lang.py has %q: the order decides ties", i, stopLists[i].code, code)
		}
		want := fixtures.Strings(t, fixtures.Field(t, stops, code))
		sameSet(t, "stop words for "+code, stopLists[i].words, want)
	}
}

func sameSet(t *testing.T, what string, got map[string]struct{}, want []string) {
	t.Helper()
	sort.Strings(want)
	var have []string
	for w := range got {
		have = append(have, w)
	}
	sort.Strings(have)
	if len(have) != len(want) {
		t.Errorf("%s: %d entries, want %d", what, len(have), len(want))
		return
	}
	for i := range have {
		if have[i] != want[i] {
			t.Errorf("%s: entry %d is %q, want %q", what, i, have[i], want[i])
			return
		}
	}
}

func TestWordSetsMatchPython(t *testing.T) {
	fx := fixtures.Load(t, "lang_tables.json")
	for name, set := range map[string]map[string]struct{}{
		"short_swedish_words":  shortSwedishWords,
		"shared_words":         sharedWords,
		"nordic_overlap_words": nordicOverlapWords,
		"en_only_words":        enOnlyWords,
		"en_collision_words":   enCollisionWords,
	} {
		sameSet(t, name, set, fixtures.Strings(t, fx.Must(name)))
	}
	var diacritics []rune
	for _, s := range fixtures.Strings(t, fx.Must("non_en_diacritics")) {
		diacritics = append(diacritics, []rune(s)[0])
	}
	sort.Slice(diacritics, func(i, j int) bool { return diacritics[i] < diacritics[j] })
	if len(diacritics) != len(nonEnDiacritics) {
		t.Fatalf("%d diacritics compiled in, %d in lang.py", len(nonEnDiacritics), len(diacritics))
	}
	for i, r := range diacritics {
		if nonEnDiacritics[i] != r {
			t.Errorf("diacritic %d is U+%04X, want U+%04X", i, nonEnDiacritics[i], r)
		}
	}
}

func TestThresholdsMatchPython(t *testing.T) {
	th := fixtures.Obj(t, fixtures.Load(t, "lang_tables.json").Must("thresholds"))
	for name, got := range map[string]float64{
		"NON_EN_DIACRITIC_RATE":         nonEnDiacriticRate,
		"ENGLISH_RESCUE_DIACRITIC_RATE": englishRescueDiacriticRate,
		"NON_LATIN_FRACTION":            nonLatinFraction,
		"NON_LATIN_MIN_FRACTION":        nonLatinMinFraction,
		"NON_LATIN_MIN_LETTERS":         nonLatinMinLetters,
		"_SHOUTED_MIN_STOPWORD":         shoutedMinStopword,
		"_SHOUTED_MIN_WORD":             shoutedMinWord,
	} {
		if want := fixtures.Float(t, th.Must(name)); got != want {
			t.Errorf("%s = %v, lang.py has %v", name, got, want)
		}
	}
}

// scriptOf is checked over all of Unicode by a digest, the way the character predicates are.
func TestScriptOfMatchesPythonEverywhere(t *testing.T) {
	want := fixtures.Str(t, fixtures.Field(t, fixtures.Load(t, "lang_detect.json").Must("unicode_digests"), "script_of"))
	h := sha256.New()
	for r := rune(0); r < 0x110000; r++ {
		h.Write([]byte(string(scriptOf(r)) + ";"))
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		t.Errorf("scriptOf disagrees with lang._script_of somewhere in Unicode")
	}
}

func TestNoNamedScriptReachesBelowTheLatinGap(t *testing.T) {
	// The two Latin cuts (U+02B0 for tallying, U+0250 for asking one letter) are interchangeable
	// only while no script range starts below U+02B0.
	for r := rune(0x0250); r <= 0x02AF; r++ {
		if scriptOf(r) != "" {
			t.Fatalf("U+%04X names a script, so the two Latin cuts are no longer equivalent", r)
		}
	}
	if scriptLookup[0] <= 0x02AF {
		t.Fatalf("a script range starts at U+%04X", scriptLookup[0])
	}
}
