package lang

import (
	"encoding/json"
	"math"
	"math/big"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/internal/unicodedata"
)

// toState rebuilds a state from the tagged form the golden stores it in, keeping dict order.
func toState(tb testing.TB, v any) any {
	tb.Helper()
	switch x := v.(type) {
	case fixtures.Object:
		out := make(Object, len(x))
		for i, m := range x {
			out[i] = Member{m.Key, toState(tb, m.Value)}
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			out[i] = toState(tb, item)
		}
		return out
	case json.Number: // a JSON number is a number, not the string type it happens to be spelled as
		f, _ := x.Float64()
		return f
	}
	return v // strings, booleans and null are leaves
}

func stateOf(tb testing.TB, row fixtures.Object) any {
	tb.Helper()
	encoded := fixtures.Field(tb, row, "state")
	value := fixtures.Field(tb, encoded, "value")
	switch fixtures.Str(tb, fixtures.Field(tb, encoded, "kind")) {
	case "string":
		return fixtures.Str(tb, value)
	case "bytes":
		list := fixtures.List(tb, value)
		out := make([]byte, len(list))
		for i, b := range list {
			out[i] = byte(fixtures.Int(tb, b))
		}
		return out
	}
	return toState(tb, value)
}

func assertProfile(t *testing.T, where string, want any, got Profile) {
	t.Helper()
	obj := fixtures.Obj(t, want)
	if len(obj) != len(got) {
		t.Fatalf("%s: script profile has %d entries, want %d (%v)", where, len(got), len(obj), obj.Keys())
	}
	for i, m := range obj {
		if string(got[i].Script) != m.Key {
			t.Fatalf("%s: script profile key %d is %q, want %q (order matters)", where, i, got[i].Script, m.Key)
		}
		if w := fixtures.Float(t, m.Value); got[i].Fraction != w {
			t.Errorf("%s: share of %s = %v, want %v", where, m.Key, got[i].Fraction, w)
		}
	}
}

func nullable(t *testing.T, v any) string { t.Helper(); return fixtures.StrOrEmpty(t, v) }

func TestDetectionParity(t *testing.T) {
	cases := fixtures.List(t, fixtures.Load(t, "lang_detect.json").Must("cases"))
	if len(cases) < 100 {
		t.Fatalf("the corpus shrank to %d cases", len(cases))
	}
	for _, raw := range cases {
		row := fixtures.Obj(t, raw)
		name := fixtures.Str(t, fixtures.Field(t, row, "name"))
		t.Run(name, func(t *testing.T) { assertCase(t, name, row) })
	}
}

func assertCase(t *testing.T, name string, row fixtures.Object) {
	state := stateOf(t, row)
	text := fixtures.Str(t, fixtures.Field(t, row, "state_text"))
	if got := StateText(state); got != text {
		t.Errorf("StateText = %q, want %q", got, text)
	}
	if got, want := DetectScript(text), Script(fixtures.Str(t, fixtures.Field(t, row, "detect_script"))); got != want {
		t.Errorf("DetectScript = %q, want %q", got, want)
	}
	assertProfile(t, "ScriptProfile", fixtures.Field(t, row, "script_profile"), ScriptProfile(text))

	wantLatin := fixtures.Field(t, row, "latin_profile")
	latin := LatinProfileOf(text)
	if want := nullable(t, fixtures.Field(t, wantLatin, "language")); latin.Language != want {
		t.Errorf("LatinProfileOf.Language = %q, want %q", latin.Language, want)
	}
	if want := fixtures.Int(t, fixtures.Field(t, wantLatin, "english_hits")); latin.EnglishHits != want {
		t.Errorf("EnglishHits = %d, want %d", latin.EnglishHits, want)
	}
	if want := fixtures.Float(t, fixtures.Field(t, wantLatin, "diacritic_rate")); latin.DiacriticRate != want {
		t.Errorf("latin DiacriticRate = %v, want %v", latin.DiacriticRate, want)
	}
	if want := fixtures.Bool(t, fixtures.Field(t, wantLatin, "looks_non_english")); latin.LooksNonEnglish != want {
		t.Errorf("LooksNonEnglish = %v, want %v", latin.LooksNonEnglish, want)
	}
	if got := GuessLatinLanguage(text); got != latin.Language {
		t.Errorf("GuessLatinLanguage = %q, want LatinProfileOf's %q", got, latin.Language)
	}

	want := fixtures.Field(t, row, "analyse")
	got := Analyze(state)
	if w := Script(fixtures.Str(t, fixtures.Field(t, want, "script"))); got.Script != w {
		t.Errorf("Analyze.Script = %q, want %q", got.Script, w)
	}
	assertProfile(t, "Analyze.ScriptProfile", fixtures.Field(t, want, "script_profile"), got.ScriptProfile)
	if w := nullable(t, fixtures.Field(t, want, "language")); got.Language != w {
		t.Errorf("Analyze.Language = %q, want %q", got.Language, w)
	}
	if w := fixtures.Bool(t, fixtures.Field(t, want, "is_english")); got.English != w {
		t.Errorf("Analyze.English = %v, want %v", got.English, w)
	}
	if w := fixtures.Bool(t, fixtures.Field(t, want, "language_undecided")); got.LanguageUndecided() != w {
		t.Errorf("Analyze.LanguageUndecided = %v, want %v", got.LanguageUndecided(), w)
	}
	if w := fixtures.Float(t, fixtures.Field(t, want, "diacritic_rate")); got.DiacriticRate != w {
		t.Errorf("Analyze.DiacriticRate = %v, want %v", got.DiacriticRate, w)
	}
	if w := fixtures.Float(t, fixtures.Field(t, want, "non_latin_fraction")); got.NonLatinFraction != w {
		t.Errorf("Analyze.NonLatinFraction = %v, want %v", got.NonLatinFraction, w)
	}
	if w := nullable(t, fixtures.Field(t, want, "mixed_segment")); got.MixedSegment != w {
		t.Errorf("Analyze.MixedSegment = %q, want %q", got.MixedSegment, w)
	}
	if w := fixtures.Bool(t, fixtures.Field(t, row, "is_english")); IsEnglish(state) != w {
		t.Errorf("IsEnglish = %v, want %v", !w, w)
	}
	if got.Language != DetectLanguage(state) {
		t.Errorf("DetectLanguage = %q, want Analyse's %q", DetectLanguage(state), got.Language)
	}
	if h, ok := row.Get("helpers"); ok {
		// The helpers were recorded over the state itself, not its 4000-character flattening.
		raw, isString := state.(string)
		if !isString {
			t.Fatalf("helpers recorded for a %T state", state)
		}
		assertHelpers(t, raw, h, latin)
	}
}

func eqStrings(t *testing.T, what string, got, want []string) {
	t.Helper()
	for i := 0; i < len(got) && i < len(want); i++ {
		if got[i] != want[i] {
			t.Errorf("%s[%d] = %q, want %q", what, i, got[i], want[i])
			return
		}
	}
	if len(got) != len(want) {
		t.Errorf("%s has %d entries, want %d", what, len(got), len(want))
	}
}

func assertHelpers(t *testing.T, text string, h any, latin LatinProfile) {
	t.Helper()
	field := func(k string) any { return fixtures.Field(t, h, k) }
	eqStrings(t, "words", words(text), fixtures.Strings(t, field("words")))
	if got, want := substituteIdentifiers(text), fixtures.Str(t, field("substitute_identifiers")); got != want {
		t.Errorf("substituteIdentifiers = %q, want %q", got, want)
	}
	eqStrings(t, "nonLatinWords", nonLatinWords(text), fixtures.Strings(t, field("non_latin_words")))
	if got, want := namedProseLanguage(text), nullable(t, field("named_prose_language")); got != want {
		t.Errorf("namedProseLanguage = %q, want %q", got, want)
	}
	if got, want := hasCodeLine(text), fixtures.Bool(t, field("has_code_line")); got != want {
		t.Errorf("hasCodeLine = %v, want %v", got, want)
	}
	if got, want := blankUpperRuns(text), fixtures.Str(t, field("blank_upper_runs")); got != want {
		t.Errorf("blankUpperRuns = %q, want %q", got, want)
	}
	if got, want := unicodedata.Lower(text), fixtures.Str(t, field("python_lower")); got != want {
		t.Errorf("Lower = %q, want %q", got, want)
	}
	if got, want := unicodedata.Strip(text), fixtures.Str(t, field("strip")); got != want {
		t.Errorf("Strip = %q, want %q", got, want)
	}
	eqStrings(t, "Fields", unicodedata.Fields(text), fixtures.Strings(t, field("split_whitespace")))
	eqStrings(t, "split on newline", strings.Split(text, "\n"), fixtures.Strings(t, field("split_newline")))
	if got, want := countAlpha(text), fixtures.Int(t, field("count_alpha")); got != want {
		t.Errorf("countAlpha = %d, want %d", got, want)
	}
	if got, want := utf8.RuneCountInString(text), fixtures.Int(t, field("code_point_length")); got != want {
		t.Errorf("rune count = %d, want %d", got, want)
	}
	var joined []bool
	for _, token := range unicodedata.Fields(text) {
		joined = append(joined, hasJoined(token))
	}
	wantJoined := fixtures.List(t, field("has_joined_tokens"))
	if len(joined) != len(wantJoined) {
		t.Fatalf("hasJoined over %d tokens, want %d", len(joined), len(wantJoined))
	}
	for i, w := range wantJoined {
		if joined[i] != fixtures.Bool(t, w) {
			t.Errorf("hasJoined(token %d) = %v", i, joined[i])
		}
	}
	ws := words(unicodedata.Lower(strings.ReplaceAll(substituteIdentifiers(text), "İ", "i")))
	if got, want := englishRescuedByWords(ws, latin.DiacriticRate), fixtures.Bool(t, field("english_rescued_by_words")); got != want {
		t.Errorf("englishRescuedByWords = %v, want %v", got, want)
	}
}

// The corpus has to reach the branches it was built for; a golden that lost its edge cases would
// pass a port that lost the rules.
func TestCorpusCoversItsBranches(t *testing.T) {
	var mixed, nonEnglish, undecided, nonLatin, named, rescued, identifiers, acronyms int
	for _, raw := range fixtures.List(t, fixtures.Load(t, "lang_detect.json").Must("cases")) {
		row := fixtures.Obj(t, raw)
		analysis := fixtures.Field(t, row, "analyse")
		if fixtures.Field(t, analysis, "mixed_segment") != nil {
			mixed++
		}
		if !fixtures.Bool(t, fixtures.Field(t, analysis, "is_english")) {
			nonEnglish++
		}
		if fixtures.Bool(t, fixtures.Field(t, analysis, "language_undecided")) {
			undecided++
		}
		if s := fixtures.Str(t, fixtures.Field(t, analysis, "script")); s != "latin" && s != "unknown" {
			nonLatin++
		}
		h, ok := row.Get("helpers")
		if !ok {
			continue
		}
		raw := fixtures.Str(t, fixtures.Field(t, fixtures.Field(t, row, "state"), "value"))
		if fixtures.Field(t, h, "named_prose_language") != nil {
			named++
		}
		if fixtures.Bool(t, fixtures.Field(t, h, "english_rescued_by_words")) {
			rescued++
		}
		if raw != fixtures.Str(t, fixtures.Field(t, h, "substitute_identifiers")) {
			identifiers++
		}
		if raw != fixtures.Str(t, fixtures.Field(t, h, "blank_upper_runs")) {
			acronyms++
		}
	}
	for _, c := range []struct {
		name      string
		got, want int
	}{
		{"cases reaching the mixed-segment scan", mixed, 5},
		{"cases routed away from English", nonEnglish, 40},
		{"undecided cases", undecided, 30},
		{"non-Latin scripts", nonLatin, 14},
		{"cases naming a prose language", named, 15},
		{"cases exercising the English rescue", rescued, 10},
		{"cases holding an identifier", identifiers, 5},
		{"cases holding an acronym", acronyms, 5},
	} {
		if c.got < c.want {
			t.Errorf("only %d %s, want at least %d", c.got, c.name, c.want)
		}
	}
}

func TestRoundHalfEvenMatchesCPython(t *testing.T) {
	halfway, distinguishing := 0, 0
	for _, raw := range fixtures.List(t, fixtures.Load(t, "lang_detect.json").Must("round_to_int")) {
		pair := fixtures.List(t, raw)
		v, want := fixtures.Float(t, pair[0]), fixtures.Int(t, pair[1])
		if got := int(math.RoundToEven(v)); got != want {
			t.Errorf("round(%v) = %d, want %d", v, got, want)
		}
		if v == float64(int(v)) {
			continue
		}
		if frac := v - float64(int(v)); frac == 0.5 || frac == -0.5 {
			halfway++
			if int(math.Floor(v+0.5)) != want {
				distinguishing++
			}
		}
	}
	if halfway < 20 || distinguishing < 10 {
		t.Errorf("only %d halfway values (%d tell half-even from half-up); the check would not notice a mode swap",
			halfway, distinguishing)
	}
}

func TestRound4MatchesPython(t *testing.T) {
	// Expected values are what CPython 3.12's round(x, 4) returns.
	for _, c := range []struct{ in, want float64 }{
		{0.12345, 0.1235}, {0.12355, 0.1235}, {2.0 / 3, 0.6667}, {0.99995, 1}, {1, 1},
		{0.00005, 0.0001}, {0.00015, 0.0001}, {0.28125, 0.2812}, {1e-7, 0},
	} {
		if got := round4(c.in); got != c.want {
			t.Errorf("round4(%v) = %v, want %v", c.in, got, c.want)
		}
	}
	if got := round4(-1e-5); got != 0 || !math.Signbit(got) {
		t.Errorf("round4(-1e-5) = %v, want -0.0 as CPython gives", got)
	}
}

func TestIdentifierScanIsLinear(t *testing.T) {
	run := strings.Repeat("a", 200_000)
	if got := substituteIdentifiers(run); got != run {
		t.Error("a run with no joiner must come back untouched")
	}
	if got := substituteIdentifiers(run + "." + run); got != " " {
		t.Errorf("one joiner makes the whole thing one identifier, got %d bytes", len(got))
	}
	if n := len(words(run)); n != 1 {
		t.Errorf("words = %d, want 1", n)
	}
	if hasJoined(run) {
		t.Error("hasJoined on an unbroken run")
	}
}

func TestTrailingNewlinesChangeNothing(t *testing.T) {
	body := "The customer opened this ticket yesterday and we asked for a screenshot.\n" +
		"Nao consigo entrar na minha conta e a senha nao funciona de jeito nenhum"
	plain := Analyze(body)
	for _, suffix := range []string{"\n", "\n\n", "\n\n\n", "\n \n"} {
		got := Analyze(body + suffix)
		if got.Language != plain.Language || got.English != plain.English ||
			got.MixedSegment != plain.MixedSegment {
			t.Errorf("a %q suffix changed the verdict: %+v vs %+v", suffix, got, plain)
		}
	}
}

func TestStateDepthLimit(t *testing.T) {
	const leaf = "Ich kann mich nicht in mein Konto einloggen und das Passwort"
	var deep any = leaf
	for i := 0; i < 6; i++ {
		deep = []any{deep}
	}
	if got := StateText(deep); got != leaf {
		t.Errorf("six levels of nesting are still read, got %q", got)
	}
	if got := StateText([]any{deep}); got != "" {
		t.Errorf("the seventh level must contribute nothing, got %q", got)
	}
}

func TestStateShapes(t *testing.T) {
	obj := Object{{"b", "second"}, {"a", "first"}}
	if got := StateText(obj); got != "second first" {
		t.Errorf("an Object reads in insertion order, got %q", got)
	}
	if got := StateText(map[string]any{"b": "second", "a": "first"}); got != "first second" {
		t.Errorf("a map reads in sorted key order, got %q", got)
	}
	if got := StateText([]byte{0xff, 0xfe}); got != "" {
		t.Errorf("invalid UTF-8 bytes are not text, got %q", got)
	}
	if got := StateText([]any{1, true, nil, 2.5, "x"}); got != "x" {
		t.Errorf("only strings and bytes are text, got %q", got)
	}
	if got := StateTextN("abcdef", 3); got != "abc" {
		t.Errorf("StateTextN = %q", got)
	}
}

func TestConcurrentUse(t *testing.T) {
	var states []any
	for _, raw := range fixtures.List(t, fixtures.Load(t, "lang_detect.json").Must("cases")) {
		states = append(states, stateOf(t, fixtures.Obj(t, raw)))
	}
	want := make([]bool, len(states))
	for i, state := range states {
		want[i] = Analyze(state).English
	}
	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i, state := range states {
				if got := Analyze(state).English; got != want[i] {
					t.Errorf("case %d changed its verdict under concurrent use", i)
				}
			}
		}()
	}
	wg.Wait()
}

// round4 against CPython's round(x, 4) on a distribution of real shares and on engineered
// halfway values, where half-up and half-even part ways.
func TestRound4MatchesPythonGolden(t *testing.T) {
	groups := fixtures.Obj(t, fixtures.Load(t, "python_json.json").Must("round4"))
	if _, ok := groups.Get("halfway"); !ok {
		t.Fatal("the halfway group is what discriminates the rules")
	}
	for _, g := range groups {
		for _, raw := range fixtures.List(t, g.Value) {
			bits, ok := new(big.Int).SetString(string(fixtures.Field(t, raw, "bits").(json.Number)), 10)
			if !ok {
				t.Fatal("bad bits")
			}
			v := math.Float64frombits(bits.Uint64())
			if got, want := round4(v), fixtures.Float(t, fixtures.Field(t, raw, "r4")); got != want {
				t.Errorf("%s: round4(%v) = %v, want %v", g.Key, v, got, want)
			}
		}
	}
}
