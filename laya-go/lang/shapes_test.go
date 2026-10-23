package lang

import (
	"encoding/hex"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
)

type named string
type namedBytes []byte

func selfList() []any {
	l := make([]any, 2)
	l[0], l[1] = german, l
	return l
}

func selfMap() map[string]any {
	m := map[string]any{"a": german}
	m["b"] = m
	return m
}

func ptr[T any](v T) *T { return &v }

const german = "Ich kann mich nicht in mein Konto einloggen und das Passwort funktioniert nicht"

// goVariants spells each shape lang.py's golden records as the Go types a caller might hold.
func goVariants() map[string][]any {
	var nilStr *string
	var nilIface any
	return map[string][]any{
		"string": {german, named(german), ptr(german), ptr(named(german)), ptr[any](german)},
		"map_of_slices": {map[string][]string{"a": {german}}, http.Header{"A": {german}},
			url.Values{"a": {german}}, map[string][]any{"a": {german}}, map[string]any{"a": []string{german}}},
		"slice_of_maps": {[]map[string]string{{"a": german}}, []map[string]any{{"a": german}},
			[]any{map[string]string{"a": german}}},
		"map_of_objects": {map[string]Object{"a": {{"b", german}}}, map[string]any{"a": Object{{"b", german}}},
			map[string]map[string]string{"a": {"b": german}}},
		"slice_of_slices": {[][]string{{german}}, [][]any{{german}}, []any{[]string{german}}},
		"tuple":           {[1]string{german}, [1]any{german}, ptr([1]string{german})},
		"seven_deep":      {[]any{[]any{[]any{[]any{[]any{[]any{[]any{german}}}}}}}},
		"six_deep":        {[]any{[]any{[]any{[]any{[]any{[]any{german}}}}}}},
		// no Python equivalent for most of these: they contribute nothing, as non-container objects do
		"numbers_and_none": {[]any{1, 2.5, nil, true}, []int{1, 2}, map[string]float64{"a": 1}, struct{ A string }{german},
			nilStr, nilIface},
		"non_string_keys": {map[int]string{1: german}, map[float64]any{2.5: german}, map[bool][]string{true: {german}},
			map[[2]int]string{{1, 2}: german}, map[any]string{1: german}},
		"self_ref_list": {selfList()},
		"self_ref_dict": {selfMap()},
		"bytes":         {[]byte(german), namedBytes(german), ptr([]byte(german))},
		"invalid_bytes": {[]byte{0xff, 0xfe}, namedBytes{0xff, 0xfe}},
		"mixed":         {map[string]any{"k": []any{german, 3, map[string]any{"x": nil}}}},
	}
}

// Python's lang.py reads any Mapping, list or tuple, and ignores everything else. Each variant
// must reach the same verdict as the Python state it stands for.
func TestStateShapesMatchPython(t *testing.T) {
	golden := fixtures.Obj(t, fixtures.Load(t, "go_extra.json").Must("state_shapes"))
	variants := goVariants()
	for _, g := range golden {
		vs, ok := variants[g.Key]
		if !ok {
			t.Fatalf("no Go variant for the recorded shape %q", g.Key)
		}
		wantEnglish := fixtures.Bool(t, fixtures.Field(t, g.Value, "is_english"))
		wantText := fixtures.Str(t, fixtures.Field(t, g.Value, "state_text"))
		wantLang := fixtures.StrOrEmpty(t, fixtures.Field(t, g.Value, "language"))
		for i, v := range vs {
			if got := StateText(v); got != wantText {
				t.Errorf("%s variant %d (%T): StateText = %q, want %q", g.Key, i, v, got, wantText)
			}
			a := Analyze(v)
			if a.English != wantEnglish || a.Language != wantLang {
				t.Errorf("%s variant %d (%T): English=%v Language=%q, want %v %q", g.Key, i, v, a.English, a.Language, wantEnglish, wantLang)
			}
			if IsEnglish(v) != wantEnglish {
				t.Errorf("%s variant %d (%T): IsEnglish disagrees", g.Key, i, v)
			}
		}
	}
	if len(golden) != len(variants) {
		t.Errorf("%d shapes recorded, %d spelled in Go", len(golden), len(variants))
	}
}

func TestInvalidUTF8MatchesPython(t *testing.T) {
	fam := fixtures.Obj(t, fixtures.Load(t, "go_extra.json").Must("invalid_utf8"))
	hexes := fixtures.Strings(t, fam.Must("hex"))
	decoded := fixtures.Strings(t, fam.Must("decoded"))
	scripts := fixtures.Strings(t, fam.Must("script"))
	texts := fixtures.Strings(t, fam.Must("state_text"))
	if len(hexes) < 3000 {
		t.Fatalf("only %d cases", len(hexes))
	}
	for i, h := range hexes {
		raw, err := hex.DecodeString(h)
		if err != nil {
			t.Fatal(err)
		}
		if got := StateText(string(raw)); got != texts[i] {
			t.Fatalf("StateText(%x) = %+q, want %+q", raw, got, texts[i])
		}
		if got := DetectScript(string(raw)); string(got) != scripts[i] {
			t.Fatalf("DetectScript(%x) = %q, want %q (decoded %+q)", raw, got, scripts[i], decoded[i])
		}
	}
}

// A pointer that points at itself, however it is spelled, must end and read as nothing.
func TestPointerLoopsTerminate(t *testing.T) {
	var viaInterface any
	viaInterface = &viaInterface
	type node struct{ Next *node }
	n := &node{}
	n.Next = n
	ring := map[string]any{}
	ring["self"] = &ring
	chain := []any{german}
	chain = append(chain, &chain)
	var pp **any
	var loop any
	pp = new(*any)
	*pp = &loop
	loop = pp
	for name, v := range map[string]any{"interface": viaInterface, "struct": n, "map via pointer": ring, "slice via pointer": chain, "double pointer": loop} {
		done := make(chan struct{})
		go func() { defer close(done); _ = StateText(v); _ = Analyze(v); _ = IsEnglish(v) }()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatalf("%s: did not terminate", name)
		}
	}
	if got := StateText(viaInterface); got != "" {
		t.Errorf("a pointer loop reads as nothing, got %q", got)
	}
	if got := StateText(chain); got != strings.Join([]string{german, german, german, german, german, german}, " ") {
		t.Errorf("a loop through a slice ends at the depth limit as Python's does, got %d chars", len(got))
	}
}

// A Go map has no insertion order, so values are read in the order fmt prints a map: numbers
// numerically, not as text, strings lexically, false before true.
func TestMapKeysAreReadInSortedOrder(t *testing.T) {
	for name, c := range map[string]struct {
		state any
		want  string
	}{
		"ints numerically":  {map[int]string{10: "j", 3: "c", 1: "a", 2: "b"}, "a b c j"},
		"floats":            {map[float64]string{2.5: "b", -1: "a", 10: "c"}, "a b c"},
		"bools":             {map[bool]string{true: "t", false: "f"}, "f t"},
		"strings":           {map[string]string{"b": "2", "a": "1", "B": "0"}, "0 1 2"},
		"arrays":            {map[[2]int]string{{2, 1}: "c", {1, 9}: "b", {1, 2}: "a"}, "a b c"},
		"mixed in an iface": {map[any]string{"x": "s", 2: "i", 1: "j"}, "j i s"},
	} {
		if got := StateText(c.state); got != c.want {
			t.Errorf("%s: StateText = %q, want %q", name, got, c.want)
		}
	}
}

func sameNameA() any {
	type key struct{ A int }
	return key{1}
}

func sameNameB() any {
	type key struct{ A, B int }
	return key{1, 2}
}

// Two distinct types that share a name must order without being compared field by field, which
// panics, and must order the same way every time.
func TestSameNamedKeyTypesDoNotPanic(t *testing.T) {
	m := map[any]string{sameNameA(): "a", sameNameB(): "b"}
	first := StateText(m)
	if first != "a b" && first != "b a" {
		t.Fatalf("StateText = %q", first)
	}
	for i := 0; i < 50; i++ {
		if got := StateText(m); got != first {
			t.Fatalf("run %d read %q, the first read %q", i, got, first)
		}
	}
}
