package pyjson

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"math/big"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
)

func TestFloatReprMatchesCPython(t *testing.T) {
	cases := fixtures.List(t, fixtures.Load(t, "python_json.json").Must("doubles"))
	covered := map[uint64]bool{}
	for _, c := range cases {
		bits, err := fixtures.Field(t, c, "bits").(json.Number).Int64()
		if err != nil {
			// above 2^63: parse unsigned
			u, ok := new(big.Int).SetString(string(fixtures.Field(t, c, "bits").(json.Number)), 10)
			if !ok {
				t.Fatalf("bad bits %v", fixtures.Field(t, c, "bits"))
			}
			bits = int64(u.Uint64())
		}
		v := math.Float64frombits(uint64(bits))
		covered[uint64(bits)] = true
		if got, want := ReprFloat(v), fixtures.Str(t, fixtures.Field(t, c, "repr")); got != want {
			t.Errorf("repr of bits %d = %q, want %q", bits, got, want)
		}
	}
	if len(cases) < 400 {
		t.Errorf("the doubles golden should be substantial, found %d", len(cases))
	}
	for _, v := range []float64{1e16, 1e-5, 0.0001, 1e23, math.Copysign(0, -1), 5e-324, 2.0} {
		if !covered[math.Float64bits(v)] {
			t.Errorf("the golden no longer covers %v", v)
		}
	}
}

// valueOf rebuilds a golden value, with integers as *big.Int and floats as float64, so the
// integer/float distinction json.dumps preserves survives the trip.
func valueOf(t *testing.T, v any) any {
	t.Helper()
	switch x := v.(type) {
	case fixtures.Object:
		out := make(Object, len(x))
		for i, m := range x {
			out[i] = Member{m.Key, valueOf(t, m.Value)}
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			out[i] = valueOf(t, item)
		}
		return out
	case json.Number:
		if n, ok := new(big.Int).SetString(string(x), 10); ok {
			return n
		}
		f, err := x.Float64()
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	return v
}

func TestDumpsMatchesCPython(t *testing.T) {
	for _, c := range fixtures.List(t, fixtures.Load(t, "python_json.json").Must("structured")) {
		want := fixtures.Str(t, fixtures.Field(t, c, "dumps"))
		got, err := Dumps(valueOf(t, fixtures.Field(t, c, "value")))
		if err != nil || got != want {
			t.Errorf("Dumps = %q (%v), want %q", got, err, want)
		}
	}
}

func TestReprMatchesCPython(t *testing.T) {
	cases := fixtures.List(t, fixtures.Load(t, "python_json.json").Must("repr_strings"))
	escaped, doubleQuoted := 0, 0
	for _, c := range cases {
		pair := fixtures.List(t, c)
		in, want := fixtures.Str(t, pair[0]), fixtures.Str(t, pair[1])
		if got := Repr(in); got != want {
			t.Errorf("Repr(%+q) = %q, want %q", in, got, want)
		}
		if strings.Contains(want, `\`) {
			escaped++
		}
		if strings.HasPrefix(want, `"`) {
			doubleQuoted++
		}
	}
	if len(cases) < 30 || escaped < 15 || doubleQuoted < 1 {
		t.Errorf("the golden lost its edge cases: %d cases, %d escaping, %d double-quoted", len(cases), escaped, doubleQuoted)
	}
}

func TestReprEscapesExactlyTheUnprintableEverywhere(t *testing.T) {
	h := sha256.New()
	for r := rune(0); r < 0x110000; r++ {
		if r >= 0xD800 && r <= 0xDFFF {
			continue
		}
		h.Write([]byte(Repr("a" + string(r) + "b")))
	}
	want := fixtures.Str(t, fixtures.Load(t, "python_json.json").Must("repr_digest"))
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		t.Error("Repr disagrees with CPython somewhere in Unicode")
	}
}

func TestPercent0MatchesCPython(t *testing.T) {
	distinguishing := 0
	for _, c := range fixtures.List(t, fixtures.Load(t, "python_json.json").Must("percent0")) {
		pair := fixtures.List(t, c)
		fraction, want := fixtures.Float(t, pair[0]), fixtures.Str(t, pair[1])
		if got := Percent0(fraction); got != want {
			t.Errorf("Percent0(%v) = %q, want %q", fraction, got, want)
		}
		if math.Floor(100*fraction+0.5) != math.RoundToEven(100*fraction) {
			distinguishing++
		}
	}
	if distinguishing < 3 {
		t.Errorf("only %d cases tell half-even from half-up", distinguishing)
	}
}

func TestNonFiniteAndNegativeZero(t *testing.T) {
	for v, want := range map[float64]string{math.NaN(): "NaN", math.Inf(1): "Infinity", math.Inf(-1): "-Infinity"} {
		if got := ReprFloat(v); got != want {
			t.Errorf("ReprFloat(%v) = %q, want %q", v, got, want)
		}
	}
	if got := ReprFloat(math.Copysign(0, -1)); got != "-0.0" {
		t.Errorf("negative zero keeps its sign, got %q", got)
	}
	if Percent0(math.NaN()) != "nan" || Percent0(math.Inf(1)) != "inf" || Percent0(math.Inf(-1)) != "-inf" {
		t.Error("Percent0 spells non-finite values in lower case, as Python percent-formatting does")
	}
}

func TestRefusals(t *testing.T) {
	if _, err := Dumps(struct{}{}); err == nil {
		t.Error("an unsupported type must be refused, not stringified")
	}
	if _, err := Dumps("\xff"); err == nil {
		t.Error("a string that is not UTF-8 must be refused")
	}
	if _, err := Dumps(Object{{[]any{1}, "x"}}); err == nil {
		t.Error("a key type CPython refuses is still refused")
	}
	deep := any("x")
	for i := 0; i < 600; i++ {
		deep = []any{deep}
	}
	if _, err := Dumps(deep); err == nil {
		t.Error("nesting past the limit must be refused")
	}
}

func TestKeysAreCoercedAsPythonCoercesThem(t *testing.T) {
	for _, c := range []struct {
		in   Object
		want string
	}{
		{Object{{1, "x"}}, `{"1": "x"}`},
		{Object{{true, 1}}, `{"true": 1}`},
		{Object{{nil, "x"}}, `{"null": "x"}`},
		{Object{{2.5, 1}}, `{"2.5": 1}`},
	} {
		if got, err := Dumps(c.in); err != nil || got != c.want {
			t.Errorf("Dumps = %q (%v), want %q", got, err, c.want)
		}
	}
	if got, _ := Dumps(map[string]any{"b": 1, "a": 2}); got != `{"a": 2, "b": 1}` {
		t.Errorf("a plain map is written in sorted key order, got %q", got)
	}
}
