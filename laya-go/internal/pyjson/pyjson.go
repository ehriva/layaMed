// Package pyjson writes values the way CPython's json.dumps(..., ensure_ascii=False) and repr()
// do.
//
// This is not a convenience. laya feeds the result straight into the tokenizer (a non-string state
// is JSON-encoded, a structured criterion likewise), so every character produced becomes token
// ids: a formatter that agrees with Python on ordinary values and differs on one float yields a
// different sequence and a different answer, with no test above the tokenizer able to say why.
// Go's own spelling differs from Python's in the places that matter: strconv switches to scientific
// notation at different magnitudes and pads the exponent differently, and json.Marshal escapes <, >
// and & and sorts map keys.
package pyjson

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/NandhaKishorM/laya/laya-go/internal/unicodedata"
)

// maxDepth bounds nesting, which also stops a self-referential []any from recursing forever.
const maxDepth = 512

// Member is one key and value of an Object.
type Member struct {
	Key   any
	Value any
}

// Object is a mapping that keeps insertion order, as a Python dict does. Keys may be strings,
// numbers, booleans or nil, which json.dumps coerces to strings. A plain map[string]any is also
// accepted and written in sorted key order.
type Object []Member

// Dumps is json.dumps(value, ensure_ascii=False): ", " and ": " separators, no escaping beyond
// the JSON minimum, and NaN and Infinity spelled as Python spells them.
//
// It accepts nil, bool, string, every integer type, *big.Int, float32, float64, []any, []string,
// Object and map[string]any, and returns an error for anything else or for a string that is not
// valid UTF-8, rather than stringifying it into the model's input.
func Dumps(value any) (string, error) {
	var b strings.Builder
	if err := write(&b, value, 0); err != nil {
		return "", err
	}
	return b.String(), nil
}

func write(b *strings.Builder, value any, depth int) error {
	if depth > maxDepth {
		return fmt.Errorf("value nested deeper than %d levels", maxDepth)
	}
	switch v := value.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		b.WriteString(strconv.FormatBool(v))
	case string:
		return writeString(b, v)
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		fmt.Fprint(b, v)
	case *big.Int:
		b.WriteString(v.String())
	case float64:
		b.WriteString(ReprFloat(v))
	case float32:
		b.WriteString(ReprFloat(float64(v)))
	case []any:
		b.WriteByte('[')
		for i, item := range v {
			if i > 0 {
				b.WriteString(", ")
			}
			if err := write(b, item, depth+1); err != nil {
				return err
			}
		}
		b.WriteByte(']')
	case []string:
		items := make([]any, len(v))
		for i, s := range v {
			items[i] = s
		}
		return write(b, items, depth)
	case Object:
		b.WriteByte('{')
		for i, m := range v {
			if i > 0 {
				b.WriteString(", ")
			}
			if err := writeMember(b, m, depth); err != nil {
				return err
			}
		}
		b.WriteByte('}')
	case map[string]any:
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		obj := make(Object, len(keys))
		for i, k := range keys {
			obj[i] = Member{k, v[k]}
		}
		return write(b, obj, depth)
	default:
		return fmt.Errorf("cannot serialise %T the way Python's json does; pass a string, bool, "+
			"integer, float, []any, Object or map[string]any", value)
	}
	return nil
}

func writeMember(b *strings.Builder, m Member, depth int) error {
	key, err := keyString(m.Key)
	if err != nil {
		return err
	}
	if err := writeString(b, key); err != nil {
		return err
	}
	b.WriteString(": ")
	return write(b, m.Value, depth+1)
}

// keyString coerces a key as json.dumps does.
func keyString(key any) (string, error) {
	switch k := key.(type) {
	case string:
		return k, nil
	case nil:
		return "null", nil
	case bool:
		return strconv.FormatBool(k), nil
	case float64:
		return ReprFloat(k), nil
	case float32:
		return ReprFloat(float64(k)), nil
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return fmt.Sprint(k), nil
	case *big.Int:
		return k.String(), nil
	}
	return "", fmt.Errorf("keys must be a string, a number, a boolean or null, as CPython "+
		"requires; got %T", key)
}

var errInvalidUTF8 = errors.New("string is not valid UTF-8, which a Python str always is")

func writeString(b *strings.Builder, s string) error {
	if !utf8.ValidString(s) {
		return errInvalidUTF8
	}
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			if r < 0x20 {
				fmt.Fprintf(b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
	return nil
}

// ReprFloat is Python's repr(float), which json.dumps also uses: the shortest digits that
// round-trip, fixed notation for decimal exponents in [-4, 16) and scientific outside it, with
// a signed two-digit exponent. NaN and the infinities are spelled as json.dumps spells them.
func ReprFloat(v float64) string {
	switch {
	case math.IsNaN(v):
		return "NaN"
	case math.IsInf(v, 1):
		return "Infinity"
	case math.IsInf(v, -1):
		return "-Infinity"
	}
	sign := ""
	if math.Signbit(v) {
		sign = "-"
		v = -v
	}
	if v == 0 {
		return sign + "0.0"
	}
	// FormatFloat's shortest form is d.ddde±XX; take its digits and exponent.
	text := strconv.FormatFloat(v, 'e', -1, 64)
	mantissa, expText, _ := strings.Cut(text, "e")
	exp, _ := strconv.Atoi(expText)
	digits := strings.Replace(mantissa, ".", "", 1)
	if exp >= -4 && exp < 16 {
		return sign + fixed(digits, exp)
	}
	var out strings.Builder
	out.WriteString(sign)
	out.WriteByte(digits[0])
	if len(digits) > 1 {
		out.WriteByte('.')
		out.WriteString(digits[1:])
	}
	out.WriteByte('e')
	if exp < 0 {
		out.WriteByte('-')
		exp = -exp
	} else {
		out.WriteByte('+')
	}
	if exp < 10 {
		out.WriteByte('0')
	}
	out.WriteString(strconv.Itoa(exp))
	return out.String()
}

func fixed(digits string, exp int) string {
	if exp < 0 {
		return "0." + strings.Repeat("0", -exp-1) + digits
	}
	whole := exp + 1
	if len(digits) <= whole {
		return digits + strings.Repeat("0", whole-len(digits)) + ".0"
	}
	return digits[:whole] + "." + digits[whole:]
}

// Repr is Python's repr(str): single-quoted unless the text holds a single quote and no double
// quote, with everything str.isprintable() rejects escaped as \xNN, \uNNNN or \UNNNNNNNN.
func Repr(s string) string {
	quote := '\''
	if strings.ContainsRune(s, '\'') && !strings.ContainsRune(s, '"') {
		quote = '"'
	}
	var b strings.Builder
	b.Grow(len(s) + 2)
	b.WriteRune(quote)
	for _, r := range s {
		switch {
		case r == quote || r == '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case unicodedata.IsPrintable(r):
			b.WriteRune(r)
		case r < 0x100:
			fmt.Fprintf(&b, `\x%02x`, r)
		case r < 0x10000:
			fmt.Fprintf(&b, `\u%04x`, r)
		default:
			fmt.Fprintf(&b, `\U%08x`, r)
		}
	}
	b.WriteRune(quote)
	return b.String()
}

// Percent0 is Python's '%.0f' % (100 * fraction): a whole-number percentage, halves to even.
func Percent0(fraction float64) string {
	scaled := 100 * fraction
	switch {
	case math.IsNaN(scaled):
		return "nan"
	case math.IsInf(scaled, 1):
		return "inf"
	case math.IsInf(scaled, -1):
		return "-inf"
	}
	return strconv.FormatFloat(scaled, 'f', 0, 64)
}
