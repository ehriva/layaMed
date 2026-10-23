// Package fixtures loads the golden files the parity tests compare against.
//
// The goldens are recorded from the Python package (laya-java/scripts/gen_fixtures.py), and CI
// regenerates them on every run into a scratch directory named by LAYA_GO_FIXTURES. Locally the
// default is laya-java/fixtures, so a plain `go test ./...` works from a checkout. Objects keep
// their key order, because in the goldens order is data: a profile's keys, a state's values.
//
// This package exists for tests and is not part of the library.
package fixtures

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Member is one key and value of an Object.
type Member struct {
	Key   string
	Value any
}

// Object is a JSON object that remembers its key order.
type Object []Member

// Get returns the value for key.
func (o Object) Get(key string) (any, bool) {
	for _, m := range o {
		if m.Key == key {
			return m.Value, true
		}
	}
	return nil, false
}

// Must returns the value for key and panics when it is absent, for use where a missing key means
// the golden itself is broken.
func (o Object) Must(key string) any {
	v, ok := o.Get(key)
	if !ok {
		panic("golden has no key " + key)
	}
	return v
}

// Keys lists the keys in order.
func (o Object) Keys() []string {
	keys := make([]string, len(o))
	for i, m := range o {
		keys[i] = m.Key
	}
	return keys
}

// extraGolden is the one golden laya-go records itself, because laya-java has no use for it.
const extraGolden = "go_extra.json"

// Dir is where the goldens are read from: LAYA_GO_FIXTURES when set, else the committed copies
// (laya-java/fixtures for the shared goldens, laya-go/fixtures for go_extra.json).
func Dir(name string) string {
	if dir := os.Getenv("LAYA_GO_FIXTURES"); dir != "" {
		return dir
	}
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..", "..")
	if name == extraGolden {
		return filepath.Join(root, "fixtures")
	}
	return filepath.Join(root, "..", "laya-java", "fixtures")
}

// Checkpoints is the root holding the english and multilingual checkpoints, or "" when unset.
func Checkpoints() string {
	return os.Getenv("LAYA_CHECKPOINTS")
}

// Load reads and parses one golden file. A missing or malformed file fails the test, never skips
// it: a golden that is absent is a broken run, not a reason to pass.
func Load(tb testing.TB, name string) Object {
	tb.Helper()
	data, err := os.ReadFile(filepath.Join(Dir(name), name))
	if err != nil {
		tb.Fatalf("golden %s: %v", name, err)
	}
	value, err := Parse(data)
	if err != nil {
		tb.Fatalf("golden %s: %v", name, err)
	}
	obj, ok := value.(Object)
	if !ok {
		tb.Fatalf("golden %s is not a JSON object", name)
	}
	return obj
}

// Parse decodes JSON into nil, bool, string, json.Number, []any and Object.
func Parse(data []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	value, err := parseValue(dec)
	if err != nil {
		return nil, err
	}
	if dec.More() {
		return nil, fmt.Errorf("trailing data after the JSON value")
	}
	return value, nil
}

func parseValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	delim, isDelim := tok.(json.Delim)
	if !isDelim {
		return tok, nil
	}
	switch delim {
	case '{':
		obj := Object{}
		for dec.More() {
			keyTok, err := dec.Token()
			if err != nil {
				return nil, err
			}
			value, err := parseValue(dec)
			if err != nil {
				return nil, err
			}
			obj = append(obj, Member{keyTok.(string), value})
		}
		_, err = dec.Token()
		return obj, err
	case '[':
		list := []any{}
		for dec.More() {
			value, err := parseValue(dec)
			if err != nil {
				return nil, err
			}
			list = append(list, value)
		}
		_, err = dec.Token()
		return list, err
	}
	return nil, fmt.Errorf("unexpected delimiter %v", delim)
}

// Field fetches a key of an object, failing the test when it is absent.
func Field(tb testing.TB, v any, key string) any {
	tb.Helper()
	obj, ok := v.(Object)
	if !ok {
		tb.Fatalf("want an object holding %q, got %T", key, v)
	}
	value, ok := obj.Get(key)
	if !ok {
		tb.Fatalf("object has no key %q (keys: %v)", key, obj.Keys())
	}
	return value
}

// Str asserts v is a string.
func Str(tb testing.TB, v any) string {
	tb.Helper()
	s, ok := v.(string)
	if !ok {
		tb.Fatalf("want a string, got %T (%v)", v, v)
	}
	return s
}

// StrOrEmpty reads a string-or-null as a string, null becoming "".
func StrOrEmpty(tb testing.TB, v any) string {
	tb.Helper()
	if v == nil {
		return ""
	}
	return Str(tb, v)
}

// Int asserts v is an integer.
func Int(tb testing.TB, v any) int {
	tb.Helper()
	n, ok := v.(json.Number)
	if !ok {
		tb.Fatalf("want a number, got %T (%v)", v, v)
	}
	i, err := n.Int64()
	if err != nil {
		tb.Fatalf("want an integer, got %v", n)
	}
	return int(i)
}

// Float asserts v is a number and returns it as a float64.
func Float(tb testing.TB, v any) float64 {
	tb.Helper()
	n, ok := v.(json.Number)
	if !ok {
		tb.Fatalf("want a number, got %T (%v)", v, v)
	}
	f, err := n.Float64()
	if err != nil {
		tb.Fatalf("want a float, got %v", n)
	}
	return f
}

// Bool asserts v is a boolean.
func Bool(tb testing.TB, v any) bool {
	tb.Helper()
	b, ok := v.(bool)
	if !ok {
		tb.Fatalf("want a bool, got %T (%v)", v, v)
	}
	return b
}

// List asserts v is an array.
func List(tb testing.TB, v any) []any {
	tb.Helper()
	l, ok := v.([]any)
	if !ok {
		tb.Fatalf("want an array, got %T", v)
	}
	return l
}

// Obj asserts v is an object.
func Obj(tb testing.TB, v any) Object {
	tb.Helper()
	o, ok := v.(Object)
	if !ok {
		tb.Fatalf("want an object, got %T", v)
	}
	return o
}

// Strings asserts v is an array of strings.
func Strings(tb testing.TB, v any) []string {
	tb.Helper()
	list := List(tb, v)
	out := make([]string, len(list))
	for i, item := range list {
		out[i] = Str(tb, item)
	}
	return out
}

// Ints asserts v is an array of integers.
func Ints(tb testing.TB, v any) []int32 {
	tb.Helper()
	list := List(tb, v)
	out := make([]int32, len(list))
	for i, item := range list {
		out[i] = int32(Int(tb, item))
	}
	return out
}

// Marshal writes a parsed value back as JSON, keeping object order, for tests that hand a
// golden's embedded document (a tokenizer spec, say) to the code under test.
func Marshal(tb testing.TB, v any) []byte {
	tb.Helper()
	var buf bytes.Buffer
	if err := marshalTo(&buf, v); err != nil {
		tb.Fatalf("marshal: %v", err)
	}
	return buf.Bytes()
}

func marshalTo(buf *bytes.Buffer, v any) error {
	switch x := v.(type) {
	case Object:
		buf.WriteByte('{')
		for i, m := range x {
			if i > 0 {
				buf.WriteByte(',')
			}
			key, _ := json.Marshal(m.Key)
			buf.Write(key)
			buf.WriteByte(':')
			if err := marshalTo(buf, m.Value); err != nil {
				return err
			}
		}
		buf.WriteByte('}')
	case []any:
		buf.WriteByte('[')
		for i, item := range x {
			if i > 0 {
				buf.WriteByte(',')
			}
			if err := marshalTo(buf, item); err != nil {
				return err
			}
		}
		buf.WriteByte(']')
	default:
		data, err := json.Marshal(x)
		if err != nil {
			return err
		}
		buf.Write(data)
	}
	return nil
}
