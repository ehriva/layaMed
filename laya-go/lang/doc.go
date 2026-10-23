// Package lang detects the script and, for Latin script, the language of a piece of text. It
// is the Go port of laya/lang.py and returns the same answers, including its tie-breaks and
// thresholds; laya uses it to decide whether a request belongs on the English checkpoint.
//
// Analyze is the entry point: it returns the dominant Script, the language (an ISO 639-1 code,
// "" when none can be named) and whether the text is English. DetectScript, ScriptProfile,
// LatinProfileOf and IsEnglish answer narrower questions.
//
// Text is read as UTF-8. Invalid bytes become U+FFFD one per maximal invalid subpart, as
// Python's bytes.decode("utf-8", "replace") does, before anything is counted.
//
// # States
//
// What Analyze reads is a "state": a string, bytes, or any map (whatever its keys), slice or
// array of them, nested up to MaxDepth levels and followed through pointers and named types.
// Like lang.py, anything else contributes nothing: a struct, a number or nil reads as no text,
// and a text with no letters is English with ScriptUnknown.
//
// A Go map has no insertion order, so its values are read in sorted key order: like fmt for a
// single key type (numbers numerically, strings lexically), and for a map[any]T with mixed key
// types by kind, then type name. A Python dict is read in insertion order, so a verdict can
// differ from a Python dict with the same items when the MaxChars budget truncates. Use Object,
// which keeps insertion order and decodes from JSON in document order, for an order-sensitive
// state.
//
// Reading a state is O(number of leaves): StateText stops once its budget is spent, but Analyze
// scans every leaf for a foreign-language one.
//
// Every function is safe for concurrent use: the tables are read-only and nothing here keeps
// state between calls.
package lang
