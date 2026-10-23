package lang_test

import (
	"encoding/json"
	"fmt"

	"github.com/NandhaKishorM/laya/laya-go/lang"
)

func ExampleAnalyze() {
	a := lang.Analyze("Ich kann mich nicht in mein Konto einloggen und das Passwort funktioniert nicht")
	fmt.Println(a.Script, a.Language, a.English)

	a = lang.Analyze("我们被收取了两次费用，请退还。")
	fmt.Printf("%s %q %v\n", a.Script, a.Language, a.English)

	// A text with no letters is English: it goes to the English checkpoint.
	a = lang.Analyze("")
	fmt.Println(a.Script, a.English)
	// Output:
	// latin de false
	// han "" false
	// unknown true
}

func ExampleIsEnglish() {
	fmt.Println(lang.IsEnglish("Please refund the duplicate charge from March."))
	fmt.Println(lang.IsEnglish("Je n'arrive pas à me connecter à mon compte depuis hier."))
	// A state may be any nesting of maps and slices; a struct, a number or nil reads as nothing.
	fmt.Println(lang.IsEnglish(map[string][]string{"message": {"No puedo entrar en mi cuenta y quiero que me devuelvan el dinero"}}))
	// Output:
	// true
	// false
	// false
}

func ExampleObject() {
	// Python reads a dict in insertion order, and a Go map has none. Decoding JSON into an
	// Object keeps the order the document had.
	var state lang.Object
	_ = json.Unmarshal([]byte(`{"title":"Refund request","body":"Ich kann mich nicht einloggen"}`), &state)
	fmt.Println(lang.StateText(state))
	// Output:
	// Refund request Ich kann mich nicht einloggen
}
