package router_test

import (
	"fmt"

	"github.com/NandhaKishorM/laya/laya-go/router"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

func ExampleRouter_Route() {
	r, _ := router.New()
	var qs schema.Questions

	for _, state := range []string{
		"Please refund the duplicate charge from March.",
		"मुझसे दो बार शुल्क लिया गया",
		"",
	} {
		d, _ := r.Route(state, qs)
		fmt.Println(d.Model, "|", d.Reason)
	}
	// An explicit model wins over everything the text says.
	d, _ := r.Route("मुझसे दो बार शुल्क लिया गया", qs, router.ForModel("en"))
	fmt.Println(d)
	// Output:
	// english | English Latin text
	// multilingual | non-Latin script (devanagari, 100% of letters); the English checkpoint cannot read it
	// multilingual | no letters detected in state; using default (multilingual)
	// RouteDecision(model='english', reason="explicit model='en'")
}

func ExampleRouter_RouteBatch() {
	r, _ := router.New(router.WithDefault("english"))
	decisions, _ := r.RouteBatch([]router.Request{
		{State: "12345"},
		{State: "Kunden wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung"},
		{State: "anything", Options: []router.CallOption{router.ForLang("de_DE.UTF-8")}},
	})
	for _, d := range decisions {
		fmt.Println(d.Model)
	}
	// Output:
	// english
	// multilingual
	// multilingual
}

func ExampleEnglishFromCode() {
	for _, code := range []string{"en_US.UTF-8", "pt-BR", "C.UTF-8", ""} {
		english, ok := router.EnglishFromCode(code)
		fmt.Printf("%q english=%v known=%v\n", code, english, ok)
	}
	// Output:
	// "en_US.UTF-8" english=true known=true
	// "pt-BR" english=false known=true
	// "C.UTF-8" english=false known=false
	// "" english=false known=false
}

func ExampleNormaliseName() {
	c, _ := router.NormaliseName("  Typed_Decisions ")
	fmt.Println(c)
	_, err := router.NormaliseName("gpt")
	fmt.Println(err)
	// Output:
	// typed-decisions
	// unknown model 'gpt'; choose one of ['english', 'multilingual', 'typed-decisions'] (or an alias: ['decisions', 'default', 'en', 'laya', 'laya-multilingual', 'laya-typed-decisions', 'ml', 'multi', 'typed', 'typed_decisions'])
}

func ExampleRouter_AddHook() {
	r, _ := router.New()
	// A hook may replace the decision; Route returns the replacement.
	_ = r.AddHook(&router.HookFuncs{OnRouteFunc: func(c *router.RouteContext) {
		fmt.Println("saw", c.Decision.Model)
		if c.Decision.Model == router.English {
			c.Decision = router.Decision{Model: router.Multilingual, Repo: c.Decision.Repo, Reason: "pinned by policy"}
		}
	}})
	d, _ := r.Route("Please refund the duplicate charge from March.", schema.Questions{})
	fmt.Println(d.Model, "|", d.Reason)
	// Output:
	// saw english
	// multilingual | pinned by policy
}
