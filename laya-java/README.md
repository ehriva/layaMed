# laya-java

JVM inference for Laya: `Agent.predict` / `Agent.predictBatch` over the BPE tokenizer, the sequence
builder, ONNX Runtime and typed answers. JDK 17+. One dependency — `com.microsoft.onnxruntime`.

Laya is a non-autoregressive "System 1" decision model: a bidirectional encoder plus a typed head
that answers `choice` / `score` / `noul` questions about a state in **one forward pass**, with
calibrated confidences. Every question about the same state becomes one row of a single batch, so
four questions about a document cost one batched encode rather than four round trips.

## Status

Implemented: tokenizer, sequence builder, config, ONNX inference (fused or split graph), answer
decoding, `predict`, `predictBatch`, usage and truncation reporting, script and language detection
(`lang.LanguageDetection`), the question presets (`Presets`), the checkpoint `Router` with its
load-and-evict lifecycle, the embedding `Shortlist` with its LRU cache, the email cleaner and
state builder (`LayaEmail`), the abstention gate (`ConfidenceGate`), scanning a state longer than
the context window (`predictLong`), schema-driven decisions (`Decisions.decide` / `decideBatch`),
and the prediction hooks (`hooks`).

Not implemented yet: the `laya-java-client` HTTP module, Android. `predictLong` dispatches no hook:
a scan is sized before a hook could rewrite what is asked, so it stays unhooked until the
reference's start probe and budget check are ported — see
[Watching and shaping a call](#watching-and-shaping-a-call) and [Router hooks](#router-hooks). **Not published to
Maven Central** — see [Installing](#installing).

## Where to go next

| | |
|---|---|
| [MODELS.md](MODELS.md) | how to get the checkpoint and the graph `Agent.open` takes |
| [samples/](samples/) | three runnable programs: quickstart, routing and shortlisting, benchmark |
| [CHANGELOG.md](CHANGELOG.md) | what is in 0.1.0, and where Java is not Python |

## Export a graph (once per checkpoint)

```bash
python scripts/export_onnx.py --model <ckpt-dir-or-hub-id> --output ./model/laya.onnx
```

That writes the fused graph. The split form (`encoder.onnx` + `head.onnx`, from
`laya-ts/scripts/export_onnx.py`) is also supported; `LayaSession` opens whichever a directory
holds, preferring the fused one.

A checkpoint directory must contain `rl_agent_config.json` and `tokenizer/` — that is where the
budgets, the fitted temperatures and the special tokens live, and none of them may be defaulted.
The two shipped checkpoints disagree on every one: english is `max_len 512 / head_max_len 192` with
fitted per-bucket temperatures, multilingual is `1024 / 256` with none, and multilingual reuses
`<bos>` as its classification token and `<eos>` as its separator.

## Ask questions

```java
import com.convaiinnovations.laya.*;
import java.nio.file.Path;
import java.util.*;

// Options are POSITIONAL, so use a LinkedHashMap: two orders are two different questions.
// A HashMap or Map.of is refused rather than silently reordered -- see Option order below.
Map<String, Object> criteria = new LinkedHashMap<>();
criteria.put("refund", "money back for a duplicate charge");
criteria.put("escalate", "pass it to a human");
criteria.put("ignore", "no action needed");

Map<String, Question> questions = new LinkedHashMap<>();
questions.put("intent", Question.choice("What does the customer want?", criteria));
questions.put("urgent", Question.noul("This needs a human today."));
questions.put("severity", Question.score("Rate the severity.",
        List.of("none", "minor", "major", "critical")));

try (Agent agent = Agent.open(Path.of("./checkpoint"), Path.of("./model"))) {
    Prediction p = agent.predict("We were billed twice for March and want a refund today.",
            questions, "en");   // the language tag may select a temperature override

    Answer.Choice intent = (Answer.Choice) p.answer("intent");
    System.out.println(intent.choice());                 // "refund"
    System.out.println(intent.probabilities());          // {refund=..., escalate=..., ignore=...}
    System.out.println(intent.answerConfidence());       // max(p), the calibrated one

    Answer.Noul urgent = (Answer.Noul) p.answer("urgent");
    System.out.println(urgent.noul());                   // P(the statement holds)

    Answer.Score severity = (Answer.Score) p.answer("severity");
    System.out.println(severity.score());                // expected value over level indices
    System.out.println(severity.legend());               // {0=none, 1=minor, 2=major, 3=critical}
}
```

`Answer` is sealed: the three shapes are a closed set, so a reader knows there is no fourth, and an
`instanceof` pattern needs no `else` that throws.

```java
Answer a = p.answer("intent");
if (a instanceof Answer.Choice c) {
    handle(c.choice(), c.probabilities());
} else if (a instanceof Answer.Score sc) {
    handle(sc.score(), sc.legend());
} else if (a instanceof Answer.Noul n) {
    handle(n.noul());
}
```

On **JDK 21 or newer** a pattern `switch` over it is exhaustive and a missing case is a compile
error. That is a language feature of 21, not of this library — on JDK 17, which this module
targets, pattern switches are a preview feature and the `instanceof` form above is the portable
one.

### Two confidences, deliberately

`answerConfidence()` is the probability mass on the reported answer — `max(p)` — and it is the
quantity temperature scaling fits and every calibration figure in laya is computed on.
`confidence()` means something different per type: normalised entropy for a choice or a score, and
`max(p, 1-p)` for a noul. **They are on different scales and must not be compared against the same
threshold.**

The calibration guarantee on `answerConfidence` is conditional, and the condition is not met by
default: it holds only after the temperatures have been fitted and validated on held-out data for
that checkpoint and option count. The shipped checkpoints are over-confident.

## Usage, and what was silently dropped

A state longer than the room left after the question's head is truncated **without an error**, and a
caller cannot reconstruct that from outside: the budget is in tokens, not characters, and the room
left moves per checkpoint and per question.

```java
Usage u = p.usage();
u.inputTokens();            // non-padding tokens fed to the encoder, across every row
u.stateTokens();            // what the state serialised to, before any budget was applied
u.stateTokensDropped();     // the worst case across this state's questions
u.truncated();              // whether any question dropped state tokens
u.truncatedQuestions();     // which ones
u.collapsedOptions();       // questions whose options no longer have a token span each
```

`collapsedOptions()` is the one a reviewer of your prompts will care about: two options whose
rendered text is identical for the first 48 tokens collapse to the same span, the model cannot tell
them apart, and **nothing else downstream can see it** because the marker count still matches. It
reports the count the question *defines*, not the number of markers that reached the sequence.

A question whose option markers do not all fit `max_len` is **refused**, not answered partially —
answering would return a distribution over whichever options survived, with the rest absent from the
answer and unchoosable.

## Many states, one call

```java
List<Object> states = List.of(emailA, emailB, emailC);
List<Prediction> out = agent.predictBatch(states, questions);        // all in one graph call
List<Prediction> out2 = agent.predictBatch(states, questions, "en",
        /* batchSize */ 16, /* sortByLength */ true);                // bounded memory, less padding
```

Results come back in the caller's order whatever the grouping was. `batchSize` bounds peak memory
and `sortByLength` cuts padding; neither changes an answer.

## Routing between checkpoints

The English checkpoint does not degrade gently off English, it collapses — 0.100 on 20-option
MASSIVE Hindi intent against 0.050 for random guessing, and it reports high confidence while doing
so. So a `Router` picks the checkpoint before the question is asked.

```java
import com.convaiinnovations.laya.Router;
import java.nio.file.Path;

try (Router router = Router.builder()
        .checkpointsRoot(Path.of("./checkpoints"))   // ./checkpoints/english, /multilingual, ...
        .maxLoaded(2)                                // a memory ceiling, not a cache hint
        .build()) {

    Router.RouteDecision decided = router.route(state);
    System.out.println(decided.model() + ": " + decided.reason());
    // multilingual: non-Latin script (devanagari, 100% of letters); the English checkpoint
    // cannot read it

    Prediction answer = router.predict(state, questions);   // routes, loads, answers
}
```

`route` runs no model and reads no disk, so it is safe on every request. Precedence, highest
first: an explicit model, an explicit task, a detected typed-decisions workflow (opt-in only), an
explicit language, a caller's hint, the built-in detection, then the configured default.

`maxLoaded` evicts the least recently used checkpoint, and `Router.Lease` keeps one open for as
long as you hold it — `predict` leases internally, so the ordinary path needs no thought. An agent
handed in with `attach` is never closed: the caller keeps ownership.

The default is multilingual, as in the reference: an unidentified Latin-script state is no
evidence of English. A deployment whose traffic is known to be English can set
`defaultCheckpoint(Checkpoint.ENGLISH)`. The *alias* `"default"` is a different thing and still names
english, as the reference's alias table does: `load("default")` loads english whatever
`defaultCheckpoint` is set to.

`predictBatch` routes many requests and answers them with one load per checkpoint used:

```java
List<Prediction> out = router.predictBatch(states, questions);     // each state routed alone
List<Prediction> mixed = router.predictBatch(List.of(
        Router.Request.of(email, Presets.email()),
        Router.Request.of(ticket, Presets.triage())
                .options(Router.RouteOptions.none().model("multilingual"))
                .maxLen(512)));
```

Requests are routed first, grouped by checkpoint in order of first appearance, and within a
checkpoint those sharing a question schema and token budget share `Agent.predictBatch` calls.
Results come back in request order and match per-request `predict` within 1e-4 (padding in a
shared forward pass moves the last bits). `Router` is a
`BatchPredictor`, so `Decisions.decideBatch(router, states, schema)` works; to pin it, pass
`(s, q) -> router.predictBatch(s, q, options)` as the runner.

## Detection on its own

`lang.LanguageDetection` is the router's evidence, usable without one:

```java
import com.convaiinnovations.laya.lang.LanguageDetection;

LanguageDetection.Analysis seen = LanguageDetection.analyse(state);
seen.script();              // "latin", "han", "devanagari", ... or "unknown"
seen.language();            // best effort, null when undecided
seen.english();             // whether the English checkpoint can read it
seen.mixedSegment();        // the line or field that made a mostly-English state non-English
```

Undecided is not English: when nothing identifies the language, non-English letters or a shared
Swedish–Danish marker still prefer the multilingual checkpoint.

## Presets

Five ready-made question sets, word for word the reference's:

```java
import com.convaiinnovations.laya.Presets;

Map<String, Question> questions = Presets.triage();        // or email(), guard(),
                                                           // moderation(), router()
Presets.stateField(questions);                             // "message" — the key it reads
```

Each call returns a fresh mutable map, so dropping a question you do not want is safe. A preset
names the state key it reads in backticks, and `stateField` reads it back out — so a caller can
place its text under the right key instead of guessing.

## Shortlisting a large label set

Choice options share one `head_max_len`, so a seventeen-label set leaves few tokens per label.
`Shortlist` embeds the state and each option, keeps the top `k`, and runs one prediction over the
reduced set — no second decision pass.

```java
import com.convaiinnovations.laya.Shortlist;

Shortlist.Embedder embedder = texts -> myBiEncoder.embed(texts);   // your vectors
Shortlist.Shortlisted out = Shortlist.predict(agent, state, questions,
        Shortlist.cached(embedder), 20);

out.prediction();                    // the answer, over the kept labels only
out.shortlist().get("intent");       // which labels survived, and their scores
```

`Shortlist.cached(...)` embeds each text once under an LRU bound, so a fixed label list is
embedded on the first call and only the new query thereafter. `k` at or above the label count is a
passthrough: the labels come back in order and **the embedder is never called**.

Both `Agent` and `Router` implement `Predictor`, so shortlisting works identically against a fixed
checkpoint or a routed one.

Without an embedder, `Shortlist.predictTournament` lets the model narrow the set itself: labels are
answered in near-equal groups of at most 16 (one prediction per round covers every group of every
question), and the winners meet in one final prediction.

```java
Shortlist.Tournament out = Shortlist.predictTournament(agent, state, questions);

out.prediction();                    // the final call's answers and usage, over the finalists
out.tournament().get("intent");      // the finalists, the label count and the rounds played
```

## Cleaning an email

`laya.email`'s cleaner and state builder. The markers cover English, Portuguese, Spanish and
French mail clients, because the router already sends the last three to the multilingual
checkpoint and an English-only cleaner left their quoted history — often a *different* request —
weighing on the answer as much as the new message.

```java
import com.convaiinnovations.laya.LayaEmail;
import java.util.Map;

String body = """
        I was charged twice for order 8812. Please refund one of them.

        Atenciosamente,
        Ana Souza
        Enviado do meu iPhone

        Esta mensagem e confidencial e de uso exclusivo do destinatario.

        Em ter., 3 de set. de 2025, Suporte <suporte@x.com> escreveu:
        > Podemos ajudar?
        """;

// quoted history, sign-off, device footer and disclaimer all go
String clean = LayaEmail.cleanEmailBody(body);

Map<String, Object> state = LayaEmail.emailState("Cobranca duplicada", body, "ana@x.com");
// {subject=Cobranca duplicada, body=I was charged twice..., from=ana@x.com}

Prediction p = agent.predict(state, LayaEmail.emailQuestions());
```

`emailState` takes the same `maxChars` budget and passes it through, and it is worth raising for
a long message: at the default the body stops after 3,000 characters, so a request arriving in
the last paragraphs never reaches the model.

What it deliberately does **not** cut is the interesting half. `From: my side the integration
works, but please refund...` is prose, not a header, so a reply header is recognised only when an
address follows it or its own `Sent:`/`Enviado:` line does. `Thanks for the quick reply.` is not
a sign-off, `Obrigado pelo retorno, mas ...` is a request, and `Is this confidential?` is a
question — a cleaner that is too eager deletes what the sender actually wrote, which is worse
than leaving one boilerplate line behind.

## Lower-level pieces

```java
import com.convaiinnovations.laya.config.AgentConfig;
import com.convaiinnovations.laya.tokenizer.Tokenizer;
import java.nio.file.Path;

Tokenizer tok = Tokenizer.fromModelDirectory(Path.of("./checkpoint"));
int[] ids = tok.encode("charged twice");
int[] capped = tok.encode(longText, 48);          // stops early; same prefix as the full encoding
String text = tok.decode(ids);                    // ids back to text; special tokens dropped
String withSpecials = tok.decode(ids, false);     // ...kept
AgentConfig cfg = AgentConfig.fromModelDirectory(Path.of("./checkpoint"));
```

## Abstaining on a low-confidence answer

```java
var gated = ConfidenceGate.apply(prediction.answers(), 0.8);   // null = no gate at all
gated.ifPresent(report -> report.forEach((id, v) -> {
    if (v.abstention() == ConfidenceGate.Abstention.ABSTAINED) {
        escalate(id, v.answer(), v.threshold());
    }
}));
```

Three states, not a boolean: `UNEVALUATED` is an answer the gate could not read, and calling
that a pass is the same mistake as calling it a flag. An **ungated** call returns an empty
`Optional`, so the presence of a report is what tells you a gate ran.

A threshold is not a claim that the number is calibrated — "about c of the answers returned at
c are correct" holds only after temperatures have been fitted and validated for that checkpoint
and question shape. Pass a `Map` instead of a scalar to gate each option-count bucket
(`choice:2`, `score:6-10`, …) at the level its calibration actually earns.

## Deciding against a JSON schema

```java
import com.convaiinnovations.laya.json.Json;

Map<String, Object> schema = (Map<String, Object>) Json.parse("""
    {"type": "object", "properties": {
       "department":  {"enum": ["billing", "support", "sales"]},
       "urgency":     {"type": "integer", "minimum": 1, "maximum": 5},
       "needs_human": {"type": "boolean"}}}""");

Decisions.Decision d = Decisions.decide(agent, ticket, schema, null, 0.8);
Map<String, Object> values = d.values();      // {"department": "billing", "urgency": 4, ...}
```

An `enum` or a `const` becomes a `choice`, a `boolean` becomes a `noul`, a bounded `integer` or
`number` becomes a `score`, and the answers come back as **the schema's own values**: the choice's
value and not the label it was shown under, so `enum: [10, 20, 30]` decides to the integer `20`.
`anyOf` / `oneOf` with one non-null branch is unwrapped (pydantic's `Optional`), a one-item
`allOf` is unwrapped with the outer keys on top, and a local `$ref` is inlined.

Anything that cannot be answered from a fixed option set — a free string, an array, a nested
object, a union of two real types — is refused with a `SchemaException` **naming the path**, which
is the only part of the message a caller holding a 32-property schema can act on.

`minConfidence` wires straight into `ConfidenceGate`: a field whose answer falls below the bar
comes back as `null`, and a field that was never answered is **absent** rather than null, because
"not believed" and "not asked" are different outcomes. The gate's full report is on
`d.gate()`. `Decisions.decideBatch(agent, states, schema)` does the same for many states in one
batched call, in input order.

`Decisions.questions(schema)` and `Decisions.answersToJson(answers, schema)` are the two halves on
their own, for a caller that runs the model itself.

There is no pydantic on the JVM, so only the JSON-schema path is ported; mapping Java records
would be a new design rather than a port.

## A state longer than the context window

```java
LongPrediction scan = agent.predictLong(wholeDocument, questions);
Answer intent = scan.answer("intent");
scan.window("intent").ifPresent(w ->
        System.out.printf("decided by window %d of %d, tokens [%d,%d)%n",
                w.index(), w.count(), w.tokenStart(), w.tokenEnd()));
```

`predict` truncates a state that exceeds `max_len` to a single window and drops the rest without
saying so. `predictLong` scans it instead: one tokenization, overlapping windows, shared graph
calls, then one answer per question. A `noul` takes the **highest** P(true) across windows — the
statement holds if any window supports it — while a choice or a score takes the **most confident**
window, which stops a localized signal being out-voted by the neutral text that makes up most of a
long document. Ties go to the earliest window.

The probability on an answer is **the deciding window's**, not a calibrated number for the whole
document, which is why `window(id)` exists: without it the number has no stated scope. The window
is absent when the state fitted one window, because then nothing was decided between windows.

`decode` follows the reference rather than tidying after it, and on the multilingual checkpoint
that means it is **lossy**: `Metaspace` prepends its marker, so `decode(encode("Hello world"))` is
`" Hello world"`. The English checkpoint happens to round-trip. Correcting the space would make
every window of a long document tokenize differently from the reference, so the loss is kept.

`Agent.using(tokenizer, config, session)` assembles an agent from parts — for a caller that already
holds them, or to drive the batching and usage accounting through a stub
`infer.InferenceSession` instead of a 1.2 GB graph.

## Watching and shaping a call

```java
agent.hooks().addHook(new Hook() {
    @Override public void onPredictEnd(PredictContext ctx) {
        metrics.record(ctx.model(), ctx.usage().inputTokens(), ctx.elapsedMs());
    }
});

// Per call: answer from a cache without the model running at all.
agent.predictBatch(states, questions, null, 0, false,
        HookCall.of(Hooks.onPredictStart(ctx -> cache.lookup(ctx.states())
                .ifPresent(ctx::skip))));
```

A hook is the one place your code runs **inside** a prediction. Implement any of the six methods
on `Hook` — the rest default to doing nothing — and install it on the agent, on a single call, or
process-wide with `Hooks.setDefaultHooks`.

`PredictContext` is the call, and it is mutable on purpose. A start hook may rewrite
`ctx.states(...)` or `ctx.questions(...)`, move the token budget with `ctx.maxLen(...)` /
`ctx.headMaxLen(...)`, or answer outright with `ctx.skip(results)` — which skips inference while
still running the end hooks. An end hook may replace `ctx.results(...)`, and sees `ctx.usage()`
totalled over the call plus `ctx.elapsedMs()`. Every hook of one call gets the **same** context,
so `ctx.runId()` pairs a start with its end without state of your own.

The order is a contract: **process-wide defaults, then installed, then per-call**, and within a
per-call `HookCall`, hook objects before the `onStart`/`onEnd` callbacks. A tracer installed to
watch what a per-call hook did only sees it if it runs after it.

`agent.hooks()` also carries the policy. `raiseErrors(false)` reports a throwing hook and carries
on, which is what a telemetry hook needs — it must not be able to fail a request. An `Error` is
rethrown whatever it says, which is the reference's rule in Java terms: it catches `Exception` and
deliberately not `BaseException`. `concurrent(false)` serialises hooks that are not re-entrant.
A `HookCall` can override `raiseErrors` and `timeout` for one call without touching the others.

`timeout(Duration)` bounds **the wait** for each hook call — not the hook, and not the request. An
overrunning hook fails the call and **keeps running**, since neither runtime can interrupt a thread
that will not cooperate, so a hook that blocks forever leaks a daemon thread per call. That much is
the reference's behaviour.

Where this port diverges, deliberately: the abandoned thread is then **cut off from the call**.
Once its deadline has passed, every `ctx` mutator it calls throws `IllegalStateException` instead
of rewriting a call that moved on without it. The reference has the identical hazard and no cheap
way to close it. Measured here before the guard, over 60 calls whose start hook overran a 50 ms
deadline by 30 ms and then assigned results: 60 of 60 late writes were accepted, and `ctx.usage`
ended up describing a different answer from `ctx.results` in 60 of 60 — in some runs the *caller*
got the abandoned hook's answer. With the guard: 0 of 60. It is a narrowing and not a proof — a
write already past its check when the deadline expires still lands, which happened in 2 to 17 of
60 once the overrun was cut to 1–5 ms, so the window is a few instructions wide rather than the
whole remainder of the call.

Both lines dispatch reports — the swallowed failure and the overrun — name the hook as
`<class>.<event>` and spell a deadline in seconds exactly as the reference's `%g` does, so
`exceeded 1s` and not `exceeded 1.0s`, and one grep works against either runtime's logs.
`Hooks.onPredictStart`/`onPredictEnd` return the reference's own `_StartAdapter`/`_EndAdapter`
classes for the same reason: the name is what appears in that line, and an anonymous Java class
has none. `Hooks.hookName(hook)` and `Hooks.seconds(duration)` are public, so your own
`onFailure` sink can produce the same text.

Two smaller divergences, both for the same reason — a number or a name that differed between the
runtimes where the port claims they do not. `Hooks.Totals` counts in `long`, because summing
per-state `int` usage into an `int` wrapped silently (three results of a billion input tokens
totalled `-1294967296`), and a Python `int` cannot. And a `PredictContext` refuses a null `states`
or `questions` by name rather than letting the copy throw a bare `NullPointerException`; before
hooks were wired in, `predictBatch(List.of(), null)` returned an empty list, which the reference
does not do either.

Scope hooks to a block with `try (var scope = agent.hooks().hooksInstalled(tracer)) { ... }`. It
removes one copy of each hook it added — not every copy by identity, which would take one the
application had installed before the block, and not a snapshot, which would undo an overlapping
block and discard anything added inside this one.

`predictLong` runs **no** hooks — not per-call, not installed, not process-wide. A scan is sized
before any hook could run, and the reference only lets hooks into `predict_long` by way of a start
probe, a post-chain budget check and two separate "a hook answered the document" paths. None of
those are ported, and without them a hook that adds one option silently re-truncates every window.
Window the state yourself and hand the windows to `predictBatch` if you need hooks over a long
document.

There is no `AsyncHook`. It exists in the reference to finish a coroutine from synchronous code,
and a JVM method call is already synchronous: a hook that wants asynchronous work composes it and
blocks on it — `ship(ctx.results()).toCompletableFuture().join()` — bounded however your runtime
wants, or by `timeout(Duration)`.

## Router hooks

`Router.hooks()` is its own registry, separate from any agent's. A router-level hook sees three
events no agent can:

| event | when |
|---|---|
| `onRoute` | after a checkpoint is chosen, from every branch of `route` |
| `onLoad` | after a checkpoint is built by `load`, `lease`, `predict` or `preload`, or attached while that build ran — not on a cache hit |
| `onEvict` | after one is dropped, by `maxLoaded`, `unload` or `unloadAll` — before the `onLoad` that caused it, as in the reference |

`Router.predict` also dispatches one `onPredictStart`/`onPredictEnd` pair for the answer. As in the
reference, routing and loading run first, so the order is `onRoute`, any `onEvict`/`onLoad`, then
the pair, and on success `elapsedMs` does not include a cold load. A routing or loading failure
reaches `onError` and `onPredictEnd`, with no start event, and its `elapsedMs` covers the routing
and the failed load. The agent's own pair still
fires for the forward pass, so a router-level hook and an agent-level hook see different spans.
Process-wide defaults (`Hooks.setDefaultHooks`) see only the router's pair: the forward pass runs
without them, as the reference runs it under `_SKIP_DEFAULTS`, so they fire once per request on
`predict` and `predictBatch` alike.

`Router.predictBatch` dispatches one pair **per request**, as `predict` does, so a start hook can
rewrite or `skip` a request before it joins a shared forward pass. The order is the reference's:
every `onRoute`, then per checkpoint its `onEvict`/`onLoad`, its requests' starts in order and
their ends in reverse. If a checkpoint's inference fails, each of its started requests gets
`onError` and its end before the failure propagates. A routing failure reaches that request, and a
loading failure every request of that checkpoint, as `onError` and `onPredictEnd` with no start —
as on `predict`, where the reference's `predict_batch` reports neither. The agent's installed hooks fire once
per forward pass, not per request. Unlike `predict`, the batch path honours `ctx.maxLen` / `ctx.headMaxLen`
from a router-level start hook, since a request's own budget is seeded there.

Hooks are dispatched outside the router's lock, so a hook may call back into the router —
`loaded()`, `load`, `unload` — without deadlocking. Two consequences of that are yours to handle:
`load()` can hand back an agent a hook unloaded while its `on_load` was being dispatched, and a
router-level `on_predict_start` that calls `router.predict` recurses until the stack runs out,
because that inner call dispatches the same hook again.

`ctx.maxLen` / `ctx.headMaxLen` set by a router-level start hook are not honoured by
`Router.predict`: `Agent.predict` takes no budget arguments. `predictBatch` honours them, and an
agent-level hook can set them on either path.

```java
router.hooks().addHook(new Hook() {
    @Override public void onRoute(PredictContext ctx) { log.info("routed to {}", ctx.model()); }
    @Override public void onEvict(PredictContext ctx) { meter.increment("evictions"); }
});
```

### What a router hook cannot do

| | |
|---|---|
| a throwing `on_evict` hook fails `Router.close()` | and the remaining evictions are not dispatched. In try-with-resources that masks the body's own failure |
| calling back into the router from `on_evict` **during `close()`** | throws `this Router is closed`: `close()` marks the router closed before unloading |
| `concurrent(false)` with an `AsyncHook` whose callback dispatches a hook on the same registry | deadlocks with no deadline, and stalls until it with one. Serial dispatch holds a lock across the callback, and the callback is on another thread, so the inner dispatch cannot re-enter that lock. A callback that dispatches nothing — `loaded()`, a cached `load` — is fine |
| `ctx.maxLen` / `ctx.headMaxLen`, and states past the first | ignored by `Router.predict`, which calls `Agent.predict` with no budget arguments; `predictBatch` honours the budgets. Both answer only the first state |

`on_route` carries its own context, not the predict pair's, so `runId` differs between them. Correlate
on the model name, not on `runId`. That context holds the chosen checkpoint in `ctx.model()` and no
states, questions or decision: unlike the reference, a hook here observes the route but cannot
replace it.

`AsyncHook.of(hook, executor)` runs a hook on a particular thread — a framework request scope, an
actor, a UI loop. It waits for the callback, so an exception still reaches the hook policy and the
mutable `PredictContext` is never read after the call has moved on. Pass a `Duration` to fail a
callback that overruns.

## Option order

A choice's options are positional: option N renders Nth and logit N is read back as the Nth label.
So the criteria map's iteration order decides both what is asked and how the answer is labelled, and
a map whose type does not define that order changes both silently. Measured on JDK 17:

| written | `LinkedHashMap` | `HashMap` | `Map.of` |
|---|---|---|---|
| `refund, escalate, ignore` | `refund, escalate, ignore` | `ignore, escalate, refund` | varies per JVM |

`HashMap` is exactly reversed here, so `probabilities().get("refund")` would have reported the logit
that belonged to `ignore`. It is at least reproducible, because `String.hashCode` is specified.
`Map.of` is worse: it salts its table per JVM, so three runs of one program gave three different
orders -- the same question answering differently after a restart.

`Question.choice` therefore refuses the unordered maps it can recognise -- `HashMap`,
`Hashtable`, `WeakHashMap`, `IdentityHashMap`, `ConcurrentHashMap` **and their subclasses**, and
`Map.of`/`Map.copyOf`/`Map.ofEntries` with two or more options -- and names the fix. A
`LinkedHashMap`, a `SortedMap`, their subclasses, and any single-option map are accepted; one
option cannot be out of order. `Shortlist.rank`/`choice` apply the same check.

It cannot recognise all of them. `Collections.unmodifiableMap`, `synchronizedMap` and
`checkedMap` are the same wrapper class whichever map they wrap, so a wrapped `HashMap` is
accepted and its order is not the one you wrote; reaching the wrapped map needs `--add-opens`
into `java.base`. Wrap a `LinkedHashMap`, or use `choiceOf`.

When there is no map to hand, build the question from the options directly and the order cannot be
lost at all:

```java
Question q = Question.choiceOf("What does the customer want?",
        Map.entry("refund", "money back for a duplicate charge"),
        Map.entry("escalate", "pass it to a human"),
        Map.entry("ignore", "no action needed"));
```

Two options sharing a label are refused there too: a map would have kept only the last, asking a
question with fewer options than were written.

## Threads

`Agent.open(model, graph)` leaves the thread count to ONNX Runtime, which is what the Python
runtime does. Pass `Agent.open(model, graph, 1)` for a request-per-thread server, where the
parallelism is already in the requests. One `Agent` is **not** safe for concurrent `predict` calls
unless ONNX Runtime is configured for it; hold one per worker, or serialise access.

## Build and test

```bash
cd laya-java
./gradlew build                 # -Xlint:all -Werror
./gradlew test                  # the suite that needs no checkpoint
```

The parity tests need a checkpoint and a graph, and **abort with an actionable message** without
them rather than passing vacuously:

```bash
LAYA_CHECKPOINTS=/path/to/checkpoints \
LAYA_ONNX_GRAPH=/path/to/model/laya.onnx \
  ./gradlew test
```

A blank value counts as absent, so a CI cell that owns no graph can set `LAYA_ONNX_GRAPH=''`
without turning an abort into a failure.

### Testing on another JDK

The artifact is compiled for 17 and the toolchain pins the **compiler** to 17, so installing a
different JDK does not change what the tests run on. `-PtestJavaVersion` moves the test JVM only —
the bytecode stays at release 17:

```bash
./gradlew test -PtestJavaVersion=24    # compiled for 17, executed on 24
```

CI runs the model-free suite on **17, 21 and 24** — three different Unicode versions (13.0, 15.0
and 16.0). The artifact is compiled for 17, so it runs on 17 and anything newer; those three are
the versions the suite is actually asserted against.

This matters more here than in most ports. `\p{L}` and `\p{N}` in `java.util.regex` follow the
JDK's own Unicode version, and `Character.isLetter` disagrees with itself across JDK 17 (Unicode
13.0) and JDK 24 (Unicode 16.0) on 751 of the code points this port has to classify — 0 of 751 on
one, 751 of 751 on the other. Everything Unicode-shaped is therefore compiled in from the
reference, and this flag is how that is checked. `TestJvmVersionTest` asserts the tests really are
running on the JDK that was asked for, because Gradle writes `<properties/>` empty into the JUnit
XML and nothing downstream can tell 17 from 24.

## Parity: generated, not asserted

`laya-java/fixtures/*.json` are generated **from** the Python package and committed. The drift check
is what keeps them honest:

```bash
python laya-java/scripts/gen_fixtures.py            # regenerate
python laya-java/scripts/gen_fixtures.py --check     # fail if a committed fixture would change
```

A family that needs a checkpoint or a graph this run cannot reach is **left alone** and reported as
unverified, so running the generator without the checkpoints cannot replace good expectations with
skip markers. `--strict` turns "could not verify" into an error, for a lane that is supposed to have
them.

Where it does *not* match Python bit for bit, and why: the ONNX logits are float32 and Python
computes the softmax in float32, while this port — like `laya-ts` and `laya-dotnet` — computes it in
`double`. Over 200,000 random logit rows the four-decimal probabilities differ on 88 of them.
Matching float32 exactly would mean reproducing NumPy's own float32 `exp` (which differs from
rounding a double `exp` on 0.65% of values) and its pairwise summation order; neither is a contract
NumPy publishes. Rounding, by contrast, *is* exact: `BigDecimal` at scale 4 with `HALF_EVEN` is
Python's `round(v, 4)` on Python's operand.

## Installing

**Not published yet.** Build from source into your local repository:

```bash
cd laya-java && ./gradlew publishToMavenLocal
```

```kotlin
dependencies { implementation("com.convaiinnovations:laya-java:0.1.0-SNAPSHOT") }
```

Publishing to Maven Central needs the `com.convaiinnovations` namespace verified by the project
owner, a Central portal token and a PGP signing key. `.github/workflows/release-java.yml` is wired
for it and has never been run; signing activates only when a key is present, so `build` and
`publishToMavenLocal` work without any of it.
