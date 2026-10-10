# Changelog

All notable changes to the `com.convaiinnovations:laya-java` artifact are documented here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Releases are tagged `laya-java-v<version>`.

## [0.1.0] - unreleased

First release: a port of the Laya Python SDK to the JVM, with every expectation recorded by
running the Python package rather than written by hand. **Not yet published to Maven Central** —
see [Installing](README.md#installing).

### Added

- `Agent`: `predict` and `predictBatch` over an exported ONNX checkpoint, with every question for
  a state answered in one batched forward pass. Typed answers — `Answer.Choice`, `Answer.Score`,
  `Answer.Noul` — as a sealed interface, so a choice is not a score at the API boundary.
- Both ONNX layouts: the fused `laya.onnx` and the split `encoder.onnx` + `head.onnx`.
- `Usage` and truncation reporting, including which question is blamed for a truncation, and
  `confidence` alongside `answerConfidence`.
- Python-compatible tokenization (BPE and Metaspace, added tokens, normalizers), sequence
  building, collation, answer decoding and calibration.
- `ConfidenceGate`: the opt-in abstention gate. Reports three states — `PASSED`, `ABSTAINED`
  and `UNEVALUATED` — because an answer that carried no usable confidence is not a pass, and a
  boolean cannot say so. Reads `answerConfidence` (what temperature scaling fits) and falls back
  to `confidence` (normalised entropy, a different scale) only when the first is unusable.
  Per-bucket thresholds are supported, since one threshold does not transfer across option
  counts. Unlike the reference it does not mutate the answers — records cannot — so `apply`
  returns a report, and an **ungated** call returns an empty `Optional` rather than a report of
  nulls: the presence of the verdict is how a caller tells "no gate ran" from "everything
  passed".
- `Agent.predictLong` and `LongPrediction`: questions answered over a state longer than the
  context window. `predict` truncates such a state to one window and drops the rest silently;
  this tokenizes it once, scans it in overlapping windows, scores them in shared graph calls and
  aggregates per question — P(true) is the **max** over windows for a `noul`, and a choice or a
  score takes the single **most-confident** window, so a localized signal is not out-voted by the
  neutral text a long document is mostly made of. Ties take the earliest window. Each answer
  carries the `Window` that decided it, because the probability reported is that window's and not
  a calibrated number for the document.
- `sequence.WindowPlan`: how a scan over a state longer than one sequence is sized —
  `stateRoom`, `budget` and `batchCap`, ported from `laya.common`. The window is capped at the
  room the questions leave, and at the *smallest* room when several are asked, because a window
  wider than that is re-truncated on the way in: its tail would reach no model while the reported
  span said it did. `Budget` carries the reference's `RuntimeWarning` text as data as well as
  logging it, since every clamp here is a decision the caller cannot otherwise see — their window
  shrinks, their stride shrinks with it, and the forward-pass count can triple.
- `hooks`: the opt-in lifecycle, ported from `laya.hooks`. `Hook` is one interface with six
  default no-op methods — the reference's structural `Hook` protocol and its concrete `BaseHook`
  are the same thing once an interface can carry defaults. `PredictContext` is the per-call,
  mutable state every hook of one call shares: a start hook may rewrite the states, the
  questions and the token budget, or answer the call outright with `ctx.skip(...)`, in which
  case the model never runs; an end hook may replace the results and sees the call's totalled
  `usage` and `elapsedMs`. `HookRegistry` is the per-agent list — `agent.hooks()` — editable
  while the agent is alive and snapshotted per call, with `hooksInstalled` scoping hooks to a
  block. `HookCall` carries per-call hooks and policy overrides. `Hooks` composes the chain —
  **process-wide defaults, then installed, then per-call**, which is a contract and not an
  implementation detail — dispatches it, bounds each hook with an optional deadline, serialises
  it behind a lock when asked, and totals usage. `Agent.predict` and `Agent.predictBatch` run
  it; `Agent.predictLong` deliberately does not, for the reason under *Known limitations*.
  Everything above is recorded from the reference in `fixtures/hooks.json`, including what a
  start hook sees, what `skip` does to inference, what a throwing hook does under each policy,
  and how a deadline is spelled in the line an overrun reports.
- Three **deliberate divergences** inside `hooks`, each because a number or a name differed
  between the runtimes where the port claims they do not:
  - A hook that outran `timeout(Duration)` is **cut off from the call**. The deadline bounds the
    WAIT, not the hook and not the request — neither runtime can interrupt a thread that will not
    cooperate — so the abandoned thread is still holding the live, mutable `PredictContext` of a
    call that moved on without it. Measured over 60 calls whose start hook overran a 50 ms
    deadline by 30 ms and then assigned results: 60 of 60 late writes were accepted, and
    `ctx.usage` ended up describing a different answer from `ctx.results` in 60 of 60; in some
    runs the caller itself got the abandoned hook's answer. Every `PredictContext` mutator now
    throws `IllegalStateException` for a thread whose deadline has passed, and the same 60 calls
    disagree 0 times. The reference has the identical hazard and no cheap way to close it. It is
    a narrowing, not a proof: with the overrun cut to 1–5 ms, so that the write and the deadline
    collide, 2 to 17 of 60 late writes still landed. The context's mutable fields are `volatile`
    for the same reason — after `join` expires there is no happens-before edge to the caller.
  - `Hooks.Totals` counts in `long`. Summing per-state `int` usage into an `int` wraps silently —
    three results of 1,000,000,000 input tokens each totalled `-1294967296` — and a Python `int`
    cannot, so the port would have been the only one of the two lying to a metering hook about
    what a call cost.
  - `PredictContext` refuses a null `states` or `questions` **by name**, rather than letting its
    own defensive copy throw a bare `NullPointerException`. Before the hooks were wired in,
    `predictBatch(List.of(), null)` returned an empty list, because the empty-state
    short-circuit ran before anything looked at the questions; the reference refuses a non-dict
    `questions` for an empty batch as much as for any other, so that empty list was this port's
    divergence rather than a ported behaviour.
- The two lines dispatch reports are **greppable across both runtimes**, which they were not.
  A deadline is formatted as CPython's `%g` formats it, so a one-second deadline reads
  `exceeded 1s` and not `exceeded 1.0s` — a whole number of seconds being the common case for a
  deadline — with the rendering recorded per value in `fixtures/hooks.json` rather than typed
  out. And `Hooks.onPredictStart` / `onPredictEnd` return NAMED classes (`_StartAdapter`,
  `_EndAdapter`, the reference's own names), because `getSimpleName()` of the anonymous classes
  they used to return is the empty string: the two advertised low-friction hooks reported
  `laya: hook .on_predict_start failed`, naming nothing, in the one line a swallowed telemetry
  failure leaves. `Hooks.hookName` and `Hooks.seconds` are public so a caller's own `onFailure`
  sink can produce the same text.
- `Agent.using(tokenizer, config, session, modelName)` and `Agent.modelName()`: the checkpoint
  name a hook reads off `PredictContext.model()`. `Agent.open` takes it from the model
  directory's own name, since `rl_agent_config.json` does not carry one.
- `Tokenizer.decode(int[])` and `decode(int[], boolean skipSpecialTokens)`: ids back to text,
  matching the `tokenizers` crate's `ByteLevel`, `Replace`, `ByteFallback`, `Fuse`, `Strip` and
  `Sequence` decoders. Byte-exact against the reference over both shipped checkpoints, including
  201,279 window slices — the shape that cuts a multi-byte character in half, where a decoder that
  agrees on every whole text can still be wrong. Decoding on the multilingual checkpoint is lossy
  by the reference's own behaviour: `Metaspace` prepends its marker and the round trip gains a
  leading space, which this port reproduces rather than corrects.
- `lang.LanguageDetection`: script and language detection, the eight public fields and the
  fourteen intermediate steps the reference exposes.
- `Presets`: the five preset question sets, compared against the reference word for word.
- `Router`: checkpoint selection with its reason strings, plus a load-and-evict lifecycle with
  leases, so a checkpoint in use is never closed under a caller.
- `Shortlist`: cosine ranking over a caller's embedder, with an LRU cache and numpy's tie order.
- `Shortlist.predictTournament`: elimination rounds over a large choice, a port of
  `predict_tournament`, with no embedder.
- `LayaEmail`: `cleanEmailBody` and `emailState`, with the English, Portuguese, Spanish and French
  marker sets, and `emailQuestions` re-exported from `Presets`.
- `Decisions`: schema-driven decisions — `decide` and `decideBatch` over a JSON schema. An
  `enum` or `const` becomes a `choice`, a `boolean` a `noul`, a bounded `integer`/`number` a
  `score`; `anyOf`/`oneOf` with one non-null branch is unwrapped (pydantic's `Optional`), a
  one-item `allOf` is unwrapped with the outer keys on top, and a local `$ref` is inlined. The
  answers come back as the schema's **values** — the choice's value, not the label it was shown
  under — a score as `minimum + argmax(probabilities)`, and a `noul` as `>= 0.5`, so exactly
  `0.5` is true. A field below `minConfidence` is `null` and a field that was never answered is
  **absent**, because "not believed" and "not asked" are different outcomes. Anything that
  cannot be answered from a fixed option set is refused with a `SchemaException` naming the
  path. Unlike the reference, the gate is not re-implemented: `decide` reads
  `ConfidenceGate`'s report, so the value a caller acts on and the state the gate reports cannot
  drift apart. Only the JSON-schema path is ported — there is no pydantic on the JVM.
- `BatchPredictor`: the batching half of `Predictor`, implemented by `Agent`. The reference
  raises `TypeError` when a runner has no `predict_batch`; here a runner that cannot batch does
  not compile, because looping `decide` N times behind the caller's back is the one thing
  neither wants to do silently.
- `Predictor`: one interface satisfied unchanged by both `Agent` and `Router`.
- `json.PythonJson`: CPython's `repr` and `%.0f`, which the router's reason strings are built from.
- `scripts/prepare_checkpoint.py`: a pinned download and export, writing the two directories
  `Agent.open` takes. See [MODELS.md](MODELS.md).

### Testing

- 1,328 tests **run** with both shipped checkpoints present (1,334 registered,
  6 aborted), and 917 **run** with no checkpoints at all
  (934 registered, 17 aborted). Both counts are tests that actually
  executed, not tests that were registered: a `@TestFactory` that aborts by assumption still
  appears in the report, so the two kinds of number are not comparable and mixing them overstates
  whichever one is the smaller. Measured on JDK 17; the aborted ones need a checkpoint, and the
  end-to-end golden additionally needs a traced graph.
- `typed-decisions` is covered end to end by `TypedDecisionsParityTest` against
  `fixtures/typed_decisions.json`: **14 tests** with the checkpoint present — nine single-state
  cases spanning all four fine-tuned workflow signatures, three batch configurations, and two
  checks on the weights themselves. Four of the 14 register without a checkpoint and one of those
  passes there, since the workflow-signature check reads only the fixture.
- That second pair of checks is not ceremony, and it was not load-bearing either until now.
  `typed-decisions` is fine-tuned from `english` and ships its `tokenizer.json` byte for byte and
  the same encoder, so opening the english checkpoint against the typed graph raises nothing and
  answers. Measured with the english checkpoint symlinked in as `typed-decisions/`: **11 of the
  13 tests passed** under that pairing — every single-state case but `over-budget`, and all three
  batch configurations, so the batch factory contributed nothing to the defence. The config check
  did NOT run "before any probability is compared": JUnit 5 orders nothing by default and the
  class shares one agent, so it ran ninth of thirteen, after all eight probability comparisons.
  It is asserted on the shared agent as it is opened now, so a mispairing fails every case that
  touches the weights and both factories fail to register at all — measured, the same pairing now
  yields 4 tests, 3 failed, and the cell's exact `min-tests` floor fails on the count as well.
  `@Order` puts the named check first in the report to match.
- What actually separates the two checkpoints is narrower than that section used to imply. Their
  `temperature_by_options` tables are **identical** — every key, every digit, `choice:11+` at
  0.10058280825614929 included — so the bucket keyset assertion, the per-bucket loop, the raw
  value and the `TEMP_MIN` clamp all pass unchanged under the wrong checkpoint. Only `max_len`
  (1024 against 512), `head_max_len` (256 against 192) and the fitted `temperature` triple
  discriminate. The triple never reached a forward pass, because every recorded question landed
  in a bucket both files ship, so `unbucketed-score` was added: a six-level score question, which
  `temp_bucket` maps to `score:6-10`, a bucket neither checkpoint ships. The lookup misses and the
  fitted `temperature[QTYPE_SCORE]` is what scales it — 1.0374 here against 1.2514 on english.
- The golden records the `.laya-revision` stamp the graph was exported under, so a `HF_REVISION`
  bump cannot re-record this family in silence: `gen_fixtures.py --check` is what notices.
  `predict.json` still records none — regenerating it needs the multilingual graph.
- Every expectation is **generated from the Python package**, never written by hand:
  `laya-java/fixtures/*.json`, regenerated by `scripts/gen_fixtures.py` and gated by `--check` in
  CI so a change to `laya/` that moves a recorded contract shows up here rather than as drift.
- The compiled-in Unicode tables are checked by a **sha256 over all 1,114,112 code points** per
  property, so they are proven exhaustively rather than on the cases someone thought to write.
- Seven CI lanes: build plus model-free tests, the same tests on JDK 17, 21 and 24 — three
  different Unicode versions, 13.0, 15.0 and 16.0 — and three parity cells split by whether they
  need a traced graph, and then by which checkpoint that graph was traced from. A graph is
  traced from exactly one checkpoint and carries its weights, so the cells cannot share one.
- `-PtestJavaVersion` runs the tests on a different JDK from the one the classes are compiled
  for, because the toolchain pins the compiler to 17 and a lane that merely installs another JDK
  tests the same thing twice.
- Three guards against a suite that is green because it tested nothing. A fixture section that
  emptied used to produce a `@TestFactory` with no tests in it, which JUnit does not fail, so
  `HooksTest.section` now refuses an empty or missing section and names it — the count floor
  alone was not enough, because with the smallest section emptied the `build` lane ran 793
  against a floor of 794 and caught it by exactly one test while the `jdk-matrix` lane, one test
  higher, landed exactly ON 794 and exited 0. The two lanes now carry floors one apart for that
  reason. A recorded value with no Java counterpart used to fall into the branch that asserts
  only about the reference, so a representable addition stopped testing Java silently; the
  timeout factory now fails on a value it has been told nothing about. And the process-wide
  default hooks are cleared after every test in the module by an auto-registered
  `DefaultHooksIsolation` extension, rather than by two classes remembering to — pinned by a
  class that deliberately does not remember.
- That matrix earned itself on its first run. The JDK 21 cell failed where 17 and 24 passed,
  because `Character.isLetter` on Unicode 15.0 agrees with CPython 15.0 EXACTLY, and a test had
  asserted that the JDK disagrees. Catching up is not a reason to drop the table -- it is the
  reason the table exists, since the agreement is two version numbers lining up by accident and
  un-happens on the next release in either direction. The assertion is now the version-independent
  one, and whether a given JDK agrees is recorded rather than required.

### Known limitations

- Not implemented: the `laya-java-client` HTTP module, Android.
- `AsyncHook` is not ported, and will not be. It exists in the reference to finish a coroutine
  from synchronous code; a JVM method call is already synchronous, so a hook that wants
  asynchronous work composes it and blocks on it in one line. Porting the wrapper would mean
  choosing a future type for every caller and owning a thread pool to await on.
- Hooks run on `predict` and `predictBatch` only. `Router` does not dispatch them — so
  `PredictContext` has no `router`/`decision` pair, rather than two fields that are always null
  — and `predictLong` deliberately does not either. The reference's `predict_long` survives
  hooks only by way of a start probe, a post-chain budget check and two separate "a hook
  answered the document" paths, none of which this port has; without them a hook that adds one
  option silently re-truncates every window, and a hook that answers the call leaves `windows`
  claiming a scan that never ran. Running hooks there without that machinery would be worse
  than not running them.
- `Hooks.withoutDefaultHooks` is a `ThreadLocal` where the reference's `_SKIP_DEFAULTS` is a
  `contextvars.ContextVar`. A `ContextVar` is copied into an asyncio task; a `ThreadLocal` is
  not inherited by a thread started inside the scope. Nothing here dispatches hooks off the
  calling thread, so there is nowhere for it to bite today.
- A timed-out hook keeps running, on both runtimes: neither Java nor Python can interrupt a
  thread that will not cooperate, so the deadline bounds the request and not the process.
- `typed-decisions` is routable — `Router` knows it and resolves its aliases — but has no recorded
  end-to-end fixtures. The routing decision is tested; a forward pass against that checkpoint is
  not.
- Only `invoice_processing`'s trained question schema is in this repository, so the other three
  typed-decisions workflows are recorded by their question-id signature with instruction text
  written for the fixture. The ids are what routing reads and the text is recorded verbatim, so
  the forward pass is real; the trained wording is not claimed to be.
- The softmax is computed in `double` where Python computes it in float32, as `laya-ts` and
  `laya-dotnet` also do. Over 200,000 random logit rows the four-decimal probabilities differ on
  88 of them. Rounding, by contrast, is exact: `BigDecimal` at scale 4 with `HALF_EVEN` is
  Python's `round(v, 4)` on Python's operand.
- Float `repr` spelling diverges from CPython on 218 of 1.7M doubles, in the branch that chooses
  scientific notation. It reaches a reason string, never a number.
- Seven tokenizer behaviours are implemented but unexercised, because no shipped checkpoint
  declares them: declared added-token ids, duplicate merge ranks, `end_of_word_suffix`,
  `prepend_scheme: "first"`, the `single_word` fallback, `model.dropout` and `truncation`.

### Where Java is not Python, and what it cost

Recorded here because each one was a defect first and a lesson second.

- `\p{L}` and `\p{N}` in `java.util.regex` follow **the JDK's** Unicode version, so the
  pre-tokenizer returned different token ids from the same jar on different JDKs — Corretto 17
  carries Unicode 13.0 and Corretto 24 carries 16.0, and they disagree on 9,917 code points. The
  classes are recorded from the reference tokenizer and compiled in, and the pattern is a scanner
  over them.
- `truncate_left = isinstance(state, list)`. A conversation serialises newest-last, so passing a
  literal `false` showed the model the *opening* of an oversized conversation and discarded the
  current turn. `usage` was byte-identical either way.
- Python's `str.strip()` takes U+0085, U+00A0, U+2007 and U+202F; `String.trim()` does not. The
  gap routed `"en"` plus an ideographic space to the multilingual checkpoint while the reason
  string still said the caller asked for English.
- Python's `\s` is 29 code points and Java's is 6, or 25 under `UNICODE_CHARACTER_CLASS`; Python's
  `\d` is Nd and Java's follows the JDK. Both are built from the recorded tables.
- Python slices a `str` by **code point**, and a negative index drops the tail rather than
  returning nothing. `substring` does neither.
- `Double.compare` ranks NaN as the largest double; `np.argsort` puts it last. In a shortlist that
  does not merely misreport a score, it changes which labels survive the cut.
- `%.0f` and `round()` are half-to-**even** in CPython; `String.format` and `Math.round` are
  half-up.
- A Java `finally` that throws **replaces** the pending exception, where Python's re-raises the
  original. The reference's hook wrapper goes out of its way to stop a failing end hook masking
  the failure it was observing, so the end-hook stage here is written out longhand rather than
  put in a `finally`, and the hook's failure is attached with `addSuppressed` where the
  reference chains it onto `__context__`.

[0.1.0]: https://github.com/NandhaKishorM/laya
