# Laya models: getting the artifacts `laya-java` reads

`laya-java` runs ONNX. It does not download anything, does not call Hugging Face, and does not
start a Python process — `Agent.open` takes two directories that already exist on disk:

| directory | holds | made by |
|---|---|---|
| the **checkpoint** | `rl_agent_config.json`, `tokenizer/tokenizer.json`, `encoder/config.json` | a download |
| the **graph** | `laya.onnx` (fused), or `encoder.onnx` + `head.onnx` (split) | the exporter |

One script does both. From the repository root:

```bash
python laya-java/scripts/prepare_checkpoint.py --checkpoint multilingual
```

It writes, under `laya-java/.work/` so the repository root stays clean and one `.gitignore`
covers it:

```
laya-java/.work/checkpoints/multilingual/      the checkpoint, as Agent.open expects it
laya-java/.work/onnx/multilingual/laya.onnx    the fused graph, as Agent.open expects it
```

Then point the SDK at them:

```bash
export LAYA_CHECKPOINTS=$PWD/laya-java/.work/checkpoints
export LAYA_ONNX_GRAPH=$PWD/laya-java/.work/onnx/multilingual
```

or pass the paths directly:

```java
Agent agent = Agent.open(
        Path.of("laya-java/.work/checkpoints/multilingual"),
        Path.of("laya-java/.work/onnx/multilingual"));
```

---

## The three checkpoints

Named as `laya.router` names them, and resolved through it — the script reads
`laya.router.DEFAULT_MODELS` rather than keeping its own copy of the repository names, so a
checkpoint the Python package knows about is one this can fetch.

| `--checkpoint` | model | tokens | notes |
|---|---|---|---|
| `english` | ModernBERT-large, 421M | 512 | English only, and it does not gently degrade off English |
| `multilingual` | mmBERT-base, 322M | 1024 | 100+ languages |
| `typed-decisions` | ModernBERT-large, 421M | 1024 | fine-tuned on four synthetic workflows |

`typed-decisions` is never selected automatically. `Router` knows it, and the aliases `typed`,
`typed_decisions`, `decisions` and `laya-typed-decisions` all resolve to it, but reaching it takes
an explicit `task` or `autoTaskDetection` — it is fine-tuned on four specific workflows and is the
wrong silent default.

Its forward pass **is** recorded, in `fixtures/typed_decisions.json`, and asserted by
`TypedDecisionsParityTest` in the `jvm parity (typed-decisions)` cell: nine single-state cases
covering all four workflow signatures, three batch configurations, and the two checks that the
weights under test are the right ones. Regenerate it with:

```bash
python laya-java/scripts/prepare_checkpoint.py --checkpoint typed-decisions
LAYA_TYPED_ONNX_GRAPH=laya-java/.work/onnx/typed-decisions/laya.onnx \
  python laya-java/scripts/gen_fixtures.py --only typed_decisions.json
```

`LAYA_TYPED_ONNX_GRAPH`, not `LAYA_ONNX_GRAPH`, and the two must never be crossed. A graph
carries the weights and nothing in it names the checkpoint it was traced from, so a wrong pairing
answers instead of failing — `typed-decisions` is fine-tuned **from** `english` and ships its
`tokenizer.json` byte for byte and its ModernBERT-large encoder. Measured with the english
checkpoint symlinked in as `typed-decisions/`: opening it against the typed-decisions graph raises
nothing and reproduces the recorded probabilities inside the golden's tolerance — `off-workflow`
answers `{billing: 0.8967, other: 0.1033}`, which is what the committed fixture records for the
right pairing — and `model` reads `laya-rl-agent-onnx` either way. **11 of the 13 tests passed**
under that pairing: every single-state case but `over-budget`, which truncates differently only
because `max_len` is 512 there and 1024 here, plus all three batch configurations.

What names the problem is the `config` section of the golden, and only three fields of it do:
`max_len` 1024 against 512, `head_max_len` 256 against 192, and the fitted base temperatures
`[1.0148, 1.0374, 1.0575]` against `[1.6369, 1.2514, 1.9834]`. The `temperature_by_options` tables
are **identical** between the two checkpoints — every key and every digit, `choice:11+` at 0.1006
included — so the bucket assertions and the `TEMP_MIN` clamp assertion catch a port bug and
nothing about a wrong pairing. That config check is asserted on the shared agent as it is opened,
not as one test among thirteen: it used to run ninth, after every probability comparison, because
JUnit 5 orders nothing by default. Measured now, the same pairing registers 4 tests and fails 3,
and the cell's exact `min-tests: 14` floor fails on the count as a second signal. The
`unbucketed-score` case is there so the fitted triple has power of its own: a six-level score
question lands in `score:6-10`, a bucket neither checkpoint ships, so the lookup misses and
`temperature[QTYPE_SCORE]` — 1.0374 here, 1.2514 on english — is what scales those logits. Only
the multilingual checkpoint fails loudly against that graph, and for an unrelated reason: a 235292
token id against a 50368-row embedding, which ONNX Runtime rejects at the `node_embedding` Gather.

The golden also records the `.laya-revision` stamp the graph carries, so a `HF_REVISION` bump
shows up as `gen_fixtures.py --check` drift instead of a silent re-record. `predict.json` records
no revision; it would have to be re-recorded against the multilingual graph to gain one.

Of the four workflows, only `invoice_processing`'s trained question schema is in this repository
(`examples/32_typed_decisions_workflow.py`). The other three are known by their question-id
signatures alone, from `laya.router._TYPED_DECISION_WORKFLOWS`, so the golden carries those ids
with instruction text written for the fixture. The ids are what the routing decision reads and the
text is recorded verbatim, so both halves are real; the trained wording is not claimed.

A prepared checkpoint is about **680–850 MB** on disk, and the fused multilingual graph is
**1.29 GB** (`laya.onnx` is a 2.8 MB graph beside a `laya.onnx.data` of external weights). The
typed-decisions checkpoint is **846 MB** and its graph is **1.69 GB** (3.6 MB beside the external
weights); the export took **39 s** locally, and recording the family another **103 s**. Budget
accordingly; the export is the slow part.

---

## Only the tokenizer, when that is all you need

Most of what this SDK does needs no weights. The tokenizer, the sequence builder, the collator,
language detection, the presets, the router's *decision* and the email cleaner are all exercised
by `tokenizer/tokenizer.json` and `rl_agent_config.json`, which are kilobytes:

```bash
python laya-java/scripts/prepare_checkpoint.py --checkpoint english --no-graph
```

That skips `model.safetensors` and the export entirely. It is what the `jvm parity (tokenizer)`
CI cell uses, and why that cell runs in seconds while the cell that needs a graph takes a cache
miss measured in tens of minutes.

---

## Pinning, and why a stamp sits next to the artifact

The download is pinned: `HF_REPO = "convaiinnovations/laya"` at revision
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, the same pin `laya-dotnet/tools/regen_golden.py`
uses. Both lanes measure the same checkpoint, so they must not drift to different revisions — a
port that matched one and not the other would look like a port bug. The script refuses to run if
a checkpoint has moved to a different repository rather than silently fetching from the new one.

Beside each artifact it writes a `.laya-revision` stamp, and "already here" means "already here at
THIS revision". Without it, `--force` would be the only way to pick up a revision bump, and a
restored CI cache would make the whole lane measure the old model while reporting on the new one:
the fixture would be re-recorded from the stale graph, both sides of the comparison would agree,
and the cell would go green having tested nothing it claims to. The stamp also records whether
weights were fetched, because a `--no-graph` checkpoint cannot be traced and must not satisfy a
request that needs to.

To re-download and re-export anyway:

```bash
python laya-java/scripts/prepare_checkpoint.py --checkpoint multilingual --force
```

To fetch a different revision (and get a different stamp):

```bash
python laya-java/scripts/prepare_checkpoint.py --checkpoint multilingual --revision <sha>
```

---

## Both graph layouts

`LayaSession.open` takes a directory and accepts either layout:

* **fused** — `laya.onnx`, one graph from ids to logits. What the exporter writes, and what
  `prepare_checkpoint.py` produces.
* **split** — `encoder.onnx` + `head.onnx`. A documented, advertised layout, so it is supported:
  the encoder returns a pooled representation and the head turns it into logits.

If you export the split layout yourself, note that the head declares `qtype` as `[B, 1]`, not a
rank-1 tensor. This port passed rank-1 at first and ORT refused every graph the only exporter for
that layout produces, with an error that read like a bad model rather than a bad client.

A directory with neither layout is an error that names what it looked for, rather than a null
later on.

---

## Exporting by hand

`prepare_checkpoint.py` shells out to the repository's own exporter, so the graph the Java tests
run is the graph `laya` ships:

```bash
PYTHONPATH=. python scripts/export_onnx.py \
    --model laya-java/.work/checkpoints/multilingual \
    --output laya-java/.work/onnx/multilingual/laya.onnx
```

`PYTHONPATH`, not just the working directory: a script's `sys.path[0]` is the *script's*
directory, so `scripts/export_onnx.py` would get `<repo>/scripts` and `import laya` fails with
`ModuleNotFoundError` even when run from the repository root. Setting it rather than installing
the package keeps the export measuring the working tree.

---

## What the SDK needs at run time

| variable | read by | meaning |
|---|---|---|
| `LAYA_CHECKPOINTS` | the tests | a directory holding `english/`, `multilingual/`, … |
| `LAYA_ONNX_GRAPH` | the tests | the directory holding `laya.onnx`, or the file itself for the generator |
| `LAYA_FIXTURE_CHECKPOINTS` | `gen_fixtures.py` | the checkpoint root when re-recording |
| `LAYA_PREDICT_MODEL` | `gen_fixtures.py` | which checkpoint the end-to-end golden is recorded from |

A **blank** value counts as absent, so a CI cell that owns no graph can set
`LAYA_ONNX_GRAPH=''` without turning a test's deliberate abort into a failure.

None of these are read by the library itself. `Agent.open` and `Router` take paths; the
environment is the test harness's business.

---

## Without a graph, tests abort rather than pass

A parity test that needs a checkpoint says so and is skipped. That is deliberate, and the CI
floors exist because a skipped parity test is not a green one:

```
$ ./gradlew test
...
PredictParityTest > singleMatches() SKIPPED
```

With `LAYA_CHECKPOINTS` and `LAYA_ONNX_GRAPH` set, nothing in the suite may abort, and
`scripts/check_test_results.py` fails the run if anything does.

For the library's own API, see the [README](README.md). For what changed, see the
[CHANGELOG](CHANGELOG.md).
