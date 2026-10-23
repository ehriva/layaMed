"""API-stability guard for laya.evals: an accidental rename or a torch import fails here.

Run: python tests/test_evals_api.py
"""
import argparse
import dataclasses
import inspect
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya import evals  # noqa: E402

PASS, FAIL = [], []


def check(name, got, expected):
    ok = got == expected
    (PASS if ok else FAIL).append(name)
    print(("  ok   " if ok else "  FAIL ") + name, "" if ok else "(got %r, expected %r)" % (got, expected))


def check_true(name, condition):
    check(name, bool(condition), True)


# --------------------------------------------------------------- fields
check("Example fields",
      [f.name for f in dataclasses.fields(evals.Example)],
      ["state", "questions", "expected", "tags", "language", "model"])
check("EvalReport fields",
      [f.name for f in dataclasses.fields(evals.EvalReport)],
      ["config", "overall", "slices", "cases"])

# --------------------------------------------------------------- evaluators
check("default evaluator names",
      sorted(e.name for e in evals.default_evaluators()),
      ["choice_accuracy", "mean_confidence", "noul_accuracy", "score_mae"])
check_true("ScoreWithin names its tolerance", evals.ScoreWithin(0.25).name == "score_within_0.25")

# --------------------------------------------------------------- exports / callables
for name in ("Dataset", "Example", "EvalError", "EvalReport", "evaluate", "ece", "assert_regression",
             "REPORT_SCHEMA", "questions_fingerprint", "file_fingerprint",
             "compare_agreement", "evaluate_agreement",
             "brier", "aurc", "selective_accuracy", "is_confidence_metric",
             # `_eval_policy` imports these three, so they are a contract between two modules in
             # this package, not internals -- a rename would break the release gate, not just a
             # caller.
             "is_coverage_metric", "coverage_definition_conflict",
             "COVERAGE_METRIC_DEFINITION"):
    check_true("laya.evals.%s exists" % name, hasattr(evals, name))

check("REPORT_SCHEMA", evals.REPORT_SCHEMA, "laya-evals-report/1")
check("the coverage-cut definition this code computes", evals.COVERAGE_METRIC_DEFINITION, 2)
check_true("EvalReport.comparable_to is part of the report contract",
           callable(getattr(evals.EvalReport, "comparable_to", None)))
check("EvalReport fields are unchanged", [f.name for f in dataclasses.fields(evals.EvalReport)],
      ["config", "overall", "slices", "cases"])

# --------------------------------------------------------------- evaluate's call signature
# `evaluate` is the harness entry point callers extend from their own scripts, so its parameter
# names and order are the contract. `sort_by_length` is core's #294 forward-pass grouping knob
# reaching a scored run through `laya-evals run --sort-by-length`; it is appended, never inserted,
# because a caller that passed `on_error` or `config` positionally must still mean the same thing.
# `min_confidence` extends the same rule: it is appended after `sort_by_length`, defaults to None
# (an unset abstention threshold means "score the raw argmax", which is what a pre-#361 run did),
# and sits after the batch size the calls that carry it can group inside.
_sig = inspect.signature(evals.evaluate)
check("evaluate signature", [p.name for p in _sig.parameters.values()],
      ["runner", "dataset", "evaluators", "batch_size", "on_error", "config",
       "sort_by_length", "min_confidence"])
check("evaluate: grouping defaults to off", _sig.parameters["sort_by_length"].default, False)
check("evaluate: abstention threshold defaults to unset",
      _sig.parameters["min_confidence"].default, None)
check_true("evaluate: the knob sits after the batch size it groups inside",
           list(_sig.parameters).index("batch_size")
           < list(_sig.parameters).index("sort_by_length"))
check_true("evaluate: the abstention knob sits after the batch size it may apply within",
           list(_sig.parameters).index("batch_size")
           < list(_sig.parameters).index("min_confidence"))

# The two batch entry points the CLI wires up: both have to take the knob by the same name, or a
# `--sort-by-length` run reports `sort_by_length_sent: false` for the surface it ships. The same
# is true for the abstention threshold; a `--min-confidence` run that silently dropped the
# argument would publish `report.config["timing"]["min_confidence"]` naming a threshold that was
# never applied -- the metrics stay identical (the gate flags, it does not overwrite the argmax),
# but the report itself would lie -- so the CLI runner shapes have to accept the kwarg under the
# same name the guard on `evaluate` checks.
from laya import evals_cli  # noqa: E402

for label, fn in (("RouterRunner.predict_batch", evals_cli.RouterRunner.predict_batch),
                  ("OnnxRunner.predict_batch", evals_cli.OnnxRunner.predict_batch)):
    params = inspect.signature(fn).parameters
    check_true("%s takes the grouping knob" % label, "sort_by_length" in params)
    check("%s defaults it to off" % label, params["sort_by_length"].default, False)
    check_true("%s takes the abstention threshold" % label, "min_confidence" in params)
    check("%s defaults the threshold to unset" % label,
          params["min_confidence"].default, None)

# The single-predict fallback path takes the threshold under the same name on both runner shapes,
# so a run without --batch-size still scores the gate rather than silently skipping it.
for label, fn in (("RouterRunner.predict", evals_cli.RouterRunner.predict),
                  ("OnnxRunner.predict", evals_cli.OnnxRunner.predict)):
    params = inspect.signature(fn).parameters
    check_true("%s takes the abstention threshold" % label, "min_confidence" in params)
    check("%s defaults the threshold to unset" % label,
          params["min_confidence"].default, None)

# A signature is not a wiring: a runner that declares the knob and drops it on the way through
# would pass both checks above. `OnnxRunner`'s call is gated in test_evals_onnx.py against a
# recording agent; this is the same witness for the router-shaped runner the CLI uses by default.
class _RecordingRouter:
    def __init__(self):
        self.calls = []

    def predict_batch(self, requests, batch_size=None, sort_by_length=False,
                      min_confidence=None):
        self.calls.append({"batch_size": batch_size, "sort_by_length": sort_by_length,
                           "min_confidence": min_confidence})
        return [{"model": "m", "answers": {}} for _ in requests]


_router = _RecordingRouter()

# --------------------------------------------------------------- torch stays out of import
probe = subprocess.run(
    [sys.executable, "-c",
     "import sys; import laya.evals; sys.exit(1 if 'torch' in sys.modules else 0)"],
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
check("import laya.evals does not import torch", probe.returncode, 0)

# --------------------------------------------------------------- argument documentation
# `_build_parser` is what a caller reads before running anything: argparse prints each
# declaration's help into `--help`, and docs/evals.md is the page that explains the command. On
# `main` five of its 22 declarations carried no help at all -- `run --on-error` printed as bare
# `--on-error {fail,skip}`, `compare --baseline` printed bare while being the one argument that
# call requires, and `validate`/`run`'s `dataset` and `compare`'s `report` had nothing under them
# -- while the page described `on_error=skip`'s effect on the batch counters without ever naming
# the flag that sets it. These read the parser the CLI really builds, so a flag added tomorrow is
# held to the same three rules as `--on-error` is today.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Backslash continuations are folded first, so a flag on the second line of a multi-line example is
# still attributed to the subcommand that started it.
_PAGE = re.sub(r"\\\n\s*", " ",
               open(os.path.join(_ROOT, "docs", "evals.md"), encoding="utf-8").read())
# Greedy, so `--min-accuracy` is one token and never also yields `--min`.
_PAGE_FLAGS = set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", _PAGE))


def _subparsers(parser):
    """{subcommand name: its parser}, read off the subparsers action the parser holds."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


_subs = _subparsers(evals_cli._build_parser())
check("laya-evals subcommands", sorted(_subs), ["agreement", "compare", "evidence", "run", "validate"])

_unhelped, _unstated, _options = [], [], {}
for _sub, _parser in sorted(_subs.items()):
    for action in _parser._actions:
        if action.dest == "help":
            continue
        site = "%s %s" % (_sub, " ".join(action.option_strings) or action.dest)
        _options.setdefault(_sub, set()).update(action.option_strings)
        if not (action.help or "").strip():
            _unhelped.append(site)
        elif action.default not in (None, False):
            # argparse prints a flag's default nowhere unless its own help text does, so a default
            # that matters has to be said: `--on-error` is documented as a choice, and the run it
            # aborts rather than reports is the half a caller cannot guess. The word `default` and
            # the quoted value must share a clause, because a help string that simply names both
            # choices -- which is what `--on-error`'s does -- would otherwise still read as if the
            # other one were the fallback after a flip.
            value = re.escape(str(action.default))
            if not re.search(r"'%s'[^;:]*\bdefault|\bdefault\b[^;:]*'%s'" % (value, value),
                             action.help):
                _unstated.append("%s (default %r)" % (site, action.default))

check("every laya-evals argument carries a help string", sorted(_unhelped), [])
check("every laya-evals default is stated in its own help", sorted(_unstated), [])

_documented = sorted("%s %s" % (sub, opt) for sub, opts in _options.items() for opt in opts
                     if opt not in _PAGE_FLAGS)
check("every laya-evals option is named in docs/evals.md", _documented, [])

_advertised = []
for line in _PAGE.splitlines():
    match = re.search(r"laya-evals\s+([a-z]+)", line)
    if not match or match.group(1) not in _options:
        continue
    _advertised += ["%s %s" % (match.group(1), flag)
                    for flag in re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", line)
                    if flag not in _options[match.group(1)]]
check("docs/evals.md advertises only options the parser defines", sorted(_advertised), [])

# --------------------------------------------------------------- run identity per route
# `docs/evals.md`'s "Run identity" table and `docs/staged-adoption.md`'s "four of those six" both
# promise `config.revisions`. What each route puts in that block is a property of the writer, not of
# the prose: the torch run reports `Router.loaded_revisions`, the `--onnx` run one entry keyed by the
# `--model` value (its default when `--model` is left out), and neither has a commit for a checkpoint
# given as a local directory. These run the real CLI on three arms against fakes -- no download, no
# onnxruntime -- so what the block holds is what the code printed here, and the two pages are then
# held to that.
import ast  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402

import laya  # noqa: E402
import laya.onnx_agent as _onnx_agent_module  # noqa: E402  (numpy-only at import time)

_IDENTITY_Q = {"intent": {"type": "choice", "instructions": "i", "criteria": {"billing": "b"}}}
_SHA = "55cf4c4e" + "0" * 32


class _RouteRouter:
    """Router-shaped: answers from two checkpoints, one Hub-pinned, one a local path."""

    def __init__(self, *a, **kw):
        self.loaded_revisions = {"english": _SHA, "multilingual": None}

    def predict(self, state, questions, **kw):
        return {"model": "english",
                "answers": {qid: {"type": "choice", "choice": "billing", "answer_confidence": 0.9}
                            for qid in questions},
                "usage": {"input_tokens": 3, "output_tokens": 0}}


class _RouteOnnxAgent:
    """`ONNXAgent` holds the source it was given and a commit only when it downloaded one
    (`laya/onnx_agent.py:101`, `:107`, `:126`), which is the pair the report block is read from."""

    def __init__(self, model_id_or_path, onnx_path=None, **kw):
        self.model_id = model_id_or_path
        self.revision = None

    def predict(self, state, questions, **kw):
        return {"model": self.model_id,
                "answers": {qid: {"type": "choice", "choice": "billing", "answer_confidence": 0.9}
                            for qid in questions},
                "usage": {"input_tokens": 3, "output_tokens": 0}}


_identity_tmp = tempfile.mkdtemp(prefix="laya_evals_identity_")
_identity_dataset = os.path.join(_identity_tmp, "data.jsonl")
with open(_identity_dataset, "w") as f:
    for _state in ("charged twice, refund me", "billed again after cancelling"):
        f.write(json.dumps({"state": _state, "questions": _IDENTITY_Q,
                            "expected": {"intent": "billing"}}) + "\n")


def _route_report(extra_args):
    global _route_n
    _route_n += 1
    path = os.path.join(_identity_tmp, "report_%d.json" % _route_n)
    label = "onnx" if "--onnx" in extra_args else "torch Router"
    with contextlib.redirect_stdout(io.StringIO()):
        rc = evals_cli.main(["run", _identity_dataset, "--json", path] + extra_args)
    check_true("run on the %s route exits 0" % label, rc == 0)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["config"]


_route_n = 0
_real_router, _real_agent = laya.Router, _onnx_agent_module.ONNXAgent
try:
    laya.Router = _RouteRouter
    _router_cfg = _route_report(["--device", "cpu"])
    _onnx_agent_module.ONNXAgent = _RouteOnnxAgent
    _onnx_cfg = _route_report(["--onnx", os.path.join(_identity_tmp, "laya.onnx"),
                               "--model", "english-ckpt"])
    _onnx_default_cfg = _route_report(["--onnx", os.path.join(_identity_tmp, "laya.onnx")])
finally:
    laya.Router, _onnx_agent_module.ONNXAgent = _real_router, _real_agent

check("the torch Router run records one entry per checkpoint it loaded",
      _router_cfg.get("revisions"), {"english": _SHA, "multilingual": None})
check("an `--onnx` run records one entry keyed by the `--model` value",
      _onnx_cfg.get("revisions"), {"english-ckpt": None})
check("an `--onnx` run with no `--model` keys the default source",
      _onnx_default_cfg.get("revisions"), {"convaiinnovations/laya": None})
check("and its `model` stays null while the block names that default",
      _onnx_default_cfg.get("model"), None)
check("both routes record the dataset, question and gate identity",
      sorted(k for k in _onnx_cfg if k in ("dataset_sha256", "questions_sha256", "thresholds")),
      sorted(k for k in _router_cfg if k in ("dataset_sha256", "questions_sha256", "thresholds")))
check_true("a local-path checkpoint is recorded as null, not dropped",
           _router_cfg.get("revisions", {"multilingual": "ABSENT"})["multilingual"] is None)
check_true("an `--onnx` run from a local checkpoint directory is null too",
           "revisions" in _onnx_cfg and _onnx_cfg["revisions"]["english-ckpt"] is None)


def _revisions_writes():
    """`(writes, writes nested inside an `if`)` for the `revisions=` keyword in `_cmd_run`.

    The block has to be written once for every route. It used to be read off
    `Router.loaded_revisions` under `if router is not None`, which left the `--onnx` report with no
    attribution at all, so the shape that keeps both routes honest is one write at the top level of
    the function -- not a count that a guard added back would leave alone.
    """
    tree = ast.parse(inspect.getsource(evals_cli._cmd_run))
    found = []

    def walk(node, depth):
        for child in ast.iter_child_nodes(node):
            if (isinstance(child, ast.keyword) and child.arg == "revisions"
                    and isinstance(child.value, ast.Name)):
                found.append(depth)
            walk(child, depth + (1 if isinstance(child, ast.If) else 0))

    walk(tree, 0)
    return (len(found), sum(1 for depth in found if depth))


check("the CLI writes the revisions block once, at the top level of the run",
      _revisions_writes(), (1, 0))

_ROW = {line.split("|")[1].strip().strip("`"): line.split("|")[2]
        for line in _PAGE.splitlines() if line.startswith("| `") and line.count("|") >= 3}
check("docs/evals.md's `revisions` row covers both routes and the `--model` key",
      [s in _ROW.get("revisions", "") for s in ("--onnx", "torch run", "`--model`")],
      [True, True, True])
check_true("docs/evals.md's `revisions` row names the local-path null",
           "`null`" in _ROW.get("revisions", ""))
_BLOCK_PARA = [p for p in _PAGE.split("\n\n") if "Both routes write the `revisions` block" in p]
check("exactly one paragraph states what each route records", len(_BLOCK_PARA), 1)
check_true("that paragraph names the default the `--onnx` run is keyed by",
           bool(_BLOCK_PARA) and "convaiinnovations/laya" in _BLOCK_PARA[0]
           and "`--model`" in _BLOCK_PARA[0])
check_true("that paragraph says a local directory is recorded as null, not dropped",
           bool(_BLOCK_PARA) and "local directory" in _BLOCK_PARA[0]
           and "recorded as `null`" in _BLOCK_PARA[0])
# The paragraph cites the two `revision` attributes the null comes from; both files still hold them.
_defaults = []
for _rel in ("laya/agent.py", "laya/onnx_agent.py"):
    _src = open(os.path.join(_ROOT, _rel), encoding="utf-8").read()
    _defaults.append("self.revision: Optional[str] = None" in _src)
check("both cited files leave `revision` None before any download", _defaults, [True, True])
_STAGED = [p for p in open(os.path.join(_ROOT, "docs", "staged-adoption.md"),
                           encoding="utf-8").read().split("\n\n")
           if "Four of those six" in p]
check("exactly one paragraph claims the four harness keys", len(_STAGED), 1)
check_true("docs/staged-adoption.md keeps the null caveat and drops the route claim",
           bool(_STAGED) and "`null`" in _STAGED[0] and "builds no Router" not in _STAGED[0])

# --------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for name in FAIL:
    print("  FAIL " + name)
if not FAIL:
    print("all eval API checks passed")
sys.exit(1 if FAIL else 0)
