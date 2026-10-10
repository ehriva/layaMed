"""Model-free checks for paired agreement and case-level uncertainty."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from laya import evals


def run_document(run_id, rows, *, teacher=True):
    cases = []
    for index, labels in enumerate(rows):
        questions = {
            "q%d" % q: {"type": "choice", "instructions": "Choose a route.",
                        "criteria": {"a": "Accept", "b": "Defer"}}
            for q in range(len(labels))
        }
        cases.append({
            "id": "case-%d" % index,
            "state": {"text": "request-%d" % index, "context": ["original"]},
            "questions": questions,
            "answers": dict(zip(questions, labels)),
            "tags": ["routing"], "language": "en",
        })
    return {"schema": "laya-agreement-run/1", "run_id": run_id,
            "provenance": {"model": "teacher" if teacher else "student",
                           "revision": "pinned", "settings": {"temperature": 0.5}},
            "cases": cases}


def documents(reference=(("a",), ("b",)), repeat=None, student=None):
    return (run_document("reference", reference),
            run_document("repeat", repeat if repeat is not None else reference),
            run_document("student", student if student is not None else reference,
                         teacher=False))


def compare(runs, **kwargs):
    return evals.compare_agreement(*runs, **kwargs)


def test_paired_rates_use_every_decision_and_preserve_abstention_denominator():
    runs = documents((("a", "a", "a"), ("b", "a")),
                     (("a", "b", "a"), ("b", "a")),
                     (("a", "b", None), ("b", "b")))
    report = compare(runs)
    assert report["schema"] == "laya-agreement-report/1"
    overall = report["overall"]
    assert overall["n_cases"] == 2
    assert overall["n_decisions"] == 5
    assert overall["teacher_repeat_agreement"] == pytest.approx(4 / 5)
    assert overall["student_teacher_agreement"] == pytest.approx(2 / 5)
    assert overall["student_coverage"] == pytest.approx(4 / 5)
    assert overall["paired_agreement_delta"] == pytest.approx(-2 / 5)
    assert len(report["cases"]) == 5
    observed = {(row["case_id"], row["qid"]):
                (row["teacher_reference"], row["teacher_repeat"], row["student"],
                 row["accepted"], row["delta"]) for row in report["cases"]}
    assert observed == {
        ("case-0", "q0"): ("a", "a", "a", True, 0),
        ("case-0", "q1"): ("a", "b", "b", True, 0),
        ("case-0", "q2"): ("a", "a", None, False, -1),
        ("case-1", "q0"): ("b", "b", "b", True, 0),
        ("case-1", "q1"): ("a", "a", "b", True, -1),
    }
    assert report["slices"]["qid"]["q2"]["n_decisions"] == 1
    assert report["slices"]["qid"]["q2"]["student_coverage"] == 0.0
    assert not {"accuracy", "choice_accuracy", "ece", "brier", "aurc"} & overall.keys()
    assert json.loads(json.dumps(report)) == report


@pytest.mark.parametrize("repeat,student,agreement", [("a", "a", 1.0), ("b", "b", 0.0)])
def test_perfect_and_zero_agreement(repeat, student, agreement):
    overall = compare(documents((("a",), ("a",)),
                                ((repeat,), (repeat,)),
                                ((student,), (student,))))["overall"]
    assert overall["teacher_repeat_agreement"] == agreement
    assert overall["student_teacher_agreement"] == agreement
    assert overall["paired_agreement_delta"] == 0.0
    assert tuple(overall["paired_delta_ci95"]) == (0.0, 0.0)


def test_student_abstention_is_not_a_missing_case_or_an_agreeing_answer():
    overall = compare(documents(student=((None,), (None,))))["overall"]
    assert overall["n_cases"] == overall["n_decisions"] == 2
    assert overall["student_coverage"] == 0.0
    assert overall["student_teacher_agreement"] == 0.0
    assert overall["paired_agreement_delta"] == -1.0
    assert tuple(overall["paired_delta_ci95"]) == (-1.0, -1.0)


def test_alignment_uses_case_ids_and_comparison_does_not_mutate_inputs():
    runs = documents((("a",), ("b",), ("a",)),
                     (("a",), ("a",), ("b",)),
                     (("b",), ("b",), ("a",)))
    original = deepcopy(runs)
    expected = compare(runs, seed=31)["overall"]
    assert runs == original
    runs[0]["cases"].reverse()
    runs[1]["cases"] = runs[1]["cases"][1:] + runs[1]["cases"][:1]
    assert compare(runs, seed=31)["overall"] == expected


def test_one_case_has_no_interval_even_with_many_questions():
    overall = compare(documents((("a",) * 30,)))["overall"]
    assert overall["n_cases"] == 1
    assert overall["n_decisions"] == 30
    assert overall["paired_delta_ci95"] is None


def test_bootstrap_resamples_whole_cases_not_correlated_questions():
    # There are just two independent cases. Resampling two cases can give only
    # delta -1, 0 or +1, regardless of how many questions each case contains.
    runs = documents((("a",) * 25, ("a",) * 25),
                     (("a",) * 25, ("b",) * 25),
                     (("b",) * 25, ("a",) * 25))
    report = compare(runs, bootstrap_samples=2000, seed=17)
    overall = report["overall"]
    assert overall["paired_agreement_delta"] == 0.0
    assert tuple(overall["paired_delta_ci95"]) == (-1.0, 1.0)
    assert compare(runs, bootstrap_samples=2000, seed=17) == report


def test_bootstrap_preserves_decision_weights_with_unequal_case_sizes():
    runs = documents((("a",) * 100, ("a",), ("a",), ("a",)),
                     (("b",) * 100, ("a",), ("a",), ("a",)),
                     (("a",) * 100, ("b",), ("b",), ("b",)))
    overall = compare(runs, bootstrap_samples=5000, seed=23)["overall"]
    assert overall["paired_agreement_delta"] == pytest.approx(97 / 103)
    # K large-case draws follows Binomial(4, 1/4). The 2.5th/97.5th percentiles
    # have K=0/3: ratios -4/4 and (300-1)/(300+1), not case-mean rates -1/0.5.
    assert overall["paired_delta_ci95"] == pytest.approx([-1.0, 299 / 301])


@pytest.mark.parametrize("side", [0, 1, 2])
@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown"])
def test_case_sets_must_match_exactly(side, change):
    runs = documents()
    if change == "missing":
        runs[side]["cases"].pop()
    elif change == "duplicate":
        runs[side]["cases"].append(deepcopy(runs[side]["cases"][0]))
    else:
        runs[side]["cases"][0]["id"] = "unrelated"
    with pytest.raises(ValueError):
        compare(runs)


@pytest.mark.parametrize("change", ["state", "prompt", "criteria_order", "criteria_text", "qid"])
def test_input_and_question_drift_are_rejected(change):
    runs = documents()
    case = runs[1]["cases"][0]
    question = case["questions"]["q0"]
    if change == "state":
        case["state"]["context"].append("changed")
    elif change == "prompt":
        question["instructions"] = "Choose the opposite route."
    elif change == "criteria_order":
        question["criteria"] = {"b": "Defer", "a": "Accept"}
    elif change == "criteria_text":
        question["criteria"]["a"] = "Reject"
    else:
        case["questions"]["other"] = case["questions"].pop("q0")
        case["answers"]["other"] = case["answers"].pop("q0")
    with pytest.raises(ValueError):
        compare(runs)


@pytest.mark.parametrize("key,value", [("model", "other"), ("revision", "other-pin"),
                                       ("settings", {"temperature": 0.9})])
def test_teacher_configuration_must_match(key, value):
    runs = documents()
    runs[1]["provenance"][key] = value
    with pytest.raises(ValueError):
        compare(runs)


def test_teacher_seed_may_differ():
    runs = documents()
    runs[0]["provenance"]["seed"] = 5
    runs[1]["provenance"]["seed"] = 8
    assert compare(runs)["overall"]["teacher_repeat_agreement"] == 1.0


def test_unknown_teacher_revision_remains_unknown_and_disclosed():
    runs = documents()
    for run in runs[:2]:
        run["provenance"]["revision"] = None
    report = compare(runs)
    assert report["config"]["provenance"]["reference"]["revision"] is None
    assert report["config"]["provenance"]["repeat"]["revision"] is None
    assert any("revision" in note.lower() for note in report["limitations"])


@pytest.mark.parametrize("name,value", [("bootstrap_samples", 99), ("bootstrap_samples", 100001),
    ("bootstrap_samples", True), ("bootstrap_samples", 100.5), ("bootstrap_samples", float("inf")),
    ("seed", -1), ("seed", True), ("seed", 1.5), ("seed", float("nan"))])
def test_bootstrap_options_require_bounded_integer_values(name, value):
    with pytest.raises(ValueError):
        compare(documents(), **{name: value})


@pytest.mark.parametrize("change", ["schema", "empty", "score", "teacher_abstention",
                                  "unknown_label", "missing_answer", "extra_answer", "nonfinite"])
def test_malformed_runs_are_rejected(change):
    runs = documents()
    case = runs[0]["cases"][0]
    if change == "schema":
        runs[0]["schema"] = "laya-agreement-run/999"
    elif change == "empty":
        for run in runs:
            run["cases"] = []
    elif change == "score":
        case["questions"]["q0"]["type"] = "score"
    elif change == "teacher_abstention":
        case["answers"]["q0"] = None
    elif change == "unknown_label":
        case["answers"]["q0"] = "unlisted"
    elif change == "missing_answer":
        case["answers"].pop("q0")
    elif change == "extra_answer":
        case["answers"]["other"] = "a"
    else:
        for run in runs:
            run["provenance"]["settings"]["temperature"] = float("nan")
    with pytest.raises(ValueError):
        compare(runs)


_DEFAULT_RESULT = object()


class RecordingRunner:
    def __init__(self, result=_DEFAULT_RESULT, mutate=False):
        self.calls = []
        self.result = result
        self.mutate = mutate

    def predict(self, state, questions, model=None):
        self.calls.append((deepcopy(state), deepcopy(questions), model))
        answers = {qid: {"type": "choice", "choice": "a"} for qid in questions}
        if self.mutate:
            state["context"].append("runner mutation")
            questions["q0"]["criteria"]["a"] = "runner mutation"
        return {"answers": answers} if self.result is _DEFAULT_RESULT else self.result


def evaluate(runner, runs, **kwargs):
    return evals.evaluate_agreement(runner, runs[0], runs[1],
                                   student_provenance=runs[2]["provenance"], **kwargs)


def test_evaluate_forwards_model_and_isolates_inputs_from_runner_mutation():
    runs = documents()
    original = deepcopy(runs)
    runner = RecordingRunner(mutate=True)
    report = evaluate(runner, runs, model="local-checkpoint")
    assert runs == original
    assert len(runner.calls) == 2
    assert all(call[2] == "local-checkpoint" for call in runner.calls)
    assert report["overall"]["student_teacher_agreement"] == 0.5


def test_evaluate_treats_abstained_argmax_as_uncovered():
    runner = RecordingRunner({"answers": {"q0": {"type": "choice", "choice": "a",
                                               "abstention": "abstained"}}})
    overall = evaluate(runner, documents())["overall"]
    assert overall["student_coverage"] == overall["student_teacher_agreement"] == 0.0


@pytest.mark.parametrize("result", [None, "invalid", {}, {"answers": {}},
    {"answers": {"q0": {"type": "choice", "choice": "unlisted"}}},
    {"answers": {"q0": {"type": "score", "score": 0.4}}},
    {"answers": {"q0": {"type": "choice", "choice": None}}}])
def test_evaluate_rejects_invalid_runner_results(result):
    with pytest.raises(ValueError):
        evaluate(RecordingRunner(result), documents())


@pytest.mark.parametrize("invalid", ["alignment", "bootstrap", "provenance", "null_state"])
def test_evaluate_validates_every_input_before_first_runner_call(invalid):
    runs = documents()
    kwargs = {}
    if invalid == "alignment":
        runs[1]["cases"][-1]["state"] = "changed final case"
    elif invalid == "bootstrap":
        kwargs["bootstrap_samples"] = True
    elif invalid == "null_state":
        for run in runs:
            run["cases"][-1]["state"] = None
    else:
        runs[2]["provenance"]["settings"] = {"temperature": float("inf")}
    runner = RecordingRunner()
    with pytest.raises(ValueError):
        evaluate(runner, runs, **kwargs)
    assert runner.calls == []


def cli(*args):
    return subprocess.run([sys.executable, "-m", "laya.evals_cli", "agreement", *map(str, args)],
                          cwd=Path(__file__).resolve().parents[1], capture_output=True,
                          text=True, encoding="utf-8", timeout=30)


def saved_runs(tmp_path):
    runs = documents()
    for run in runs:
        run["cases"][0]["state"] = "退款申请"
    paths = [tmp_path / (run["run_id"] + ".json") for run in runs]
    for path, run in zip(paths, runs):
        path.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    return paths, runs


def test_cli_saved_run_replay_matches_public_api(tmp_path):
    paths, runs = saved_runs(tmp_path)
    expected = compare(runs, bootstrap_samples=100, seed=12)
    args = [*paths, "--bootstrap-samples", "100", "--seed", "12"]
    result = cli(*args)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected
    output = tmp_path / "结果.json"
    result = cli(*args, "--json", output)
    assert result.returncode == 0, result.stderr
    assert not result.stdout
    assert json.loads(output.read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize("invalid", ["duplicate_key", "nan", "malformed", "nested", "missing", "misaligned"])
def test_cli_refuses_bad_inputs_without_a_report(tmp_path, invalid):
    paths, runs = saved_runs(tmp_path)
    if invalid == "duplicate_key":
        # Reject ambiguity even when the duplicate has the same value.
        raw = paths[0].read_text(encoding="utf-8")
        paths[0].write_text(raw.replace('"schema":', '"schema":"laya-agreement-run/1","schema":', 1),
                            encoding="utf-8")
    elif invalid == "nan":
        runs[0]["provenance"]["settings"]["temperature"] = float("nan")
        paths[0].write_text(json.dumps(runs[0]), encoding="utf-8")
    elif invalid == "malformed":
        paths[0].write_text("{broken", encoding="utf-8")
    elif invalid == "nested":
        paths[0].write_text("[" * 5000 + "0" + "]" * 5000, encoding="utf-8")
    elif invalid == "missing":
        paths[0].unlink()
    else:
        runs[2]["cases"].pop()
        paths[2].write_text(json.dumps(runs[2]), encoding="utf-8")
    output = tmp_path / "report.json"
    result = cli(*paths, "--json", output)
    assert result.returncode == 2
    assert "laya-evals:" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()
