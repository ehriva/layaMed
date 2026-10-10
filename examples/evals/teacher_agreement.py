"""Compare a local Laya checkpoint with two published teacher observations.

From a checkout::

    python -m pip install -e . pyarrow
    python examples/evals/teacher_agreement.py --output-dir /tmp/laya-agreement
    laya-evals agreement /tmp/laya-agreement/reference.json \
        /tmp/laya-agreement/repeat.json /tmp/laya-agreement/student.json

The first 32 test rows and teacher repeats 0/1 are fixed before student inference.
The parquet is downloaded once, checked against its SHA256, and kept in the output
directory. Pass --parquet to reuse a downloaded copy. Only the example needs pyarrow.
The pinned English Laya checkpoint runs locally on CPU; no teacher API is called.

Data credit: LocalGate's MIT-licensed localgate/mmlu-pro-closed dataset, derived
from TIGER-Lab/MMLU-Pro. The publisher's protocol and generation manifest are at:
https://huggingface.co/datasets/localgate/mmlu-pro-closed/blob/de3c3d7bd8870e6ec48f03284f3e48042bae9ecf/protocol.md
https://huggingface.co/datasets/localgate/mmlu-pro-closed/blob/de3c3d7bd8870e6ec48f03284f3e48042bae9ecf/provenance/labels_manifest.json

Teacher picks are real saved generations, not reconstructed from aggregate scores.
The publisher did not record the teacher's immutable model revision; null preserves
that uncertainty. Laya receives the same question and ordered options as typed
inputs, not the teacher's chain-of-thought prompt. This small panel demonstrates
the evaluation pipeline: agreement is neither accuracy nor a performance ceiling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from copy import deepcopy
from pathlib import Path

DATASET = "localgate/mmlu-pro-closed"
DATASET_REVISION = "de3c3d7bd8870e6ec48f03284f3e48042bae9ecf"
PARQUET_SHA256 = "7b1ee7627f19a6ac322aa677a6483549e5ea6e136b6992dba6da0f8a2227176f"
PARQUET_URL = (
    f"https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REVISION}"
    "/data/test-00000-of-00001.parquet"
)
PROTOCOL_SHA256 = "6f0ac97a38225f523a61803d25d6868e90813ef6873a8170a7e2f2a9e056ef9b"
MODEL = "convaiinnovations/laya"
MODEL_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
COUNT = 32
PROMPT_TEMPLATE = (
    "<bos><|turn>system\n"
    "The following are multiple choice questions (with answers) about {category}. "
    'Think step by step and then finish your answer with "the answer is (X)" '
    "where X is the correct letter choice.<turn|>\n"
    "<|turn>user\n"
    "Question:\n{question}\nOptions:\n{ordered_options}\n"
    "Answer: Let's think step by step.<turn|>\n"
    "<|turn>model\n"
)


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def teacher_runs(parquet: Path) -> tuple[dict, dict]:
    import pyarrow.parquet as pq

    if parquet.exists():
        data = parquet.read_bytes()
    else:
        request = urllib.request.Request(
            PARQUET_URL, headers={"User-Agent": "Laya-agreement-example"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != PARQUET_SHA256:
        raise ValueError(f"parquet SHA256 mismatch: {actual}")
    if not parquet.exists():
        parquet.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(data)

    table = pq.read_table(parquet)
    if table.num_rows < COUNT:
        raise ValueError(f"expected at least {COUNT} rows in the pinned parquet")
    rows = table.slice(0, COUNT).to_pylist()
    if len({row["question_id"] for row in rows}) != COUNT:
        raise ValueError("duplicate question_id in the fixed selection")
    runs = []
    for repeat, seed in ((0, 42), (1, 43)):
        cases = []
        for row in rows:
            criteria = dict(zip("ABCDEFGHIJ", row["options"]))
            if row["k"] != 5 or len(row["picks"]) != 5:
                raise ValueError(f"expected five teacher observations for {row['question_id']}")
            if row["picks"][repeat] not in criteria:
                raise ValueError(
                    f"invalid teacher choice for {row['question_id']}, repeat {repeat}; "
                    "the fixed selection is not filtered or backfilled"
                )
            cases.append({
                "id": str(row["question_id"]),
                "state": row["question"],
                "questions": {"answer": {
                    "type": "choice",
                    "instructions": "Choose the correct answer to the question.",
                    "criteria": criteria,
                }},
                "answers": {"answer": row["picks"][repeat]},
                "tags": [row["category"]],
                "language": "en",
            })
        runs.append({
            "schema": "laya-agreement-run/1",
            "run_id": f"gemma-seed-{seed}",
            "provenance": {
                "model": "google/gemma-4-E2B-it",
                "revision": None,
                "settings": {
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_gen_toks": 6144,
                    "until": ["Question:"],
                    "thinking": False,
                    "num_fewshot": 0,
                    "prompt_template": PROMPT_TEMPLATE,
                    "option_line_template": "{letter}. {option}",
                    "protocol_sha256": PROTOCOL_SHA256,
                },
                "seed": seed,
                "source": {
                    "dataset": DATASET,
                    "revision": DATASET_REVISION,
                    "split": "test",
                    "selection": "first 32 rows in the pinned parquet",
                    "file_sha256": PARQUET_SHA256,
                    "teacher_revision_note": "not recorded in the publisher's manifest",
                },
            },
            "cases": cases,
        })
    return runs[0], runs[1]


class LocalRunner:
    """Adapt one loaded Agent to the evaluation runner's model keyword."""

    def __init__(self, agent):
        self.agent = agent
        self.calls = 0

    def predict(self, state, questions, *, model=None):
        if model is not None:
            raise ValueError("this example uses a single fixed checkpoint")
        result = self.agent.predict(state, questions, lang="en")
        self.calls += 1
        if self.calls % 8 == 0:
            print(f"Evaluated {self.calls}/{COUNT} cases", file=sys.stderr, flush=True)
        return result


def student_run(reference: dict, report: dict) -> dict:
    """Persist actual report observations for offline CLI replay, without inference."""
    cases = deepcopy(reference["cases"])
    by_id = {case["id"]: case for case in cases}
    for case in cases:
        case["answers"] = {}
    for record in report["cases"]:
        by_id[record["case_id"]]["answers"][record["qid"]] = record["student"]
    return {
        "schema": "laya-agreement-run/1",
        "run_id": report["config"]["run_ids"]["student"],
        "provenance": report["config"]["provenance"]["student"],
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, help="reuse the pinned test parquet")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet = args.parquet or args.output_dir / "teacher-test.parquet"
    reference, repeat = teacher_runs(parquet)
    write_json(args.output_dir / "reference.json", reference)
    write_json(args.output_dir / "repeat.json", repeat)

    import torch
    import transformers

    import laya
    from laya import evals

    torch.set_num_threads(2)
    torch.manual_seed(0)
    provenance = {
        "model": MODEL,
        "revision": MODEL_REVISION,
        "settings": {
            "device": "cpu", "backend": "eager", "lang": "en",
            "min_confidence": None, "torch_threads": 2,
            "laya_version": laya.__version__, "torch_version": str(torch.__version__),
            "transformers_version": transformers.__version__,
        },
        "seed": 0,
    }
    print(f"Loading {MODEL}@{MODEL_REVISION} on CPU", file=sys.stderr, flush=True)
    with laya.load(MODEL, device="cpu", revision=MODEL_REVISION, backend="eager") as agent:
        report = evals.evaluate_agreement(
            LocalRunner(agent), reference, repeat, student_provenance=provenance,
            bootstrap_samples=2000, seed=0,
        )
    student = student_run(reference, report)
    # The saved run must reproduce the full report, including input fingerprints.
    if evals.compare_agreement(reference, repeat, student) != report:
        raise RuntimeError("saved observations did not reproduce the live evaluation")
    write_json(args.output_dir / "student.json", student)
    write_json(args.output_dir / "report.json", report)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
