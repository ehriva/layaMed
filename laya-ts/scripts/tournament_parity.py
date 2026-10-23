"""Record a local PyTorch reference for the TypeScript tournament example.

Export the same checkpoint with laya-ts/scripts/export_onnx.py first, then run from the
repository root (with the checkout installed):

    python laya-ts/scripts/tournament_parity.py --model-dir CHECKPOINT \
        --onnx-dir SPLIT_EXPORT --fixture /tmp/tournament-python.json
    node laya-ts/examples/tournament.mjs --model-dir SPLIT_EXPORT \
        --fixture /tmp/tournament-python.json --report /tmp/tournament-node.json

Only inputs are reused from the Java golden fixture; its predictions and its
single-graph ONNX results are never used as the reference. Both runtimes execute
real models independently. This is a small parity experiment, not an accuracy or
throughput benchmark. Each path is warmed once before its measured call(s).
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from copy import deepcopy
from pathlib import Path

SCHEMA = "laya-ts-tournament-parity/1"
CASES = Path(__file__).resolve().parents[2] / "laya-java/fixtures/tournament_golden.json"
OPTIONS = {"lang": "en", "max_len": 512, "head_max_len": 192, "min_confidence": 0.9}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class Recording:
    def __init__(self, agent):
        self.agent, self.calls = agent, []

    def predict(self, state, questions, **kwargs):
        call = {"state": deepcopy(state), "questions": deepcopy(questions),
                "kwargs": deepcopy(kwargs)}
        started = time.perf_counter()
        result = self.agent.predict(state, questions, **kwargs)
        call.update(answers=deepcopy(result["answers"]), usage=deepcopy(result["usage"]),
                    elapsed_ms=(time.perf_counter() - started) * 1000)
        self.calls.append(call)
        return result


def measure(agent, run, repeats):
    run(Recording(agent))  # warm this path; excluded from reported timings
    elapsed, reference, result = [], None, None
    for _ in range(repeats):
        recorder = Recording(agent)
        started = time.perf_counter()
        result = run(recorder)
        elapsed.append((time.perf_counter() - started) * 1000)
        if reference is None:
            reference = recorder.calls
        elif [call["answers"] for call in recorder.calls] != [call["answers"] for call in reference]:
            raise RuntimeError("the reference changed across repeated calls")
    return {"calls": reference, "result": result, "model_calls": len(reference),
            "elapsed_ms": elapsed, "median_ms": statistics.median(elapsed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="local safetensors checkpoint")
    parser.add_argument("--onnx-dir", type=Path, required=True, help="split export of that checkpoint")
    parser.add_argument("--fixture", type=Path, required=True, help="write the new Python reference JSON")
    parser.add_argument("--repeats", type=int, default=1, help="measured calls per path after one warmup")
    args = parser.parse_args()
    checkpoint, exported = args.model_dir.resolve(), args.onnx_dir.resolve()
    if not checkpoint.is_dir() or not exported.is_dir():
        parser.error("--model-dir and --onnx-dir must be existing local directories")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.fixture.resolve() == CASES or args.fixture.resolve().is_relative_to(exported):
        parser.error("--fixture must be outside the export directory and must not overwrite the input cases")
    required = ("encoder.onnx", "head.onnx", "rl_agent_config.json", "tokenizer.json")
    for name in required:
        if not (exported / name).is_file():
            parser.error("split export is missing " + name)
    # Agent reads tokenizer/, while the split exporter may use a root-level copy.
    source_names = ("model.safetensors", "rl_agent_config.json", "tokenizer/tokenizer.json")
    for name in source_names:
        if not (checkpoint / name).is_file():
            parser.error("local Python checkpoint is missing " + name)
    source_hashes = {name: digest(checkpoint / name) for name in source_names}
    for name in ("encoder/config.json", "tokenizer.json"):
        source_hashes[name] = digest(checkpoint / name) if (checkpoint / name).is_file() else None
    # Include external tensors as well as the graphs, regardless of their filenames.
    export_hashes = {path.relative_to(exported).as_posix(): digest(path)
                     for path in sorted(exported.rglob("*")) if path.is_file()}
    for source, target in (("rl_agent_config.json", "rl_agent_config.json"),
                           ("tokenizer/tokenizer.json", "tokenizer.json")):
        if source_hashes[source] != export_hashes[target]:
            parser.error("checkpoint and split export differ: " + target)

    source_cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    cases = [{key: deepcopy(case[key]) for key in ("name", "state", "questions", "group_size")}
             for case in source_cases]
    # The same 40 labels at size 4 exercise two narrowing rounds without new test text.
    two_rounds = deepcopy(next(case for case in cases if case["name"] == "banking-default-group"))
    two_rounds.update(name="banking-two-rounds", group_size=4)
    cases.append(two_rounds)

    # Loading must use local checkpoint/cache files even when optional config is absent.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    import transformers

    import laya

    torch.set_num_threads(2)
    torch.manual_seed(0)
    document = {
        "schema": SCHEMA,
        "checkpoint_sha256": source_hashes,
        "onnx_sha256": export_hashes,
        "case_source_sha256": digest(CASES),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "laya": laya.__version__, "torch": str(torch.__version__),
                        "transformers": transformers.__version__, "torch_threads": 2,
                        "device": "cpu", "backend": "eager"},
        "predict_kwargs": OPTIONS,
        "repeats": args.repeats,
        "cases": [],
    }
    with laya.load(str(checkpoint), device="cpu", backend="eager") as agent:
        for case in cases:
            state, questions = case["state"], case["questions"]
            group = 16 if case["group_size"] is None else case["group_size"]
            plain = measure(agent, lambda recorder, s=state, q=questions: recorder.predict(s, q, **OPTIONS),
                            args.repeats)
            tournament = measure(
                agent, lambda recorder, s=state, q=questions, g=group: laya.predict_tournament(
                    recorder, s, q, group_size=g, **OPTIONS), args.repeats)
            document["cases"].append(dict(case, plain=plain, tournament=tournament))
            print(json.dumps({"case": case["name"], "plain_ms": plain["median_ms"],
                              "tournament_ms": tournament["median_ms"],
                              "model_calls": tournament["model_calls"]}), flush=True)
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
