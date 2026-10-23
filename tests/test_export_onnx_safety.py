"""The shipped ONNX exporter must refuse checkpoints without safe weights, and it must
not put a graph in ``--out-dir`` before the checks that gate it have passed."""

import importlib.util
import io
import os
import sys
import tempfile
import types
from contextlib import redirect_stdout
from pathlib import Path

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "laya_ts_export_onnx", root / "laya-ts" / "scripts" / "export_onnx.py"
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


with tempfile.TemporaryDirectory() as directory:
    model_dir = Path(directory)
    (model_dir / "pytorch_model.bin").write_bytes(b"legacy weights must never be loaded")
    try:
        exporter._checkpoint_weights_path(str(model_dir))
    except SystemExit as error:
        assert "model.safetensors" in str(error)
    else:
        raise AssertionError("a legacy .bin checkpoint must be rejected")

    weights = model_dir / "model.safetensors"
    weights.touch()
    assert exporter._checkpoint_weights_path(str(model_dir)) == str(weights)


class _TracedModule:
    """Base for the encoder/head wrappers ``main`` defines: it only calls ``eval`` on them."""

    def eval(self):
        return self


class _Model:
    """The built network, reduced to what ``main`` touches: five submodules and ``eval``."""

    encoder = head = type_emb = scorer = act_head = None

    def load_state_dict(self, state, strict=False):
        pass

    def eval(self):
        return self

    def float(self):
        return self


def _checkpoint(directory):
    """Write the smallest checkpoint dir ``main`` accepts: safe weights, config, tokenizer."""
    model_dir = Path(directory)
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "rl_agent_config.json").write_text('{"decoder": {"local_attention": 128}}',
                                                     encoding="utf-8")
    (model_dir / "tokenizer.json").write_text('{"vocab": {}}', encoding="utf-8")
    return str(model_dir)


def _fake_torch():
    """A ``torch`` module carrying only what ``main`` calls, with tensor work stubbed out."""
    torch = types.ModuleType("torch")
    torch.long = "int64"
    torch.ones = lambda *shape, dtype=None: shape
    for name in ("nn", "onnx", "export"):
        setattr(torch, name, types.ModuleType("torch." + name))
    torch.nn.Module = _TracedModule
    torch.export.Dim = lambda name, min=None, max=None: (name, min, max)
    return torch


def _fake_safetensors():
    root = types.ModuleType("safetensors")
    root.torch = types.ModuleType("safetensors.torch")
    root.torch.load_file = lambda path: {}
    return root


def _export(model_dir, out_dir, events, *, export_fails=False, verify_fails=False, extra=(),
            sidecars=()):
    """Drive ``main`` against faked torch and read-back seams; return its stdout.

    ``events`` records ``(stage, graph_path, out_dir_existed)`` in call order for every
    trace and every verification, which is what pins the ordering: while a stage is
    recorded, ``--out-dir`` must not exist yet and the path must not be inside it.
    ``SystemExit`` from the exporter propagates, leaving the events it reached.
    """
    torch = _fake_torch()

    def export(module, inputs, path, **kwargs):
        events.append(("trace", path, os.path.isdir(out_dir)))
        if export_fails:
            raise SystemExit("tracing failed")
        with open(path, "wb") as handle:
            handle.write(b"graph")
        # A graph whose initializers pass the protobuf limit is written as `<name>.onnx` plus a
        # `<name>.onnx.data` sidecar that the `.onnx` names by a path relative to its own
        # directory, so the two only stay loadable if they travel together.
        for suffix in sidecars:
            with open(path + suffix, "wb") as handle:
                handle.write(b"weights")

    def verify(enc_path, head_path, *args, **kwargs):
        events.append(("verify", (enc_path, head_path), os.path.isdir(out_dir)))
        if verify_fails:
            raise SystemExit("verification failed: logits max abs diff 2.0e-04 > 1e-4")

    def run_ref(model, head, seq, batch=1):
        return ("hidden", "logits", "act", "pos", "mask", "qtype", "att")

    torch.onnx.export = export
    safetensors = _fake_safetensors()
    names = ("torch", "torch.nn", "torch.onnx", "torch.export",
             "safetensors", "safetensors.torch")
    saved = {name: sys.modules.get(name) for name in names}
    patched = (exporter.build_model, exporter._run_ref, exporter._verify)
    printed = io.StringIO()
    try:
        sys.modules.update({"torch": torch, "torch.nn": torch.nn, "torch.onnx": torch.onnx,
                            "torch.export": torch.export, "safetensors": safetensors,
                            "safetensors.torch": safetensors.torch})
        exporter.build_model = lambda cfg, encoder_dir=None, pretrained=False: _Model()
        exporter._run_ref = run_ref
        exporter._verify = verify
        with redirect_stdout(printed):
            exporter.main(["--model-dir", model_dir, "--out-dir", out_dir,
                           "--seq-len", "16", "--verify-len", "512", *extra])
    finally:
        exporter.build_model, exporter._run_ref, exporter._verify = patched
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return printed.getvalue()


with tempfile.TemporaryDirectory() as directory:
    model_dir = _checkpoint(directory)

    # A run whose 1e-4 check fails must leave no graph behind for the Node/Web providers to
    # load, and must not even create --out-dir: the old code traced straight into it.
    failed = os.path.join(directory, "failed")
    events = []
    try:
        _export(model_dir, failed, events, verify_fails=True)
        raise AssertionError("a failed verification must abort the export")
    except SystemExit as error:
        assert "verification failed" in str(error), error
    assert [stage for stage, _, _ in events] == ["trace", "trace", "verify"], events
    assert not os.path.exists(failed), os.listdir(directory)

    # Same for a trace that dies partway: the first graph is useless without the second.
    torn = os.path.join(directory, "torn")
    events = []
    try:
        _export(model_dir, torn, events, export_fails=True)
        raise AssertionError("a failed trace must abort the export")
    except SystemExit as error:
        assert "tracing failed" in str(error), error
    assert [stage for stage, _, _ in events] == ["trace"], events
    assert not os.path.exists(torn), os.listdir(directory)

    # The passing run: every stage reads a staged graph while --out-dir is still absent,
    # then both graphs land in it. The staging dir is the temp dir the exporter made.
    ok = os.path.join(directory, "model")
    events = []
    printed = _export(model_dir, ok, events)
    assert [stage for stage, _, _ in events] == ["trace", "trace", "verify", "verify", "verify"], events
    assert all(existed is False for _, _, existed in events), events
    staged = os.path.dirname(events[0][1])
    assert staged != ok and os.path.basename(staged).startswith("laya-onnx-"), staged
    assert not os.path.exists(staged), "the staging dir must be cleaned up"
    assert events[2][1] == (os.path.join(staged, "encoder.onnx"),
                            os.path.join(staged, "head.onnx")), events[2]
    assert sorted(os.listdir(ok)) == ["encoder.onnx", "head.onnx", "rl_agent_config.json",
                                      "tokenizer.json"], os.listdir(ok)
    for name in ("encoder.onnx", "head.onnx"):
        with open(os.path.join(ok, name), "rb") as handle:
            assert handle.read() == b"graph", name
    assert printed == ("wrote %s and %s\n" % (os.path.join(ok, "encoder.onnx"),
                                              os.path.join(ok, "head.onnx"))), printed

    # ``--no-verify`` still ships: it skips the checks, not the move.
    skipped = os.path.join(directory, "no-verify")
    events = []
    _export(model_dir, skipped, events, extra=("--no-verify",))
    assert [stage for stage, _, _ in events] == ["trace", "trace"], events
    assert sorted(os.listdir(skipped)) == ["encoder.onnx", "head.onnx", "rl_agent_config.json",
                                           "tokenizer.json"], os.listdir(skipped)

    # Everything the staging directory holds has to move, not the two graph names. torch writes
    # `encoder.onnx.data` beside `encoder.onnx` once the initializers pass the protobuf limit, and
    # the `.onnx` records that sidecar relative to its own directory, so a move of the two names
    # alone published a graph whose external data had just been deleted with the staging dir.
    # onnxruntime refuses to load that: `External data path does not exist:
    # .../onnx-split/english/encoder.onnx.data`, which is how the .NET parity lane found it.
    external = os.path.join(directory, "external-data")
    events = []
    _export(model_dir, external, events, sidecars=(".data",))
    assert sorted(os.listdir(external)) == [
        "encoder.onnx", "encoder.onnx.data", "head.onnx", "head.onnx.data",
        "rl_agent_config.json", "tokenizer.json"], os.listdir(external)
    for name in ("encoder.onnx.data", "head.onnx.data"):
        with open(os.path.join(external, name), "rb") as handle:
            assert handle.read() == b"weights", name
    assert not os.path.exists(os.path.dirname(events[0][1])), "the staging dir must be cleaned up"

    # And a failed run still publishes nothing, sidecars included.
    external_failed = os.path.join(directory, "external-data-failed")
    events = []
    try:
        _export(model_dir, external_failed, events, verify_fails=True, sidecars=(".data",))
        raise AssertionError("a failed verification must abort the export")
    except SystemExit as error:
        assert "verification failed" in str(error), error
    assert not os.path.exists(external_failed), os.listdir(directory)
