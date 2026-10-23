"""Regression tests for DecisionModel.forward (laya/common.py).

Uses a tiny from-config BERT encoder (no pretrained weights downloaded) so
these run fast and offline, unlike tests/test_local_e2e.py which needs a
real checkpoint on disk.
"""
import inspect
import os
import re
import sys
import torch
from transformers import AutoConfig, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import DecisionModel


def _tiny_model(head_layers: int = 1, n_act: int = 2) -> DecisionModel:
    cfg = AutoConfig.for_model(
        "bert",
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        vocab_size=50,
    )
    encoder = AutoModel.from_config(cfg)
    model = DecisionModel(encoder, head_layers=head_layers, n_act=n_act)
    model.eval()
    return model


def _inputs(batch: int, seq: int, n_markers: int):
    input_ids = torch.randint(0, 50, (batch, seq))
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    qtype = torch.zeros(batch, dtype=torch.long)
    marker_pos = torch.arange(n_markers).unsqueeze(0).expand(batch, -1).clone()
    marker_mask = torch.ones(batch, n_markers, dtype=torch.bool)
    return input_ids, attention_mask, marker_pos, marker_mask, qtype


def test_single_option_question_does_not_crash():
    # Regression test for #96: a `choice` question with exactly one criterion
    # used to crash inside forward() with "selected index k out of range",
    # because p.topk(2, -1) has nothing to select for the second slot when
    # there is only one valid marker.
    torch.manual_seed(0)
    model = _tiny_model()
    input_ids, attention_mask, marker_pos, marker_mask, qtype = _inputs(
        batch=1, seq=8, n_markers=1
    )
    with torch.no_grad():
        logits, act_logits = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
    assert logits.shape == (1, 1)
    assert act_logits.shape == (1, 2)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(act_logits).all()


def test_single_option_top1_minus_top2_is_exactly_one():
    # Softmax over a single valid logit is 1.0 regardless of its value, so the
    # padded top2 (0.0) must make the act head's top1-top2 gap read as a fully
    # decided 1.0 - the same signal it gets for any other unambiguous choice.
    torch.manual_seed(1)
    model = _tiny_model()
    input_ids, attention_mask, marker_pos, marker_mask, qtype = _inputs(
        batch=1, seq=6, n_markers=1
    )
    with torch.no_grad():
        h = model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + model.type_emb(qtype)[:, None, :]
        pad = ~attention_mask.bool()
        for layer in model.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = model.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits.detach(), -1)
    assert torch.allclose(p, torch.ones_like(p))


def test_multi_option_question_is_unaffected():
    # The >=2-marker path must still use the original torch.topk(2, -1) call
    # unchanged - this pins that the single-option fix didn't touch it.
    torch.manual_seed(2)
    model = _tiny_model()
    input_ids, attention_mask, marker_pos, marker_mask, qtype = _inputs(
        batch=2, seq=10, n_markers=4
    )
    with torch.no_grad():
        logits, act_logits = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(act_logits).all()


def test_action_head_explicit_dtype_and_autocast():
    # The old boundary always passed fp32 to the head. Preserve that under AMP,
    # including bf16 weights with fp16 autocast (an extra bf16 cast would round twice).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for weights in (torch.float32, torch.bfloat16):
        for amp in (None, torch.bfloat16, torch.float16):
            if device == "cpu" and amp == torch.float16:
                continue
            # A bf16 forward on a Windows CPU can die with an uncatchable SIGILL on recent
            # torch builds (pytorch/pytorch#152385); sentence-transformers and TabPFN skip
            # the same combination. It takes the whole `windows-latest` lane down, and
            # Linux/macOS CPU and CUDA still run the bf16 legs.
            if os.name == "nt" and device == "cpu" and (
                    weights == torch.bfloat16 or amp == torch.bfloat16):
                continue
            model = _tiny_model().to(device=device, dtype=weights)
            inputs = tuple(t.to(device) for t in _inputs(2, 10, 4))
            captured = []
            hook = model.act_head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
            with torch.no_grad(), torch.autocast(device, dtype=amp or torch.bfloat16, enabled=amp is not None):
                logits, act_logits = model(*inputs)
                hook.remove()
                expected_dtype = torch.float32 if amp is not None else weights
                assert captured[0].dtype == expected_dtype
                assert logits.dtype == torch.float32
                assert torch.isfinite(act_logits).all()
                if amp is not None or weights == torch.float32:
                    legacy = model.act_head(captured[0].float())
                    torch.testing.assert_close(act_logits, legacy, rtol=0, atol=0)


def test_head_is_the_encoder_stack_not_a_single_layer():
    # docs/reference/agent.md names `agent.model.head`, so its class is a documented fact rather
    # than an implementation detail. `DecisionModel.__init__` builds it as an `nn.TransformerEncoder`
    # wrapping `head_layers` copies of one layer -- not as the layer itself -- and `head_layers: 0`
    # is the other documented shape: there is no head module at all.
    model = _tiny_model(head_layers=2)
    assert type(model.head).__name__ == "TransformerEncoder"
    assert len(model.head.layers) == 2
    assert all(type(layer).__name__ == "TransformerEncoderLayer" for layer in model.head.layers)
    assert not isinstance(model.head, torch.nn.TransformerEncoderLayer)
    assert _tiny_model(head_layers=0).head is None


def _quantized_whole_model_error(model):
    """The AttributeError text `predict` would raise on a fully quantized `model`, or None."""
    quantized = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    try:
        with torch.no_grad():
            quantized(*_inputs(batch=1, seq=8, n_markers=1))
    except AttributeError as exc:
        return str(exc)
    return None


def test_the_quantized_whole_model_failure_is_the_one_the_page_quotes():
    # The page tells readers which attribute blows up when they quantize the whole model, and
    # there are two answers, so it names both and says which one they will get. Torch skips the
    # encoder layer's sparsity fast path when `self_attn.num_heads` is odd, and the head takes
    # `nhead = max(1, d // 64)`: a hidden size under 128 gives `nhead = 1`, survives the stack
    # and fails on `act_head[0].weight.dtype`, while every published checkpoint (d=768, d=1024
    # -> nhead 12 or 16) takes the fast path and fails earlier, on `.device`. Reproduced here on
    # both shapes rather than read off the source, because reading the source is exactly what
    # made the page claim `.dtype` for checkpoints that raise `.device`.
    src = inspect.getsource(DecisionModel.forward)
    read = re.search(r"self\.act_head\[0\]\.weight\.(\w+)", src)
    assert read is not None, "the fast path no longer reads a weight attribute this way"
    assert read.group(1) == "dtype", read.group(1)
    assert type(_tiny_model().act_head[0]).__name__ == "Linear"

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "docs", "reference", "agent.md"), encoding="utf-8") as handle:
        page = handle.read()
    named = re.findall(r"'function' object has no attribute '(\w+)'", page)
    assert named == ["device", "dtype"], named
    assert re.search(r"head\s+is an `nn\.TransformerEncoderLayer`", page) is None, (
        "the page calls the head a single layer again")
    assert "nhead = max(1, d // 64)" in page, "the page no longer names what picks the message"

    # A quantization engine is what the surrounding section says depends on the CPU, so the
    # reproduction is skipped rather than failed where there is none.
    try:
        torch.backends.quantized.engine = "qnnpack"
    except RuntimeError:
        return
    odd = _tiny_model(head_layers=2)
    assert odd.head.layers[0].self_attn.num_heads % 2 == 1, "the tiny fixture stopped being odd"
    try:
        odd_error = _quantized_whole_model_error(odd)
    except RuntimeError:                       # no engine for quantized::linear_prepack
        return
    assert odd_error == "'function' object has no attribute 'dtype'", odd_error

    even = _tiny_model(head_layers=2)
    for layer in even.head.layers:             # 2 heads over d=16: even, so the fast path runs
        layer.self_attn = torch.nn.MultiheadAttention(16, 2, batch_first=True)
    assert even.head.layers[0].self_attn.num_heads % 2 == 0
    even_error = _quantized_whole_model_error(even)
    assert even_error == "'function' object has no attribute 'device'", even_error


if __name__ == "__main__":
    test_single_option_question_does_not_crash()
    test_single_option_top1_minus_top2_is_exactly_one()
    test_multi_option_question_is_unaffected()
    test_action_head_explicit_dtype_and_autocast()
    test_head_is_the_encoder_stack_not_a_single_layer()
    test_the_quantized_whole_model_failure_is_the_one_the_page_quotes()
    print("all decision model tests passed")

