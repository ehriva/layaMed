# Agent

`laya.Agent` loads one checkpoint and answers typed questions about a state. `laya.load` is
a shortcut for `Agent(...)`, and `laya.RLAgent` is an alias of `Agent`. `ONNXAgent` runs an
exported ONNX model on CPU; import it from `laya.onnx_agent`.

::: laya.agent.Agent

::: laya.agent.load

::: laya.onnx_agent.ONNXAgent

## Quantized export

`scripts/export_onnx.py --quantize` writes an INT8 weight-only quantized copy beside the fp32
export (`laya.onnx` also produces `laya.int8.onnx`). Dynamic quantization converts the `MatMul`
weights to int8 with the activation scale computed per input at run time, so no calibration
dataset is needed, and `ONNXAgent` loads the result by pointing `onnx_path` at it. On CPU it is
roughly 2x faster than the eager model and ~1.8x faster than the fp32 ONNX graph, and 1.4-2.8x
smaller depending on the checkpoint.

INT8 trades real accuracy, so it is a size/latency option, not a free one — do not use it where
the calibrated probability or confidence matters. Scales are **per-tensor** by default; `--per-channel`
opts into per-channel weights but on the dynamic path that collapses the decision model (agreement
with the eager model dropped to ~32% on the English checkpoint and ~40% on the multilingual one,
vs ~67% / ~83% per-tensor; see issue #790). Even per-tensor drifts noticeably on the larger
checkpoint; accuracy-safe int8 would need QAT or SmoothQuant-style outlier handling. The int8
graph is CPU-only: ONNX Runtime has no INT8 MatMul kernel on the CUDAExecutionProvider, and a GPU
provider silently falls back per node.

```bash
python scripts/export_onnx.py --model convaiinnovations/laya --output laya.onnx --quantize
```

## In-process dynamic quantization

The quantized export above is the int8 path `ONNXAgent` loads. `torch.ao.quantization.quantize_dynamic`
on a live `Agent` is a different operation.

[Issue #1065](https://github.com/NandhaKishorM/laya/issues/1065) reports that the default fbgemm
engine crashes with SIGILL (illegal instruction) on a CPU with SSE4.2 and no AVX2 (laya 0.3.22,
torch 2.14). Check the CPU first. `torch.backends.cpu.get_cpu_capability()` returns `"NO AVX"`
when AVX2 is unavailable, and `"AVX2"` or `"AVX512"` when it is. On Linux,
`grep -m1 flags /proc/cpuinfo` prints the raw flags (`avx2` is absent in the report). The
setting that ran there is qnnpack, applied only to the encoder:

```python
import torch
import torch.nn as nn

torch.backends.quantized.engine = "qnnpack"
torch.ao.quantization.quantize_dynamic(
    agent.model.encoder, {nn.Linear}, dtype=torch.qint8, inplace=True)
```

Encoder-only quantization does run, with either engine the CPU can execute. It is not a speedup
you can assume, and it changes scores. On the machine in #1065, qnnpack took 3.7 s against 3.0 s
in fp32 for 20 states, and one candidate's `noul` went from 0.15 to 0.04. A spot check on the
multilingual checkpoint with torch 2.14.1 (capability `AVX512`, four threads, short states) saw
the same kind of move: qnnpack took 2.1 s against 0.57 s, and one `noul` went from 0.90 to 0.13.
fbgemm was faster on that CPU (0.39 s) and still moved the score, to 0.15. Validate latency and
probabilities on your own data.

Do not quantize the whole `agent.model`. That replaces the decision head's `nn.Linear` layers
with dynamically quantized linears, whose `weight` is a method rather than a tensor. The head
is an `nn.TransformerEncoderLayer`, and its fast path reads `.device` on those weights, so
`predict` raises `AttributeError: 'function' object has no attribute 'device'`. Leave the
head in fp32.
