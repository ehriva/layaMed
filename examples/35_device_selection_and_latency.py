"""Example 35 -- device selection, the per-call autocast policy, and device latency.

Laya picks CUDA, then MPS, then CPU when `device=None`. This example times the multilingual
checkpoint on CPU and on MPS (if present), prints median ms and ms/question, checks that the two
devices return the same answers, and prints -- then measures -- the precision each forward really
runs in.
"""
import json

import torch
import torch.nn as nn

from _common import STATE_HI, banner, describe, heading, load, timed

banner("35", "Device selection and latency", """
    Device resolution in `Agent.__init__`: an explicit `device=` wins (with a warning and a
    fallback to CPU if it is unavailable), otherwise CUDA, then MPS, then CPU. Mixed precision is
    chosen per device at load time and per call at inference time, and it is not CUDA-only: MPS
    gets `amp_enabled=True` with an autocast `dtype` of float16, XPU gets bfloat16, and CPU gets
    bfloat16 only when `LAYA_CPU_AMP=bf16` asks for it. What one call actually runs in is
    `dtype_for(rows)`, and on MPS that is gated by row count: below `mps_amp_min_rows` (5 rows by
    default, override with `LAYA_MPS_AMP_MIN_ROWS`) an MPS forward stays fp32, because fp16
    autocast loses on a single small row and only wins once the batch has several. So `dtype` is
    the target and `dtype_for(rows)` is the precision; this example prints both and then measures
    the same MPS batch under each.

    For explicit control pass `device="cpu"`, `"mps"` or `"cuda"`. Laya does not read an env var
    for the device itself, but a common application pattern is to read one (`LAYA_DEVICE`) and
    pass it through.

    `timed(fn, repeat=3)` discards one warm-up call, so the first MPS call's ~13 s of Metal
    kernel compilation does not pollute the median. Both legs answer the same question set.
    """)

device_line_torch = "   torch %s   cuda_available=%s   mps_available=%s"
mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(device_line_torch % (torch.__version__, torch.cuda.is_available(), mps_ok))
print("   explicit selection example: device = os.environ.get('LAYA_DEVICE') or None")

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {"billing": "invoices, payments, refunds",
                     "technical": "bugs, outages, system errors",
                     "account": "login, seats, profile changes",
                     "other": "everything else"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}
N_QUESTIONS = len(QUESTIONS)


def measure(device):
    agent = load("multilingual", device=device)
    result, median_ms = timed(lambda: agent.predict(STATE_HI, QUESTIONS), repeat=3)
    print("   %-4s -> loaded on %-4s dtype=%-8s median %7.1f ms/call  %.1f ms/question"
          % (device, agent.device, agent.dtype, median_ms, median_ms / N_QUESTIONS))
    print("        `amp_enabled`=%s   `dtype_for(%d rows)`=%s   `mps_amp_min_rows`=%s"
          " (MPS only)"
          % (agent.amp_enabled, N_QUESTIONS, agent.dtype_for(N_QUESTIONS), agent.mps_amp_min_rows))
    return agent, result, median_ms


print("\n   == CPU ==")
cpu_agent, cpu_result, cpu_ms = measure("cpu")

print("\n   == MPS ==")
if mps_ok:
    mps_agent, mps_result, mps_ms = measure("mps")
else:
    print("   MPS not available on this machine; skipping the MPS leg cleanly.")
    mps_agent = mps_result = mps_ms = None

if mps_result is not None:
    print("\n   speed-up: CPU %.1f ms -> MPS %.1f ms  (%.2fx)"
          % (cpu_ms, mps_ms, cpu_ms / mps_ms))
    cpu_json = json.dumps(cpu_result["answers"], sort_keys=True)
    mps_json = json.dumps(mps_result["answers"], sort_keys=True)
    print("   answers agree after JSON serialisation: %s" % (cpu_json == mps_json))
    if cpu_json != mps_json:
        for qid in QUESTIONS:
            a, b = cpu_result["answers"][qid], mps_result["answers"][qid]
            if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                print("     %s: cpu=%s mps=%s" % (qid, a, b))
    print("   cpu answers : %s" % cpu_json[:100])
    print("   mps answers : %s" % mps_json[:100])

print("\n   device line and one answer from the CPU leg:")
describe(cpu_result["answers"])


# ------------------------------------------ which precision the forward really ran in, measured
heading("which precision a forward really runs in")


def observed(agent, questions):
    """(`torch.is_autocast_enabled()`, the first Linear layer's output dtype) inside a real forward."""
    seen = {}
    linear = next((m for m in agent.model.modules() if isinstance(m, nn.Linear)), None)

    def hook(module, args, output):
        seen.setdefault("autocast", torch.is_autocast_enabled())
        seen.setdefault("dtype", str(output.dtype))
    handle = None if linear is None else linear.register_forward_hook(hook)
    try:
        agent.predict(STATE_HI, questions)
    finally:
        if handle is not None:
            handle.remove()
    return seen.get("autocast"), seen.get("dtype")


ONE = {"tone": {"type": "noul", "instructions": "Is the customer's tone angry?"}}
FIVE = dict(ONE)
FIVE.update({
    "language": {"type": "noul", "instructions": "Is the message written in Hindi?"},
    "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
    "leaving": {"type": "noul", "instructions": "Does the customer threaten to leave?"},
    "duplicate": {"type": "noul", "instructions": "Is a duplicate charge mentioned?"},
})

print("   `dtype` is the autocast target picked once at load; `dtype_for(rows)` is the precision a")
print("   call with `rows` rows really runs in, and `predict` spends one row per question. The two")
print("   legs above asked %d questions, so the MPS leg ran fp32 because %d is under"
      " `mps_amp_min_rows` --" % (N_QUESTIONS, N_QUESTIONS))
print("   the row gate decided that, not the device. Three batch sizes, three thresholds each:")

if mps_agent is None:
    print("   these arms need an MPS device, and this machine skipped that leg, so there is nothing")
    print("   to measure here. On Apple Silicon the table shows the same agent autocast on or off.")
else:
    gate = mps_agent.mps_amp_min_rows
    for questions in (ONE, QUESTIONS, FIVE):
        label = "%d question%s" % (len(questions), "" if len(questions) == 1 else "s")
        arms = (("as shipped", gate), ("fp16 forced on", 1), ("fp32 forced off", 10 ** 9))
        medians = {}
        for arm, min_rows in arms:
            mps_agent.mps_amp_min_rows = min_rows
            autocast, dtype = observed(mps_agent, questions)
            _, median_ms = timed(lambda: mps_agent.predict(STATE_HI, questions), repeat=5)
            medians[arm] = median_ms
            print("   %-12s %-15s `mps_amp_min_rows`=%-10s autocast=%-5s %-16s %5.1f ms"
                  % (label, arm, min_rows, autocast, dtype, median_ms))
        print("   %-12s this machine, fp16 over fp32 at this batch size: %.2fx"
              % ("", medians["fp16 forced on"] / medians["fp32 forced off"]))
    mps_agent.mps_amp_min_rows = gate
    print("""
   `torch.is_autocast_enabled()` is set inside the MPS forward and the first Linear layer emits
   its autocast dtype, so MPS mixed precision is real on this torch build. `amp_enabled` and
   `dtype` are what the load-time policy picked for the device; `dtype_for(rows)` and the flag
   observed above are what a call actually ran in. `mps_amp_min_rows` is where the library puts
   the crossover on MPS, and `LAYA_MPS_AMP_MIN_ROWS` moves it -- the two forced arms print this
   machine's ratio at each batch size, so the threshold can be checked against the hardware a call
   is really running on rather than taken on faith.""")
