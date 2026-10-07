"""Concurrent sequence building must not race on the shared fast tokenizer.

`build_sequence` asks the tokenizer for truncated option text (`truncation=True`), and that mutates
the underlying Rust object through `enable_truncation`. One tokenizer is parsed per checkpoint
directory and shared across Agents, so two threads encoding at once used to raise
`RuntimeError: Already borrowed`. `common.encode_text` serialises those calls; this guards it.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from laya.agent import Agent  # noqa: E402
from laya.common import _TOKENIZE_LOCK, build_sequence  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, (": " + detail) if detail else ""))


VOCAB = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
for i in range(200):
    VOCAB["tok%d" % i] = 5 + i

TOKENIZER = PreTrainedTokenizerFast(
    tokenizer_object=Tokenizer(WordLevel(VOCAB, unk_token="[UNK]")),
    pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
    sep_token="[SEP]", mask_token="[MASK]",
)

# Long enough that truncation does real work on every option, not just enable itself.
QUESTION = {
    "t": "choice",
    "ins": "Which team should handle `message`?",
    "crit": {"billing": " ".join("tok%d" % (i * 3 % 200) for i in range(80)),
             "technical": " ".join("tok%d" % (i * 7 % 200) for i in range(80))},
}
STATE = " ".join("tok%d" % (i * 11 % 200) for i in range(600))

errors, results = [], []


def worker():
    try:
        for _ in range(40):
            seq, markers = build_sequence(TOKENIZER, STATE, QUESTION, 512, 192)
            results.append((tuple(seq), tuple(markers)))
    except Exception as e:  # noqa: BLE001
        errors.append("%s: %s" % (type(e).__name__, e))


threads = [threading.Thread(target=worker) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()

check_true("concurrency/four threads finish without an error", not errors,
           errors[0] if errors else "")
check("concurrency/every thread built the same sequence", len(set(results)), 1)
check("concurrency/all calls produced output", len(results), 4 * 40)


# `Agent.predict_long` tokenizes the whole state before it splits it into windows. That encode has
# to hold the same lock: a scan running next to a `predict()` on the shared tokenizer otherwise
# raises `RuntimeError: Already borrowed` in both calls.
class _LockProbeTok:
    """A tokenizer stand-in that records every encode made while the tokenize lock was free."""

    mask_token = "[MASK]"
    cls_token_id, sep_token_id, mask_token_id = 2, 3, 4

    def __init__(self):
        self.calls, self.unlocked = 0, 0

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        free = []

        def probe():
            # The lock is re-entrant, so only another thread can tell whether it is held.
            if _TOKENIZE_LOCK.acquire(blocking=False):
                _TOKENIZE_LOCK.release()
                free.append(True)

        t = threading.Thread(target=probe)
        t.start()
        t.join()
        self.calls += 1
        self.unlocked += len(free)
        ids = list(range(len(text)))
        return {"input_ids": ids[:max_length] if (truncation and max_length) else ids}

    def decode(self, ids):
        return "w%d_%d" % (ids[0], ids[-1]) if ids else "w"


def _scan_agent():
    a = Agent.__new__(Agent)
    a.cfg = {"max_len": 100, "head_max_len": 20}
    a.tok = _LockProbeTok()
    a._to_internal = staticmethod(Agent._to_internal).__func__
    answer = {"flag": {"type": "noul", "noul": 0.5, "confidence": 0.5, "answer_confidence": 0.5}}
    a.predict_batch = lambda states, questions, **kw: [
        {"model": "stub", "answers": dict(answer), "usage": {"input_tokens": 1}} for _ in states]
    return a


scan_agent = _scan_agent()
scanned = scan_agent.predict_long("x" * 400, {"flag": {"type": "noul", "instructions": "?"}})
check_true("predict_long/the document was split into several windows", scanned["usage"]["windows"] > 1,
           "windows=%r" % scanned["usage"]["windows"])
check_true("predict_long/the tokenizer was called", scan_agent.tok.calls > 0)
check("predict_long/every encode holds the tokenize lock", scan_agent.tok.unlocked, 0)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
sys.exit(1 if FAIL else 0)
