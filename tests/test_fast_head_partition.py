"""The fast path must partition the decision head's attention by the head's own head count.

The TileLang fast path only runs on CUDA (importing `laya.fast` pulls in `tilelang`), so this is a
source-level pin -- it reads the files as text, imports nothing, and runs in the ordinary CI lane.
It guards the fix for the head/encoder head-count mismatch: the decision head is built with
`nhead = max(1, d // 64)` (laya/common.py), which differs from the encoder's
`cfg.num_attention_heads` whenever the encoder head_dim != 64, and reusing the encoder partition
for the head's attention silently mixes the wrong channels on any such checkpoint.

Run: python tests/test_fast_head_partition.py
"""
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST = open(os.path.join(_ROOT, "laya", "fast.py"), encoding="utf-8").read()
COMMON = open(os.path.join(_ROOT, "laya", "common.py"), encoding="utf-8").read()

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else "%s %s" % (name, detail))


# The decision head's own head partition is defined, by the same rule common.py builds it with.
check("fast/defines H_head = max(1, self.D // 64)",
      re.search(r"self\.H_head\s*=\s*max\(\s*1\s*,\s*self\.D\s*//\s*64\s*\)", FAST) is not None)
check("fast/defines Dh_head from H_head",
      re.search(r"self\.Dh_head\s*=\s*self\.D\s*//\s*self\.H_head", FAST) is not None)
check("common/head built with nhead = max(1, d // 64)",
      re.search(r"nhead\s*=\s*max\(\s*1\s*,\s*d\s*//\s*64\s*\)", COMMON) is not None,
      "if this rule moves, fast.py's H_head must move with it")

# The head's attention reshape and kernel use the head partition, not the encoder's.
check("fast/head attention reshapes by H_head/Dh_head",
      "qkv.view(B, L, 3, self.H_head, self.Dh_head)" in FAST)
check("fast/head has its own attn kernel builder",
      re.search(r"def attn_k_head\b", FAST) is not None)
check("fast/head kernel is built with H_head/Dh_head",
      re.search(r"attn_kernel\([^)]*self\.H_head,\s*self\.Dh_head", FAST) is not None)

# The encoder's attention is unchanged -- it legitimately uses cfg.num_attention_heads.
check("fast/encoder attention still uses self.H/self.Dh",
      "qkv.view(B, L, 3, self.H, self.Dh)" in FAST)
# And the head no longer reuses the encoder partition anywhere.
check("fast/no head reshape left on the encoder partition",
      FAST.count("qkv.view(B, L, 3, self.H, self.Dh)") == 1,
      "exactly the encoder call should remain")


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
