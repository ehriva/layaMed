"""The score-question rule shared by the guardrail wrappers."""
from typing import Any, Dict


def score_violation_probability(answer: Dict[str, Any], levels: int = 0) -> float:
    """Probability that a `score` answer is at or above the middle of its scale.

    `score` is the expected level (0..k-1), not a probability, so a guard threshold cannot be
    compared with it directly. `levels` is the number of levels the question defines; it is only
    used when the answer carries no `probabilities`, and then the normalised expected level stands
    in for the distribution.
    """
    probs = answer.get("probabilities")
    # `probabilities` is read straight from a remote or caller decision answer, so both its outer
    # shape and its values are arbitrary JSON: a non-mapping (a list, string or number) raises at
    # `len`/`.get`, and a value that is not a real number raises inside `float` -- while a boolean
    # would read as 0.0/1.0 silently because `bool` subclasses `int`, turning a spurious `true` into
    # a full-probability violation. Normalise anything that is not a non-empty mapping to the empty
    # distribution, and sum a distribution only when every level in range holds a real number;
    # otherwise fall through to the same normalised expected level the empty path already uses, so a
    # malformed body degrades the guardrail screen instead of crashing it. A well-formed numeric
    # distribution -- including numeric strings, which `float()` already accepted -- sums unchanged.
    if not isinstance(probs, dict) or not probs:
        probs = {}
    k = len(probs) or levels
    if k < 2:
        return 0.0
    if probs:
        total = 0.0
        usable = True
        for i in range(k // 2, k):
            value = probs.get(str(i), 0.0)
            if isinstance(value, bool):
                usable = False
                break
            try:
                total += float(value)
            except (TypeError, ValueError):
                usable = False
                break
        if usable:
            return total
    # No usable distribution: stand in with the normalised expected level. `score` rides the same
    # arbitrary-answer path, so a non-number or a bool cannot be divided -- read it as 0.0 instead.
    score = answer.get("score", 0.0)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return 0.0
    return score / (k - 1)
