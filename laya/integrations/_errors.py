"""Exceptions shared by the framework integrations."""
from typing import Any, Dict


class LayaLowConfidenceError(ValueError):
    """Raised by a guarded integration when a decision falls below the confidence threshold.

    Only `LayaCrewRouter` and `LayaSingleSelector` raise it, and only at the caller's request:
    the gate needs `confidence_threshold > 0.0`, a `fallback_agent_index` / `fallback_index`
    absorbs the low-confidence decision when one is set, and with neither the decision is
    returned as it came. So `raise_on_low_confidence=True` (default `False`) is what turns it
    into an exception, and it covers a tool selection as well as agent routing.
    """

    def __init__(self, message: str, confidence: float, threshold: float, raw_decision: Dict[str, Any]):
        super().__init__(message)
        self.confidence = confidence
        self.threshold = threshold
        self.raw_decision = raw_decision
