"""Utility functions for cleaning and validating model outputs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class FormatIssue:
    field: str
    message: str


def load_model_payload(raw_text: str) -> Dict[str, Any]:
    """Parse a model response that is expected to be JSON."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:  # pragma: no cover - placeholder behavior
        raise ValueError("Model response was not valid JSON") from exc


def validate_payload(payload: Dict[str, Any]) -> List[FormatIssue]:
    """Check for required keys and value types."""
    issues: List[FormatIssue] = []
    if "summary" not in payload:
        issues.append(FormatIssue(field="summary", message="Missing required key."))
    if not isinstance(payload.get("scores", []), list):
        issues.append(FormatIssue(field="scores", message="Expected list of scores."))
    return issues


def heal_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Apply light-weight fixes to common issues."""
    healed = dict(payload)
    if "summary" not in healed:
        healed["summary"] = "TODO: add summary"
    scores = healed.get("scores")
    if not isinstance(scores, list):
        healed["scores"] = []
    return healed
