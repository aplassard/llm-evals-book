"""Simple classification helper used in the Conformance & Control Checks chapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    accepted: bool


def classify(probabilities: Dict[str, float], threshold: float = 0.6) -> ClassificationResult:
    """Return the highest probability label and whether it meets the acceptance threshold."""
    label, probability = max(probabilities.items(), key=lambda item: item[1])
    return ClassificationResult(label=label, confidence=probability, accepted=probability >= threshold)


def audit_trail(result: ClassificationResult) -> str:
    """Generate an audit string for logging or downstream review."""
    status = "accepted" if result.accepted else "rejected"
    return f"label={result.label} confidence={result.confidence:.2f} status={status}"


def batch_classify(batch: Iterable[Dict[str, float]], threshold: float = 0.6) -> Tuple[ClassificationResult, ...]:
    """Evaluate a batch of probability dictionaries."""
    return tuple(classify(probabilities, threshold=threshold) for probabilities in batch)
