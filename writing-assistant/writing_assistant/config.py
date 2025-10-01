"""Configuration helpers for writing assistant tooling."""

from __future__ import annotations

from pathlib import Path


def resolve_repo_root() -> Path:
    """Return the repository root assuming this file lives in writing-assistant/."""

    return Path(__file__).resolve().parents[2]
