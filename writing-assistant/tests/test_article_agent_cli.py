from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from writing_assistant.article_agent_cli import main


REQUIRED_ENV_VARS = ("OPENROUTER_API_KEY", "TAVILY_API_KEY")


def _skip_if_missing_keys() -> None:
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        pytest.skip(
            f"Missing required environment variables for integration tests: {', '.join(missing)}"
        )


def run_cli(args: list[str]) -> dict:
    _skip_if_missing_keys()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = main(args)
    assert exit_code == 0
    output = buffer.getvalue()
    return json.loads(output)


def test_known_reference_returns_expected_metadata():
    data = run_cli(
        [
            "--name",
            "Hinton and Salakhutdinov 2006 (Science)",
            "--details",
            "Science paper on deep autoencoders demonstrating MNIST",
            "--status",
            "known",
        ]
    )

    assert isinstance(data, dict), "CLI should return a JSON object"
    items = data.get("items", [])
    assert items, "Items list must not be empty"

    entry = items[0]
    assert entry["title"] == "Reducing the Dimensionality of Data with Neural Networks"
    assert entry["publicationTitle"] == "Science"
    assert entry.get("doi") == "10.1126/science.1127647"

    evidence = data.get("context", {}).get("evidence", [])
    assert evidence, "At least one evidence source should be provided"
    assert any("science.org" in e.get("source", "") for e in evidence)


def test_unknown_reference_produces_candidates():
    data = run_cli(
        [
            "--name",
            "Article/blog: 'There are no new ideas, just new datasets'",
            "--details",
            "Locate the original article attributed to Jack Morris about datasets driving breakthroughs",
            "--status",
            "unknown",
        ]
    )

    container = data if isinstance(data, dict) else {"items": data}
    items = container.get("items", [])
    assert items, "Unknown entries should still return at least one candidate"

    for item in items:
        assert item.get("title"), "Candidate entries must include a title"
        assert item.get("url"), "Candidate entries must include a URL for follow-up"
