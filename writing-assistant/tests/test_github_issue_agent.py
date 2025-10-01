from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from writing_assistant.github_issue_agent import build_user_prompt, format_issue_markdown


@pytest.fixture()
def sample_note(tmp_path: Path) -> tuple[dict, Path]:
    data = {
        "text_summary": "Outline follow-up actions from the latest walking note.",
        "notes": ["Example note"],
        "articles_to_find": [
            {
                "name": "Benchmarking Safety Evaluations",
                "details": "Confirm publication venue and DOI",
                "status": "known",
            },
            {
                "name": "Emergent Reasoning Techniques",
                "details": "Locate original preprint",
                "status": "unknown",
            },
        ],
        "topics_to_review": [
            {
                "topic": "Dataset governance",
                "details": ["Summarise policy guidance", "Collect recent academic coverage"],
            }
        ],
    }
    json_path = tmp_path / "cleaned_notes" / "demo.json"
    json_path.parent.mkdir(parents=True)
    json_path.write_text(json.dumps(data))
    return data, json_path


def test_format_issue_markdown_structure(sample_note: tuple[dict, Path]) -> None:
    data, json_path = sample_note
    body = format_issue_markdown(data, "cleaned_notes/demo.json")

    assert body.startswith("Outline follow-up actions"), "Summary should lead the issue body"
    assert "## Articles to Find" in body
    assert "- [ ] Benchmarking Safety Evaluations" in body
    assert "## Topics to Review" in body
    assert "- [ ] Dataset governance" in body
    assert "cleaned_notes/demo.json" in body.splitlines()[-1]


def test_build_user_prompt_references_tools(sample_note: tuple[dict, Path]) -> None:
    data, json_path = sample_note
    body = format_issue_markdown(data, "cleaned_notes/demo.json")

    prompt = build_user_prompt(
        note=data,
        issue_body=body,
        repo="andrewplassard/llm-evals-book",
        pr_number=42,
        json_rel_path="cleaned_notes/demo.json",
    )

    assert "create_issue" in prompt
    assert "comment_on_pr" in prompt
    assert "cleaned_notes/demo.json" in prompt
    assert "Benchmarking Safety Evaluations" in prompt
    assert "Pull request number: #42" in prompt
