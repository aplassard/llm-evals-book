from __future__ import annotations

from textwrap import dedent

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from writing_assistant.research_issue_agent import (
    format_comment,
    mark_articles_completed,
    parse_issue_articles,
    IssueArticle,
    ResearchResult,
)
from writing_assistant.zotero_sync import ZoteroSyncResult


def sample_issue_body() -> str:
    return dedent(
        """
        Summary line explaining context.

        ## Articles to Find
        - [ ] Benchmarking Safety Evaluations (known)
          - Confirm venue and DOI
        - [x] Already Completed Reference (unknown)
          - Done previously

        ## Topics to Review
        - [ ] Dataset governance follow-up
        """
    ).strip()


def test_parse_issue_articles_extracts_entries() -> None:
    body = sample_issue_body()
    articles = parse_issue_articles(body)

    assert len(articles) == 2
    first = articles[0]
    assert first.name == "Benchmarking Safety Evaluations"
    assert first.status == "known"
    assert not first.checked
    assert first.details == ["Confirm venue and DOI"]

    second = articles[1]
    assert second.checked


def test_mark_articles_completed_updates_only_specified_indices() -> None:
    body = sample_issue_body()
    updated = mark_articles_completed(body, [0])

    assert "- [x] Benchmarking Safety Evaluations" in updated
    assert "- [x] Already Completed Reference" in updated


def test_format_comment_includes_title_and_sources() -> None:
    article = IssueArticle(
        name="Benchmarking Safety Evaluations",
        status="known",
        details=["Confirm venue and DOI"],
        checked=False,
        line_index=2,
    )

    structured = {
        "items": [
            {
                "title": "Reducing the Dimensionality of Data with Neural Networks",
                "publicationTitle": "Science",
                "url": "https://www.science.org/doi/10.1126/science.1127647",
                "itemType": "journalArticle",
            }
        ],
        "context": {
            "evidence": [
                {"source": "https://www.science.org/article"},
                {"source": "https://scholar.google.com"},
            ]
        },
    }

    result = ResearchResult(
        article_index=0,
        article=article,
        structured=structured,
        raw_output="",
    )
    result.zotero = ZoteroSyncResult(
        key="ABCDEF12",
        select_uri="zotero://select/items/ABCDEF12",
        web_url="https://www.zotero.org/users/123/items/ABCDEF12",
        existed=False,
    )

    comment = format_comment([result])
    assert "Benchmarking Safety Evaluations" in comment
    assert "Science" in comment
    assert "https://www.science.org/doi/10.1126/science.1127647" in comment
    assert "Evidence sources" in comment
    assert "zotero://select/items/ABCDEF12" in comment
