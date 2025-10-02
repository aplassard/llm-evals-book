"""CLI wrapper for the GitHub issue research agent."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from typing import Optional

from writing_assistant.research_issue_agent import run_research_workflow


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research GitHub issue article checklists and post results.",
    )
    parser.add_argument(
        "--repo",
        help="Repository in owner/name format (default: derived from git origin).",
    )
    parser.add_argument("--issue", required=True, type=int, help="Issue number to process.")
    parser.add_argument(
        "--selection-model",
        default="x-ai/grok-4-fast",
        help="OpenRouter model used for selecting articles (default: x-ai/grok-4-fast).",
    )
    parser.add_argument(
        "--article-model",
        default="x-ai/grok-4-fast",
        help="OpenRouter model used by the article agent (default: x-ai/grok-4-fast).",
    )
    parser.add_argument(
        "--article-max-iterations",
        type=int,
        default=8,
        help="Maximum LangGraph iterations for each article research (default: 8).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging.",
    )
    return parser.parse_args(argv)


def derive_repo_slug(explicit: Optional[str]) -> str:
    if explicit:
        return explicit

    try:
        origin = (
            subprocess.check_output(
                ["git", "remote", "get-url", "origin"],
                text=True,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise EnvironmentError(
            "Unable to determine repository slug; pass --repo explicitly."
        )

    origin = origin.rstrip(".git")
    if origin.startswith("git@github.com:"):
        slug = origin[len("git@github.com:") :]
    elif origin.startswith("https://github.com/"):
        slug = origin[len("https://github.com/") :]
    else:
        raise EnvironmentError(
            f"Could not derive owner/repo from origin URL '{origin}'. Pass --repo."
        )
    return slug


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise EnvironmentError("GITHUB_TOKEN must be set for GitHub API access.")

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise EnvironmentError("OPENROUTER_API_KEY must be set for the language models.")
    if openrouter_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = openrouter_key

    repo_slug = derive_repo_slug(args.repo)

    state = run_research_workflow(
        repo=repo_slug,
        issue_number=args.issue,
        github_token=github_token,
        selection_model=args.selection_model,
        article_model=args.article_model,
        article_iterations=args.article_max_iterations,
    )

    output = {
        "repo": repo_slug,
        "selected_indices": state.selected_indices,
        "results": [
            {
                "article_index": result.article_index,
                "article_name": result.article.name,
                "title": result.structured.get("items", [{}])[0].get("title")
                if isinstance(result.structured, dict)
                else None,
            }
            for result in state.results
        ],
        "comment_posted": state.comment_body is not None,
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
