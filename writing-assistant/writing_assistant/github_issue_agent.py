"""Automation agent for creating GitHub issues from cleaned walking notes."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import requests
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from writing_assistant.config import resolve_repo_root

LOGGER = logging.getLogger("writing_assistant.github_issue_agent")
GITHUB_API_BASE = "https://api.github.com"


class CreateIssueInput(BaseModel):
    """Arguments expected by the ``create_issue`` tool."""

    title: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)


class CommentOnPrInput(BaseModel):
    """Arguments for adding a review comment to the pull request."""

    body: str = Field(..., min_length=1)


def format_issue_markdown(note: Dict[str, Any], json_rel_path: str) -> str:
    """Return a Markdown issue body based on the structured note output."""

    summary = (note.get("text_summary") or "No summary provided.").strip()
    articles = note.get("articles_to_find") or []
    topics = note.get("topics_to_review") or []

    def _format_articles(entries: List[Dict[str, Any]]) -> List[str]:
        lines: List[str] = []
        if not entries:
            lines.append("- [ ] _No specific articles collected yet_")
            return lines
        for entry in entries:
            name = entry.get("name") or "Untitled reference"
            details = (entry.get("details") or "").strip()
            status = entry.get("status")
            label_parts = [name]
            if status:
                label_parts.append(f"({status})")
            line = f"- [ ] {' '.join(label_parts).strip()}"
            lines.append(line)
            if details:
                lines.append(f"  - {details}")
        return lines

    def _format_topics(entries: List[Dict[str, Any]]) -> List[str]:
        lines: List[str] = []
        if not entries:
            lines.append("- [ ] _No follow-up topics recorded_")
            return lines
        for entry in entries:
            topic = entry.get("topic") or "Untitled topic"
            detail_lines = entry.get("details") or []
            lines.append(f"- [ ] {topic}")
            for detail in detail_lines:
                if detail:
                    lines.append(f"  - {detail}")
        return lines

    output_lines = [summary, "", "## Articles to Find"]
    output_lines.extend(_format_articles(articles))
    output_lines.extend(["", "## Topics to Review"])
    output_lines.extend(_format_topics(topics))
    output_lines.extend(["", f"**Source JSON:** `{json_rel_path}`"])

    return "\n".join(output_lines).strip() + "\n"


def build_system_prompt() -> str:
    return (
        "You are a meticulous GitHub project assistant. "
        "Use the available tools to file follow-up work items for a cleaned walking note. "
        "Always call `create_issue` exactly once to open an issue containing the prepared body, "
        "then call `comment_on_pr` exactly once to leave a short note on the pull request linking "
        "to the new issue. Do not fabricate tool results."
    )


def build_user_prompt(
    note: Dict[str, Any],
    issue_body: str,
    repo: str,
    pr_number: int,
    json_rel_path: str,
) -> str:
    pretty_json = json.dumps(note, indent=2, ensure_ascii=False)
    instructions = (
        "Repository: {repo}\n"
        "Pull request number: #{pr}\n"
        "Relative JSON path: {path}\n"
        "\n"
        "Structured note data:\n"
        "{json}\n"
        "\n"
        "Prepared issue body (use this verbatim unless you make minor grammatical fixes):\n"
        "{body}\n"
        "\n"
        "Required actions:\n"
        "1. Synthesize a concise, action-oriented issue title capturing the transcript follow-up work.\n"
        "2. Call `create_issue` exactly once using that title and the prepared body.\n"
        "3. After receiving the issue URL, compose a short PR comment explaining that the follow-up "
        "   is tracked there and call `comment_on_pr` exactly once. The comment must mention the issue "
        "   number and link, and restate the JSON path for quick reference.\n"
        "4. End the conversation after both tool calls succeed."
    )
    return instructions.format(
        repo=repo,
        pr=pr_number,
        path=json_rel_path,
        json=pretty_json,
        body=issue_body,
    )


def make_github_tools(
    repo: str,
    pr_number: int,
    token: str,
    state: Dict[str, Any],
) -> List[Any]:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "writing-assistant-automation",
        }
    )

    @tool("create_issue", args_schema=CreateIssueInput)
    def create_issue_tool(title: str, body: str) -> str:
        response = session.post(
            f"{GITHUB_API_BASE}/repos/{repo}/issues",
            json={"title": title, "body": body},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        result = {
            "number": payload.get("number"),
            "html_url": payload.get("html_url"),
            "title": payload.get("title"),
        }
        state["issue"] = result
        return json.dumps(result)

    @tool("comment_on_pr", args_schema=CommentOnPrInput)
    def comment_on_pr_tool(body: str) -> str:
        response = session.post(
            f"{GITHUB_API_BASE}/repos/{repo}/issues/{pr_number}/comments",
            json={"body": body},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        result = {
            "id": payload.get("id"),
            "html_url": payload.get("html_url"),
        }
        state["comment"] = result
        return json.dumps(result)

    return [create_issue_tool, comment_on_pr_tool]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a GitHub issue and PR comment for a cleaned walking note",
    )
    parser.add_argument("--json-path", required=True, type=Path, help="Path to the cleaned note JSON file.")
    parser.add_argument("--repo", required=True, help="Repository in owner/name format.")
    parser.add_argument("--pr-number", required=True, type=int, help="Pull request number to comment on.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=6,
        help="Maximum reasoning/tool iterations for the agent (default: 6).",
    )
    parser.add_argument(
        "--model",
        default="x-ai/grok-4-fast",
        help="OpenRouter model to use for the agent (default: x-ai/grok-4-fast).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for troubleshooting.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {args.json_path}")

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise EnvironmentError("GITHUB_TOKEN must be set to create issues and comments.")

    openrouter_token = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_token:
        raise EnvironmentError("OPENROUTER_API_KEY must be set for the language model.")

    note_data = json.loads(args.json_path.read_text())
    repo_root = resolve_repo_root()
    try:
        json_rel_path = args.json_path.resolve().relative_to(repo_root)
    except ValueError:
        json_rel_path = args.json_path.resolve()

    issue_body = format_issue_markdown(note_data, str(json_rel_path))
    user_prompt = build_user_prompt(
        note=note_data,
        issue_body=issue_body,
        repo=args.repo,
        pr_number=args.pr_number,
        json_rel_path=str(json_rel_path),
    )

    shared_state: Dict[str, Any] = {}
    tools = make_github_tools(
        repo=args.repo,
        pr_number=args.pr_number,
        token=github_token,
        state=shared_state,
    )

    llm = ChatOpenAI(
        model=args.model,
        temperature=0,
        timeout=120,
        api_key=None,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/andrewplassard/llm-evals-book",
            "X-Title": "Writing Assistant Issue Agent",
        },
    )

    system_prompt = build_system_prompt()
    agent = create_react_agent(
        llm,
        tools,
        prompt=SystemMessage(content=system_prompt),
    )

    state: Dict[str, Any] = {
        "input": user_prompt,
        "max_iterations": args.max_iterations,
    }

    LOGGER.info(
        "Running GitHub issue agent for repo=%s, pr=%s, json=%s",
        args.repo,
        args.pr_number,
        json_rel_path,
    )

    result = agent.invoke(state)
    LOGGER.debug("Agent result: %s", result)

    if "issue" not in shared_state:
        raise RuntimeError("Agent completed without creating an issue via the tool.")
    if "comment" not in shared_state:
        raise RuntimeError("Agent completed without commenting on the pull request.")

    issue_info = shared_state["issue"]
    comment_info = shared_state["comment"]

    LOGGER.info(
        "Created issue #%s (%s) and comment %s",
        issue_info.get("number"),
        issue_info.get("html_url"),
        comment_info.get("html_url"),
    )

    print(json.dumps({
        "issue": issue_info,
        "comment": comment_info,
        "final": result.get("output"),
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
