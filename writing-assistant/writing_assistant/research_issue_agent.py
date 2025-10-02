"""Agent for researching GitHub issue checklists of articles to find."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from writing_assistant.article_agent_cli import run_agent
from writing_assistant.graph import ArticleTask, build_graph, build_user_prompt
from writing_assistant.zotero_sync import (
    ZoteroSyncError,
    ZoteroSyncResult,
    sync_structured_item,
)


LOGGER = logging.getLogger("writing_assistant.research_issue_agent")


class GitHubClient:
    """Minimal GitHub REST wrapper for issues."""

    def __init__(self, token: str, repo: str) -> None:
        self.repo = repo
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "writing-assistant-research-issue-agent",
            }
        )
        self._session = session

    def _url(self, path: str) -> str:
        return f"https://api.github.com/repos/{self.repo}{path}"

    def get_issue(self, issue_number: int) -> Dict[str, Any]:
        response = self._session.get(self._url(f"/issues/{issue_number}"), timeout=30)
        response.raise_for_status()
        return response.json()

    def comment_on_issue(self, issue_number: int, body: str) -> Dict[str, Any]:
        response = self._session.post(
            self._url(f"/issues/{issue_number}/comments"),
            json={"body": body},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def update_issue_body(self, issue_number: int, body: str) -> Dict[str, Any]:
        response = self._session.patch(
            self._url(f"/issues/{issue_number}"),
            json={"body": body},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()


@dataclass
class IssueArticle:
    name: str
    status: str
    details: List[str]
    checked: bool
    line_index: int


@dataclass
class ResearchResult:
    article_index: int
    article: IssueArticle
    structured: Dict[str, Any]
    raw_output: str
    zotero: Optional[ZoteroSyncResult] = None
    zotero_error: Optional[str] = None


@dataclass
class IssueTopic:
    topic: str
    details: List[str]
    checked: bool
    line_index: int


@dataclass
class TopicResearchResult:
    topic_index: int
    topic: IssueTopic
    structured: Dict[str, Any]
    raw_output: str
    zotero: List[ZoteroSyncResult] = field(default_factory=list)
    zotero_errors: List[str] = field(default_factory=list)


@dataclass
class ResearchState:
    repo: str
    issue_number: int
    issue_body: Optional[str] = None
    issue_title: Optional[str] = None
    summary_text: str = ""
    articles: List[IssueArticle] = field(default_factory=list)
    topics: List[IssueTopic] = field(default_factory=list)
    selected_indices: List[int] = field(default_factory=list)
    selected_topic_indices: List[int] = field(default_factory=list)
    results: List[ResearchResult] = field(default_factory=list)
    topic_results: List[TopicResearchResult] = field(default_factory=list)
    comment_body: Optional[str] = None
    updated_issue_body: Optional[str] = None
    article_comment: Optional[str] = None
    topic_comment: Optional[str] = None


ARTICLE_LINE = re.compile(r"^- \[( |x)\] (.+)$", re.IGNORECASE)
STATUS_PATTERN = re.compile(r"\((known|unknown)\)", re.IGNORECASE)


def parse_issue_articles(body: str) -> List[IssueArticle]:
    """Extract article checklist entries from the issue body."""

    lines = body.splitlines()
    articles: List[IssueArticle] = []
    in_section = False
    current_details: List[str] = []
    current_article: Optional[IssueArticle] = None

    def finalize_current() -> None:
        nonlocal current_article, current_details
        if current_article:
            current_article.details = current_details
            articles.append(current_article)
        current_article = None
        current_details = []

    for idx, raw_line in enumerate(lines):
        line = raw_line.rstrip()
        heading_match = re.match(r"^##\s+(.+)$", line)
        if heading_match:
            heading = heading_match.group(1).strip().lower()
            if current_article:
                finalize_current()
            in_section = heading.startswith("articles to find")
            continue

        if not in_section:
            continue

        match = ARTICLE_LINE.match(line.strip())
        if match:
            if current_article:
                finalize_current()
            checked = match.group(1).lower() == "x"
            remainder = match.group(2).strip()
            status_match = STATUS_PATTERN.search(remainder)
            status = status_match.group(1).lower() if status_match else "unknown"
            name = STATUS_PATTERN.sub("", remainder).strip()
            current_article = IssueArticle(
                name=name,
                status=status,
                details=[],
                checked=checked,
                line_index=idx,
            )
            continue

        if current_article and line.lstrip().startswith("-"):
            detail_text = line.strip().lstrip("- ")
            current_details.append(detail_text)

    if current_article:
        finalize_current()

    return articles


TOPIC_LINE = re.compile(r"^- \[( |x)\] (.+)$", re.IGNORECASE)


def parse_issue_topics(body: str) -> List[IssueTopic]:
    lines = body.splitlines()
    topics: List[IssueTopic] = []
    in_section = False
    current_details: List[str] = []
    current_topic: Optional[IssueTopic] = None

    def finalize_current() -> None:
        nonlocal current_topic, current_details
        if current_topic:
            current_topic.details = current_details
            topics.append(current_topic)
        current_topic = None
        current_details = []

    for idx, raw_line in enumerate(lines):
        line = raw_line.rstrip()
        heading_match = re.match(r"^##\s+(.+)$", line)
        if heading_match:
            heading = heading_match.group(1).strip().lower()
            if current_topic:
                finalize_current()
            in_section = heading.startswith("topics to review")
            continue

        if not in_section:
            continue

        match = TOPIC_LINE.match(line.strip())
        if match:
            if current_topic:
                finalize_current()
            checked = match.group(1).lower() == "x"
            topic_name = match.group(2).strip()
            current_topic = IssueTopic(
                topic=topic_name,
                details=[],
                checked=checked,
                line_index=idx,
            )
            continue

        if current_topic and line.lstrip().startswith("-"):
            detail_text = line.strip().lstrip("- ")
            current_details.append(detail_text)

    if current_topic:
        finalize_current()

    return topics


def mark_articles_completed(body: str, indices: List[int]) -> str:
    """Return an updated issue body with the specified article indices checked."""

    if not indices:
        return body

    lines = body.splitlines()
    target = set(indices)
    articles = parse_issue_articles(body)

    for pos, article in enumerate(articles):
        if pos in target and not article.checked:
            line = lines[article.line_index]
            lines[article.line_index] = line.replace("[ ]", "[x]", 1)

    return "\n".join(lines)


def mark_topics_completed(body: str, indices: List[int]) -> str:
    if not indices:
        return body

    lines = body.splitlines()
    target = set(indices)
    topics = parse_issue_topics(body)

    for pos, topic in enumerate(topics):
        if pos in target and not topic.checked:
            line = lines[topic.line_index]
            lines[topic.line_index] = line.replace("[ ]", "[x]", 1)

    return "\n".join(lines)


def format_comment(results: List[ResearchResult]) -> str:
    if not results:
        return (
            "No unchecked articles required research at this time. "
            "The checklist remains unchanged."
        )

    parts = ["### Article Research Results"]
    for result in results:
        article = result.article
        parts.append(
            f"- **{article.name}** (_status: {article.status}_):"\
        )
        items = result.structured.get("items", [])
        if isinstance(items, list) and items:
            top = items[0] if isinstance(items[0], dict) else {}
            title = top.get("title") or "Untitled"
            url = top.get("url") or top.get("doi")
            pub = top.get("publicationTitle") or top.get("conferenceName") or top.get("publisher")
            parts.append(f"  - Title: {title}")
            if pub:
                parts.append(f"  - Venue: {pub}")
            if url:
                parts.append(f"  - Link: {url}")
        else:
            parts.append("  - No structured entries were returned.")
        context = result.structured.get("context", {}) if isinstance(result.structured, dict) else {}
        evidence = context.get("evidence") if isinstance(context, dict) else None
        if isinstance(evidence, list) and evidence:
            sources = [ev.get("source") for ev in evidence if isinstance(ev, dict) and ev.get("source")]
            if sources:
                parts.append("  - Evidence sources: " + ", ".join(sources))

        if result.zotero:
            label = "Existing Zotero item" if result.zotero.existed else "Added to Zotero"
            zotero_line = f"  - Zotero: [{label}]({result.zotero.select_uri})"
            if result.zotero.web_url:
                zotero_line += f" ([Web]({result.zotero.web_url}))"
            parts.append(zotero_line)
        elif result.zotero_error:
            parts.append(f"  - Zotero sync failed: {result.zotero_error}")

    return "\n".join(parts)


def extract_issue_summary(body: str) -> str:
    sections = re.split(r"^##\s+", body, maxsplit=1, flags=re.MULTILINE)
    summary = sections[0].strip()
    return summary


def format_topics_comment(results: List[TopicResearchResult]) -> str:
    if not results:
        return ""

    parts = ["### Topic Research Results"]
    for result in results:
        topic = result.topic
        parts.append(f"- **{topic.topic}**:")
        if topic.details:
            parts.append("  - Focus points:")
            for detail in topic.details:
                parts.append(f"    * {detail}")

        items = result.structured.get("items", []) if isinstance(result.structured, dict) else []
        if isinstance(items, list) and items:
            parts.append("  - References:")
            for idx, entry in enumerate(items, 1):
                if not isinstance(entry, dict):
                    continue
                title = entry.get("title") or "Untitled"
                venue = entry.get("publicationTitle") or entry.get("conferenceName") or entry.get("publisher")
                link = entry.get("url") or entry.get("doi")
                line = f"    {idx}. {title}"
                if venue:
                    line += f" — {venue}"
                if link:
                    line += f" ({link})"
                parts.append(line)
        else:
            parts.append("  - No references were produced.")

        if result.zotero:
            parts.append("  - Zotero items:")
            for sync in result.zotero:
                label = "Existing" if sync.existed else "Added"
                line = f"    * {label}: {sync.select_uri}"
                if sync.web_url:
                    line += f" (web: {sync.web_url})"
                parts.append(line)
        if result.zotero_errors:
            parts.append("  - Zotero sync issues:")
            for err in result.zotero_errors:
                parts.append(f"    * {err}")

        notes = []
        if isinstance(result.structured, dict):
            context = result.structured.get("context", {})
            if isinstance(context, dict):
                notes_field = context.get("notes") or context.get("summary")
                if isinstance(notes_field, list):
                    notes.extend(str(entry) for entry in notes_field)
                elif isinstance(notes_field, str) and notes_field.strip():
                    notes.append(notes_field.strip())
        if notes:
            parts.append("  - Notes:")
            for note in notes:
                parts.append(f"    * {note}")

    return "\n".join(parts)


def select_articles_with_llm(
    llm: ChatOpenAI,
    issue_title: str,
    summary: str,
    articles: List[IssueArticle],
) -> List[int]:
    unchecked = [
        (idx, article)
        for idx, article in enumerate(articles)
        if not article.checked
    ]
    if not unchecked:
        return []

    prompt_lines = [
        "You review issue checklists and decide which articles should be researched now.",
        f"Issue title: {issue_title}",
        "Issue summary:",
        summary or "(none)",
        "\nUnchecked articles:",
    ]
    for position, article in unchecked:
        detail_text = "; ".join(article.details) if article.details else "(no extra details)"
        prompt_lines.append(
            f"- index={position}: name='{article.name}' status={article.status} details={detail_text}"
        )

    prompt_lines.append(
        "\nRespond with JSON: {\"selected\": [indices you plan to research now]}.")
    message = "\n".join(prompt_lines)

    response = llm.invoke([HumanMessage(content=message)])
    try:
        payload = json.loads(response.content)
        indices = payload.get("selected", [])
        return [int(idx) for idx in indices if isinstance(idx, int)]
    except (json.JSONDecodeError, ValueError, TypeError):
        return [idx for idx, _ in unchecked]


def select_topics_with_llm(
    llm: ChatOpenAI,
    issue_title: str,
    summary: str,
    topics: List[IssueTopic],
) -> List[int]:
    unchecked = [
        (idx, topic)
        for idx, topic in enumerate(topics)
        if not topic.checked
    ]
    if not unchecked:
        return []

    prompt_lines = [
        "You triage follow-up topics and decide which ones should be researched now.",
        f"Issue title: {issue_title}",
        "Issue summary:",
        summary or "(none)",
        "\nUnchecked topics:",
    ]
    for position, topic in unchecked:
        detail_text = "; ".join(topic.details) if topic.details else "(no extra details)"
        prompt_lines.append(
            f"- index={position}: topic='{topic.topic}' details={detail_text}"
        )

    prompt_lines.append(
        "\nRespond with JSON: {\"selected\": [indices you plan to research now]}.")
    message = "\n".join(prompt_lines)

    response = llm.invoke([HumanMessage(content=message)])
    try:
        payload = json.loads(response.content)
        indices = payload.get("selected", [])
        return [int(idx) for idx in indices if isinstance(idx, int)]
    except (json.JSONDecodeError, ValueError, TypeError):
        return [idx for idx, _ in unchecked]


def _invoke_article_agent(
    task: ArticleTask,
    summary_text: str,
    max_iterations: int,
    extra_instruction: Optional[str] = None,
) -> tuple[Dict[str, Any], Any]:
    user_prompt = build_user_prompt(task, summary_text, extra_instruction)
    agent = build_graph(task)
    result = run_agent(agent, user_prompt, max_iterations)
    output_text = result.get("output")
    if output_text is None:
        messages = result.get("messages") if isinstance(result, dict) else None
        if messages:
            final = messages[-1]
            output_text = getattr(final, "content", None)
    if output_text is None:
        raise RuntimeError("Article agent did not return output text.")
    if isinstance(output_text, str):
        structured = json.loads(output_text)
    else:
        structured = output_text
    return structured, output_text


def run_article_research(
    article: IssueArticle,
    summary_text: str,
    model: str,
    max_iterations: int,
) -> ResearchResult:
    article_task = ArticleTask(
        name=article.name,
        details=" ".join(article.details),
        status=article.status,
    )

    structured, raw_output = _invoke_article_agent(
        article_task,
        summary_text,
        max_iterations,
    )
    return ResearchResult(
        article_index=-1,
        article=article,
        structured=structured,
        raw_output=raw_output,
    )


def run_topic_research(
    topic: IssueTopic,
    summary_text: str,
    max_iterations: int,
) -> TopicResearchResult:
    details_text = "\n".join(f"- {detail}" for detail in topic.details)
    extra_instruction = (
        "You are researching a broader topic with multiple focus points. "
        "Ensure the JSON schema above is followed. For each item you return, align it with the focus points. "
        "Add helpful `notes` entries summarising how each reference supports the requested details. "
        "Focus points:\n"
        f"{details_text}" if details_text else ""
    )

    article_task = ArticleTask(
        name=topic.topic,
        details=" ".join(topic.details),
        status="unknown",
    )

    structured, raw_output = _invoke_article_agent(
        article_task,
        summary_text,
        max_iterations,
        extra_instruction=extra_instruction,
    )

    return TopicResearchResult(
        topic_index=-1,
        topic=topic,
        structured=structured,
        raw_output=raw_output,
    )


def build_research_graph(
    client: GitHubClient,
    selection_llm: ChatOpenAI,
    article_model: str,
    article_iterations: int,
) -> StateGraph[ResearchState]:
    graph = StateGraph(ResearchState)

    def fetch_issue_node(state: ResearchState) -> ResearchState:
        try:
            issue = client.get_issue(state.issue_number)
        except requests.HTTPError as exc:  # type: ignore[attr-defined]
            raise RuntimeError(
                f"Failed to fetch issue #{state.issue_number} from {state.repo}: {exc}"
            ) from exc
        body = issue.get("body", "")
        state.issue_body = body
        state.issue_title = issue.get("title") or ""
        state.summary_text = extract_issue_summary(body)
        state.articles = parse_issue_articles(body)
        state.topics = parse_issue_topics(body)
        return state

    def select_articles_node(state: ResearchState) -> ResearchState:
        if not state.articles:
            state.selected_indices = []
            return state
        selected = select_articles_with_llm(
            selection_llm,
            state.issue_title or "",
            state.summary_text,
            state.articles,
        )
        state.selected_indices = selected
        return state

    def research_articles_node(state: ResearchState) -> ResearchState:
        results: List[ResearchResult] = []
        for idx in state.selected_indices:
            if idx < 0 or idx >= len(state.articles):
                continue
            article = state.articles[idx]
            LOGGER.info("Running article agent for '%s'", article.name)
            research = run_article_research(article, state.summary_text, article_model, article_iterations)
            research.article_index = idx
            try:
                sync_result = sync_structured_item(research.structured)
                research.zotero = sync_result
                LOGGER.info(
                    "Synced '%s' to Zotero (key=%s, existed=%s)",
                    article.name,
                    sync_result.key,
                    sync_result.existed,
                )
            except ZoteroSyncError as exc:  # pragma: no cover - depends on live API
                research.zotero_error = str(exc)
                LOGGER.warning("Zotero sync failed for '%s': %s", article.name, exc)
            results.append(research)
        state.results = results
        state.article_comment = format_comment(results)
        state.updated_issue_body = mark_articles_completed(
            state.issue_body or "",
            [res.article_index for res in results],
        )
        return state

    def select_topics_node(state: ResearchState) -> ResearchState:
        if not state.topics:
            state.selected_topic_indices = []
            return state
        selected = select_topics_with_llm(
            selection_llm,
            state.issue_title or "",
            state.summary_text,
            state.topics,
        )
        state.selected_topic_indices = selected
        return state

    def research_topics_node(state: ResearchState) -> ResearchState:
        if not state.selected_topic_indices:
            state.topic_results = []
            return state

        results: List[TopicResearchResult] = []
        for idx in state.selected_topic_indices:
            if idx < 0 or idx >= len(state.topics):
                continue
            topic = state.topics[idx]
            LOGGER.info("Researching topic '%s'", topic.topic)
            topic_result = run_topic_research(
                topic,
                state.summary_text,
                article_iterations,
            )
            topic_result.topic_index = idx

            for entry in topic_result.structured.get("items", []) if isinstance(topic_result.structured, dict) else []:
                if not isinstance(entry, dict):
                    continue
                payload = {"items": [entry], "context": topic_result.structured.get("context")}
                try:
                    sync_result = sync_structured_item(payload)
                    topic_result.zotero.append(sync_result)
                except ZoteroSyncError as exc:  # pragma: no cover - depends on live API
                    topic_result.zotero_errors.append(str(exc))
                    LOGGER.warning("Zotero sync failed for topic '%s': %s", topic.topic, exc)

            results.append(topic_result)

        state.topic_results = results
        state.topic_comment = format_topics_comment(results)
        updated_body = state.updated_issue_body or state.issue_body or ""
        state.updated_issue_body = mark_topics_completed(
            updated_body,
            [res.topic_index for res in results],
        )
        return state

    def update_issue_node(state: ResearchState) -> ResearchState:
        sections: List[str] = []
        if state.article_comment:
            sections.append(state.article_comment)
        if state.topic_comment:
            sections.append(state.topic_comment)
        comment_text = "\n\n".join(section for section in sections if section)
        state.comment_body = comment_text or None

        if state.comment_body:
            client.comment_on_issue(state.issue_number, state.comment_body)

        if state.updated_issue_body and state.updated_issue_body != (state.issue_body or ""):
            client.update_issue_body(state.issue_number, state.updated_issue_body)
        return state

    graph.add_node("fetch_issue", fetch_issue_node)
    graph.add_node("select_articles", select_articles_node)
    graph.add_node("research_articles", research_articles_node)
    graph.add_node("update_issue", update_issue_node)

    graph.set_entry_point("fetch_issue")

    graph.add_edge("fetch_issue", "select_articles")

    def branch_after_article_selection(state: ResearchState) -> str:
        if not state.selected_indices:
            return "skip"
        return "research"

    graph.add_conditional_edges(
        "select_articles",
        branch_after_article_selection,
        {
            "skip": "select_topics",
            "research": "research_articles",
        },
    )

    graph.add_node("select_topics", select_topics_node)
    graph.add_node("research_topics", research_topics_node)

    graph.add_edge("research_articles", "select_topics")

    def branch_after_topic_selection(state: ResearchState) -> str:
        if not state.selected_topic_indices:
            return "skip"
        return "research"

    graph.add_conditional_edges(
        "select_topics",
        branch_after_topic_selection,
        {
            "skip": "update_issue",
            "research": "research_topics",
        },
    )

    graph.add_edge("research_topics", "update_issue")
    graph.set_finish_point("update_issue")
    return graph


def run_research_workflow(
    repo: str,
    issue_number: int,
    github_token: str,
    selection_model: str,
    article_model: str,
    article_iterations: int,
) -> ResearchState:
    client = GitHubClient(github_token, repo)

    selection_llm = ChatOpenAI(
        model=selection_model,
        api_key=None,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/andrewplassard/llm-evals-book",
            "X-Title": "Writing Assistant Research Issue Agent",
        },
        temperature=0,
        timeout=120,
    )

    graph = build_research_graph(client, selection_llm, article_model, article_iterations)
    app = graph.compile()
    initial_state = ResearchState(repo=repo, issue_number=issue_number)
    result_state = app.invoke(initial_state)
    if isinstance(result_state, dict):
        result_state = ResearchState(**result_state)
    return result_state
