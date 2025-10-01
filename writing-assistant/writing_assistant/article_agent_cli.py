"""Command-line interface for researching references via LangGraph."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

from writing_assistant.graph import ArticleTask, build_graph, build_user_prompt
from langchain_core.messages import AIMessage, HumanMessage


LOGGER = logging.getLogger("writing_assistant.article_agent_cli")

REQUIRED_ENV_VARS = ("OPENROUTER_API_KEY", "TAVILY_API_KEY")


def ensure_environment() -> None:
    """Validate required environment variables and normalise expected aliases."""

    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        missing_str = ", ".join(missing)
        raise EnvironmentError(
            f"Missing required environment variables: {missing_str}. "
            "Set them in the shell or provide an env file via `uv run --env-file`."
        )

    if os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the writing assistant research agent for a single reference."
    )
    parser.add_argument("--name", required=True, help="Display name of the reference request.")
    parser.add_argument(
        "--details",
        required=True,
        help="Additional context describing what should be located.",
    )
    parser.add_argument(
        "--status",
        required=True,
        type=lambda value: value.lower(),
        choices=["known", "unknown"],
        help="Whether the reference is already known or needs discovery.",
    )
    parser.add_argument(
        "--summary",
        help="Optional transcript summary to ground the research prompt.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="Maximum tool+LLM iterations for the agent (defaults to 8).",
    )
    parser.add_argument(
        "--raw-output",
        action="store_true",
        help="Print the agent output verbatim instead of attempting JSON parsing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging to stderr for troubleshooting.",
    )
    return parser.parse_args(argv)


def run_agent(agent: Any, prompt: str, max_iterations: int) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "input": prompt,
        "messages": [HumanMessage(content=prompt)],
    }
    if max_iterations:
        # LangGraph's ReAct agent accepts an optional max_iterations control.
        state["max_iterations"] = max_iterations
    result = agent.invoke(state)
    if not isinstance(result, dict):
        return {"output": result}
    return result


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ensure_environment()

    article_task = ArticleTask(name=args.name, details=args.details, status=args.status)
    note_summary = args.summary or ""

    LOGGER.info(
        "Building agent for '%s' (status=%s)", article_task.name, article_task.status
    )
    agent = build_graph(article_task)

    user_prompt = build_user_prompt(article_task, note_summary)
    LOGGER.debug("User prompt:\n%s", user_prompt)

    try:
        result = run_agent(agent, user_prompt, args.max_iterations)
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        print("Interrupted by user", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - log unexpected failures
        LOGGER.exception("Agent execution failed: %s", exc)
        raise

    output_text = result.get("output")
    if output_text is None and "messages" in result:
        messages = result["messages"]
        if messages:
            final = messages[-1]
            if isinstance(final, dict):
                output_text = final.get("content")
            elif isinstance(final, AIMessage):
                output_text = final.content
            else:
                output_text = getattr(final, "content", None)

    if output_text is None:
        print(json.dumps(result, indent=2, default=str), file=sys.stderr)
        raise RuntimeError("Agent completed without returning output text.")

    if args.raw_output:
        LOGGER.info("Returning raw agent output")
        print(output_text)
        return 0

    try:
        structured = json.loads(output_text)
    except json.JSONDecodeError:
        print("Agent output was not valid JSON. Use --raw-output to inspect details.", file=sys.stderr)
        print(output_text)
        return 1

    if article_task.status == "known":
        tokens = [
            token
            for token in re.split(r"[^a-z0-9]+", article_task.name.lower())
            if len(token) >= 4
        ]
        if tokens:
            def text_matches(text: str) -> bool:
                lowered = text.lower()
                return any(token in lowered for token in tokens)

            items = structured if isinstance(structured, list) else structured.get("items", [])
            if not isinstance(items, list):
                items = []

            match_found = False
            for item in items:
                if not isinstance(item, dict):
                    continue
                title = item.get("title", "")
                creators = " ".join(
                    f"{c.get('firstName', '')} {c.get('lastName', '')}"
                    for c in item.get("creators", [])
                    if isinstance(c, dict)
                )
                notes_val = item.get("notes", [])
                if isinstance(notes_val, list):
                    note_blob = " ".join(
                        elem if isinstance(elem, str) else json.dumps(elem, default=str)
                        for elem in notes_val
                    )
                else:
                    note_blob = str(notes_val)
                combined = " ".join(filter(None, [title, creators, note_blob]))
                if text_matches(combined):
                    match_found = True
                    break

            if not match_found:
                LOGGER.error(
                    "Known reference check failed; output does not mention any of %s tokens.",
                    tokens,
                )
                print(json.dumps(structured, indent=2), file=sys.stderr)
                raise RuntimeError(
                    "Agent returned a reference that does not match the requested known citation."
                )

    LOGGER.info("Agent completed successfully")
    print(json.dumps(structured, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
