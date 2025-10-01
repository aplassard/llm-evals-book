"""Thin wrapper to execute the GitHub issue automation CLI."""

from __future__ import annotations

from writing_assistant.github_issue_agent import main


if __name__ == "__main__":
    raise SystemExit(main())
