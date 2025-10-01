"""Configuration helpers for writing assistant tooling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def load_dotenv(dotenv_path: Path) -> Dict[str, str]:
    """Load key/value pairs from a .env file into the environment.

    Parameters
    ----------
    dotenv_path:
        Path to the .env file. The file must contain simple ``KEY=VALUE`` pairs.

    Returns
    -------
    dict
        Mapping of keys that were loaded. Existing environment variables are
        left untouched so callers can override values explicitly.
    """

    if not dotenv_path.exists():
        raise FileNotFoundError(f".env file not found at {dotenv_path}")

    loaded: Dict[str, str] = {}
    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        loaded[key] = value
        os.environ.setdefault(key, value)

    required = ("OPENROUTER_API_KEY", "TAVILY_API_KEY")
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        missing_str = ", ".join(missing)
        raise EnvironmentError(
            f"Missing required environment variables: {missing_str}. "
            "Ensure they are set in the shell or provided via the .env file."
        )

    # Map OpenRouter credentials to the names expected by LangChain/OpenAI clients.
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = openrouter_key

    return loaded


def resolve_repo_root() -> Path:
    """Return the repository root assuming this file lives in writing-assistant/."""

    return Path(__file__).resolve().parents[2]
