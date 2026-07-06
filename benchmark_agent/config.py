from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = REPO_ROOT / "prompts" / "tool-count-accuracy-benchmark-agent.ko.md"


def openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def benchmark_prompt_path() -> Path:
    raw = os.getenv("BENCHMARK_SYSTEM_PROMPT_PATH")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_PROMPT_PATH
