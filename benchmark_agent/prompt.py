from __future__ import annotations

import re
from pathlib import Path

from benchmark_agent.config import benchmark_prompt_path


def _load_fenced_system_prompt(md_path: Path) -> str:
    """Extract the first ``` ... ``` block from the markdown file."""
    text = md_path.read_text(encoding="utf-8")
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if not match:
        raise ValueError(f"No ``` fenced block found in {md_path}")
    return match.group(1).strip()

_ACTIVE_BLOCK = re.compile(
    r"(ACTIVE_TOOLS \(한 줄에 하나, 하이픈 목록\):\n)(?:- [^\n]*\n)+",
    re.MULTILINE,
)


def inject_active_tools(system_prompt: str, active_tool_names: list[str]) -> str:
    """Replace the ACTIVE_TOOLS bullet list inside the fenced system prompt."""
    if active_tool_names:
        bullets = "".join(f"- {name}\n" for name in active_tool_names)
    else:
        bullets = "- (없음 — 이번 실행에서 도구 미노출)\n"

    def repl(m: re.Match[str]) -> str:
        return m.group(1) + bullets

    updated, n = _ACTIVE_BLOCK.subn(repl, system_prompt, count=1)
    if n == 1:
        return updated
    suffix = (
        "\n\n### 시스템 주입 ACTIVE_TOOLS\n"
        "ACTIVE_TOOLS (한 줄에 하나, 하이픈 목록):\n"
        f"{bullets}"
    )
    return system_prompt.rstrip() + suffix


def load_benchmark_system_prompt(
    md_path: Path | None = None,
    *,
    active_tool_names: list[str] | None = None,
) -> str:
    path = md_path or benchmark_prompt_path()
    base = _load_fenced_system_prompt(path)
    return inject_active_tools(base, list(active_tool_names or []))
