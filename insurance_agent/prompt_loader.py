from __future__ import annotations

import re
from pathlib import Path


def load_fenced_system_prompt(md_path: Path) -> str:
    """Extract the first ``` ... ``` block from the markdown file."""
    text = md_path.read_text(encoding="utf-8")
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if not match:
        raise ValueError(f"No ``` fenced block found in {md_path}")
    return match.group(1).strip()
