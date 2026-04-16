from __future__ import annotations

from collections.abc import Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage

from insurance_agent.nodes.chat import with_system_prompt


def iter_assistant_text(
    llm: BaseChatModel,
    system_prompt: str,
    conversation: list[BaseMessage],
) -> Iterator[str]:
    """Token/text stream for the next assistant turn (same context rules as the graph chat node)."""
    messages = with_system_prompt(conversation, system_prompt)
    for chunk in llm.stream(messages):
        if not isinstance(chunk, AIMessageChunk):
            continue
        part = chunk.content
        if isinstance(part, str) and part:
            yield part
        elif isinstance(part, list):
            for block in part:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text") or ""
                    if text:
                        yield text
