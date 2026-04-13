from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Graph state. Extend with fields (e.g. retrieval_context) as features grow."""

    messages: Annotated[list[AnyMessage], add_messages]
