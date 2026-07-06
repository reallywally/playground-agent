from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class BenchmarkState(TypedDict):
    """LangGraph state for tool-count benchmark runs."""

    messages: Annotated[list[AnyMessage], add_messages]
    tool_trace: Annotated[list[dict], operator.add]
