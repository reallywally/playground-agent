from __future__ import annotations

from collections.abc import Callable

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import Runnable

from insurance_agent.state import AgentState


def make_chat_node(
    llm: Runnable,
    system_prompt: str,
) -> Callable[[AgentState], dict[str, list[BaseMessage]]]:
    """Returns a LangGraph node: append system once, then model reply to latest messages."""

    def chat_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        messages = list(state["messages"])
        has_system = any(isinstance(m, SystemMessage) for m in messages)
        if not has_system:
            messages = [SystemMessage(content=system_prompt), *messages]
        response = llm.invoke(messages)
        if isinstance(response, AIMessage):
            return {"messages": [response]}
        return {"messages": [AIMessage(content=str(response))]}

    return chat_node
