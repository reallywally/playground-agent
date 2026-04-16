from __future__ import annotations

from collections.abc import Callable

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import Runnable

from insurance_agent.state import AgentState


def with_system_prompt(messages: list[BaseMessage], system_prompt: str) -> list[BaseMessage]:
    """Prepend a single system message for the LLM call when state does not already include one."""
    msgs = list(messages)
    if not any(isinstance(m, SystemMessage) for m in msgs):
        return [SystemMessage(content=system_prompt), *msgs]
    return msgs


def make_chat_node(
    llm: Runnable,
    system_prompt: str,
) -> Callable[[AgentState], dict[str, list[BaseMessage]]]:
    """Returns a LangGraph node: append system once, then model reply to latest messages."""

    def chat_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        messages = with_system_prompt(list(state["messages"]), system_prompt)
        response = llm.invoke(messages)
        if isinstance(response, AIMessage):
            return {"messages": [response]}
        return {"messages": [AIMessage(content=str(response))]}

    return chat_node
