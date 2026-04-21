from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from benchmark_agent.config import benchmark_prompt_path, openai_model
from benchmark_agent.prompt import load_benchmark_system_prompt
from benchmark_agent.state import BenchmarkState
from benchmark_agent.tools import build_tool_catalog, tool_result_to_content, tools_for_active_names


def _with_system_prompt(messages: list[BaseMessage], system_prompt: str) -> list[BaseMessage]:
    msgs = list(messages)
    if not any(isinstance(m, SystemMessage) for m in msgs):
        return [SystemMessage(content=system_prompt), *msgs]
    return msgs


def _route_after_agent(state: BenchmarkState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def _make_tools_node(tool_map: dict[str, Any]):
    def tools_node(state: BenchmarkState) -> dict[str, Any]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}
        traces: list[dict[str, Any]] = []
        msgs: list[ToolMessage] = []
        for tc in last.tool_calls:
            name = tc.get("name") or ""
            raw_args = tc.get("args")
            args: dict[str, Any] = dict(raw_args) if isinstance(raw_args, dict) else {}
            call_id = tc.get("id") or ""
            traces.append({"name": name, "arguments": args})
            tool = tool_map.get(name)
            if tool is None:
                content = tool_result_to_content({"error": f"unknown_or_inactive_tool:{name}"})
            else:
                try:
                    out = tool.invoke(args)
                    content = tool_result_to_content(out)
                except Exception as e:  # noqa: BLE001 — surface to model as tool output
                    content = tool_result_to_content({"error": str(e)})
            msgs.append(ToolMessage(content=content, tool_call_id=call_id))
        return {"messages": msgs, "tool_trace": traces}

    return tools_node


def build_benchmark_bundle(
    *,
    active_tool_names: list[str] | None = None,
    prompt_md_path: Path | None = None,
    model_name: str | None = None,
    temperature: float = 0.3,
):
    """
    Compiled graph, LLM, system prompt, active tool map (for extensions / tests).
    """
    active = list(active_tool_names or [])
    path = prompt_md_path or benchmark_prompt_path()
    system = load_benchmark_system_prompt(path, active_tool_names=active)
    catalog = build_tool_catalog()
    tools_list = tools_for_active_names(active, catalog=catalog) if active else []
    tool_map = {t.name: t for t in tools_list}

    llm = ChatOpenAI(model=model_name or openai_model(), temperature=temperature)
    runnable: Runnable = llm.bind_tools(tools_list) if tools_list else llm

    def agent_node(state: BenchmarkState) -> dict[str, Any]:
        messages = _with_system_prompt(list(state["messages"]), system)
        response = runnable.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(BenchmarkState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", _make_tools_node(tool_map))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile(), llm, system, tool_map


def build_benchmark_agent(
    *,
    active_tool_names: list[str] | None = None,
    prompt_md_path: Path | None = None,
    model_name: str | None = None,
    temperature: float = 0.3,
):
    app, _, _, _ = build_benchmark_bundle(
        active_tool_names=active_tool_names,
        prompt_md_path=prompt_md_path,
        model_name=model_name,
        temperature=temperature,
    )
    return app
