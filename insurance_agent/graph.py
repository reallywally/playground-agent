from __future__ import annotations

from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from insurance_agent.config import openai_model, system_prompt_path
from insurance_agent.nodes.chat import make_chat_node
from insurance_agent.prompt_loader import load_fenced_system_prompt
from insurance_agent.state import AgentState


def build_agent_bundle(
    *,
    system_prompt: str | None = None,
    prompt_md_path: Path | None = None,
    model_name: str | None = None,
    temperature: float = 0.4,
):
    """
    Graph + LLM + resolved system prompt (for Streamlit streaming and CLI extensions).
    """
    path = prompt_md_path or system_prompt_path()
    resolved = system_prompt if system_prompt is not None else load_fenced_system_prompt(path)
    llm = ChatOpenAI(model=model_name or openai_model(), temperature=temperature)
    graph = StateGraph(AgentState)
    graph.add_node("chat", make_chat_node(llm, resolved))
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile(), llm, resolved


def build_agent(
    *,
    system_prompt: str | None = None,
    prompt_md_path: Path | None = None,
    model_name: str | None = None,
    temperature: float = 0.4,
):
    """
    Compile a minimal single-node graph. Swap/chain nodes here later (RAG, tools, routing).
    """
    app, _, _ = build_agent_bundle(
        system_prompt=system_prompt,
        prompt_md_path=prompt_md_path,
        model_name=model_name,
        temperature=temperature,
    )
    return app
