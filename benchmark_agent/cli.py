from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from benchmark_agent.config import benchmark_prompt_path
from benchmark_agent.grading import grade_tool_trace
from benchmark_agent.graph import build_benchmark_bundle
from benchmark_agent.tools import all_tool_names


def _print_trace(tool_trace: list[dict]) -> None:
    print(json.dumps(tool_trace, ensure_ascii=False, indent=2), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Tool-count / accuracy benchmark agent (LangGraph)")
    parser.add_argument(
        "-m",
        "--message",
        help="Single user message; if omitted with --eval, uses case['user']",
    )
    parser.add_argument(
        "--active-tools",
        default="",
        help="Comma-separated tool names to expose (empty = no tools)",
    )
    parser.add_argument(
        "--all-tools",
        action="store_true",
        help="Expose the full catalog (30 tools)",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print catalog tool names and exit",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=None,
        help="Override markdown path for fenced system prompt",
    )
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument(
        "--eval",
        dest="eval_path",
        type=Path,
        default=None,
        help="JSON case file (id, user, active_tools, expected_tool_names, ...)",
    )
    parser.add_argument(
        "--dump-trace",
        action="store_true",
        help="Print tool_trace (SSOT) to stderr as JSON",
    )
    args = parser.parse_args(argv)

    if args.list_tools:
        for name in all_tool_names():
            print(name)
        return 0

    if args.all_tools:
        active = all_tool_names()
    else:
        active = [t.strip() for t in args.active_tools.split(",") if t.strip()]

    prompt_path = args.prompt or benchmark_prompt_path()

    if args.eval_path:
        case = json.loads(args.eval_path.read_text(encoding="utf-8"))
        user = case.get("user") or args.message
        if not user:
            print("eval case must include 'user' or pass --message", file=sys.stderr)
            return 2
        if "active_tools" in case:
            active = list(case["active_tools"])
    elif not args.message:
        print("Provide --message or --eval", file=sys.stderr)
        return 2
    else:
        case = None
        user = args.message

    app, _, _, _ = build_benchmark_bundle(
        active_tool_names=active,
        prompt_md_path=prompt_path,
        model_name=args.model,
        temperature=args.temperature,
    )

    result = app.invoke(
        {"messages": [HumanMessage(content=user)], "tool_trace": []},
        config={"recursion_limit": 40},
    )

    tool_trace = list(result.get("tool_trace") or [])
    if args.dump_trace:
        _print_trace(tool_trace)

    last = result["messages"][-1]
    text = last.content if isinstance(last, AIMessage) else str(last)

    if case is not None:
        grade = grade_tool_trace(
            tool_trace,
            expected_tool_names=list(case.get("expected_tool_names") or []),
            match_mode=str(case.get("match_mode") or "strict_order"),
            allow_extra=bool(case.get("allow_extra", False)),
            expected_args=case.get("expected_args"),
            args_match=str(case.get("args_match") or "canonical_json"),
            forbidden_tool_names=case.get("forbidden_tool_names"),
        )
        summary = {k: grade[k] for k in ("tool_trace_ok", "tool_name_ok", "tool_args_ok", "forbidden_hits", "extra_calls")}
        print(json.dumps({"grade": summary, "reply": text}, ensure_ascii=False, indent=2))
        return 0 if grade["tool_trace_ok"] else 1

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
