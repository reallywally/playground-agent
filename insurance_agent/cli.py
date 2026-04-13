from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from insurance_agent.graph import build_agent


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Insurance consultation agent (LangGraph)")
    parser.add_argument(
        "-m",
        "--message",
        help="Single user message; if omitted, reads lines from stdin until EOF",
    )
    args = parser.parse_args(argv)

    agent = build_agent()

    if args.message:
        result = agent.invoke({"messages": [HumanMessage(content=args.message)]})
        last = result["messages"][-1]
        print(last.content)
        return 0

    print("보험 상담 에이전트 (종료: Ctrl+D)", file=sys.stderr)
    messages = []
    try:
        while True:
            line = input("You: ").strip()
            if not line:
                continue
            messages.append(HumanMessage(content=line))
            result = agent.invoke({"messages": messages})
            messages = result["messages"]
            reply = messages[-1].content
            print(f"Agent: {reply}\n")
    except EOFError:
        print("", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
