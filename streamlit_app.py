from __future__ import annotations

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
import streamlit as st

from insurance_agent.config import openai_model
from insurance_agent.graph import build_agent_bundle
from insurance_agent.streaming import iter_assistant_text


def _gpt_like_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.25rem; padding-bottom: 4rem; max-width: 52rem; }
          [data-testid="stChatMessage"] { padding: 0.65rem 0.85rem; }
          div[data-testid="stVerticalBlockBorderWrapper"] > div {
            border-radius: 12px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session() -> None:
    if "lc_messages" not in st.session_state:
        st.session_state.lc_messages: list[BaseMessage] = []
    if "bundle" not in st.session_state:
        st.session_state.bundle = build_agent_bundle()


def main() -> None:
    load_dotenv()
    st.set_page_config(
        page_title="보험 가입 상담",
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="auto",
    )
    _gpt_like_css()
    _init_session()

    _, llm, system_prompt = st.session_state.bundle

    with st.sidebar:
        st.subheader("세션")
        if st.button("새 대화", use_container_width=True):
            st.session_state.lc_messages = []
            st.rerun()
        st.divider()
        st.text(f"모델: {openai_model()}")

    st.title("보험 가입 상담")
    st.caption("정보성 상담 보조입니다. 최종 가입·약관 해석은 담당 전문가에게 확인하세요.")

    for m in st.session_state.lc_messages:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(m.content)

    if prompt := st.chat_input("메시지를 입력하세요…"):
        st.session_state.lc_messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            stream = iter_assistant_text(llm, system_prompt, st.session_state.lc_messages)
            full = st.write_stream(stream)
        st.session_state.lc_messages.append(AIMessage(content=full or ""))


if __name__ == "__main__":
    main()
