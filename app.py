"""
app.py
=======
Streamlit web interface for the Flashcard Quiz Agent.

Run:
    streamlit run app.py

Architecture
------------
* The GroqAgent is instantiated once and stored in `st.session_state` so it
  survives across Streamlit reruns with its full sliding-window memory intact.
* Tool-call events are captured via the `on_tool_event` callback added to
  `GroqAgent.chat()`.  Each event is a dict with keys:
      - `tool_name`  (str)
      - `arguments`  (dict)
      - `result`     (str -- raw JSON from the tool)
  Events are collected into a list during a single `chat()` call and
  rendered as collapsible `st.expander` blocks below the user message and
  above the final assistant reply.
* The sidebar recomputes KPI metrics directly from `FLASHCARD_DB` (the
  shared in-memory dict in `tools/flashcards.py`) so the numbers are always
  in sync with the agent's last action.
* Quick-action buttons write a `pending_prompt` key into `st.session_state`
  which the main panel reads and submits as if the user typed it, giving a
  single clean code path for all input.
"""

from __future__ import annotations

import json
import os
import sys

# ---------------------------------------------------------------------------
# Environment bootstrap -- must happen before importing GroqAgent so that
# GROQ_API_KEY is in os.environ before the Groq client is constructed.
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the FIRST Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Flashcard Quiz Agent",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Validate API key before anything agent-related
# ---------------------------------------------------------------------------
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    st.error(
        "**GROQ_API_KEY is not set.**\n\n"
        "Add it to your .env file and restart: streamlit run app.py",
        icon="🔑",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Local imports (after env is loaded)
# ---------------------------------------------------------------------------
from core.agent import GroqAgent  # noqa: E402
from tools.flashcards import FLASHCARD_DB, _MASTERED_STREAK  # noqa: E402

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Sidebar width */
    [data-testid="stSidebar"] {min-width: 320px; max-width: 360px;}

    /* KPI metric cards */
    [data-testid="stMetric"] {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 12px 16px;
        border: 1px solid #313244;
    }
    [data-testid="stMetricLabel"] {font-size: 0.78rem; color: #a6adc8;}
    [data-testid="stMetricValue"] {font-size: 1.5rem; font-weight: 700;}

    /* Tool call expander */
    details[data-testid="stExpander"] summary {
        font-size: 0.85rem;
        color: #cba6f7;
    }

    /* Header badge row */
    .badge-row {display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 6px;}
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-purple {background:#313244; color:#cba6f7; border:1px solid #cba6f7;}
    .badge-green  {background:#1e3a2f; color:#a6e3a1; border:1px solid #a6e3a1;}
    .badge-blue   {background:#1e2a3a; color:#89dceb; border:1px solid #89dceb;}

    /* Quick-action buttons */
    div[data-testid="stButton"] > button {
        width: 100%;
        border-radius: 8px;
        font-size: 0.82rem;
        padding: 6px 4px;
    }

    /* Deck inspector dataframe */
    [data-testid="stDataFrame"] {border-radius: 8px; overflow: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "agent" not in st.session_state:
    st.session_state.agent = GroqAgent(api_key=GROQ_API_KEY, model=GROQ_MODEL)

if "messages" not in st.session_state:
    # Each item: {"role": "user"|"assistant", "content": str,
    #             "tool_events": list[dict] | None}
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ---------------------------------------------------------------------------
# Sidebar helper: compute live KPIs from the shared in-memory DB
# ---------------------------------------------------------------------------
def _compute_kpis() -> dict:
    cards = list(FLASHCARD_DB.get("cards", {}).values())
    total = len(cards)
    unattempted = sum(1 for c in cards if c["total_attempts"] == 0)
    mastered = sum(
        1 for c in cards if c.get("consecutive_correct", 0) >= _MASTERED_STREAK
    )
    weak_spots = sum(
        1
        for c in cards
        if c["incorrect_count"] > 0
        and c.get("consecutive_correct", 0) < _MASTERED_STREAK
    )
    return {
        "total": total,
        "unattempted": unattempted,
        "mastered": mastered,
        "weak_spots": weak_spots,
        "cards": cards,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🃏 Deck Analytics")
    st.divider()

    kpi = _compute_kpis()

    # KPI metric cards
    col1, col2 = st.columns(2)
    col1.metric("📚 Total Cards", kpi["total"])
    col2.metric("🆕 Unattempted", kpi["unattempted"])

    col3, col4 = st.columns(2)
    col3.metric("🔥 Weak Spots", kpi["weak_spots"])
    col4.metric("✅ Mastered", kpi["mastered"])

    st.divider()
    st.markdown("### ⚡ Quick Actions")

    quick_actions = [
        ("⚡ Quiz Adaptive", "quiz me"),
        ("🎯 Focus Weak Spots", "quiz me on weak topics"),
        (
            "🆕 Unattempted Card",
            "Show me a card I have never seen before. Use unattempted mode.",
        ),
        (
            "📊 Full Deck Summary",
            "Give me a summary of all cards, showing mastery and urgency scores.",
        ),
    ]

    for label, prompt in quick_actions:
        if st.button(label, key=f"qa_{label}"):
            st.session_state.pending_prompt = prompt

    st.divider()

    # Live Deck Inspector
    with st.expander("🔍 Live Deck Inspector", expanded=False):
        cards = kpi["cards"]
        if not cards:
            st.info("No cards in the deck yet. Ask the agent to add some!")
        else:
            import pandas as pd

            rows = []
            for c in sorted(cards, key=lambda x: x["id"]):
                consec = c.get("consecutive_correct", 0)
                mastered = consec >= _MASTERED_STREAK
                rows.append(
                    {
                        "ID": c["id"],
                        "Question": c["question"][:55] + ("..." if len(c["question"]) > 55 else ""),
                        "Answer": c["answer"][:40] + ("..." if len(c["answer"]) > 40 else ""),
                        "Attempts": c["total_attempts"],
                        "Incorrect": c["incorrect_count"],
                        "Consec OK": consec,
                        "Mastered": "Yes" if mastered else "No",
                    }
                )

            df = pd.DataFrame(rows).set_index("ID")
            st.dataframe(
                df,
                use_container_width=True,
                height=min(35 * len(rows) + 38, 400),
            )


# ---------------------------------------------------------------------------
# Main panel header
# ---------------------------------------------------------------------------
st.markdown("# 🃏 Flashcard Quiz Agent")
st.markdown(
    f"""
    <div class="badge-row">
        <span class="badge badge-purple">Model: {GROQ_MODEL}</span>
        <span class="badge badge-green">Engine: ReAct Loop</span>
        <span class="badge badge-blue">Memory: Sliding Window (6 turns)</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()


# ---------------------------------------------------------------------------
# Render tool-call events as collapsible expanders
# ---------------------------------------------------------------------------
def _render_tool_events(tool_events: list[dict]) -> None:
    for event in tool_events:
        tool_name = event.get("tool_name", "unknown_tool")
        arguments = event.get("arguments", {})
        raw_result = event.get("result", "")

        try:
            result_obj = json.loads(raw_result)
        except (json.JSONDecodeError, TypeError):
            result_obj = raw_result

        with st.expander(f"Tool Executed: {tool_name}", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Arguments**")
                st.json(arguments)
            with col_b:
                st.markdown("**Result**")
                st.json(result_obj)


# ---------------------------------------------------------------------------
# Render existing chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        if role == "assistant" and msg.get("tool_events"):
            _render_tool_events(msg["tool_events"])
        st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Core: run the agent and update session state
# ---------------------------------------------------------------------------
def _run_agent(user_input: str) -> None:
    user_input = user_input.strip()
    if not user_input:
        return

    # Render + persist user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "tool_events": None}
    )

    # Collect tool events via callback
    collected_events: list[dict] = []

    def _on_tool_event(event: dict) -> None:
        collected_events.append(event)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Agent thinking..."):
            final_response = st.session_state.agent.chat(
                user_input,
                on_tool_event=_on_tool_event,
            )

        # Render tool traces
        if collected_events:
            _render_tool_events(collected_events)

        # Render final reply
        st.markdown(final_response)

    # Persist to session state
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_response,
            "tool_events": collected_events if collected_events else None,
        }
    )


# ---------------------------------------------------------------------------
# Input: consume quick-action OR chat input
# ---------------------------------------------------------------------------
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    _run_agent(prompt)
    st.rerun()

if user_text := st.chat_input("Ask me to quiz you, add cards, or get a summary..."):
    _run_agent(user_text)
    st.rerun()
