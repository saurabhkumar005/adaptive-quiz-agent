"""
app.py  --  FlashCard Quest: Gamified Streamlit Interface
==========================================================
Two-tab interface:
  Tab 1  ->  Quiz Game  (gamified card-by-card mode with XP, streaks, levels, achievements)
  Tab 2  ->  Ask Agent  (free-form chat with full tool-call traces)

Game Flow (Tab 1):
  home  ->  question  ->  feedback  ->  next question (loops)

Direct tool calls are used in the game tab for instant feedback (no LLM latency).
The full GroqAgent with its ReAct loop powers Tab 2.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FlashCard Quest",
    page_icon="\U0001f0cf",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# API / env
# ---------------------------------------------------------------------------
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    st.error("**GROQ_API_KEY is not set.** Add it to `.env` and restart: `streamlit run app.py`", icon="\U0001f511")
    st.stop()

# ---------------------------------------------------------------------------
# Local imports (after env loaded so Groq client sees the key)
# ---------------------------------------------------------------------------
from core.agent import GroqAgent
from tools.flashcards import (
    FLASHCARD_DB,
    _MASTERED_STREAK,
    quiz_me       as _quiz_me_raw,
    record_answer as _record_answer_raw,
)
import pandas as pd

# ---------------------------------------------------------------------------
# CSS  --  Game Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }

/* sidebar */
[data-testid="stSidebar"] {
    background: rgba(12, 10, 35, 0.97);
    border-right: 1px solid rgba(124, 58, 237, 0.25);
    min-width: 300px !important;
    max-width: 330px !important;
}

/* metric cards */
[data-testid="stMetric"] {
    background: rgba(124, 58, 237, 0.1);
    border: 1px solid rgba(124, 58, 237, 0.28);
    border-radius: 12px;
    padding: 10px 14px;
}
[data-testid="stMetricLabel"] { color: #a78bfa !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.35rem !important; font-weight: 800 !important; }

/* tabs */
[data-testid="stTabs"] [role="tab"] {
    font-size: 1rem; font-weight: 700;
    padding: 10px 28px; border-radius: 10px 10px 0 0;
    color: #94a3b8;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: rgba(124,58,237,0.18);
    border-bottom: 3px solid #7c3aed;
    color: #c4b5fd;
}

/* question card */
.q-card {
    background: linear-gradient(145deg, rgba(124,58,237,0.13), rgba(59,130,246,0.1));
    border: 2px solid rgba(124,58,237,0.55);
    border-radius: 22px;
    padding: 40px 44px;
    margin: 18px 0;
    text-align: center;
    position: relative;
    box-shadow: 0 0 60px rgba(124,58,237,0.18), 0 8px 40px rgba(0,0,0,0.5);
    animation: cardGlow 3s ease-in-out infinite;
}
@keyframes cardGlow {
    0%,100% { box-shadow: 0 0 40px rgba(124,58,237,0.22), 0 8px 40px rgba(0,0,0,0.5); }
    50%      { box-shadow: 0 0 75px rgba(124,58,237,0.48), 0 8px 40px rgba(0,0,0,0.5); }
}
.q-card .badge-top {
    position: absolute; top: -14px; left: 50%; transform: translateX(-50%);
    background: linear-gradient(90deg, #7c3aed, #3b82f6);
    color: #fff; font-size: 0.68rem; font-weight: 800; letter-spacing: 0.12em;
    padding: 5px 20px; border-radius: 999px; text-transform: uppercase; white-space: nowrap;
}
.q-card h2 { color: #f8fafc; font-size: 1.55rem; font-weight: 700; margin: 14px 0 6px; line-height: 1.55; }
.q-card .q-meta { color: #64748b; font-size: 0.78rem; margin-top: 10px; }

/* feedback panels */
.fb-correct {
    background: linear-gradient(135deg, rgba(16,185,129,0.14), rgba(5,150,105,0.08));
    border: 2px solid rgba(16,185,129,0.55);
    border-radius: 18px; padding: 28px 32px; text-align: center;
    animation: popIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.fb-wrong {
    background: linear-gradient(135deg, rgba(239,68,68,0.14), rgba(185,28,28,0.08));
    border: 2px solid rgba(239,68,68,0.5);
    border-radius: 18px; padding: 28px 32px; text-align: center;
    animation: shake 0.5s ease;
}
.fb-skip {
    background: rgba(100,116,139,0.12);
    border: 1px solid rgba(100,116,139,0.35);
    border-radius: 18px; padding: 28px 32px; text-align: center;
}
@keyframes popIn {
    0%   { transform: scale(0.78); opacity: 0; }
    100% { transform: scale(1);    opacity: 1; }
}
@keyframes shake {
    0%,100% { transform: translateX(0); }
    18%     { transform: translateX(-9px); }
    36%     { transform: translateX(9px); }
    54%     { transform: translateX(-5px); }
    72%     { transform: translateX(5px); }
}

/* XP bar */
.xp-wrap { background: rgba(255,255,255,0.07); border-radius: 999px; height: 9px; margin: 6px 0 2px; overflow: hidden; }
.xp-fill  { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #7c3aed, #06b6d4); transition: width 0.9s ease; }

/* level badge */
.lvl-badge {
    display: flex; align-items: center; justify-content: center; flex-direction: column;
    background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(59,130,246,0.14));
    border: 2px solid rgba(124,58,237,0.42); border-radius: 16px;
    padding: 14px 10px; margin-bottom: 10px;
}
.lvl-badge .lvl-icon { font-size: 2.4rem; line-height: 1; }
.lvl-badge .lvl-name { color: #c4b5fd; font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 5px; }
.lvl-badge .lvl-xp   { color: #475569; font-size: 0.68rem; margin-top: 2px; }

/* streak badge */
.streak-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: linear-gradient(90deg, #f59e0b, #ef4444);
    color: #fff; font-weight: 900; font-size: 1rem;
    padding: 5px 16px; border-radius: 999px;
    box-shadow: 0 0 22px rgba(245,158,11,0.55);
    animation: streakPulse 1.4s ease-in-out infinite;
}
@keyframes streakPulse {
    0%,100% { box-shadow: 0 0 18px rgba(245,158,11,0.4); }
    50%      { box-shadow: 0 0 36px rgba(245,158,11,0.85); }
}

/* achievement pill */
.ach-pill {
    display: inline-flex; align-items: center; gap: 4px;
    background: rgba(251,191,36,0.13); border: 1px solid rgba(251,191,36,0.38);
    color: #fbbf24; font-size: 0.76rem; font-weight: 700;
    padding: 4px 11px; border-radius: 999px; margin: 3px 2px;
    animation: popIn 0.35s ease;
}

/* hero */
.hero { text-align: center; padding: 32px 20px 20px; }
.hero h1 { font-size: 2.8rem; font-weight: 900; color: #f8fafc; margin: 0; letter-spacing: -1px; }
.hero p  { color: #94a3b8; font-size: 1.05rem; margin-top: 8px; }

/* primary buttons */
button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
    border: none !important; color: #fff !important; font-weight: 700 !important;
    border-radius: 11px !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.38) !important;
    transition: all 0.18s !important;
}
button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.62) !important;
}

/* mode card */
.mode-card {
    background: rgba(124,58,237,0.09); border: 1px solid rgba(124,58,237,0.3);
    border-radius: 14px; padding: 16px; margin: 6px 0;
}
.mc-icon  { font-size: 1.6rem; }
.mc-title { color: #c4b5fd; font-weight: 700; font-size: 0.92rem; margin: 4px 0 2px; }
.mc-desc  { color: #64748b; font-size: 0.76rem; }

/* chat messages */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.035) !important;
    border-radius: 12px !important; margin-bottom: 8px !important;
    border: 1px solid rgba(255,255,255,0.055) !important;
}

/* expander */
details[data-testid="stExpander"] summary { color: #a78bfa; font-size: 0.84rem; }

/* progress bar */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #7c3aed, #06b6d4) !important;
    border-radius: 999px;
}

[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STOP_WORDS = {
    "a","an","the","is","are","was","were","of","in","on","at","to",
    "for","with","and","or","it","its","this","that","be","do","did",
    "has","have","had","by","as","from","but","not","no","so","if",
    "what","which","who","when","where","how","why","can","could",
    "would","should",
}

XP_LEVELS = [
    (0,   "\U0001f331", "Seedling",   50),
    (50,  "\U0001f4d6", "Scholar",   150),
    (150, "\U0001f393", "Graduate",  300),
    (300, "\u2b50",     "Expert",    500),
    (500, "\U0001f52e", "Wizard",    750),
    (750, "\U0001f3c6", "Legend",   9999),
]

ACHIEVEMENT_DEFS = [
    ("first_blood",  "\U0001f3af", "First Blood",    "Got your first correct answer!"),
    ("streak_3",     "\U0001f525", "On Fire!",        "3 answers correct in a row!"),
    ("streak_5",     "\u26a1",     "Lightning!",      "5 answers correct in a row!"),
    ("streak_10",    "\U0001f32a", "Unstoppable!",    "10 answers correct in a row!"),
    ("ten_done",     "\U0001f4da", "Bookworm",        "Answered 10 questions this session!"),
    ("twenty_done",  "\U0001f9e0", "Brainiac",        "Answered 20 questions this session!"),
    ("accuracy_ace", "\U0001f48e", "Ace",             "Session accuracy above 90%!"),
    ("comeback",     "\U0001f985", "Comeback King",   "Correct after 2 wrong answers in a row!"),
]
ACH_MAP = {a[0]: a for a in ACHIEVEMENT_DEFS}

EXAMPLE_PROMPTS = {
    "\u2795 Add Cards": [
        "Add a card: Q: What is Docker? / A: OS-level virtualisation platform",
        "Add 5 cards on Python data structures",
        "Generate 8 cards on SQL joins",
        "Add 3 cards on basics of machine learning",
        "Add a card: Q: What is recursion? / A: A function that calls itself",
    ],
    "\U0001f3af Quiz Me": [
        "Quiz me",
        "Quiz me on Docker",
        "Quiz me with 3 questions back to back",
        "Show me a card I have never seen before. Use unattempted mode.",
        "Focus on my weakest cards only",
        "Quiz me on machine learning",
    ],
    "\U0001f4ca Stats & Insights": [
        "Give me a full deck summary with mastery and urgency scores",
        "Which topics am I struggling with the most?",
        "How many cards have I mastered so far?",
        "What is my single weakest card right now?",
        "Show me my accuracy across all topics",
    ],
    "\U0001f4a1 Learn + Save": [
        "What is a RAG pipeline? Then add it as a card.",
        "Explain gradient descent in 2 sentences and add it as a card",
        "What is the CAP theorem? Save it to my deck.",
        "Explain ACID properties and create 4 cards on it",
    ],
}

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "agent": None,
    "messages": [],
    "pending_prompt": None,
    "game_phase": "home",
    "current_card": None,
    "priority_reason": "",
    "last_correct": None,
    "last_xp": 0,
    "correct_answer_text": "",
    "wrong_streak_count": 0,
    "session_xp": 0,
    "session_streak": 0,
    "session_best_streak": 0,
    "session_correct": 0,
    "session_total": 0,
    "session_achievements": [],
    "last_wrong_count": 0,
    "quiz_topic": "",
    "quiz_mode": "adaptive",
    "total_xp": 0,
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.agent is None:
    st.session_state.agent = GroqAgent(api_key=GROQ_API_KEY, model=GROQ_MODEL)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fuzzy_match(user_ans: str, correct_ans: str) -> bool:
    def norm(s: str) -> str:
        s = re.sub(r"[^\w\s]", " ", s.lower().strip())
        return " ".join(w for w in s.split() if w not in STOP_WORDS)
    u, c = norm(user_ans), norm(correct_ans)
    if not u:
        return False
    if u == c or u in c or c in u:
        return True
    uw, cw = set(u.split()), set(c.split())
    if cw and len(uw & cw) / len(cw) >= 0.65:
        return True
    return False


def _get_level(xp: int):
    for min_xp, icon, name, next_xp in XP_LEVELS:
        if xp < next_xp:
            pct = (xp - min_xp) / max(next_xp - min_xp, 1)
            return icon, name, round(pct, 3), next_xp - xp
    return "\U0001f3c6", "Legend", 1.0, 0


def _calc_xp(streak: int, card: dict) -> int:
    base = 10
    streak_bonus = min(streak * 3, 30)
    new_card_bonus = 8 if card.get("total_attempts", 0) == 0 else 0
    return base + streak_bonus + new_card_bonus


def _kpi() -> dict:
    cards = list(FLASHCARD_DB.get("cards", {}).values())
    total = len(cards)
    unattempted = sum(1 for c in cards if c["total_attempts"] == 0)
    mastered = sum(1 for c in cards if c.get("consecutive_correct", 0) >= _MASTERED_STREAK)
    weak = sum(1 for c in cards if c["incorrect_count"] > 0
               and c.get("consecutive_correct", 0) < _MASTERED_STREAK)
    return {"total": total, "unattempted": unattempted, "mastered": mastered,
            "weak": weak, "cards": cards}


def _check_achievements():
    s = st.session_state
    unlocked = {a[0] for a in s.session_achievements}
    candidates = []
    if s.session_correct >= 1 and "first_blood" not in unlocked:
        candidates.append("first_blood")
    if s.session_streak >= 3 and "streak_3" not in unlocked:
        candidates.append("streak_3")
    if s.session_streak >= 5 and "streak_5" not in unlocked:
        candidates.append("streak_5")
    if s.session_streak >= 10 and "streak_10" not in unlocked:
        candidates.append("streak_10")
    if s.session_total >= 10 and "ten_done" not in unlocked:
        candidates.append("ten_done")
    if s.session_total >= 20 and "twenty_done" not in unlocked:
        candidates.append("twenty_done")
    if (s.session_total >= 5 and s.session_correct / max(s.session_total, 1) >= 0.90
            and "accuracy_ace" not in unlocked):
        candidates.append("accuracy_ace")
    if s.last_correct and s.last_wrong_count >= 2 and "comeback" not in unlocked:
        candidates.append("comeback")
    for aid in candidates:
        _, icon, name, desc = ACH_MAP[aid]
        s.session_achievements.append((aid, icon, name, desc))
        st.toast(f"{icon} **{name}** unlocked! {desc}", icon="\U0001f3c5")


def _render_tool_events(tool_events: list[dict]) -> None:
    for event in tool_events:
        tn = event.get("tool_name", "?")
        args = event.get("arguments", {})
        raw = event.get("result", "")
        try:
            res = json.loads(raw)
        except Exception:
            res = raw
        with st.expander(f"\u2699\ufe0f Tool: `{tn}`", expanded=False):
            ca, cb = st.columns(2)
            ca.markdown("**Arguments**"); ca.json(args)
            cb.markdown("**Result**");    cb.json(res)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    total_xp = st.session_state.total_xp
    icon, lvl_name, pct, xp_to_next = _get_level(total_xp)

    st.markdown(f"""
    <div class="lvl-badge">
      <div class="lvl-icon">{icon}</div>
      <div class="lvl-name">{lvl_name}</div>
      <div class="lvl-xp">{total_xp} XP {"&bull; " + str(xp_to_next) + " to next level" if xp_to_next < 9000 else "&bull; MAX LEVEL"}</div>
    </div>
    <div class="xp-wrap"><div class="xp-fill" style="width:{int(pct*100)}%;"></div></div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### \U0001f3ae Session Stats")

    streak = st.session_state.session_streak
    if streak >= 3:
        st.markdown(
            f'<div style="text-align:center;margin:6px 0">'
            f'<span class="streak-pill">\U0001f525 {streak} STREAK!</span></div>',
            unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("\u26a1 XP",      st.session_state.session_xp)
    c2.metric("\u2705 Correct", f"{st.session_state.session_correct}/{st.session_state.session_total}")
    c3, c4 = st.columns(2)
    c3.metric("\U0001f525 Streak", streak)
    c4.metric("\U0001f3c6 Best",   st.session_state.session_best_streak)

    if st.session_state.session_total > 0:
        acc = int(100 * st.session_state.session_correct / st.session_state.session_total)
        st.progress(acc / 100, text=f"Accuracy: {acc}%")

    st.divider()
    st.markdown("#### \U0001f0cf Deck Health")
    kpi = _kpi()
    k1, k2 = st.columns(2)
    k1.metric("\U0001f4da Total",  kpi["total"])
    k2.metric("\U0001f195 Unseen", kpi["unattempted"])
    k3, k4 = st.columns(2)
    k3.metric("\U0001f525 Weak",    kpi["weak"])
    k4.metric("\u2705 Mastered",   kpi["mastered"])

    if kpi["total"] > 0:
        mp = kpi["mastered"] / kpi["total"]
        st.progress(mp, text=f"Deck mastery: {int(mp*100)}%")

    if st.session_state.session_achievements:
        st.divider()
        st.markdown("#### \U0001f3c5 Achievements")
        html = "".join(
            f'<span class="ach-pill">{icon} {name}</span>'
            for _, icon, name, _ in st.session_state.session_achievements
        )
        st.markdown(html, unsafe_allow_html=True)

    st.divider()
    with st.expander("\U0001f50d Deck Inspector", expanded=False):
        cards_list = kpi["cards"]
        if not cards_list:
            st.info("No cards yet! Ask the agent to add some.")
        else:
            rows = []
            for c in sorted(cards_list, key=lambda x: x["id"]):
                consec = c.get("consecutive_correct", 0)
                rows.append({
                    "ID":       c["id"],
                    "Question": c["question"][:42] + ("\u2026" if len(c["question"]) > 42 else ""),
                    "Tries":    c["total_attempts"],
                    "Wrong":    c["incorrect_count"],
                    "Streak":   consec,
                    "Done":     "\u2705" if consec >= _MASTERED_STREAK else "\u274c",
                })
            df = pd.DataFrame(rows).set_index("ID")
            st.dataframe(df, use_container_width=True, height=min(38 * len(rows) + 38, 340))

    if st.button("\U0001f504 Reset Session", use_container_width=True):
        for _rk in ["session_xp", "session_streak", "session_best_streak",
                     "session_correct", "session_total", "last_wrong_count", "wrong_streak_count"]:
            st.session_state[_rk] = 0
        st.session_state.session_achievements = []
        st.session_state.game_phase = "home"
        st.session_state.current_card = None
        st.session_state.last_correct = None
        st.rerun()

# ---------------------------------------------------------------------------
# MAIN PANEL  -- Two tabs
# ---------------------------------------------------------------------------
tab_game, tab_chat = st.tabs(["\U0001f3ae  Quiz Game", "\U0001f4ac  Ask the Agent"])


# ============================================================================
# TAB 1 -- QUIZ GAME
# ============================================================================
with tab_game:
    phase = st.session_state.game_phase
    kpi   = _kpi()

    # --- HOME ---
    if phase == "home":
        st.markdown("""
        <div class="hero">
          <h1>\U0001f0cf FlashCard Quest</h1>
          <p>Train your brain &bull; Master your deck &bull; Beat your streak</p>
        </div>
        """, unsafe_allow_html=True)

        if kpi["total"] == 0:
            st.info(
                "\U0001f4ed **Your deck is empty!**\n\n"
                "Switch to the **\U0001f4ac Ask the Agent** tab and type something like:\n"
                "`Add 5 cards on Python data structures` \u2014 then come back to play!"
            )
        else:
            qs1, qs2, qs3, qs4 = st.columns(4)
            qs1.metric("\U0001f4da Cards",   kpi["total"])
            qs2.metric("\U0001f195 Unseen",  kpi["unattempted"])
            qs3.metric("\U0001f525 Weak",    kpi["weak"])
            qs4.metric("\u2705 Mastered",    kpi["mastered"])

            st.markdown("---")
            st.markdown("### \U0001f3af Choose Your Game Mode")

            mode_cols = st.columns(3)
            modes = [
                ("adaptive",    "\U0001f3b2", "Adaptive AI",
                 "Smart priority: weak spots + unseen cards. Best for daily practice."),
                ("unattempted", "\U0001f195", "Explorer",
                 "Only shows cards you have NEVER seen. Perfect for new material."),
                ("weakest",     "\U0001f525", "Weak Spot Blitz",
                 "Hammers your worst cards until you nail them. Intense review."),
            ]

            for col, (mode_val, m_icon, m_title, m_desc) in zip(mode_cols, modes):
                with col:
                    selected = st.session_state.quiz_mode == mode_val
                    border = "2px solid #7c3aed" if selected else "1px solid rgba(124,58,237,0.3)"
                    bg     = "rgba(124,58,237,0.22)" if selected else "rgba(124,58,237,0.08)"
                    chk    = " \u2713" if selected else ""
                    st.markdown(f"""
                    <div class="mode-card" style="border:{border};background:{bg}">
                      <div class="mc-icon">{m_icon}</div>
                      <div class="mc-title">{m_title}{chk}</div>
                      <div class="mc-desc">{m_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    btn_type = "primary" if selected else "secondary"
                    if st.button(f"Select {m_title}", key=f"mode_{mode_val}",
                                 use_container_width=True, type=btn_type):
                        st.session_state.quiz_mode = mode_val
                        st.rerun()

            st.markdown("---")

            all_words = sorted({
                w for c in kpi["cards"]
                for w in re.findall(r"\b[a-zA-Z]{4,}\b", c["question"].lower())
                if w not in STOP_WORDS
            })
            topic_opts = ["\U0001f310 All Topics"] + all_words
            chosen_topic = st.selectbox(
                "\U0001f4cc Filter by Topic (optional)", topic_opts,
                help="Narrow the quiz to a keyword found in your card questions."
            )
            st.session_state.quiz_topic = (
                "" if chosen_topic == "\U0001f310 All Topics" else chosen_topic
            )

            st.markdown("<br>", unsafe_allow_html=True)
            launch_col = st.columns([1, 2, 1])[1]
            with launch_col:
                if st.button("\U0001f680  Start Quiz!", use_container_width=True, type="primary"):
                    st.session_state.game_phase   = "question"
                    st.session_state.current_card = None
                    st.rerun()

    # --- QUESTION ---
    elif phase == "question":
        if st.session_state.current_card is None:
            raw    = _quiz_me_raw(
                mode  = st.session_state.quiz_mode,
                topic = st.session_state.quiz_topic or None,
                count = 1,
            )
            result = json.loads(raw)
            status = result.get("status", "")

            if status in ("empty", "no_match", "no_unattempted"):
                st.warning(f"\u26a0\ufe0f {result.get('message', 'No cards available.')}")
                if st.button("\U0001f3e0 Back to Home"):
                    st.session_state.game_phase = "home"
                    st.rerun()
                st.stop()

            card = result.get("card") or (result.get("cards") or [{}])[0]
            st.session_state.current_card        = card
            st.session_state.correct_answer_text = card.get("answer", "")
            st.session_state.priority_reason     = result.get("priority_reason", "")

        card   = st.session_state.current_card
        streak = st.session_state.session_streak

        hdr1, hdr2, hdr3 = st.columns([3, 2, 2])
        with hdr1:
            mode_label = {
                "adaptive":    "\U0001f3b2 Adaptive",
                "unattempted": "\U0001f195 Explorer",
                "weakest":     "\U0001f525 Weak Spot",
            }.get(st.session_state.quiz_mode, "Quiz")
            topic_label = f" \u2014 {st.session_state.quiz_topic}" if st.session_state.quiz_topic else ""
            st.markdown(f"### {mode_label}{topic_label}")
        with hdr2:
            if streak >= 3:
                st.markdown(
                    f'<div style="text-align:center;padding-top:8px">'
                    f'<span class="streak-pill">\U0001f525 {streak} streak!</span></div>',
                    unsafe_allow_html=True)
        with hdr3:
            st.markdown(
                f'<div style="text-align:right;padding-top:12px;color:#c4b5fd;font-weight:700">'
                f'\u26a1 {st.session_state.session_xp} XP</div>',
                unsafe_allow_html=True)

        mode_badge = {
            "adaptive":    "Adaptive Pick",
            "unattempted": "New Card!",
            "weakest":     "Weak Spot",
        }.get(st.session_state.quiz_mode, "Question")

        meta_parts: list[str] = [f"Card #{card['id']}"]
        if card.get("total_attempts", 0) == 0:
            meta_parts.append("\U0001f195 First time!")
        else:
            meta_parts.append(f"Seen {card['total_attempts']}x")
        if card.get("incorrect_count", 0) > 0:
            meta_parts.append(f"\u274c {card['incorrect_count']} mistake(s)")
        if card.get("consecutive_correct", 0) >= _MASTERED_STREAK:
            meta_parts.append("\u2705 Mastered")

        st.markdown(f"""
        <div class="q-card">
          <div class="badge-top">\U0001f3ae {mode_badge}</div>
          <h2>{card["question"]}</h2>
          <div class="q-meta">{"  &bull;  ".join(meta_parts)}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.priority_reason:
            st.caption(f"\U0001f4a1 {st.session_state.priority_reason}")

        with st.form("ans_form", clear_on_submit=True):
            user_ans = st.text_input(
                "Your Answer",
                placeholder="Type your answer here and press Enter or click Submit...",
                label_visibility="collapsed",
            )
            btn1, btn2, btn3 = st.columns([4, 2, 2])
            submitted = btn1.form_submit_button("\u2705  Submit Answer", use_container_width=True, type="primary")
            skipped   = btn2.form_submit_button("\u23ed\ufe0f  Skip Card",    use_container_width=True)
            go_home   = btn3.form_submit_button("\U0001f3e0  Home",           use_container_width=True)

        if go_home:
            st.session_state.game_phase   = "home"
            st.session_state.current_card = None
            st.rerun()

        if skipped:
            st.session_state.last_correct     = None
            st.session_state.session_streak   = 0
            st.session_state.last_xp          = 0
            st.session_state.game_phase       = "feedback"
            st.session_state.current_card     = None
            st.rerun()

        if submitted:
            if not user_ans.strip():
                st.warning("Please type your answer before submitting!")
            else:
                is_correct   = _fuzzy_match(user_ans.strip(), card["answer"])
                _record_answer_raw(card["id"], is_correct)
                prev_wrong   = st.session_state.wrong_streak_count
                st.session_state.session_total += 1

                if is_correct:
                    st.session_state.session_correct     += 1
                    st.session_state.session_streak      += 1
                    st.session_state.session_best_streak  = max(
                        st.session_state.session_best_streak,
                        st.session_state.session_streak,
                    )
                    xp = _calc_xp(st.session_state.session_streak, card)
                    st.session_state.session_xp   += xp
                    st.session_state.total_xp     += xp
                    st.session_state.last_xp       = xp
                    st.session_state.last_wrong_count   = prev_wrong
                    st.session_state.wrong_streak_count = 0
                else:
                    st.session_state.session_streak     = 0
                    st.session_state.wrong_streak_count = prev_wrong + 1
                    st.session_state.last_xp            = 0

                st.session_state.last_correct = is_correct
                st.session_state.game_phase   = "feedback"
                st.session_state.current_card = None
                _check_achievements()
                st.rerun()

    # --- FEEDBACK ---
    elif phase == "feedback":
        is_correct  = st.session_state.last_correct
        answer_text = st.session_state.correct_answer_text
        xp_gained   = st.session_state.last_xp
        streak      = st.session_state.session_streak

        if is_correct is None:
            st.markdown(f"""
            <div class="fb-skip">
              <div style="font-size:3rem">\u23ed\ufe0f</div>
              <h2 style="color:#94a3b8;margin:8px 0">Card Skipped</h2>
              <p style="color:#64748b">No worries \u2014 it will come back around.</p>
              <p style="color:#94a3b8;margin-top:12px">The answer was:<br>
                <strong style="color:#f8fafc;font-size:1.1rem">{answer_text}</strong></p>
            </div>
            """, unsafe_allow_html=True)

        elif is_correct:
            if streak >= 5:
                st.balloons()
            fire       = "\U0001f525" if streak >= 3 else "\u2705"
            streak_msg = f"  &bull;  \U0001f525 {streak} in a row!" if streak >= 2 else ""
            fire_prefix = "ON FIRE!&nbsp;&nbsp;" if streak >= 3 else ""
            st.markdown(f"""
            <div class="fb-correct">
              <div style="font-size:3.8rem">{fire}</div>
              <h2 style="color:#10b981;margin:8px 0">{fire_prefix}Correct!</h2>
              <p style="color:#6ee7b7;font-size:1.1rem;font-weight:700">+{xp_gained} XP earned{streak_msg}</p>
              <p style="color:#94a3b8;margin-top:8px">Answer: <strong style="color:#f8fafc">{answer_text}</strong></p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="fb-wrong">
              <div style="font-size:3.8rem">\u274c</div>
              <h2 style="color:#ef4444;margin:8px 0">Not Quite!</h2>
              <p style="color:#94a3b8">The correct answer is:</p>
              <div style="background:rgba(255,255,255,0.06);border-radius:10px;padding:12px 18px;margin:10px 0">
                <strong style="color:#f8fafc;font-size:1.15rem">{answer_text}</strong>
              </div>
              <p style="color:#475569;font-size:0.82rem">This card is flagged for extra review \U0001f501</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("\u25b6\ufe0f  Next Question", use_container_width=True, type="primary"):
                st.session_state.game_phase   = "question"
                st.session_state.current_card = None
                st.session_state.last_correct = None
                st.rerun()
        with b2:
            if st.button("\U0001f504  Change Mode", use_container_width=True):
                st.session_state.game_phase   = "home"
                st.session_state.current_card = None
                st.rerun()
        with b3:
            if st.button("\U0001f3e0  Home", use_container_width=True):
                st.session_state.game_phase   = "home"
                st.session_state.current_card = None
                st.rerun()

        st.markdown("---")
        st.markdown("#### \U0001f4ca Session Scoreboard")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("\u26a1 XP",       st.session_state.session_xp)
        sc2.metric("\u2705 Correct",  st.session_state.session_correct)
        sc3.metric("\U0001f4dd Answered", st.session_state.session_total)
        acc_val = int(100 * st.session_state.session_correct / max(st.session_state.session_total, 1))
        sc4.metric("\U0001f3af Accuracy", f"{acc_val}%")

        if st.session_state.session_best_streak >= 2:
            st.info(
                f"\U0001f3c6 Best streak this session: "
                f"**{st.session_state.session_best_streak}** correct in a row!"
            )
        if st.session_state.session_achievements:
            ach_html = "".join(
                f'<span class="ach-pill">{ico} {nm}</span>'
                for _, ico, nm, _ in st.session_state.session_achievements
            )
            st.markdown("\U0001f3c5 **Unlocked:** " + ach_html, unsafe_allow_html=True)


# ============================================================================
# TAB 2 -- ASK THE AGENT
# ============================================================================
with tab_chat:
    st.markdown("### \U0001f4ac Chat with Your AI Study Agent")
    st.caption(
        f"Model: `{GROQ_MODEL}`  \u2022  Engine: ReAct Loop  \u2022  "
        "Memory: Sliding Window (6 turns)"
    )

    st.markdown("#### \U0001f4a1 Example Prompts \u2014 click any to send instantly")
    for category, prompts in EXAMPLE_PROMPTS.items():
        with st.expander(category, expanded=False):
            cols = st.columns(2)
            for i, p in enumerate(prompts):
                label = p if len(p) <= 58 else p[:55] + "\u2026"
                if cols[i % 2].button(label, key=f"ep_{hash(p)}", use_container_width=True):
                    st.session_state.pending_prompt = p

    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("tool_events"):
                _render_tool_events(msg["tool_events"])
            st.markdown(msg["content"])

    def _run_agent(user_input: str) -> None:
        user_input = user_input.strip()
        if not user_input:
            return
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append(
            {"role": "user", "content": user_input, "tool_events": None}
        )
        collected: list[dict] = []

        def _cb(ev: dict) -> None:
            collected.append(ev)

        with st.chat_message("assistant"):
            with st.spinner("\U0001f914 Agent thinking\u2026"):
                reply = st.session_state.agent.chat(user_input, on_tool_event=_cb)
            if collected:
                _render_tool_events(collected)
            st.markdown(reply)

        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "tool_events": collected or None,
        })

    if st.session_state.pending_prompt:
        p = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        _run_agent(p)
        st.rerun()

    if user_text := st.chat_input("Ask the agent to quiz you, add cards, or show stats\u2026"):
        _run_agent(user_text)
        st.rerun()
