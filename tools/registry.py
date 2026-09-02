"""
tools/registry.py
==================
Dispatch Map (Tool Registry) pattern implementation.

``TOOL_REGISTRY`` maps every tool name string to its callable, enabling O(1)
dispatch in the agent loop with no if/elif chains.

``AVAILABLE_SCHEMAS`` is the list of OpenAI-compatible JSON tool schemas sent
to the Groq API on every inference call so the model knows which tools exist.
"""

from __future__ import annotations

from .flashcards import add_card, quiz_me, record_answer, get_stats

# ---------------------------------------------------------------------------
# OpenAI-compatible JSON schemas
# ---------------------------------------------------------------------------

ADD_CARD_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "add_card",
        "description": (
            "Add a new flashcard to the study database. "
            "Call this when the user provides BOTH a question AND an answer. "
            "If the user asks you to create cards on a topic WITHOUT providing answers, "
            "you MUST generate accurate questions and answers from your own knowledge "
            "and call add_card autonomously -- do NOT ask the user for the answer. "
            "Only ask the user for the answer when the topic is personal/subjective "
            "(e.g. 'add a card about my meeting on Tuesday')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question text for the flashcard.",
                },
                "answer": {
                    "type": "string",
                    "description": "The correct answer text for the flashcard.",
                },
            },
            "required": ["question", "answer"],
            "additionalProperties": False,
        },
    },
}

QUIZ_ME_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "quiz_me",
        "description": (
            "Retrieve the highest-priority flashcard the student should be quizzed on next. "
            "Supports three modes: 'adaptive' (default, uses urgency score formula), "
            "'unattempted' (only cards never seen before), 'weakest' (highest error count). "
            "Also accepts an optional 'topic' string for keyword filtering -- use this when "
            "the user says 'quiz me on docker' or 'test me on ACID'. "
            "Always call this before presenting any quiz question. "
            "If the database is empty the tool returns status='empty' -- relay that message "
            "to the user and ask them to add cards first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["adaptive", "unattempted", "weakest"],
                    "description": (
                        "Selection strategy. "
                        "'adaptive' (default): urgency score formula balancing weak spots "
                        "and unattempted cards. "
                        "'unattempted': strictly picks cards with total_attempts == 0. "
                        "'weakest': strictly picks the card with the highest incorrect_count."
                    ),
                },
                "topic": {
                    "type": "string",
                    "description": (
                        "Optional keyword to filter cards by topic. "
                        "Case-insensitive substring match on question and answer text. "
                        "E.g. 'docker', 'ACID', 'BST'."
                    ),
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

RECORD_ANSWER_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "record_answer",
        "description": (
            "Record whether the student answered a flashcard correctly or incorrectly. "
            "Always call this after evaluating the student response so the adaptive "
            "engine can update weak-spot priorities. "
            "Correct answers increment consecutive_correct and decay incorrect_count. "
            "Incorrect answers reset the streak and increment incorrect_count."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "integer",
                    "description": "The numeric ID of the flashcard that was just answered.",
                },
                "is_correct": {
                    "type": "boolean",
                    "description": "True if the student answered correctly, False otherwise.",
                },
            },
            "required": ["card_id", "is_correct"],
            "additionalProperties": False,
        },
    },
}

GET_STATS_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_stats",
        "description": (
            "Return a full summary of the flashcard database: total card count, "
            "unattempted count, mastered count, per-card accuracy, urgency scores, "
            "and a ranked list of weak spots. Call this when the user asks for a "
            "summary, diagnostic, or wants to know which topics they are struggling with."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}

# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

#: List of schemas sent to the Groq API on every call.
AVAILABLE_SCHEMAS: list[dict] = [
    ADD_CARD_SCHEMA,
    QUIZ_ME_SCHEMA,
    RECORD_ANSWER_SCHEMA,
    GET_STATS_SCHEMA,
]

#: O(1) dispatch map: tool_name -> callable.
TOOL_REGISTRY: dict[str, callable] = {
    "add_card":      add_card,
    "quiz_me":       quiz_me,
    "record_answer": record_answer,
    "get_stats":     get_stats,
}
