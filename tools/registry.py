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

from .flashcards import add_card, quiz_me, record_answer

# ---------------------------------------------------------------------------
# OpenAI-compatible JSON schemas
# ---------------------------------------------------------------------------

ADD_CARD_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "add_card",
        "description": (
            "Add a new flashcard to the study database. "
            "Call this whenever the user wants to create or save a new question-answer pair."
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
            "The tool automatically selects the card with the most incorrect answers first "
            "(adaptive weak-spot targeting). Call this before presenting any quiz question."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
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
            "Always call this after evaluating the student's response so the adaptive "
            "engine can update weak-spot priorities."
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

# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

#: List of schemas sent to the Groq API on every call.
AVAILABLE_SCHEMAS: list[dict] = [
    ADD_CARD_SCHEMA,
    QUIZ_ME_SCHEMA,
    RECORD_ANSWER_SCHEMA,
]

#: O(1) dispatch map: tool_name -> callable.
TOOL_REGISTRY: dict[str, callable] = {
    "add_card": add_card,
    "quiz_me": quiz_me,
    "record_answer": record_answer,
}
