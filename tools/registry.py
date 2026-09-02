"""
tools/registry.py
==================
Dispatch Map (Tool Registry) pattern implementation.

TOOL_REGISTRY maps every tool name string to its callable, enabling O(1)
dispatch in the agent loop with no if/elif chains.

AVAILABLE_SCHEMAS is the list of OpenAI-compatible JSON tool schemas sent
to the Groq API on every inference call so the model knows which tools exist.

Tools in this registry
-----------------------
  add_card            -- single-card insertion (1-3 explicit Q&A pairs)
  add_cards_batch     -- bulk insertion in one tool call (4+ cards)
  quiz_me             -- adaptive priority selection: mode / topic / count
  record_answer       -- metric update with correct/incorrect decay
  get_stats           -- full database diagnostic summary
"""

from __future__ import annotations

from .flashcards import add_card, add_cards_batch, quiz_me, record_answer, get_stats

# ---------------------------------------------------------------------------
# OpenAI-compatible JSON schemas
# ---------------------------------------------------------------------------

ADD_CARD_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "add_card",
        "description": (
            "Add a single flashcard to the study database. "
            "Use this for 1-3 cards when the user provides explicit Q&A pairs. "
            "For bulk creation (4+ cards) or autonomous generation on a topic, "
            "use add_cards_batch to avoid mid-task abandonment from token-budget limits. "
            "If the user asks to create cards on a general knowledge topic WITHOUT answers, "
            "generate accurate Q&A from your own knowledge -- do NOT ask the user."
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

ADD_CARDS_BATCH_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "add_cards_batch",
        "description": (
            "Insert multiple flashcards in a single tool call. "
            "ALWAYS use this instead of multiple add_card calls when: "
            "(a) the user requests 4 or more cards, "
            "(b) the user says 'add N cards on [topic]', or "
            "(c) you are autonomously generating cards on a general knowledge topic. "
            "Accepts a list of {question, answer} objects and inserts them atomically. "
            "Resolves generation derailment from per-turn token-budget exhaustion "
            "that previously caused the model to stop after 3 cards and hallucinate a mode switch."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cards": {
                    "type": "array",
                    "description": (
                        "List of flashcard objects to insert. "
                        "Each must have 'question' and 'answer' string fields."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question text.",
                            },
                            "answer": {
                                "type": "string",
                                "description": "The correct answer text.",
                            },
                        },
                        "required": ["question", "answer"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["cards"],
            "additionalProperties": False,
        },
    },
}

QUIZ_ME_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "quiz_me",
        "description": (
            "Retrieve the highest-priority flashcard(s) for the student to answer next. "
            "ALWAYS call this before presenting any quiz question -- never rely on memory. "
            "Returns status='empty' if deck is empty, status='no_match' if topic has no cards. "
            "Use topic= to filter by keyword (e.g. topic='docker' for 'quiz me on docker'). "
            "Use count=N to fetch N cards in ONE call (e.g. count=5 for '5 questions at once'). "
            "NEVER fire multiple quiz_me calls in one turn -- use the count parameter instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["adaptive", "unattempted", "weakest"],
                    "description": (
                        "Selection strategy. "
                        "'adaptive' (default): urgency-score formula balancing weak spots and unattempted cards. "
                        "'unattempted': strictly picks cards with total_attempts == 0. "
                        "'weakest': strictly picks cards with the highest incorrect_count."
                    ),
                },
                "topic": {
                    "type": "string",
                    "description": (
                        "Optional keyword to filter cards by topic. "
                        "Case-insensitive substring match on question and answer text. "
                        "Examples: 'docker', 'ACID', 'BST', 'india', 'bollywood'."
                    ),
                },
                "count": {
                    "type": "integer",
                    "description": (
                        "Number of cards to return (default 1, minimum 1). "
                        "Use this when the user asks for multiple questions at once. "
                        "Example: 'give me 5 questions' -> count=5. "
                        "Returns at most the number of available matching candidates."
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
            "Incorrect answers reset the streak and increment incorrect_count. "
            "Use is_correct=True for partial/fuzzy matches when the core entity is "
            "unambiguously identified (e.g. 'Ashutosh' for 'Ashutosh Gowariker' is correct)."
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
    ADD_CARDS_BATCH_SCHEMA,
    QUIZ_ME_SCHEMA,
    RECORD_ANSWER_SCHEMA,
    GET_STATS_SCHEMA,
]

#: O(1) dispatch map: tool_name -> callable.
TOOL_REGISTRY: dict[str, callable] = {
    "add_card":        add_card,
    "add_cards_batch": add_cards_batch,
    "quiz_me":         quiz_me,
    "record_answer":   record_answer,
    "get_stats":       get_stats,
}
