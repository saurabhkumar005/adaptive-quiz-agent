"""
tools/flashcards.py
====================
Persistent flashcard state store and the three agent tools:
  - add_card(question, answer)      → adds a new card
  - quiz_me()                       → returns the highest-priority weak card
  - record_answer(card_id, is_correct) → updates card metrics

State is persisted to ``flashcard_db.json`` in the project root so that
weak-spot data survives across agent sessions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).resolve().parent.parent / "flashcard_db.json"


def _load_db() -> dict[str, Any]:
    """Load the flashcard database from disk, or return a fresh empty store."""
    if _DB_PATH.exists():
        try:
            with open(_DB_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {"next_id": 1, "cards": {}}


def _save_db(db: dict[str, Any]) -> None:
    """Persist the flashcard database to disk atomically."""
    tmp = _DB_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(db, fh, indent=2, ensure_ascii=False)
    tmp.replace(_DB_PATH)


# ---------------------------------------------------------------------------
# In-memory state (loaded once per process, flushed on every write)
# ---------------------------------------------------------------------------

FLASHCARD_DB: dict[str, Any] = _load_db()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def add_card(question: str, answer: str) -> str:
    """
    Add a new flashcard to the database.

    Parameters
    ----------
    question : str
        The question text for the flashcard.
    answer : str
        The correct answer text for the flashcard.

    Returns
    -------
    str
        A JSON-encoded confirmation object containing the new card's id.
    """
    card_id = FLASHCARD_DB["next_id"]
    FLASHCARD_DB["next_id"] += 1

    FLASHCARD_DB["cards"][str(card_id)] = {
        "id": card_id,
        "question": question,
        "answer": answer,
        "incorrect_count": 0,
        "total_attempts": 0,
    }

    _save_db(FLASHCARD_DB)

    result = {
        "status": "success",
        "message": f"Flashcard #{card_id} added successfully.",
        "card": {
            "id": card_id,
            "question": question,
            "answer": answer,
        },
    }
    return json.dumps(result)


def quiz_me() -> str:
    """
    Return the highest-priority flashcard for the student to answer.

    Selection strategy (agentic adaptive logic):
      1. Cards with ``incorrect_count > 0`` are filtered and sorted descending
         by ``incorrect_count`` — the weakest card is served first.
      2. If no card has been answered incorrectly yet, fall back to the card
         with the fewest ``total_attempts`` (least-seen card).
      3. If the database is empty, return an instructional prompt.

    Returns
    -------
    str
        A JSON-encoded object containing the selected card's id and question,
        plus metadata to help the agent explain priority reasoning.
    """
    cards = list(FLASHCARD_DB["cards"].values())

    if not cards:
        return json.dumps({
            "status": "empty",
            "message": "No flashcards found. Use add_card to add some first!",
        })

    # Weak cards: at least one incorrect answer recorded
    weak_cards = [c for c in cards if c["incorrect_count"] > 0]

    if weak_cards:
        chosen = max(weak_cards, key=lambda c: c["incorrect_count"])
        priority = "weak_spot"
    else:
        # Fall back: least-attempted card (highest novelty)
        chosen = min(cards, key=lambda c: c["total_attempts"])
        priority = "least_seen"

    return json.dumps({
        "status": "success",
        "priority": priority,
        "card": {
            "id": chosen["id"],
            "question": chosen["question"],
            "incorrect_count": chosen["incorrect_count"],
            "total_attempts": chosen["total_attempts"],
        },
    })


def record_answer(card_id: int, is_correct: bool) -> str:
    """
    Record whether a student answered a specific flashcard correctly.

    Parameters
    ----------
    card_id : int
        The numeric identifier of the card that was answered.
    is_correct : bool
        ``True`` if the student answered correctly, ``False`` otherwise.

    Returns
    -------
    str
        A JSON-encoded object with the updated card metrics.
    """
    key = str(card_id)
    if key not in FLASHCARD_DB["cards"]:
        return json.dumps({
            "status": "error",
            "message": f"Card #{card_id} not found in the database.",
        })

    card = FLASHCARD_DB["cards"][key]
    card["total_attempts"] += 1
    if not is_correct:
        card["incorrect_count"] += 1

    _save_db(FLASHCARD_DB)

    return json.dumps({
        "status": "success",
        "card_id": card_id,
        "is_correct": is_correct,
        "updated_metrics": {
            "incorrect_count": card["incorrect_count"],
            "total_attempts": card["total_attempts"],
        },
        "message": (
            "✅ Correct! Card metrics updated."
            if is_correct
            else f"❌ Incorrect. Card #{card_id} is now prioritised for review."
        ),
    })
