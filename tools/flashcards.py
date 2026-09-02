"""
tools/flashcards.py
====================
Persistent flashcard state store and the four agent tools:
  - add_card(question, answer)         → adds a new card
  - quiz_me()                          → returns the highest-priority weak card
  - record_answer(card_id, is_correct) → updates card metrics
  - get_stats()                        → returns a full database summary

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
    question = question.strip()
    answer = answer.strip()

    if not question or not answer:
        return json.dumps({
            "status": "error",
            "message": "Both 'question' and 'answer' must be non-empty strings.",
        })

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
      3. If the database is empty, return a clear instructional message.

    Returns
    -------
    str
        A JSON-encoded object containing the selected card's id, question,
        answer (for agent evaluation only), and priority metadata.
    """
    cards = list(FLASHCARD_DB["cards"].values())

    if not cards:
        return json.dumps({
            "status": "empty",
            "message": (
                "Your flashcard deck is empty! "
                "Please add some cards first using add_card before quizzing."
            ),
        })

    # Weak cards: at least one incorrect answer recorded
    weak_cards = [c for c in cards if c["incorrect_count"] > 0]

    if weak_cards:
        chosen = max(weak_cards, key=lambda c: c["incorrect_count"])
        priority = "weak_spot"
        priority_reason = (
            f"This card has been answered incorrectly "
            f"{chosen['incorrect_count']} time(s) — highest error count."
        )
    else:
        # Fall back: least-attempted card (highest novelty)
        chosen = min(cards, key=lambda c: c["total_attempts"])
        priority = "least_seen"
        priority_reason = (
            f"No errors recorded yet. Serving the least-seen card "
            f"({chosen['total_attempts']} attempts)."
        )

    return json.dumps({
        "status": "success",
        "priority": priority,
        "priority_reason": priority_reason,
        "card": {
            "id": chosen["id"],
            "question": chosen["question"],
            "answer": chosen["answer"],
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
            else f"❌ Incorrect. Card #{card_id} is now prioritised for review "
                 f"(total errors: {card['incorrect_count']})."
        ),
    })


def get_stats() -> str:
    """
    Return a complete summary of the flashcard database state.

    Includes total card count, per-card metrics, and a ranked list
    of weak spots so the agent can give an accurate diagnostic summary.

    Returns
    -------
    str
        A JSON-encoded statistics object.
    """
    cards = list(FLASHCARD_DB["cards"].values())
    total = len(cards)

    if total == 0:
        return json.dumps({
            "status": "empty",
            "total_cards": 0,
            "message": "No flashcards in the database yet.",
        })

    # Sort: most incorrect first, then most attempted
    sorted_cards = sorted(
        cards,
        key=lambda c: (-c["incorrect_count"], -c["total_attempts"])
    )

    card_summaries = [
        {
            "id": c["id"],
            "question": c["question"],
            "incorrect_count": c["incorrect_count"],
            "total_attempts": c["total_attempts"],
            "accuracy": (
                f"{round((1 - c['incorrect_count'] / c['total_attempts']) * 100)}%"
                if c["total_attempts"] > 0 else "not attempted"
            ),
        }
        for c in sorted_cards
    ]

    weakest = sorted_cards[0] if sorted_cards[0]["incorrect_count"] > 0 else None

    return json.dumps({
        "status": "success",
        "total_cards": total,
        "weakest_card": {
            "id": weakest["id"],
            "question": weakest["question"],
            "incorrect_count": weakest["incorrect_count"],
        } if weakest else None,
        "all_cards": card_summaries,
    })
