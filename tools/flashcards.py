"""
tools/flashcards.py
====================
Persistent flashcard state store and the four agent tools:
  - add_card(question, answer)               -> adds a new card
  - quiz_me(mode, topic)                     -> returns the highest-priority card
                                               using the smart adaptive urgency score
  - record_answer(card_id, is_correct)       -> updates card metrics with decay
  - get_stats()                              -> returns a full database summary

State is persisted to ``flashcard_db.json`` in the project root so that
weak-spot data survives across agent sessions.

Bug fixes included in this revision
-------------------------------------
* Card Starvation / Priority Lockout: ``incorrect_count`` now decays on
  correct answers via ``consecutive_correct`` tracking. Unattempted cards
  receive a base urgency of 5.0, so they are always explored before stale
  mastered cards.
* Parameterised ``quiz_me``: accepts ``mode`` (adaptive / unattempted /
  weakest) and ``topic`` (case-insensitive substring filter).
* ``record_answer`` decay: correct answers now decrement ``incorrect_count``
  (floor 0) and increment ``consecutive_correct``; incorrect answers reset
  ``consecutive_correct`` to 0 and increment ``incorrect_count``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
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
# Urgency score (Smart Adaptive Priority Algorithm)
# ---------------------------------------------------------------------------

_MASTERED_STREAK = 2          # consecutive_correct threshold for mastered state
_UNATTEMPTED_WEIGHT = 5.0     # base weight for cards never attempted
_MASTERED_CAP = 0.5           # weight ceiling once a card is mastered


def _urgency(card: dict[str, Any]) -> float:
    """
    Calculate a priority urgency score for a single card.

    Rules
    -----
    * Unattempted (``total_attempts == 0``):       weight = 5.0
    * Mastered    (``consecutive_correct >= 2``):  weight capped at 0.5
    * Otherwise:  weight = (incorrect_count x 3.0) - (consecutive_correct x 1.5)
      -- minimum 0.0 to avoid negative values confusing selection.
    """
    if card["total_attempts"] == 0:
        return _UNATTEMPTED_WEIGHT

    if card.get("consecutive_correct", 0) >= _MASTERED_STREAK:
        return _MASTERED_CAP

    raw = (card["incorrect_count"] * 3.0) - (card.get("consecutive_correct", 0) * 1.5)
    return max(raw, 0.0)


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
        A JSON-encoded confirmation object containing the new card id.
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
        "consecutive_correct": 0,
        "last_attempted": None,
    }

    _save_db(FLASHCARD_DB)

    return json.dumps({
        "status": "success",
        "message": f"Flashcard #{card_id} added successfully.",
        "card": {
            "id": card_id,
            "question": question,
            "answer": answer,
        },
    })


def quiz_me(mode: str = "adaptive", topic: str | None = None) -> str:
    """
    Return the highest-priority flashcard using the smart adaptive algorithm.

    Parameters
    ----------
    mode : str, optional
        Selection strategy:
        * ``"adaptive"``    -- urgency-score formula balancing weak spots and
                               unattempted cards (default).
        * ``"unattempted"`` -- strictly serve cards with total_attempts == 0.
        * ``"weakest"``     -- strictly serve the card with the highest historical
                               incorrect_count.
    topic : str or None, optional
        If provided, only cards whose question or answer contains this
        substring (case-insensitive) are considered.

    Returns
    -------
    str
        A JSON-encoded object containing the selected card and priority metadata.
    """
    all_cards = list(FLASHCARD_DB["cards"].values())

    if not all_cards:
        return json.dumps({
            "status": "empty",
            "message": (
                "Your flashcard deck is empty! "
                "Please add some cards first using add_card before quizzing."
            ),
        })

    # ---- Topic filter -------------------------------------------------------
    if topic:
        needle = topic.lower()
        candidates = [
            c for c in all_cards
            if needle in c["question"].lower() or needle in c["answer"].lower()
        ]
        if not candidates:
            return json.dumps({
                "status": "no_match",
                "message": (
                    f"No cards matched topic '{topic}'. "
                    "Try a different keyword or add cards on that topic first."
                ),
            })
        topic_note = f" matching topic '{topic}'"
    else:
        candidates = all_cards
        topic_note = ""

    # ---- Mode dispatch -------------------------------------------------------
    mode = (mode or "adaptive").lower().strip()

    if mode == "unattempted":
        pool = [c for c in candidates if c["total_attempts"] == 0]
        if not pool:
            return json.dumps({
                "status": "no_unattempted",
                "message": (
                    f"All cards{topic_note} have been attempted at least once. "
                    "Try mode='adaptive' to review weak spots."
                ),
            })
        chosen = pool[0]
        priority = "unattempted"
        priority_reason = f"Unattempted card{topic_note} -- never been served before."

    elif mode == "weakest":
        pool = sorted(candidates, key=lambda c: c["incorrect_count"], reverse=True)
        chosen = pool[0]
        priority = "weakest"
        priority_reason = (
            f"Weakest card{topic_note} -- highest historical incorrect_count "
            f"({chosen['incorrect_count']} error(s))."
        )

    else:  # adaptive (default)
        scored = [(c, _urgency(c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        chosen, score = scored[0]

        if chosen["total_attempts"] == 0:
            priority = "unattempted"
            priority_reason = (
                f"Unattempted card{topic_note} -- base urgency {score:.2f}. "
                "Serving to ensure full deck coverage."
            )
        elif chosen.get("consecutive_correct", 0) >= _MASTERED_STREAK:
            priority = "mastered_review"
            priority_reason = (
                f"All cards{topic_note} appear mastered (streak >= {_MASTERED_STREAK}). "
                f"Serving the least-mastered card for periodic review (urgency {score:.2f})."
            )
        else:
            priority = "weak_spot"
            priority_reason = (
                f"Weak spot{topic_note} -- urgency score {score:.2f} "
                f"(incorrect_count={chosen['incorrect_count']}, "
                f"consecutive_correct={chosen.get('consecutive_correct', 0)})."
            )

    return json.dumps({
        "status": "success",
        "mode": mode,
        "priority": priority,
        "priority_reason": priority_reason,
        "card": {
            "id": chosen["id"],
            "question": chosen["question"],
            "answer": chosen["answer"],
            "incorrect_count": chosen["incorrect_count"],
            "total_attempts": chosen["total_attempts"],
            "consecutive_correct": chosen.get("consecutive_correct", 0),
        },
    })


def record_answer(card_id: int, is_correct: bool) -> str:
    """
    Record whether a student answered a specific flashcard correctly.

    Metric update rules
    -------------------
    Correct answer:
      * total_attempts      += 1
      * consecutive_correct += 1
      * incorrect_count     -= 1  (floor 0) -- weight-decay fix

    Incorrect answer:
      * total_attempts      += 1
      * consecutive_correct  = 0  (streak broken)
      * incorrect_count     += 1

    Parameters
    ----------
    card_id : int
        The numeric identifier of the card that was answered.
    is_correct : bool
        True if the student answered correctly, False otherwise.

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

    # Ensure legacy cards loaded from disk have the new fields
    card.setdefault("consecutive_correct", 0)
    card.setdefault("last_attempted", None)

    card["total_attempts"] += 1
    card["last_attempted"] = datetime.now(timezone.utc).isoformat()

    if is_correct:
        card["consecutive_correct"] += 1
        if card["incorrect_count"] > 0:
            card["incorrect_count"] -= 1   # decay: correct answers reduce error weight
    else:
        card["consecutive_correct"] = 0    # streak reset on wrong answer
        card["incorrect_count"] += 1

    _save_db(FLASHCARD_DB)

    mastered = card["consecutive_correct"] >= _MASTERED_STREAK
    mastery_note = " Card is now mastered!" if (is_correct and mastered) else ""

    return json.dumps({
        "status": "success",
        "card_id": card_id,
        "is_correct": is_correct,
        "updated_metrics": {
            "incorrect_count": card["incorrect_count"],
            "total_attempts": card["total_attempts"],
            "consecutive_correct": card["consecutive_correct"],
            "mastered": mastered,
        },
        "message": (
            f"Correct! Card metrics updated.{mastery_note}"
            if is_correct
            else (
                f"Incorrect. Card #{card_id} is now prioritised for review "
                f"(total errors: {card['incorrect_count']})."
            )
        ),
    })


def get_stats() -> str:
    """
    Return a complete summary of the flashcard database state.

    Includes total card count, per-card metrics (with mastery status and
    urgency score), and a ranked list of weak spots so the agent can give
    an accurate diagnostic summary.

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

    # Sort: highest urgency first
    sorted_cards = sorted(cards, key=_urgency, reverse=True)

    card_summaries = [
        {
            "id": c["id"],
            "question": c["question"],
            "incorrect_count": c["incorrect_count"],
            "total_attempts": c["total_attempts"],
            "consecutive_correct": c.get("consecutive_correct", 0),
            "mastered": c.get("consecutive_correct", 0) >= _MASTERED_STREAK,
            "urgency_score": round(_urgency(c), 2),
            "accuracy": (
                f"{round((1 - c['incorrect_count'] / c['total_attempts']) * 100)}%"
                if c["total_attempts"] > 0 else "not attempted"
            ),
        }
        for c in sorted_cards
    ]

    weakest = next(
        (c for c in sorted_cards if c["incorrect_count"] > 0),
        None
    )

    unattempted_count = sum(1 for c in cards if c["total_attempts"] == 0)
    mastered_count = sum(
        1 for c in cards if c.get("consecutive_correct", 0) >= _MASTERED_STREAK
    )

    return json.dumps({
        "status": "success",
        "total_cards": total,
        "unattempted_cards": unattempted_count,
        "mastered_cards": mastered_count,
        "weakest_card": {
            "id": weakest["id"],
            "question": weakest["question"],
            "incorrect_count": weakest["incorrect_count"],
        } if weakest else None,
        "all_cards": card_summaries,
    })
