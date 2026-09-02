"""
tools/flashcards.py
====================
Persistent flashcard state store and the five agent tools:
  - add_card(question, answer)               -> adds a single new card
  - add_cards_batch(cards)                   -> inserts a list of cards in one operation
  - quiz_me(mode, topic, count)              -> returns top-N highest-priority cards
  - record_answer(card_id, is_correct)       -> updates card metrics with decay
  - get_stats()                              -> returns a full database summary

State is persisted to ``flashcard_db.json`` in the project root so that
weak-spot data survives across agent sessions.

Additions in this revision
--------------------------
* ``add_cards_batch``: Accepts a list of {question, answer} dicts and inserts
  all cards in a single tool call. Resolves Generation Derailment caused by
  firing 20+ individual add_card calls in one turn (token-budget exhaustion).
* ``quiz_me`` count parameter: Accepts count=N to return the top-N priority
  cards in one call. Resolves Parallel Read Duplication where the agent fired
  quiz_me 3x in one turn and received the same card 3 times.

Previous bug fixes
------------------
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


def add_cards_batch(cards: list[dict]) -> str:
    """
    Insert multiple flashcards in a single tool call.

    This is the preferred method for bulk creation (4+ cards) because it
    avoids per-turn token-budget exhaustion that causes mid-task abandonment
    when the agent tries to fire 20+ sequential add_card calls.

    Parameters
    ----------
    cards : list of dict
        Each element must have ``"question"`` and ``"answer"`` string keys.
        Invalid or empty entries are skipped and reported in the response.

    Returns
    -------
    str
        A JSON-encoded object with ``count``, ``cards`` (added), and
        ``skipped`` (entries that failed validation).

    Examples
    --------
    add_cards_batch(cards=[
        {"question": "What is Docker?", "answer": "OS-level virtualization"},
        {"question": "What is Kubernetes?", "answer": "Container orchestration platform"},
    ])
    """
    if not isinstance(cards, list) or len(cards) == 0:
        return json.dumps({
            "status": "error",
            "message": "cards must be a non-empty list of {question, answer} objects.",
        })

    added: list[dict] = []
    skipped: list[dict] = []

    for entry in cards:
        question = str(entry.get("question", "")).strip()
        answer = str(entry.get("answer", "")).strip()

        if not question or not answer:
            skipped.append({"entry": entry, "reason": "Missing or empty question/answer."})
            continue

        card_id = FLASHCARD_DB["next_id"]
        FLASHCARD_DB["next_id"] += 1

        new_card = {
            "id": card_id,
            "question": question,
            "answer": answer,
            "incorrect_count": 0,
            "total_attempts": 0,
            "consecutive_correct": 0,
            "last_attempted": None,
        }
        FLASHCARD_DB["cards"][str(card_id)] = new_card
        added.append({"id": card_id, "question": question, "answer": answer})

    if added:
        _save_db(FLASHCARD_DB)

    return json.dumps({
        "status": "success" if added else "error",
        "count": len(added),
        "skipped": len(skipped),
        "message": (
            f"{len(added)} flashcard(s) added successfully."
            + (f" {len(skipped)} entry/entries skipped due to validation errors." if skipped else "")
        ),
        "cards": added,
        "skipped_details": skipped,
    })


def quiz_me(mode: str = "adaptive", topic: str | None = None, count: int = 1) -> str:
    """
    Return the top-N highest-priority flashcards using the smart adaptive algorithm.

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
    count : int, optional
        Number of cards to return (default 1). Allows the agent to serve
        multiple questions in one call instead of firing duplicate parallel
        calls that return the same top card every time. Capped at the number
        of available unique candidates.

    Returns
    -------
    str
        A JSON-encoded object with a ``cards`` list (length = count_returned)
        plus shared priority metadata. When count=1 a legacy ``card`` key is
        also present for backward compatibility with record_answer workflows.
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

    # ---- Input validation ---------------------------------------------------
    try:
        count = max(1, int(count))
    except (TypeError, ValueError):
        count = 1

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
        sorted_pool = pool
        priority = "unattempted"
        priority_reason = f"Unattempted card(s){topic_note} -- never been served before."

    elif mode == "weakest":
        sorted_pool = sorted(candidates, key=lambda c: c["incorrect_count"], reverse=True)
        priority = "weakest"
        priority_reason = (
            f"Weakest card(s){topic_note} -- sorted by highest historical incorrect_count."
        )

    else:  # adaptive (default)
        scored = sorted(
            [(c, _urgency(c)) for c in candidates],
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_pool = [c for c, _ in scored]
        top_card, top_score = scored[0]

        if top_card["total_attempts"] == 0:
            priority = "unattempted"
            priority_reason = (
                f"Unattempted card(s){topic_note} -- base urgency {top_score:.2f}. "
                "Serving to ensure full deck coverage."
            )
        elif top_card.get("consecutive_correct", 0) >= _MASTERED_STREAK:
            priority = "mastered_review"
            priority_reason = (
                f"All cards{topic_note} appear mastered (streak >= {_MASTERED_STREAK}). "
                f"Serving for periodic review (urgency {top_score:.2f})."
            )
        else:
            priority = "weak_spot"
            priority_reason = (
                f"Weak spot(s){topic_note} -- top urgency score {top_score:.2f} "
                f"(incorrect_count={top_card['incorrect_count']}, "
                f"consecutive_correct={top_card.get('consecutive_correct', 0)})."
            )

    # ---- Deduplicate & cap at count -----------------------------------------
    seen_ids: set[int] = set()
    selected: list[dict] = []
    for c in sorted_pool:
        if c["id"] not in seen_ids:
            seen_ids.add(c["id"])
            selected.append({
                "id": c["id"],
                "question": c["question"],
                "answer": c["answer"],
                "incorrect_count": c["incorrect_count"],
                "total_attempts": c["total_attempts"],
                "consecutive_correct": c.get("consecutive_correct", 0),
            })
        if len(selected) >= count:
            break

    result: dict = {
        "status": "success",
        "mode": mode,
        "count_requested": count,
        "count_returned": len(selected),
        "priority": priority,
        "priority_reason": priority_reason,
        "cards": selected,
    }
    # Backward-compat: single-card alias when count=1
    if len(selected) == 1:
        result["card"] = selected[0]

    return json.dumps(result)


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
