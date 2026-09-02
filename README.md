# Flashcard Quiz Agent

An autonomous, adaptive flashcard study tool powered by the [Groq](https://groq.com) inference API. The agent quizzes you interactively, remembers which cards you struggle with, and **automatically prioritises your weak spots** -- not just random questions.

---

## Tools & Capabilities

The agent exposes four tools through a dispatch-map registry (`tools/registry.py`). `add_card(question, answer)` persists a new flashcard record to a JSON-backed state store; when a user asks the agent to generate cards on a general knowledge topic without providing answers, the agent autonomously constructs accurate Q&A pairs from its own knowledge before calling this tool. `quiz_me(mode, topic)` implements the smart adaptive priority algorithm: it scores every candidate card with an urgency formula -- unattempted cards receive a fixed weight of 5.0, mastered cards (consecutive_correct >= 2) are capped at 0.5, and all others are scored as `(incorrect_count * 3.0) - (consecutive_correct * 1.5)` -- then returns the highest-scoring card along with a natural-language explanation of why it was chosen; the optional `topic` argument filters candidates by case-insensitive keyword match so requests like "quiz me on docker" are handled precisely. `record_answer(card_id, is_correct)` updates a card's metrics atomically: a correct answer increments `consecutive_correct` and decrements `incorrect_count` (floor zero) to decay the error weight, while an incorrect answer resets the streak and increments the error counter. `get_stats()` returns a ranked summary of every card including urgency score, mastery state, accuracy percentage, and unattempted/mastered aggregates so the agent can deliver accurate session diagnostics.

## Memory Architecture

The agent maintains two complementary and independent memory tiers. Short-term conversational memory is implemented as a **sliding window** over the message history (`core/agent.py`): the system prompt is permanently anchored at index 0, and only the most recent six non-system turns are kept in the active context at any time; this bounds the prompt size for cost and latency efficiency while retaining enough recent dialogue history for coherent multi-turn interaction without hallucinating earlier context. Long-term **persistent state** is stored on disk in `flashcard_db.json` via an atomic write pattern (write to `.tmp`, then `rename`) that prevents data corruption on crash; each card record stores `id`, `question`, `answer`, `incorrect_count`, `total_attempts`, `consecutive_correct`, and `last_attempted`, and this data survives process restarts completely -- when `quiz_me()` is called in a brand-new session it reads the persisted urgency data and immediately re-serves the card the student has historically struggled with most, making the adaptive behaviour durable and session-independent.

## Engineering Challenges & Production Bug Fixes

Two critical production bugs were identified and resolved during testing. The first was **Card Starvation and Priority Lockout**: the original `quiz_me()` used a static error counter (`incorrect_count`) that never decayed, so a card answered incorrectly once and then correctly seven times in a row still retained `incorrect_count: 1` and was returned by every subsequent `quiz_me()` call, trapping the agent in an infinite loop serving the same card forever while all other cards -- including unattempted ones -- were permanently starved. The fix introduces a `consecutive_correct` streak field: each correct answer decrements `incorrect_count` (floor zero) and increments the streak; reaching a streak of 2 transitions the card to a mastered state (urgency capped at 0.5) that is deliberately deprioritised in favour of unattempted cards (urgency 5.0), solving both the starvation loop and the cold-start coverage gap. Additionally, `quiz_me()` was parameterised with `mode` and `topic` arguments -- the original function accepted no parameters, making it impossible for the agent to honour user intents like "quiz me on docker" or "show me cards I have not seen yet", causing silent tool blindness. The second bug was **Scratchpad and Meta-Reasoning Bleed**: when trapped serving the same card, the model entered token-repetition loops and leaked its internal conflict monologue directly into the user response. The system prompt was hardened with an explicit CRITICAL directive forbidding the model from outputting internal deliberation, policy commentary, or ellipsis sequences, and the execution loop was updated to detect empty or whitespace-only final responses and substitute a safe fallback message rather than surfacing a blank or garbled reply.

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd flashcard_agent

# 2. Create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate      # Git Bash on Windows
# .\venv\Scripts\Activate.ps1    # PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Edit .env and set GROQ_API_KEY=<your_key>

# 5. Run the interactive agent
python main.py
```

---

## Project Structure

```
flashcard_agent/
├── .env.example          # API key template (never commit .env)
├── .gitignore
├── requirements.txt
├── README.md
├── main.py               # Interactive CLI entry point
├── generate_demo_notebook.py  # Regenerates demo.ipynb
├── tools/
│   ├── __init__.py
│   ├── flashcards.py     # add_card, quiz_me, record_answer, get_stats + JSON persistence
│   └── registry.py       # TOOL_REGISTRY (dispatch map) + AVAILABLE_SCHEMAS
├── core/
│   ├── __init__.py
│   └── agent.py          # GroqAgent: ReAct loop + sliding-window memory
└── demo.ipynb            # Notebook demonstrating all scenarios and bug fixes
```

---

## Architecture Notes

- **ReAct loop**: The agent runs a continuous plan-act loop -- it calls tools, reads results, reasons over them, and loops until it has a final answer. It never just generates text directly.
- **Dispatch Map pattern**: `TOOL_REGISTRY` is a plain `dict[str, callable]`. The loop executes `TOOL_REGISTRY[tool_name](**kwargs)` -- O(1) lookup, zero `if/elif` chains.
- **Urgency-score algorithm**: Cards are scored as `(incorrect_count * 3.0) - (consecutive_correct * 1.5)`, capped at 0.5 for mastered cards and fixed at 5.0 for unattempted cards.
- **Atomic writes**: `flashcard_db.json` is written to a `.tmp` file first, then renamed, preventing data corruption on crash.
- **Security**: All credentials loaded via `python-dotenv`. No hardcoded keys anywhere in the codebase.
