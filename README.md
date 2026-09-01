# 🃏 Flashcard Quiz Agent — CSE476 CA1 Project 1

**Topic T26 · Study Domain · Solo Submission**

---

## What it does

An autonomous, adaptive flashcard agent that quizzes students and prioritises their weak spots. It executes a multi-step plan-act loop using the Groq inference API (`openai/gpt-oss-120b`), making real tool calls to manage its own flashcard database rather than generating static text responses.

---

## Tools

The agent exposes two primary tools. `add_card(question, answer)` accepts a question string and its correct answer, stores a new flashcard record in a JSON-persisted database (`flashcard_db.json`), and returns a JSON confirmation containing the assigned card ID. `quiz_me()` queries the full card database, filters for cards with at least one recorded incorrect answer, and returns the card with the highest `incorrect_count` so the agent always targets the student's weakest area first; if no incorrect answers exist yet it falls back to the least-attempted card, ensuring every new card eventually gets coverage. A third internal tool, `record_answer(card_id, is_correct)`, is called automatically by the agent after evaluating the student's reply — it increments `total_attempts` and, when the answer was wrong, also increments `incorrect_count`, then flushes the updated record to disk so priority data persists across sessions.

## Memory

The agent maintains two complementary forms of memory. Short-term conversational memory is implemented as a sliding-window over the message history: the system prompt is permanently anchored at index 0, and only the most recent six non-system turns are kept in the active context, preventing context-window overflow while preserving enough dialogue history for coherent multi-turn interaction. Long-term weak-spot memory is stored on disk in `flashcard_db.json` — specifically in each card's `incorrect_count` field — which survives process restarts. When `quiz_me()` is called in a new session it reads these persisted counts and immediately re-serves the card the student has historically struggled with most, making the agent's adaptive behaviour durable rather than session-scoped.

## Honest Failure

During early development with a smaller 8B-parameter model, the agent repeatedly produced malformed tool-call arguments: the JSON payload would be corrupted by `<|channel|>commentary` token sequences bleeding into the structured output, causing `json.JSONDecodeError` in the dispatch loop and breaking multi-step reasoning chains entirely. Switching to the 120B-parameter `openai/gpt-oss-120b` model eliminated the token bleed, but the fix was not solely model size — the system prompt was also hardened with an explicit rule ("Output ONLY valid JSON when filling tool arguments; never output raw commentary outside the `content` field") that anchors the model's output format and prevents regression if a smaller model is substituted in future.

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd flashcard_agent

# 2. Create and activate a virtual environment
python -m venv venv
# Windows Git Bash:
source venv/Scripts/activate

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
├── tools/
│   ├── __init__.py
│   ├── flashcards.py     # add_card, quiz_me, record_answer + JSON persistence
│   └── registry.py       # TOOL_REGISTRY (dispatch map) + AVAILABLE_SCHEMAS
├── core/
│   ├── __init__.py
│   └── agent.py          # GroqAgent: ReAct loop + sliding-window memory
└── demo.ipynb            # Executed notebook demonstrating 3 goals
```

---

## Rubric Compliance Checklist

| Criterion | Evidence |
|---|---|
| ≥ 2 working tools | `add_card` + `quiz_me` (+ `record_answer`) |
| Multi-step plan-act loop | `GroqAgent._run_loop()` in `core/agent.py` |
| Memory across turns | Sliding window + `incorrect_count` persisted to `flashcard_db.json` |
| Adaptive (not random) selection | `quiz_me()` sorts by `incorrect_count` descending |
| Honest failure documented | Token-bleed issue → 120B model + strict prompt guardrails |
| Running notebook | `demo.ipynb` with 3 goals and multi-step tool traces |

---

## Architecture Notes

- **Dispatch Map pattern**: `TOOL_REGISTRY` is a plain `dict[str, callable]`. The agent loop does `TOOL_REGISTRY[tool_name](**kwargs)` — O(1) lookup, zero `if/elif` chains.
- **Security**: All credentials loaded via `python-dotenv`. No hardcoded keys anywhere in the codebase.
- **Atomic writes**: `flashcard_db.json` is written to a `.tmp` file first, then renamed, preventing corruption on crash.
