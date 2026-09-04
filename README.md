# Flashcard Quiz Agent

An autonomous, adaptive flashcard study tool powered by the [Groq](https://groq.com) inference API and a **Streamlit web UI**.

The agent quizzes you interactively, remembers which cards you struggle with, and **automatically prioritises your weak spots** — not just random questions.
Use it through the browser UI **or** the classic terminal CLI — both interfaces share the exact same agent brain.

---

## ✨ Features at a Glance

| Feature | Details |
|---|---|
| **ReAct Agent Loop** | Reason → Act → Observe → Repeat until a final answer is ready |
| **Sliding-Window Memory** | System prompt always anchored; last 6 turns kept in context |
| **Smart Adaptive Priority** | Urgency scoring — weak spots promoted, mastered cards retired |
| **Persistent Deck** | `flashcard_db.json` survives restarts; atomic write prevents corruption |
| **Streamlit Web UI** | Live KPI sidebar, tool-call traces, quick-action buttons, deck inspector |
| **Terminal CLI** | Classic `python main.py` workflow — fully independent of the UI |
| **Batch Card Creation** | `add_cards_batch` inserts 4+ cards in a single tool call |

---

## 📁 Project Structure

```
flashcard_agent/
├── .env.example              # API key template  ← copy this to .env
├── .gitignore
├── requirements.txt
├── README.md
├── app.py                    # ✨ Streamlit web interface
├── main.py                   # Terminal CLI entry point
├── flashcard_db.json         # Auto-created on first run (gitignored)
├── core/
│   ├── __init__.py
│   └── agent.py              # GroqAgent: ReAct loop + sliding-window memory
└── tools/
    ├── __init__.py
    ├── flashcards.py         # add_card, add_cards_batch, quiz_me, record_answer, get_stats
    └── registry.py           # TOOL_REGISTRY (dispatch map) + AVAILABLE_SCHEMAS
```

---

## 🚀 Quick Start — Step-by-Step from Zero

### Step 1 — Get a Free Groq API Key

1. Go to **https://console.groq.com/keys**
2. Sign up or log in (free account, no credit card required)
3. Click **"Create API Key"**, give it a name, and **copy the key**
   *(it looks like `gsk_xxxxxxxxxxxxxxxxxxxx`)*

> **⚠ Important:** Keep your API key secret. Never commit `.env` to git — it is already in `.gitignore`.

---

### Step 2 — Clone the Repository

```bash
git clone https://github.com/your-username/flashcard_agent.git
cd flashcard_agent
```

---

### Step 3 — Create a Python Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Git Bash / CMD):**
```bash
python -m venv venv
source venv/Scripts/activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> **Tip:** Your prompt will show `(venv)` once the environment is active.

---

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `groq`, `streamlit`, `pandas`, `python-dotenv`, and Jupyter tools.

---

### Step 5 — Configure Your API Key

```bash
# Windows (PowerShell / Git Bash)
cp .env.example .env

# Windows (CMD only)
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Now open `.env` in any text editor and replace the placeholder:

```env
# .env  — DO NOT COMMIT THIS FILE
GROQ_API_KEY=your_groq_api_key_here    # ← paste the key from Step 1 here
GROQ_MODEL=openai/gpt-oss-120b
```

> **Where to get the key:** https://console.groq.com/keys — free tier gives generous daily limits.

---

### Step 6 — Run the Agent

#### Option A: Streamlit Web UI *(recommended)*

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

**What you get in the UI:**
- Live sidebar with **Total Cards / Unattempted / Weak Spots / Mastered** KPI metrics (auto-refreshed after every agent turn)
- One-click **Quick Action** buttons:
  - ⚡ Quiz Adaptive — picks your highest-priority card
  - 🎯 Focus Weak Spots — targets cards you are struggling with
  - 🆕 Unattempted Card — shows a brand-new card you have never seen
  - 📊 Full Deck Summary — full mastery and urgency report
- Expandable **Live Deck Inspector** — a DataFrame table of every card with its stats
- Full **chat interface** with collapsible **tool-call traces** showing exactly which tools the agent called, with what arguments, and what they returned

#### Option B: Terminal CLI

```bash
python main.py
```

Same agent, same persistent deck — just a plain text REPL in your terminal.

---

## 💬 Example Prompts

Once the agent is running (in either mode), try these:

```
Add a card: Q: What is Docker? / A: OS-level virtualisation platform
Add 5 cards on Python data structures
Quiz me
Quiz me on Docker
Quiz me on weak topics
Show me a card I have never seen before. Use unattempted mode.
Give me a summary of all cards, showing mastery and urgency scores.
```

---

## ⚙️ How It Works

### ReAct Loop (`core/agent.py`)

```
User message
    │
    ▼
Groq API  (model reasons + selects a tool)
    │
    ├── Tool call?  ──► Execute tool ──► Append result to history ──► loop back
    │
    └── No tool?    ──► Final text response ──► Return to user
```

### Adaptive Priority Algorithm (`tools/flashcards.py`)

| Card State | Urgency Score |
|---|---|
| Never attempted | **5.0** (highest — ensures full deck coverage first) |
| Has errors, not yet mastered | `(incorrect_count × 3.0) − (consecutive_correct × 1.5)` |
| Mastered (consecutive correct ≥ 2) | **0.5** (deprioritised — periodic refresh only) |

### Tool Registry (`tools/registry.py`)

| Tool | Purpose |
|---|---|
| `add_card(question, answer)` | Add a single flashcard |
| `add_cards_batch(cards)` | Bulk-insert 4+ cards in one tool call |
| `quiz_me(mode, topic, count)` | Fetch highest-priority card(s) |
| `record_answer(card_id, is_correct)` | Update metrics with correct/incorrect decay |
| `get_stats()` | Full deck diagnostic summary |

---

## 🔐 Security Notes

- `.env` is listed in `.gitignore` — your API key will **never** be accidentally committed
- `.env.example` is committed as a safe template with a placeholder value only
- No API keys are hardcoded anywhere in the source code

---

## 📋 Requirements

- Python **3.10** or newer
- A free [Groq API key](https://console.groq.com/keys)
- Internet connection (calls the Groq inference API)

---

## 🏗️ Architecture Notes

- **Dispatch Map pattern** — `TOOL_REGISTRY` is a plain `dict[str, callable]`: O(1) tool lookup, zero `if/elif` chains
- **Atomic writes** — `flashcard_db.json` is written to `.tmp` then renamed: crash-safe persistence
- **Sliding window** — Only the last 6 non-system turns are kept in the LLM context: cost-efficient and hallucination-resistant
- **Streamlit session state** — `GroqAgent` is instantiated once per browser tab and lives in `st.session_state`: full memory persists across UI reruns
- **Tool-call visibility** — The UI captures every tool event via an `on_tool_event` callback injected into `GroqAgent.chat()`: the terminal CLI is completely unaffected (callback defaults to `None`)