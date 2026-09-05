# 🃏 FlashCard Quest — Autonomous Flashcard Quiz Agent

An autonomous, **gamified** flashcard study tool powered by the [Groq](https://groq.com) inference API, a full **Streamlit Game UI**, and a **self-healing multi-model fallback cascade**.

The agent quizzes you interactively, remembers which cards you struggle with, and **automatically prioritises your weak spots** — not just random questions.
Use it through the immersive **browser game UI** or the classic **terminal CLI** — both share the exact same agent brain.

---

## ✨ Features at a Glance

| Feature | Details |
|---|---|
| 🎮 **Gamified Quiz UI** | XP system, streaks, levels, 8 achievements, animated feedback — feels like a game |
| 🤖 **ReAct Agent Loop** | Reason → Act → Observe → Repeat until a final answer is ready |
| 🔁 **Multi-Model Fallback Cascade** | Auto-failover across 4 Groq models on rate-limit / API errors — zero downtime |
| 🎟️ **Free-Trial Gatekeeper + BYOK** | 5 free interactions on the host key; paste your own key for unlimited usage |
| 🧠 **Sliding-Window Memory** | System prompt always anchored; last 6 turns kept in context |
| 📊 **Smart Adaptive Priority** | Urgency scoring — weak spots promoted, mastered cards retired |
| 💾 **Persistent Deck** | `flashcard_db.json` survives restarts; atomic write prevents corruption |
| 🃏 **3 Quiz Modes** | Adaptive AI / Explorer (unseen only) / Weak Spot Blitz |
| 💡 **Example Prompt Chips** | Clickable one-tap prompts for adding cards, quizzing, stats, and learning |
| 💻 **Terminal CLI** | `python main.py` workflow — fully independent of the UI |
| 🔁 **Batch Card Creation** | `add_cards_batch` inserts 4+ cards in a single tool call |

---

## 📁 Project Structure

```
flashcard_agent/
├── .env.example              # API key template  ← copy this to .env
├── .gitignore
├── requirements.txt
├── README.md
├── app.py                    # 🎮 Gamified Streamlit web interface
├── main.py                   # Terminal CLI entry point
├── flashcard_db.json         # Auto-created on first run (gitignored)
├── core/
│   ├── __init__.py
│   └── agent.py              # GroqAgent: ReAct loop + fallback cascade + memory
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

Now open `.env` in any text editor and fill in your values:

```env
# .env  — DO NOT COMMIT THIS FILE
GROQ_API_KEY=your_groq_api_key_here         # ← paste the key from Step 1

# Primary model (default works great)
GROQ_MODEL=openai/gpt-oss-120b

# Optional: override the auto-failover cascade (comma-separated, tried left-to-right)
# GROQ_FALLBACK_MODELS=openai/gpt-oss-120b,qwen/qwen3.8-27b,qwen/qwen3.6-27b,openai/gpt-oss-20b
```

> **Where to get the key:** https://console.groq.com/keys — free tier gives generous daily request limits.

---

### Step 6 — Run the Agent

#### Option A: Gamified Streamlit Web UI *(recommended)*

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

#### Option B: Terminal CLI

```bash
python main.py
```

Same agent, same persistent deck — just a plain text REPL in your terminal.

---

## 🎮 Web UI — Full Feature Walkthrough

### Quiz Game Tab

The 🎮 **Quiz Game** tab is a self-contained game loop — no prompting knowledge needed:

| Phase | What happens |
|---|---|
| **🏠 Home** | Choose a quiz mode and optional topic filter, then click **🚀 Start Quiz!** |
| **❓ Question** | An animated, glowing question card appears. Type your answer and hit **Submit**. |
| **✅/❌ Feedback** | Instant animated result (correct pops in, wrong shakes). See XP earned, streak, and the correct answer. Click **▶️ Next Question** to keep going. |

**3 Quiz Modes:**

| Mode | Best for |
|---|---|
| 🎲 **Adaptive AI** | Daily practice — picks the most urgent card using the priority algorithm |
| 🆕 **Explorer** | Learning new material — only shows cards you have never seen |
| 🔥 **Weak Spot Blitz** | Intensive review — hammers your worst cards first |

**Game Mechanics:**

| Mechanic | Detail |
|---|---|
| ⚡ **XP System** | +10 base per correct answer, +3 per streak level (capped at +30), +8 bonus for new cards |
| 🔥 **Streak Tracker** | Resets on wrong/skip; animated flame pill appears at streak ≥ 3 |
| 🏆 **6 Levels** | 🌱 Seedling → 📖 Scholar → 🎓 Graduate → ⭐ Expert → 🔮 Wizard → 🏆 Legend |
| 📊 **XP Progress Bar** | Animated gradient bar shows progress to next level |
| 🏅 **8 Achievements** | First Blood · On Fire! (3) · Lightning! (5) · Unstoppable! (10) · Bookworm · Brainiac · Ace (90% acc) · Comeback King |
| ⏭️ **Skip Cards** | Reveals the answer without penalising accuracy; resets streak |

### Ask the Agent Tab

The 💬 **Ask the Agent** tab is a full conversational interface:

- 4 expandable **Example Prompt categories** with one-tap chips that fire instantly:
  - ➕ Add Cards
  - 🎯 Quiz Me
  - 📊 Stats & Insights
  - 💡 Learn + Save
- Collapsible **tool-call traces** showing exactly which tool ran, with what arguments, and what it returned
- Full chat history preserved across Streamlit reruns

### Sidebar

| Element | What it shows |
|---|---|
| 🔑 **API Key input** | Paste your Groq key for unlimited usage; green badge when active |
| 🎟️ **Free trial bar** | Amber badge + progress bar showing X/5 interactions used |
| 🤖 **Active Model badge** | Which model in the cascade handled the last turn |
| 🌱 **Level badge** | Current XP level with name and animated XP bar |
| 📊 **Session stats** | XP, Correct/Total, Streak, Best Streak, Accuracy bar |
| 🃏 **Deck Health** | Total / Unseen / Weak / Mastered + mastery progress bar |
| 🏅 **Achievements** | All unlocked achievement pills |
| 🔍 **Deck Inspector** | Full DataFrame of every card with stats (expandable) |

---

## 🔑 BYOK & Free-Trial Gatekeeper

When deployed as a shared demo or hosted app:

- **Guests get 5 free interactions total** on the host's Groq API key across both Quiz Mode and Agent Mode
- Each quiz question answered and each agent conversation turn counts towards the 5-turn free quota
- Once the 5 chances are reached, both the Quiz Game and Agent Chat are locked behind a polite gate banner explaining how to get a free Groq key
- Users can paste their own key (`gsk_...`) into the sidebar at any time to get **unlimited usage immediately**
- The key is used only within the browser session — it is never stored on disk or transmitted anywhere except directly to Groq's API

---

## 🔁 Multi-Model Fallback Cascade

The agent never goes down due to a single model being rate-limited:

```
1st  openai/gpt-oss-120b    ← Primary (highest reasoning)
  ↓  (429 / 5xx)
2nd  qwen/qwen3.8-27b       ← Strong function-calling fallback
  ↓  (429 / 5xx)
3rd  qwen/qwen3.6-27b       ← Lightweight reasoning fallback
  ↓  (429 / 5xx)
4th  openai/gpt-oss-20b     ← Final resort
  ↓  (all exhausted)
Clean user-facing error returned
```

**How to customise the cascade** — add to `.env`:
```env
GROQ_FALLBACK_MODELS=openai/gpt-oss-120b,qwen/qwen3.8-27b,qwen/qwen3.6-27b,openai/gpt-oss-20b
```

The sidebar's **🤖 Active Model** badge always shows which model is currently handling requests.

---

## 💬 Example Prompts

Once the agent is running, try these in the **Ask the Agent** tab:

```
Add a card: Q: What is Docker? / A: OS-level virtualisation platform
Add 5 cards on Python data structures
Generate 8 cards on SQL joins
Quiz me
Quiz me on Docker
Quiz me with 3 questions back to back
Focus on my weakest cards only
Show me a card I have never seen before. Use unattempted mode.
Give me a full deck summary with mastery and urgency scores.
What is a RAG pipeline? Then add it as a card.
Explain ACID properties and create 4 cards on it.
```

---

## ⚙️ How It Works

### ReAct Loop + Fallback Cascade (`core/agent.py`)

```
User message
    │
    ▼
Try Model[0] → Model[1] → Model[2] → Model[3]  (on RateLimitError / APIError)
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

- `.env` is in `.gitignore` — your API key will **never** be accidentally committed
- `.env.example` is committed as a safe template with placeholder values only
- No API keys are hardcoded anywhere in the source code
- BYOK keys live only in `st.session_state` — ephemeral, never written to disk

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
- **Multi-model cascade** — `GroqAgent._models[]` is loaded from `GROQ_FALLBACK_MODELS` env var or `DEFAULT_FALLBACK_MODELS`; `_current_model_index` advances on transient errors
- **Streamlit session state** — `GroqAgent` is instantiated once per browser tab; re-created only when the user's API key changes
- **Tool-call visibility** — The UI captures every tool event via an `on_tool_event` callback injected into `GroqAgent.chat()`; the terminal CLI is completely unaffected (callback defaults to `None`)
- **Unified Free-Trial Gatekeeper** — Both Quiz Game turns and Agent Chat turns increment the trial counter on the host key. Once the 5-interaction quota is reached, both tabs require a custom BYOK key to resume.
