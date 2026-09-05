"""
main.py
========
Interactive CLI entry point for the Flashcard Quiz Agent.

Usage
-----
    python main.py

Type ``quit``, ``exit``, or ``q`` (case-insensitive) to end the session.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment bootstrap — MUST happen before importing the agent so that
# os.environ is populated before Groq client construction.
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    print(
        "❌  GROQ_API_KEY is not set.\n"
        "    Copy .env.example → .env and add your key, then try again."
    )
    sys.exit(1)

# Local imports after env is loaded
from core.agent import GroqAgent  # noqa: E402

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = r"""
╔═══════════════════════════════════════════════════════════════╗
║               🃏  Flashcard Quiz Agent                       ║
║                                                               ║
║  Commands you can try:                                        ║
║    • "Add a card: Q: <question> / A: <answer>"               ║
║    • "Quiz me"  /  "Ask me a question"                       ║
║    • Answer the question shown                                ║
║    • "Quiz me again" (agent will pick your weakest card)     ║
║    • Type 'quit' or 'exit' to end the session                ║
╚═══════════════════════════════════════════════════════════════╝
"""


def main() -> None:
    """Run the interactive flashcard CLI."""
    print(BANNER)
    print(f"🤖  Model  : {GROQ_MODEL}")
    print(f"🧠  Memory : sliding window (last 6 turns)\n")

    agent = GroqAgent(api_key=GROQ_API_KEY, model=GROQ_MODEL)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋  Session ended. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q", "end", "stop"}:
            print("\n👋  Session ended. Goodbye!")
            break

        response = agent.chat(user_input)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
