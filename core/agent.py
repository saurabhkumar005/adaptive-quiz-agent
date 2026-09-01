"""
core/agent.py
==============
GroqAgent — the autonomous ReAct (Reason + Act) brain.

Architecture decisions
----------------------
* **Sliding-window memory**: The system prompt is always kept at index 0.
  Only the most recent ``MEMORY_WINDOW`` conversation turns are retained so
  that the context stays within the model's effective reasoning window while
  still giving the agent enough history to maintain coherent multi-turn
  dialogue.

* **Dispatch Map pattern**: Tool calls are dispatched via ``TOOL_REGISTRY``
  (a plain Python dict), giving O(1) lookup with zero if/elif branching.

* **Plan-act loop**: The agent loops continuously, executing tool calls and
  appending their results back into the message history until the model
  produces a final text response with no pending tool calls.
"""

from __future__ import annotations

import json
import os
from typing import Any

from groq import Groq

from tools.registry import AVAILABLE_SCHEMAS, TOOL_REGISTRY

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of *non-system* messages to keep in the sliding window.
# Each "turn" = 1 user message + 1+ assistant/tool messages, so 6 gives
# roughly 3 full question-answer cycles of context.
MEMORY_WINDOW: int = 6

SYSTEM_PROMPT: str = """You are an intelligent, adaptive Flashcard Quiz Agent.

Your sole purpose is to help the student learn efficiently by:
1. Adding flashcards they request via the `add_card` tool.
2. Quizzing them using the `quiz_me` tool — which ALWAYS returns the card the
   student struggles with most (highest incorrect_count). You MUST call
   `quiz_me` before presenting any question; never invent questions yourself.
3. Evaluating their answer and calling `record_answer` with the correct
   card_id and a boolean `is_correct` value.
4. Autonomously looping back to quiz the student again when they ask for
   another question, again using `quiz_me` so the weakest card is served.

Strict behavioural rules:
- Always use tools; never answer quiz questions from memory.
- After calling `quiz_me`, present ONLY the question text to the student —
  do NOT reveal the answer field.
- The `quiz_me` tool result contains an "answer" field. You MUST use EXACTLY
  that "answer" field — and nothing else — to evaluate the student's response.
  NEVER use your own knowledge or training data to decide if the answer is
  correct or to tell the student what the correct answer is. The stored answer
  is the ground truth, full stop.
- Set is_correct=true in `record_answer` ONLY when the student's reply matches
  (or is a reasonable paraphrase of) the stored "answer" field from `quiz_me`.
- When the student is wrong, reveal the stored "answer" value verbatim.
- Keep explanations short and encouraging.
- Output ONLY valid JSON when filling tool arguments. Never output raw
  commentary outside of the `content` field.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class GroqAgent:
    """
    Autonomous flashcard quiz agent powered by the Groq inference API.

    Parameters
    ----------
    api_key : str
        Groq API key loaded from the environment.
    model : str
        Groq model identifier (e.g. ``"openai/gpt-oss-120b"``).
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._client = Groq(api_key=api_key)
        self._model = model
        # Message history; index 0 is always the system prompt.
        self._history: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Process a user message through the full ReAct plan-act loop and
        return the agent's final natural-language response.

        Parameters
        ----------
        user_message : str
            Raw text from the user / student.

        Returns
        -------
        str
            The agent's final response after completing all tool calls.
        """
        self._append_user_message(user_message)
        return self._run_loop()

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def _append_user_message(self, text: str) -> None:
        """Add a user turn and trim the sliding window."""
        self._history.append({"role": "user", "content": text})
        self._trim_history()

    def _trim_history(self) -> None:
        """
        Enforce the sliding-window memory limit.

        Keep index 0 (system prompt) untouched; retain only the last
        ``MEMORY_WINDOW`` non-system messages.
        """
        non_system = self._history[1:]
        if len(non_system) > MEMORY_WINDOW:
            self._history = [self._history[0]] + non_system[-MEMORY_WINDOW:]

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> str:
        """
        Continuous plan-act execution loop.

        Each iteration:
        1. Sends the current message history + tool schemas to Groq.
        2. Checks for tool calls in the response.
        3. Dispatches each tool call via the TOOL_REGISTRY (O(1) lookup).
        4. Appends the tool result with role ``"tool"`` and matching id.
        5. Repeats until the model returns a final text response.
        """
        while True:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._history,
                tools=AVAILABLE_SCHEMAS,
                tool_choice="auto",
            )

            assistant_msg = response.choices[0].message
            tool_calls = assistant_msg.tool_calls  # None or list

            if not tool_calls:
                # ── Terminal state: the model produced a final answer ──
                final_text = assistant_msg.content or ""
                self._history.append(
                    {"role": "assistant", "content": final_text}
                )
                self._trim_history()
                return final_text

            # ── Intermediate state: one or more tool calls requested ──
            # Build a clean assistant message dict. We deliberately avoid
            # model_dump() because the Groq SDK injects extra fields such as
            # 'annotations' that the Groq API itself rejects with a 400 error
            # when they appear in the message history.
            clean_tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
            self._history.append({
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": clean_tool_calls,
            })

            for tc in tool_calls:
                tool_name = tc.function.name
                raw_args = tc.function.arguments

                print(f"\n[⚙️  Agent paused to use tool: {tool_name}]")

                # Parse arguments safely
                try:
                    kwargs: dict[str, Any] = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError as exc:
                    tool_result = json.dumps({
                        "status": "error",
                        "message": f"Failed to parse tool arguments: {exc}",
                    })
                else:
                    # O(1) dispatch — no if/elif chains
                    executor = TOOL_REGISTRY.get(tool_name)
                    if executor is None:
                        tool_result = json.dumps({
                            "status": "error",
                            "message": f"Unknown tool '{tool_name}'.",
                        })
                    else:
                        try:
                            tool_result = executor(**kwargs)
                        except Exception as exc:  # noqa: BLE001
                            tool_result = json.dumps({
                                "status": "error",
                                "message": f"Tool execution error: {exc}",
                            })

                print(f"[📦 Tool result]: {tool_result}\n")

                # Append the tool result so the model can reason over it
                self._history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

            # Loop again — the model will now reason over the tool results
            self._trim_history()
