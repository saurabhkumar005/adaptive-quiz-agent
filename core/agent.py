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

* **BadRequestError recovery**: When the Groq API rejects a turn due to
  malformed model-generated tool JSON (code: tool_use_failed), we strip the
  bad assistant message from history and retry once with a plain text nudge.
"""

from __future__ import annotations

import json
import os
from typing import Any

from groq import Groq, BadRequestError

from tools.registry import AVAILABLE_SCHEMAS, TOOL_REGISTRY

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of *non-system* messages to keep in the sliding window.
# Each "turn" = 1 user message + 1+ assistant/tool messages, so 10 gives
# roughly 4-5 full question-answer cycles of context.
MEMORY_WINDOW: int = 10

SYSTEM_PROMPT: str = """You are an intelligent, adaptive Flashcard Quiz Agent.

Your sole purpose is to help the student learn efficiently by:
1. Adding flashcards they request via the `add_card` tool.
2. Quizzing them using the `quiz_me` tool — which ALWAYS returns the card the
   student struggles with most (highest incorrect_count). You MUST call
   `quiz_me` before presenting any question; NEVER invent or recall questions
   from your own knowledge.
3. Evaluating their answer and calling `record_answer` with the correct
   card_id and a boolean `is_correct` value.
4. Autonomously looping back to quiz the student again when they ask for
   another question, again using `quiz_me` so the weakest card is served.

Strict behavioural rules:
- NEVER present a quiz question without first calling `quiz_me`. This applies
  even if the user says "answer the question" or "give me a question". You
  must ALWAYS call `quiz_me` first to get the question from the database.
- After calling `quiz_me`, format the question like this:
      📖 **Question:** <question text here>
  Do NOT reveal the answer field under any circumstances.
- The `quiz_me` tool result contains an "answer" field. You MUST use EXACTLY
  that "answer" field — and nothing else — to evaluate the student's response.
  NEVER use your own knowledge or training data to judge correctness. The
  stored answer is the ground truth, full stop.
- Set is_correct=true in `record_answer` ONLY when the student's reply matches
  (or is a reasonable paraphrase of) the stored "answer" field from `quiz_me`.
- When the student is wrong, reveal the stored "answer" value verbatim, e.g.:
      ❌ Not quite! The correct answer is: **<answer>**
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

        Error handling:
        - ``BadRequestError`` with code ``tool_use_failed`` means the model
          emitted malformed tool-argument JSON on this turn. We pop the bad
          assistant message, inject a plain-text correction prompt, and retry
          once so the agent recovers gracefully without crashing.
        """
        retried = False  # allow at most one self-correction per turn

        while True:
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=self._history,
                    tools=AVAILABLE_SCHEMAS,
                    tool_choice="auto",
                )
            except BadRequestError as exc:
                # ── Malformed tool-argument JSON produced by the model ──
                error_body = exc.body or {}
                code = error_body.get("error", {}).get("code", "")

                if code == "tool_use_failed" and not retried:
                    retried = True
                    # Remove the last assistant turn that had bad JSON and
                    # inject a corrective user nudge so the model tries again
                    # with valid arguments.
                    self._history = [
                        m for m in self._history
                        if not (m.get("role") == "assistant" and m.get("tool_calls"))
                    ]
                    self._history.append({
                        "role": "user",
                        "content": (
                            "[SYSTEM NOTE] Your last tool call contained invalid JSON. "
                            "Please retry using only valid JSON for all tool arguments."
                        ),
                    })
                    print("[⚠️  Malformed tool JSON detected — retrying automatically...]")
                    continue  # retry the loop

                # Any other 400 — surface a clean message rather than crash
                fallback = (
                    "I ran into a temporary issue communicating with the AI. "
                    "Please try again."
                )
                self._history.append({"role": "assistant", "content": fallback})
                return fallback

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
            retried = False  # reset retry flag for the next sub-loop iteration
