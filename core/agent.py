"""
core/agent.py
==============
GroqAgent -- the autonomous ReAct (Reason + Act) brain.

Architecture decisions
----------------------
* Sliding-window memory: The system prompt is always kept at index 0.
  Only the most recent ``MEMORY_WINDOW`` conversation turns are retained so
  that the context stays within the model effective reasoning window while
  still giving the agent enough history to maintain coherent multi-turn
  dialogue. Window size = 6 non-system turns (system prompt + last 6 turns).

* Dispatch Map pattern: Tool calls are dispatched via ``TOOL_REGISTRY``
  (a plain Python dict), giving O(1) lookup with zero if/elif branching.

* Plan-act loop: The agent loops continuously, executing tool calls and
  appending their results back into the message history until the model
  produces a final text response with no pending tool calls.

* BadRequestError recovery: When the Groq API rejects a turn due to
  malformed model-generated tool JSON (code: tool_use_failed), we strip the
  bad assistant message from history and retry once with a plain text nudge.

* Empty / garbled response guard: If the model returns an empty string or
  pure whitespace as its final text, a friendly fallback is returned instead
  of surfacing a blank response to the user.

Bug fixes in this revision
---------------------------
* Autonomous Q&A generation: System prompt now explicitly instructs the model
  to generate questions and answers from its own knowledge when the user asks
  for cards on a general topic without providing an answer.
* Monologue bleed guardrail: System prompt contains a CRITICAL directive
  preventing the model from leaking internal deliberation or ellipsis loops.
* Empty-response guard: The execution loop detects empty/whitespace assistant
  content and returns a safe fallback instead of an invisible blank response.
"""

from __future__ import annotations

import json
from typing import Any

from groq import Groq, BadRequestError

from tools.registry import AVAILABLE_SCHEMAS, TOOL_REGISTRY

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of *non-system* messages to keep in the sliding window.
# 6 gives roughly 3 full question-answer cycles of context -- enough for
# coherent dialogue while keeping the prompt lean.
MEMORY_WINDOW: int = 6

SYSTEM_PROMPT: str = """You are an intelligent, adaptive Flashcard Quiz Agent.

You have four tools available:
  - add_card(question, answer)          -- save a new flashcard to the database
  - quiz_me(mode, topic)               -- fetch the highest-priority card
  - record_answer(card_id, is_correct) -- log the result of a quiz attempt
  - get_stats()                        -- get a full database summary with accuracy data

CRITICAL: Never output internal deliberation, policy commentary, reasoning chains,
or ellipsis loops ('...'). Output ONLY the final user-facing response. If you are
unsure of your next step, take the most helpful action silently via a tool call.
Never show the user your internal monologue.

━━━ ADDING CARDS ━━━

- When a user asks you to add one or more cards WITH both a question and an answer,
  call add_card() once per card immediately.
- When a user asks you to CREATE or GENERATE cards on a general knowledge topic
  WITHOUT providing the answers (e.g. "create 3 cards on Python decorators" or
  "add cards about Docker"), you MUST autonomously generate accurate questions
  AND answers from your own knowledge and call add_card() for each one.
  Do NOT ask the user for the answers -- generate them yourself.
- Only ask the user for an answer when the topic is personal or subjective
  (e.g. "add a card about my boss's birthday" -- that answer is unknowable to you).
- If the user provides multiple cards in one message, call add_card() sequentially
  for each one -- do not merge them.

━━━ QUIZZING ━━━

- You MUST call quiz_me() before showing any question -- every single time.
  Never invent questions from your own knowledge.
- When the user specifies a topic (e.g. "quiz me on docker"), call quiz_me with
  topic="docker". When they ask for unattempted or weakest cards, pass the
  appropriate mode argument.
- If quiz_me() returns status="empty", tell the user their deck is empty.
- After quiz_me() succeeds, present ONLY the question in this format:
      Question: <question text>
  Never reveal the answer field to the student.
- Mention the priority_reason from the tool result so the student understands
  why that card was chosen.

━━━ EVALUATING ANSWERS ━━━

- Use ONLY the "answer" field returned by quiz_me() as ground truth. NEVER
  use your own training knowledge to judge correctness.
- If the student reply matches (or is a reasonable paraphrase) of that
  stored answer -> call record_answer(card_id, is_correct=True).
- If it does not match -> call record_answer(card_id, is_correct=False), then
  reveal the correct answer verbatim:
      Not quite! The correct answer is: <stored answer>
- After a correct answer, check the updated_metrics.mastered field -- if True,
  congratulate the student on mastering that card.

━━━ SESSION SUMMARY ━━━

- When the user asks for a summary, how many cards they have, or which topic
  they are struggling with -- call get_stats() and present the results clearly.

━━━ STRICT RULES ━━━
- Output ONLY valid JSON in tool arguments. No commentary inside JSON.
- Never hallucinate answers, card IDs, or database state.
- Keep all replies concise and encouraging.
- CRITICAL: No ellipsis loops, no '...', no internal monologue in output.
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
        return the agent final natural-language response.

        Parameters
        ----------
        user_message : str
            Raw text from the user / student.

        Returns
        -------
        str
            The agent final response after completing all tool calls.
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
        4. Appends the tool result with role "tool" and matching id.
        5. Repeats until the model returns a final text response.

        Error handling:
        - ``BadRequestError`` with code ``tool_use_failed`` means the model
          emitted malformed tool-argument JSON on this turn. We pop the bad
          assistant message, inject a plain-text correction prompt, and retry
          once so the agent recovers gracefully without crashing.
        - Empty / garbled final responses: if the model returns empty or
          whitespace-only content, a safe fallback message is returned.
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
                # ---- Malformed tool-argument JSON produced by the model ----
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
                    print("[Warning: Malformed tool JSON detected -- retrying automatically...]")
                    continue  # retry the loop

                # Any other 400 -- surface a clean message rather than crash
                fallback = (
                    "I ran into a temporary issue communicating with the AI. "
                    "Please try again."
                )
                self._history.append({"role": "assistant", "content": fallback})
                return fallback

            assistant_msg = response.choices[0].message
            tool_calls = assistant_msg.tool_calls  # None or list

            if not tool_calls:
                # ---- Terminal state: the model produced a final answer ----
                final_text = (assistant_msg.content or "").strip()

                # Empty / garbled response guard
                if not final_text:
                    final_text = (
                        "I processed your request but did not produce a text response. "
                        "Please try rephrasing or ask me to quiz you or add a card."
                    )
                    print("[Warning: Empty assistant response detected -- using fallback]")

                self._history.append(
                    {"role": "assistant", "content": final_text}
                )
                self._trim_history()
                return final_text

            # ---- Intermediate state: one or more tool calls requested ----
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

                print(f"\n[Agent paused to use tool: {tool_name}]")

                # Parse arguments safely
                try:
                    kwargs: dict[str, Any] = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError as exc:
                    tool_result = json.dumps({
                        "status": "error",
                        "message": f"Failed to parse tool arguments: {exc}",
                    })
                else:
                    # O(1) dispatch -- no if/elif chains
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

                print(f"[Tool result]: {tool_result}\n")

                # Append the tool result so the model can reason over it
                self._history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

            # Loop again -- the model will now reason over the tool results
            self._trim_history()
            retried = False  # reset retry flag for the next sub-loop iteration
