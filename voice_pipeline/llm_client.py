from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from openai import OpenAI


@dataclass
class ChatConfig:
    """
    Configuration for LLM session.
    """

    model: str
    system_prompt: str
    max_turns: int = 10  # number of user/assistant pairs to keep


class ChatSession:
    """
    Simple conversational wrapper around the OpenAI chat completions API.

    Keeps a bounded in-memory history of user/assistant turns and rebuilds
    the messages list on each call to generate_reply.
    """

    def __init__(self, client: OpenAI, config: ChatConfig) -> None:
        self._client = client
        self._config = config
        self._history: List[Dict[str, str]] = []

    def _trim_history(self) -> None:
        max_msgs = 2 * self._config.max_turns
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    def add_user_message(self, text: str) -> None:
        self._history.append({"role": "user", "content": text})
        self._trim_history()

    def add_assistant_message(self, text: str) -> None:
        self._history.append({"role": "assistant", "content": text})
        self._trim_history()

    def generate_reply(self, user_text: str) -> str:
        """
        Append a user message, call the model, append the assistant reply,
        and return the reply content.
        """
        self.add_user_message(user_text)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._config.system_prompt.strip()}
        ]
        messages.extend(self._history)

        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
        )
        reply = (response.choices[0].message.content or "").strip()
        self.add_assistant_message(reply)
        return reply


