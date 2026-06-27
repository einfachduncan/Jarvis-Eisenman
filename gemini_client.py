"""Gemini client and persistent in-process chat session."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import ConfigurationError, Settings, get_settings


DEFAULT_PROMPT_PATH = Path(__file__).with_name("prompts") / "jarvis.txt"


class GeminiError(RuntimeError):
    """Raised when Gemini cannot produce a usable response."""


def _create_client(api_key: str) -> Any:
    from google import genai

    return genai.Client(api_key=api_key)


def _create_chat(client: Any, settings: Settings, system_prompt: str) -> Any:
    from google.genai import types

    return client.chats.create(
        model=settings.model,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )


def _load_system_prompt(path: Path) -> str:
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot read the JARVIS prompt file: {path}"
        ) from exc

    if not prompt:
        raise ConfigurationError(f"The JARVIS prompt file is empty: {path}")

    return prompt


def _validate_prompt(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("The prompt must not be empty.")
    return prompt.strip()


@dataclass(frozen=True)
class ChatMessage:
    """One successfully completed message in the local chat history."""

    role: str
    content: str


class ChatSession:
    """A stateful Gemini chat with JARVIS instructions and local history."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        prompt_path: Path | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._system_prompt = _load_system_prompt(
            prompt_path or DEFAULT_PROMPT_PATH
        )
        self._messages: list[ChatMessage] = []
        try:
            self._client = client or _create_client(self._settings.api_key)
            self._chat = _create_chat(
                self._client, self._settings, self._system_prompt
            )
        except Exception as exc:
            raise GeminiError(
                f"Cannot initialize the Gemini chat: {exc}"
            ) from exc

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        """Return an immutable snapshot of completed chat messages."""
        return tuple(self._messages)

    def clear(self) -> None:
        """Clear local and Gemini conversation state."""
        try:
            fresh_chat = _create_chat(
                self._client, self._settings, self._system_prompt
            )
        except Exception as exc:
            raise GeminiError(
                f"Cannot reset the Gemini chat: {exc}"
            ) from exc

        self._chat = fresh_chat
        self._messages.clear()

    def send(self, prompt: str) -> str:
        """Send a message without streaming and record the completed exchange."""
        clean_prompt = _validate_prompt(prompt)

        try:
            response = self._chat.send_message(clean_prompt)
            text = response.text
        except Exception as exc:
            raise GeminiError(f"Gemini request failed: {exc}") from exc

        clean_response = _validate_response(text)
        self._record_exchange(clean_prompt, clean_response)
        return clean_response

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield response chunks and record the exchange after it completes."""
        clean_prompt = _validate_prompt(prompt)
        chunks: list[str] = []

        try:
            for chunk in self._chat.send_message_stream(clean_prompt):
                text = chunk.text
                if text:
                    chunks.append(text)
                    yield text
        except Exception as exc:
            raise GeminiError(f"Gemini request failed: {exc}") from exc

        clean_response = _validate_response("".join(chunks))
        self._record_exchange(clean_prompt, clean_response)

    def _record_exchange(self, prompt: str, response: str) -> None:
        self._messages.extend(
            (
                ChatMessage(role="user", content=prompt),
                ChatMessage(role="assistant", content=response),
            )
        )


def _validate_response(text: str | None) -> str:
    if not text or not text.strip():
        raise GeminiError("Gemini returned an empty response.")
    return text.strip()


def ask(prompt: str) -> str:
    """Send one independent prompt to Gemini and return its text response.

    ``ChatSession`` should be used for multi-turn conversations. This helper
    remains available for simple one-off requests.
    """
    return ChatSession().send(prompt)
