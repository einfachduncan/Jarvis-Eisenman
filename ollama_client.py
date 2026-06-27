"""Persistent streaming chat session backed by a local Ollama server."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from config import OllamaSettings, get_ollama_settings


DEFAULT_PROMPT_PATH = Path(__file__).with_name("prompts") / "jarvis.txt"


class OllamaError(RuntimeError):
    """Raised when Ollama cannot produce a usable response."""


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


def _load_system_prompt(path: Path) -> str:
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise OllamaError(f"Cannot read the JARVIS prompt file: {path}") from exc
    if not prompt:
        raise OllamaError(f"The JARVIS prompt file is empty: {path}")
    return prompt


class OllamaChatSession:
    """Stateful Ollama conversation that commits only completed exchanges."""

    def __init__(
        self,
        settings: OllamaSettings | None = None,
        http_client: Any = requests,
        prompt_path: Path | None = None,
    ) -> None:
        self.settings = settings or get_ollama_settings()
        self._http = http_client
        self._system_prompt = _load_system_prompt(
            prompt_path or DEFAULT_PROMPT_PATH
        )
        self._history: list[ChatMessage] = []

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        return tuple(self._history)

    def clear(self) -> None:
        self._history.clear()

    def stream(self, prompt: str) -> Iterator[str]:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("Prompt must not be empty.")

        pending_user = ChatMessage("user", clean_prompt)
        messages = [
            {"role": "system", "content": self._system_prompt},
            *(
                {"role": message.role, "content": message.content}
                for message in (*self._history, pending_user)
            ),
        ]
        response = None
        answer_parts: list[str] = []

        try:
            response = self._http.post(
                f"{self.settings.url}/api/chat",
                json={
                    "model": self.settings.model,
                    "messages": messages,
                    "stream": True,
                    "think": False,
                    "options": {
                        "num_predict": 120,
                        "temperature": 0.7,
                        "num_ctx": 4096,
                    },
                },
                timeout=self.settings.timeout,
                stream=True,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in data:
                    raise OllamaError(str(data["error"]))

                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    answer_parts.append(chunk)
                    yield chunk
                if data.get("done"):
                    break
        except OllamaError:
            raise
        except (requests.RequestException, OSError, UnicodeError) as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc
        finally:
            if response is not None:
                response.close()

        answer = "".join(answer_parts).strip()
        if not answer:
            raise OllamaError("Ollama returned an empty response.")

        self._history.extend(
            (
                pending_user,
                ChatMessage("assistant", answer),
            )
        )
