from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator

import requests


class OllamaError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class OllamaChatSession:
    def __init__(self) -> None:
        self.model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self._history: list[ChatMessage] = []

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        return tuple(self._history)

    def clear(self) -> None:
        self._history.clear()

    def stream(self, prompt: str) -> Iterator[str]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt must not be empty.")

        self._history.append(ChatMessage("user", prompt))

        messages = [
            {
                "role": "system",
                "content": (
                    "Du bist JARVIS, ein hilfreicher deutscher KI-Assistent. "
                    "Antworte immer sichtbar auf Deutsch. "
                    "Gib niemals eine leere Antwort. "
                    "Nutze kein Thinking-Format."
                ),
            }
        ]

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        try:
            response = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "num_predict": 120,
                        "temperature": 0.7,
                        "num_ctx": 4096,
                    },
                },
                timeout=120,
                stream=True,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc

        answer_parts: list[str] = []

        for line in response.iter_lines():
            if not line:
                continue

            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            if "error" in data:
                raise OllamaError(str(data["error"]))

            message = data.get("message", {})
            chunk = message.get("content", "")

            if chunk:
                answer_parts.append(chunk)
                yield chunk

            if data.get("done"):
                break

        answer = "".join(answer_parts).strip()

        if not answer:
            answer = "Ich habe gerade keine brauchbare Antwort erzeugt."
            yield answer

        self._history.append(ChatMessage("assistant", answer))