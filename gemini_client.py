"""Small, stateless interface to the Gemini API."""

from __future__ import annotations

from typing import Any

from config import Settings, get_settings


class GeminiError(RuntimeError):
    """Raised when Gemini cannot produce a usable response."""


def _create_client(api_key: str) -> Any:
    from google import genai

    return genai.Client(api_key=api_key)


def ask(prompt: str) -> str:
    """Send one prompt to Gemini and return its text response.

    Every call is independent. Conversation memory is intentionally outside
    the scope of phase 1.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("The prompt must not be empty.")

    settings: Settings = get_settings()

    try:
        client = _create_client(settings.api_key)
        response = client.models.generate_content(
            model=settings.model,
            contents=prompt.strip(),
        )
        text = response.text
    except Exception as exc:
        raise GeminiError(f"Gemini request failed: {exc}") from exc

    if not text or not text.strip():
        raise GeminiError("Gemini returned an empty response.")

    return text.strip()
