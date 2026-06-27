"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_MODEL = "gemini-3.5-flash"


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing."""


@dataclass(frozen=True)
class Settings:
    """Validated settings required by the Gemini client."""

    api_key: str
    model: str = DEFAULT_MODEL


def get_settings() -> Settings:
    """Load and validate settings from the project-local .env file."""
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    if not api_key:
        raise ConfigurationError(
            "GEMINI_API_KEY is missing. Add your API key to the .env file."
        )

    return Settings(api_key=api_key, model=model)
