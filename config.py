"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_WHISPER_MODEL = "base"


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing."""


@dataclass(frozen=True)
class Settings:
    """Validated settings required by the Gemini client."""

    api_key: str
    model: str = DEFAULT_MODEL


@dataclass(frozen=True)
class VoiceSettings:
    """Settings for microphone capture, Whisper, and speech output."""

    model: str = DEFAULT_WHISPER_MODEL
    language: str | None = None
    device: str = "cpu"
    compute_type: str = "int8"
    sample_rate: int = 16_000
    block_duration: float = 0.1
    silence_threshold: float = 0.015
    silence_duration: float = 1.0
    start_timeout: float = 8.0
    max_duration: float = 30.0
    tts_rate: int = 185
    tts_volume: float = 1.0


@dataclass(frozen=True)
class WakeWordSettings:
    """Settings for the OpenWakeWord background listener."""

    model: str = "hey_jarvis"
    model_directory: Path = (
        Path(__file__).with_name(".models") / "openwakeword"
    )
    threshold: float = 0.5
    sample_rate: int = 16_000
    frame_duration: float = 0.08
    startup_timeout: float = 120.0
    microphone_release_timeout: float = 3.0


def _load_environment() -> None:
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)


def get_settings() -> Settings:
    """Load and validate settings from the project-local .env file."""
    _load_environment()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    if not api_key:
        raise ConfigurationError(
            "GEMINI_API_KEY is missing. Add your API key to the .env file."
        )

    return Settings(api_key=api_key, model=model)


def get_voice_settings() -> VoiceSettings:
    """Load and validate optional voice settings."""
    _load_environment()

    try:
        settings = VoiceSettings(
            model=os.getenv("WHISPER_MODEL", DEFAULT_WHISPER_MODEL).strip()
            or DEFAULT_WHISPER_MODEL,
            language=os.getenv("WHISPER_LANGUAGE", "").strip() or None,
            device=os.getenv("WHISPER_DEVICE", "cpu").strip() or "cpu",
            compute_type=(
                os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip() or "int8"
            ),
            sample_rate=int(os.getenv("VOICE_SAMPLE_RATE", "16000")),
            silence_threshold=float(
                os.getenv("VOICE_SILENCE_THRESHOLD", "0.015")
            ),
            silence_duration=float(
                os.getenv("VOICE_SILENCE_DURATION", "1.0")
            ),
            start_timeout=float(os.getenv("VOICE_START_TIMEOUT", "8.0")),
            max_duration=float(os.getenv("VOICE_MAX_DURATION", "30.0")),
            tts_rate=int(os.getenv("VOICE_TTS_RATE", "185")),
            tts_volume=float(os.getenv("VOICE_TTS_VOLUME", "1.0")),
        )
    except ValueError as exc:
        raise ConfigurationError(
            f"Invalid numeric voice configuration: {exc}"
        ) from exc

    positive_values = {
        "VOICE_SAMPLE_RATE": settings.sample_rate,
        "VOICE_SILENCE_THRESHOLD": settings.silence_threshold,
        "VOICE_SILENCE_DURATION": settings.silence_duration,
        "VOICE_START_TIMEOUT": settings.start_timeout,
        "VOICE_MAX_DURATION": settings.max_duration,
        "VOICE_TTS_RATE": settings.tts_rate,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ConfigurationError(f"{invalid[0]} must be greater than zero.")
    if not 0.0 <= settings.tts_volume <= 1.0:
        raise ConfigurationError("VOICE_TTS_VOLUME must be between 0 and 1.")

    return settings


def get_wake_word_settings() -> WakeWordSettings:
    """Load and validate wake-word settings."""
    _load_environment()

    try:
        settings = WakeWordSettings(
            model=os.getenv("WAKE_WORD_MODEL", "hey_jarvis").strip()
            or "hey_jarvis",
            model_directory=Path(
                os.getenv("WAKE_WORD_MODEL_DIR", "").strip()
                or Path(__file__).with_name(".models") / "openwakeword"
            ).expanduser(),
            threshold=float(os.getenv("WAKE_WORD_THRESHOLD", "0.5")),
            sample_rate=int(os.getenv("WAKE_WORD_SAMPLE_RATE", "16000")),
            frame_duration=float(
                os.getenv("WAKE_WORD_FRAME_DURATION", "0.08")
            ),
            startup_timeout=float(
                os.getenv("WAKE_WORD_STARTUP_TIMEOUT", "120.0")
            ),
            microphone_release_timeout=float(
                os.getenv("WAKE_WORD_RELEASE_TIMEOUT", "3.0")
            ),
        )
    except ValueError as exc:
        raise ConfigurationError(
            f"Invalid wake-word configuration: {exc}"
        ) from exc

    if not 0.0 < settings.threshold <= 1.0:
        raise ConfigurationError(
            "WAKE_WORD_THRESHOLD must be greater than 0 and at most 1."
        )
    positive_values = {
        "WAKE_WORD_SAMPLE_RATE": settings.sample_rate,
        "WAKE_WORD_FRAME_DURATION": settings.frame_duration,
        "WAKE_WORD_STARTUP_TIMEOUT": settings.startup_timeout,
        "WAKE_WORD_RELEASE_TIMEOUT": settings.microphone_release_timeout,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ConfigurationError(f"{invalid[0]} must be greater than zero.")

    return settings
