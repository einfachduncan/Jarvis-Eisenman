"""Terminal entry point for the phase-3 JARVIS text and voice chat."""

from __future__ import annotations

import sys
from collections.abc import Callable

from config import ConfigurationError
from gemini_client import ChatMessage, ChatSession, GeminiError
from voice import Voice, VoiceError


HELP_TEXT = (
    "Available commands:",
    "  help     Show this help",
    "  history  Show the current chat history",
    "  clear    Clear the current chat history",
    "  voice    Record one spoken prompt",
    "  exit     Close JARVIS",
)


def _format_history(messages: tuple[ChatMessage, ...]) -> tuple[str, ...]:
    if not messages:
        return ("Chat history is empty.",)

    lines = ["Chat history:"]
    for index, message in enumerate(messages, start=1):
        speaker = "You" if message.role == "user" else "JARVIS"
        lines.append(f"{index}. {speaker}: {message.content}")
    return tuple(lines)


def _stdout_write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def run_chat(
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    stream_output_fn: Callable[[str], None] | None = None,
    session_factory: Callable[[], ChatSession] = ChatSession,
    voice_factory: Callable[[], Voice] = Voice,
) -> int:
    """Run the interactive terminal chat and return a process exit code."""
    write = stream_output_fn or _stdout_write

    try:
        session = session_factory()
    except (ConfigurationError, GeminiError) as exc:
        output_fn(f"Configuration error: {exc}")
        return 1

    output_fn("JARVIS Gemini Chat")
    output_fn("Type help to see the available commands.")
    voice: Voice | None = None

    while True:
        try:
            prompt = input_fn("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nGoodbye!")
            return 0

        command = prompt.lower().removeprefix("/")

        if command == "exit":
            output_fn("Goodbye!")
            return 0

        if not prompt:
            continue

        if command == "help":
            for line in HELP_TEXT:
                output_fn(line)
            continue

        if command == "history":
            for line in _format_history(session.history):
                output_fn(line)
            continue

        if command == "clear":
            try:
                session.clear()
                output_fn("Chat history cleared.")
            except (ConfigurationError, GeminiError) as exc:
                output_fn(f"Error: {exc}")
            continue

        spoken_turn = False
        if command == "voice":
            try:
                if voice is None:
                    voice = voice_factory()
                output_fn("Listening...")
                prompt = voice.listen()
                output_fn(f"You (voice): {prompt}")
                spoken_turn = True
            except KeyboardInterrupt:
                output_fn("\nVoice input cancelled.")
                continue
            except (ConfigurationError, VoiceError) as exc:
                output_fn(f"Voice error: {exc}")
                continue

        response_parts: list[str] = []
        try:
            output_fn("JARVIS:")
            for chunk in session.stream(prompt):
                response_parts.append(chunk)
                write(chunk)
            write("\n")
        except (ConfigurationError, GeminiError, ValueError) as exc:
            write("\n")
            output_fn(f"Error: {exc}")
            continue

        if spoken_turn and voice is not None:
            try:
                voice.speak("".join(response_parts), blocking=True)
            except KeyboardInterrupt:
                voice.interrupt(wait=True)
                output_fn("\nSpeech interrupted.")
            except (VoiceError, ValueError) as exc:
                output_fn(f"Voice error: {exc}")


def main() -> int:
    return run_chat()


if __name__ == "__main__":
    raise SystemExit(main())
