"""Terminal entry point for the phase-1 Gemini chat."""

from __future__ import annotations

from collections.abc import Callable

from config import ConfigurationError, get_settings
from gemini_client import GeminiError, ask


EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}


def run_chat(
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    ask_fn: Callable[[str], str] = ask,
) -> int:
    """Run the interactive terminal chat and return a process exit code."""
    try:
        get_settings()
    except ConfigurationError as exc:
        output_fn(f"Configuration error: {exc}")
        return 1

    output_fn("Javis Gemini Chat")
    output_fn("Type /exit to close the chat.")

    while True:
        try:
            prompt = input_fn("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nGoodbye!")
            return 0

        if prompt.lower() in EXIT_COMMANDS:
            output_fn("Goodbye!")
            return 0

        if not prompt:
            continue

        try:
            output_fn(f"Gemini: {ask_fn(prompt)}")
        except (ConfigurationError, GeminiError, ValueError) as exc:
            output_fn(f"Error: {exc}")


def main() -> int:
    return run_chat()


if __name__ == "__main__":
    raise SystemExit(main())
