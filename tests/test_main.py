import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from config import ConfigurationError
from gemini_client import ChatMessage, GeminiError
from main import run_chat


class ChatTests(unittest.TestCase):
    def test_chat_streams_prompt_and_exits(self):
        answers = iter(["Hello", "exit"])
        output = Mock()
        streamed: list[str] = []
        session = Mock()
        session.stream.return_value = iter(["Hi ", "there"])

        result = run_chat(
            lambda _prompt: next(answers),
            output,
            streamed.append,
            lambda: session,
        )

        self.assertEqual(result, 0)
        session.stream.assert_called_once_with("Hello")
        self.assertEqual(streamed, ["Hi ", "there", "\n"])
        output.assert_any_call("JARVIS:")
        output.assert_any_call("Goodbye!")

    def test_configuration_error_stops_startup(self):
        output = Mock()

        result = run_chat(
            output_fn=output,
            session_factory=Mock(side_effect=ConfigurationError("missing key")),
        )

        self.assertEqual(result, 1)
        output.assert_called_once_with("Configuration error: missing key")

    def test_help_command_does_not_reach_gemini(self):
        answers = iter(["help", "exit"])
        output = Mock()
        session = Mock()

        run_chat(
            lambda _prompt: next(answers),
            output,
            Mock(),
            lambda: session,
        )

        session.stream.assert_not_called()
        output.assert_any_call("Available commands:")

    def test_history_command_displays_messages(self):
        answers = iter(["history", "exit"])
        output = Mock()
        session = SimpleNamespace(
            history=(
                ChatMessage("user", "Hello"),
                ChatMessage("assistant", "Good evening."),
            )
        )

        run_chat(
            lambda _prompt: next(answers),
            output,
            Mock(),
            lambda: session,
        )

        output.assert_any_call("1. You: Hello")
        output.assert_any_call("2. JARVIS: Good evening.")

    def test_empty_history_is_reported(self):
        answers = iter(["history", "exit"])
        output = Mock()
        session = SimpleNamespace(history=())

        run_chat(
            lambda _prompt: next(answers),
            output,
            Mock(),
            lambda: session,
        )

        output.assert_any_call("Chat history is empty.")

    def test_clear_command_resets_session(self):
        answers = iter(["clear", "exit"])
        output = Mock()
        session = Mock()

        run_chat(
            lambda _prompt: next(answers),
            output,
            Mock(),
            lambda: session,
        )

        session.clear.assert_called_once_with()
        output.assert_any_call("Chat history cleared.")

    def test_request_error_does_not_close_chat(self):
        answers = iter(["Hello", "exit"])
        output = Mock()
        streamed: list[str] = []
        session = Mock()
        session.stream.side_effect = GeminiError("offline")

        result = run_chat(
            lambda _prompt: next(answers),
            output,
            streamed.append,
            lambda: session,
        )

        self.assertEqual(result, 0)
        output.assert_any_call("Error: offline")

    def test_keyboard_interrupt_exits_cleanly(self):
        output = Mock()

        result = run_chat(
            input_fn=Mock(side_effect=KeyboardInterrupt),
            output_fn=output,
            session_factory=Mock(),
        )

        self.assertEqual(result, 0)
        output.assert_any_call("\nGoodbye!")
