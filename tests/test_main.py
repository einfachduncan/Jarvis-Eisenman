import unittest
from unittest.mock import Mock, patch

from config import ConfigurationError
from gemini_client import GeminiError
from main import run_chat


class ChatTests(unittest.TestCase):
    @patch("main.get_settings")
    def test_chat_sends_prompt_and_exits(self, get_settings_mock):
        answers = iter(["Hello", "/exit"])
        output = Mock()
        ask_mock = Mock(return_value="Hi there")

        result = run_chat(lambda _prompt: next(answers), output, ask_mock)

        self.assertEqual(result, 0)
        ask_mock.assert_called_once_with("Hello")
        output.assert_any_call("Gemini: Hi there")
        output.assert_any_call("Goodbye!")

    @patch("main.get_settings", side_effect=ConfigurationError("missing key"))
    def test_configuration_error_stops_startup(self, _get_settings_mock):
        output = Mock()

        self.assertEqual(run_chat(output_fn=output), 1)
        output.assert_called_once_with("Configuration error: missing key")

    @patch("main.get_settings")
    def test_request_error_does_not_close_chat(self, get_settings_mock):
        answers = iter(["Hello", "/exit"])
        output = Mock()
        ask_mock = Mock(side_effect=GeminiError("offline"))

        result = run_chat(lambda _prompt: next(answers), output, ask_mock)

        self.assertEqual(result, 0)
        output.assert_any_call("Error: offline")

    @patch("main.get_settings")
    def test_keyboard_interrupt_exits_cleanly(self, get_settings_mock):
        output = Mock()

        result = run_chat(
            input_fn=Mock(side_effect=KeyboardInterrupt),
            output_fn=output,
        )

        self.assertEqual(result, 0)
        output.assert_any_call("\nGoodbye!")
