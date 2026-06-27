import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from config import Settings
from gemini_client import GeminiError, ask


class AskTests(unittest.TestCase):
    def test_empty_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            ask("   ")

    @patch("gemini_client._create_client")
    @patch("gemini_client.get_settings")
    def test_prompt_is_sent_and_response_is_trimmed(
        self, get_settings_mock, create_client_mock
    ):
        get_settings_mock.return_value = Settings("secret", "test-model")
        generate = Mock(return_value=SimpleNamespace(text="  Hello!  "))
        create_client_mock.return_value.models.generate_content = generate

        self.assertEqual(ask("  Hi  "), "Hello!")
        create_client_mock.assert_called_once_with("secret")
        generate.assert_called_once_with(model="test-model", contents="Hi")

    @patch("gemini_client._create_client")
    @patch("gemini_client.get_settings")
    def test_api_errors_are_wrapped(self, get_settings_mock, create_client_mock):
        get_settings_mock.return_value = Settings("secret")
        create_client_mock.side_effect = RuntimeError("network unavailable")

        with self.assertRaisesRegex(GeminiError, "network unavailable"):
            ask("Hello")

    @patch("gemini_client._create_client")
    @patch("gemini_client.get_settings")
    def test_empty_api_response_is_rejected(
        self, get_settings_mock, create_client_mock
    ):
        get_settings_mock.return_value = Settings("secret")
        create_client_mock.return_value.models.generate_content.return_value = (
            SimpleNamespace(text=None)
        )

        with self.assertRaisesRegex(GeminiError, "empty response"):
            ask("Hello")
