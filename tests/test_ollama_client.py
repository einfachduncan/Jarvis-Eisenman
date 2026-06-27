import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import requests

from config import OllamaSettings
from ollama_client import ChatMessage, OllamaChatSession, OllamaError


class OllamaChatSessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.prompt_path = Path(self.temporary_directory.name) / "jarvis.txt"
        self.prompt_path.write_text("You are JARVIS.", encoding="utf-8")
        self.http = Mock()
        self.response = Mock()
        self.response.iter_lines.return_value = iter(
            [
                b'{"message":{"content":"Hello "},"done":false}',
                b'{"message":{"content":"there."},"done":true}',
            ]
        )
        self.http.post.return_value = self.response
        self.session = OllamaChatSession(
            settings=OllamaSettings(
                model="test-model",
                url="http://localhost:11434",
                timeout=10,
            ),
            http_client=self.http,
            prompt_path=self.prompt_path,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_stream_records_completed_exchange(self):
        self.assertEqual(list(self.session.stream(" Hi ")), ["Hello ", "there."])
        self.assertEqual(
            self.session.history,
            (
                ChatMessage("user", "Hi"),
                ChatMessage("assistant", "Hello there."),
            ),
        )
        payload = self.http.post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["messages"][0]["content"], "You are JARVIS.")
        self.response.close.assert_called_once_with()

    def test_failed_request_does_not_change_history(self):
        self.http.post.side_effect = requests.ConnectionError("offline")

        with self.assertRaisesRegex(OllamaError, "offline"):
            list(self.session.stream("Hello"))

        self.assertEqual(self.session.history, ())

    def test_api_stream_error_does_not_change_history(self):
        self.response.iter_lines.return_value = iter(
            [b'{"error":"model unavailable"}']
        )

        with self.assertRaisesRegex(OllamaError, "model unavailable"):
            list(self.session.stream("Hello"))

        self.assertEqual(self.session.history, ())
        self.response.close.assert_called_once_with()

    def test_empty_response_is_rejected(self):
        self.response.iter_lines.return_value = iter(
            [b'{"message":{"content":""},"done":true}']
        )

        with self.assertRaisesRegex(OllamaError, "empty response"):
            list(self.session.stream("Hello"))

        self.assertEqual(self.session.history, ())

    def test_clear_removes_history(self):
        list(self.session.stream("Hello"))
        self.session.clear()
        self.assertEqual(self.session.history, ())

    def test_empty_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            list(self.session.stream("  "))
