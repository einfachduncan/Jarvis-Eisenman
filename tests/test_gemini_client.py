import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from config import ConfigurationError, Settings
from gemini_client import ChatMessage, ChatSession, GeminiError, ask


class ChatSessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.prompt_path = Path(self.temporary_directory.name) / "jarvis.txt"
        self.prompt_path.write_text("You are JARVIS.", encoding="utf-8")
        self.client = Mock()
        self.chat = Mock()
        self.client.chats.create.return_value = self.chat

    def tearDown(self):
        self.temporary_directory.cleanup()

    @patch("gemini_client._create_chat")
    def test_session_loads_prompt_and_creates_chat(self, create_chat_mock):
        session = ChatSession(
            settings=Settings("secret", "test-model"),
            client=self.client,
            prompt_path=self.prompt_path,
        )

        create_chat_mock.assert_called_once_with(
            self.client,
            Settings("secret", "test-model"),
            "You are JARVIS.",
        )
        self.assertEqual(session.history, ())

    @patch(
        "gemini_client._create_chat",
        side_effect=RuntimeError("model unavailable"),
    )
    def test_session_initialization_error_is_wrapped(self, _create_chat_mock):
        with self.assertRaisesRegex(GeminiError, "model unavailable"):
            ChatSession(
                settings=Settings("secret"),
                client=self.client,
                prompt_path=self.prompt_path,
            )

    def test_missing_prompt_file_is_rejected(self):
        with self.assertRaisesRegex(ConfigurationError, "Cannot read"):
            ChatSession(
                settings=Settings("secret"),
                client=self.client,
                prompt_path=self.prompt_path.with_name("missing.txt"),
            )

    def test_empty_prompt_is_rejected(self):
        session = ChatSession(
            settings=Settings("secret"),
            client=self.client,
            prompt_path=self.prompt_path,
        )

        with self.assertRaises(ValueError):
            session.send("   ")

    def test_send_records_completed_exchange(self):
        self.chat.send_message.return_value = SimpleNamespace(
            text="  At your service.  "
        )
        session = ChatSession(
            settings=Settings("secret"),
            client=self.client,
            prompt_path=self.prompt_path,
        )

        result = session.send("  Hello  ")

        self.assertEqual(result, "At your service.")
        self.assertEqual(
            session.history,
            (
                ChatMessage("user", "Hello"),
                ChatMessage("assistant", "At your service."),
            ),
        )
        self.chat.send_message.assert_called_once_with("Hello")

    def test_stream_yields_chunks_and_records_exchange(self):
        self.chat.send_message_stream.return_value = iter(
            [
                SimpleNamespace(text="Good "),
                SimpleNamespace(text=None),
                SimpleNamespace(text="evening."),
            ]
        )
        session = ChatSession(
            settings=Settings("secret"),
            client=self.client,
            prompt_path=self.prompt_path,
        )

        self.assertEqual(list(session.stream("Hello")), ["Good ", "evening."])
        self.assertEqual(
            session.history,
            (
                ChatMessage("user", "Hello"),
                ChatMessage("assistant", "Good evening."),
            ),
        )

    def test_api_error_is_wrapped_and_not_recorded(self):
        self.chat.send_message.side_effect = RuntimeError("offline")
        session = ChatSession(
            settings=Settings("secret"),
            client=self.client,
            prompt_path=self.prompt_path,
        )

        with self.assertRaisesRegex(GeminiError, "offline"):
            session.send("Hello")
        self.assertEqual(session.history, ())

    def test_empty_stream_is_rejected(self):
        self.chat.send_message_stream.return_value = iter(
            [SimpleNamespace(text=None)]
        )
        session = ChatSession(
            settings=Settings("secret"),
            client=self.client,
            prompt_path=self.prompt_path,
        )

        with self.assertRaisesRegex(GeminiError, "empty response"):
            list(session.stream("Hello"))
        self.assertEqual(session.history, ())

    @patch("gemini_client._create_chat")
    def test_clear_recreates_chat_and_removes_history(self, create_chat_mock):
        first_chat = Mock()
        second_chat = Mock()
        first_chat.send_message.return_value = SimpleNamespace(text="Reply")
        create_chat_mock.side_effect = [first_chat, second_chat]
        session = ChatSession(
            settings=Settings("secret"),
            client=self.client,
            prompt_path=self.prompt_path,
        )
        session.send("Hello")

        session.clear()

        self.assertEqual(session.history, ())
        self.assertEqual(create_chat_mock.call_count, 2)

    @patch("gemini_client._create_chat")
    def test_failed_clear_preserves_existing_history(self, create_chat_mock):
        first_chat = Mock()
        first_chat.send_message.return_value = SimpleNamespace(text="Reply")
        create_chat_mock.side_effect = [
            first_chat,
            RuntimeError("cannot reset"),
        ]
        session = ChatSession(
            settings=Settings("secret"),
            client=self.client,
            prompt_path=self.prompt_path,
        )
        session.send("Hello")

        with self.assertRaisesRegex(GeminiError, "cannot reset"):
            session.clear()

        self.assertEqual(
            session.history,
            (
                ChatMessage("user", "Hello"),
                ChatMessage("assistant", "Reply"),
            ),
        )


class AskTests(unittest.TestCase):
    @patch("gemini_client.ChatSession")
    def test_ask_uses_a_new_session(self, session_class_mock):
        session_class_mock.return_value.send.return_value = "Hello"

        self.assertEqual(ask("Hi"), "Hello")
        session_class_mock.return_value.send.assert_called_once_with("Hi")
