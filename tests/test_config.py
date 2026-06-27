import os
import unittest
from unittest.mock import patch

from config import (
    ConfigurationError,
    get_ollama_settings,
    get_voice_settings,
    get_wake_word_settings,
)


class OllamaSettingsTests(unittest.TestCase):
    @patch("config.load_dotenv")
    def test_settings_are_loaded_and_trimmed(self, _load_dotenv):
        environment = {
            "OLLAMA_MODEL": "  qwen3:8b  ",
            "OLLAMA_URL": "  http://localhost:11434/  ",
            "OLLAMA_TIMEOUT": "30",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = get_ollama_settings()

        self.assertEqual(settings.model, "qwen3:8b")
        self.assertEqual(settings.url, "http://localhost:11434")
        self.assertEqual(settings.timeout, 30)

    @patch("config.load_dotenv")
    def test_invalid_url_is_rejected(self, _load_dotenv):
        with patch.dict(
            os.environ,
            {"OLLAMA_URL": "localhost:11434"},
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigurationError, "OLLAMA_URL"):
                get_ollama_settings()


class VoiceSettingsTests(unittest.TestCase):
    @patch("config.load_dotenv")
    def test_voice_settings_are_loaded(self, _load_dotenv):
        environment = {
            "WHISPER_MODEL": "small",
            "WHISPER_LANGUAGE": "de",
            "VOICE_SAMPLE_RATE": "22050",
            "PIPER_MODEL_PATH": "models/piper/test.onnx",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = get_voice_settings()

        self.assertEqual(settings.model, "small")
        self.assertEqual(settings.language, "de")
        self.assertEqual(settings.sample_rate, 22050)
        self.assertEqual(settings.piper_model_path.name, "test.onnx")

    @patch("config.load_dotenv")
    def test_invalid_voice_number_is_rejected(self, _load_dotenv):
        with patch.dict(
            os.environ,
            {"VOICE_SAMPLE_RATE": "not-a-number"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ConfigurationError, "Invalid numeric"
            ):
                get_voice_settings()

    @patch("config.load_dotenv")
    def test_piper_model_path_is_loaded(self, _load_dotenv):
        with patch.dict(
            os.environ,
            {"PIPER_MODEL_PATH": "models/custom.onnx"},
            clear=True,
        ):
            settings = get_voice_settings()
        self.assertEqual(settings.piper_model_path.name, "custom.onnx")


class WakeWordSettingsTests(unittest.TestCase):
    @patch("config.load_dotenv")
    def test_wake_word_settings_are_loaded(self, _load_dotenv):
        environment = {
            "WAKE_WORD_MODEL": "custom_model",
            "WAKE_WORD_THRESHOLD": "0.65",
            "WAKE_WORD_FRAME_DURATION": "0.1",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = get_wake_word_settings()

        self.assertEqual(settings.model, "custom_model")
        self.assertEqual(settings.model_directory.name, "openwakeword")
        self.assertEqual(settings.threshold, 0.65)
        self.assertEqual(settings.frame_duration, 0.1)

    @patch("config.load_dotenv")
    def test_invalid_wake_word_threshold_is_rejected(self, _load_dotenv):
        with patch.dict(
            os.environ,
            {"WAKE_WORD_THRESHOLD": "1.5"},
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigurationError, "THRESHOLD"):
                get_wake_word_settings()
