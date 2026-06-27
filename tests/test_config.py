import os
import unittest
from unittest.mock import patch

from config import (
    ConfigurationError,
    DEFAULT_MODEL,
    get_settings,
    get_voice_settings,
)


class SettingsTests(unittest.TestCase):
    @patch("config.load_dotenv")
    def test_missing_api_key_is_rejected(self, _load_dotenv):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigurationError):
                get_settings()

    @patch("config.load_dotenv")
    def test_settings_are_trimmed(self, _load_dotenv):
        environment = {
            "GEMINI_API_KEY": "  secret  ",
            "GEMINI_MODEL": "  custom-model  ",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = get_settings()

        self.assertEqual(settings.api_key, "secret")
        self.assertEqual(settings.model, "custom-model")

    @patch("config.load_dotenv")
    def test_empty_model_uses_default(self, _load_dotenv):
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "secret", "GEMINI_MODEL": "  "},
            clear=True,
        ):
            self.assertEqual(get_settings().model, DEFAULT_MODEL)


class VoiceSettingsTests(unittest.TestCase):
    @patch("config.load_dotenv")
    def test_voice_settings_are_loaded(self, _load_dotenv):
        environment = {
            "WHISPER_MODEL": "small",
            "WHISPER_LANGUAGE": "de",
            "VOICE_SAMPLE_RATE": "22050",
            "VOICE_TTS_VOLUME": "0.7",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = get_voice_settings()

        self.assertEqual(settings.model, "small")
        self.assertEqual(settings.language, "de")
        self.assertEqual(settings.sample_rate, 22050)
        self.assertEqual(settings.tts_volume, 0.7)

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
    def test_invalid_tts_volume_is_rejected(self, _load_dotenv):
        with patch.dict(
            os.environ,
            {"VOICE_TTS_VOLUME": "1.5"},
            clear=True,
        ):
            with self.assertRaisesRegex(ConfigurationError, "between 0 and 1"):
                get_voice_settings()
