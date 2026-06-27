import os
import unittest
from unittest.mock import patch

from config import ConfigurationError, DEFAULT_MODEL, get_settings


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
