import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from piper_tts import PiperTTS


class PiperTTSTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temporary_directory.name) / "voice.onnx"
        self.model_path.write_bytes(b"model")
        Path(f"{self.model_path}.json").write_text("{}", encoding="utf-8")
        self.process = Mock()
        self.process.communicate.return_value = ("", "")
        self.process.returncode = 0
        self.process_factory = Mock(return_value=self.process)
        self.sounddevice = Mock()
        self.soundfile = Mock()
        self.soundfile.read.return_value = ([0.1, 0.2], 22050)
        self.tts = PiperTTS(
            self.model_path,
            process_factory=self.process_factory,
            sounddevice_module=self.sounddevice,
            soundfile_module=self.soundfile,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_speak_uses_active_python_and_plays_wave(self):
        self.tts.speak("Hallo")

        command = self.process_factory.call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, "-m", "piper"])
        self.process.communicate.assert_called_once_with("Hallo")
        self.sounddevice.play.assert_called_once()
        self.sounddevice.wait.assert_called_once_with()

    def test_piper_error_is_reported(self):
        self.process.returncode = 2
        self.process.communicate.return_value = ("", "bad model")

        with self.assertRaisesRegex(RuntimeError, "bad model"):
            self.tts.speak("Hallo")

    def test_stop_terminates_process_and_audio(self):
        self.tts._process = self.process
        self.process.poll.return_value = None

        self.tts.stop()

        self.process.terminate.assert_called_once_with()
        self.sounddevice.stop.assert_called_once_with()

    def test_missing_voice_files_are_downloaded(self):
        missing_model = Path(self.temporary_directory.name) / "missing.onnx"
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.iter_content.return_value = [b"downloaded"]
        http = Mock()
        http.get.return_value = response
        tts = PiperTTS(
            missing_model,
            http_client=http,
            process_factory=self.process_factory,
            sounddevice_module=self.sounddevice,
            soundfile_module=self.soundfile,
        )

        tts.speak("Hallo")

        self.assertTrue(missing_model.is_file())
        self.assertTrue(Path(f"{missing_model}.json").is_file())
        self.assertEqual(http.get.call_count, 2)
