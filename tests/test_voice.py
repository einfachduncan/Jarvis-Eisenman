import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from config import VoiceSettings
from voice import PowerShellTTSEngine, Voice, VoiceError, _create_tts_engine


class FakeInputStream:
    def __init__(self, blocks):
        self.blocks = iter(blocks)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False

    def read(self, _block_size):
        return next(self.blocks), False


class FakeSoundDevice:
    def __init__(self, blocks):
        self.blocks = blocks
        self.arguments = None

    def InputStream(self, **kwargs):
        self.arguments = kwargs
        return FakeInputStream(self.blocks)


class VoiceTests(unittest.TestCase):
    def setUp(self):
        self.settings = VoiceSettings(
            sample_rate=100,
            block_duration=0.1,
            silence_threshold=0.1,
            silence_duration=0.2,
            start_timeout=5.0,
            max_duration=5.0,
        )

    def test_transcribe_uses_whisper_and_joins_segments(self):
        model = Mock()
        model.transcribe.return_value = (
            iter(
                [
                    SimpleNamespace(text=" Hello "),
                    SimpleNamespace(text="world. "),
                ]
            ),
            SimpleNamespace(language="en"),
        )
        factory = Mock(return_value=model)
        voice = Voice(settings=self.settings, model_factory=factory)
        audio = np.ones(100, dtype=np.float32)

        self.assertEqual(voice.transcribe(audio), "Hello world.")
        factory.assert_called_once_with(self.settings)
        passed_audio = model.transcribe.call_args.args[0]
        self.assertEqual(passed_audio.dtype, np.float32)
        self.assertTrue(model.transcribe.call_args.kwargs["vad_filter"])

    def test_empty_transcription_is_rejected(self):
        model = Mock()
        model.transcribe.return_value = (iter([]), SimpleNamespace())
        voice = Voice(
            settings=self.settings,
            model_factory=Mock(return_value=model),
        )

        with self.assertRaisesRegex(VoiceError, "did not recognize"):
            voice.transcribe(np.ones(10, dtype=np.float32))

    def test_recording_stops_after_speech_and_silence(self):
        quiet = np.zeros((10, 1), dtype=np.float32)
        speech = np.full((10, 1), 0.5, dtype=np.float32)
        sounddevice = FakeSoundDevice(
            [quiet, speech, speech, quiet, quiet]
        )
        model = Mock()
        model.transcribe.return_value = (
            iter([SimpleNamespace(text="Recorded")]),
            SimpleNamespace(),
        )
        voice = Voice(
            settings=self.settings,
            model_factory=Mock(return_value=model),
            sounddevice_module=sounddevice,
        )

        self.assertEqual(voice.listen(), "Recorded")
        recorded = model.transcribe.call_args.args[0]
        self.assertGreater(recorded.size, 0)
        self.assertEqual(sounddevice.arguments["samplerate"], 100)
        self.assertEqual(sounddevice.arguments["channels"], 1)

    def test_microphone_failure_is_wrapped(self):
        sounddevice = Mock()
        sounddevice.InputStream.side_effect = RuntimeError("no device")
        voice = Voice(
            settings=self.settings,
            sounddevice_module=sounddevice,
        )

        with self.assertRaisesRegex(VoiceError, "no device"):
            voice.listen()

    def test_speak_configures_tts_engine(self):
        engine = Mock()
        voice = Voice(
            settings=self.settings,
            tts_factory=Mock(return_value=engine),
        )

        voice.speak("At your service.", blocking=True)

        engine.setProperty.assert_any_call("rate", self.settings.tts_rate)
        engine.setProperty.assert_any_call(
            "volume", self.settings.tts_volume
        )
        engine.say.assert_called_once_with("At your service.")
        engine.runAndWait.assert_called_once_with()
        self.assertFalse(voice.is_speaking)

    def test_interrupt_stops_background_speech(self):
        release = threading.Event()
        engine = Mock()
        engine.runAndWait.side_effect = release.wait
        engine.stop.side_effect = release.set
        voice = Voice(
            settings=self.settings,
            tts_factory=Mock(return_value=engine),
        )

        voice.speak("Long response", blocking=False)
        self.assertTrue(engine.say.wait(timeout=1.0))
        voice.interrupt(wait=True)

        engine.stop.assert_called()
        self.assertFalse(voice.is_speaking)

    def test_tts_failure_is_wrapped(self):
        engine = Mock()
        engine.runAndWait.side_effect = RuntimeError("speaker unavailable")
        voice = Voice(
            settings=self.settings,
            tts_factory=Mock(return_value=engine),
        )

        with self.assertRaisesRegex(VoiceError, "speaker unavailable"):
            voice.speak("Hello", blocking=True)

    def test_windows_tts_falls_back_when_pyttsx3_is_unavailable(self):
        with patch(
            "pyttsx3.init", side_effect=RuntimeError("SAPI unavailable")
        ), patch("voice.sys.platform", "win32"):
            engine = _create_tts_engine()

        self.assertIsInstance(engine, PowerShellTTSEngine)
