import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from config import VoiceSettings
from voice import Voice, VoiceError


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

        result = voice.transcribe(np.ones(100, dtype=np.float32))

        self.assertEqual(result, "Hello world.")
        factory.assert_called_once_with(self.settings)
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
        self.assertEqual(sounddevice.arguments["samplerate"], 100)

    def test_microphone_failure_is_wrapped(self):
        sounddevice = Mock()
        sounddevice.InputStream.side_effect = RuntimeError("no device")
        voice = Voice(
            settings=self.settings,
            sounddevice_module=sounddevice,
        )

        with self.assertRaisesRegex(VoiceError, "no device"):
            voice.listen()

    def test_speak_uses_configured_piper_model(self):
        tts = Mock()
        factory = Mock(return_value=tts)
        voice = Voice(settings=self.settings, tts_factory=factory)

        voice.speak("At your service.", blocking=True)

        factory.assert_called_once_with(self.settings.piper_model_path)
        tts.speak.assert_called_once_with("At your service.")
        self.assertFalse(voice.is_speaking)

    def test_interrupt_stops_background_speech(self):
        started = threading.Event()
        release = threading.Event()
        tts = Mock()

        def speak(_text):
            started.set()
            release.wait()

        def stop():
            release.set()

        tts.speak.side_effect = speak
        tts.stop.side_effect = stop
        voice = Voice(
            settings=self.settings,
            tts_factory=Mock(return_value=tts),
        )

        voice.speak("Long response", blocking=False)
        self.assertTrue(started.wait(1.0))
        voice.interrupt(wait=True)

        tts.stop.assert_called_once_with()
        self.assertFalse(voice.is_speaking)

    def test_tts_failure_is_wrapped(self):
        tts = Mock()
        tts.speak.side_effect = RuntimeError("speaker unavailable")
        voice = Voice(
            settings=self.settings,
            tts_factory=Mock(return_value=tts),
        )

        with self.assertRaisesRegex(VoiceError, "speaker unavailable"):
            voice.speak("Hello", blocking=True)
