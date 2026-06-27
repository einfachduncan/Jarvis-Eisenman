import queue
import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from config import WakeWordSettings
from wakeword import (
    WakeWordService,
    _create_wake_word_model,
    _prediction_score,
    _wake_word_worker,
)


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


class FakeModel:
    def __init__(self, scores):
        self.scores = iter(scores)

    def predict(self, _audio):
        return {"hey_jarvis": next(self.scores)}


class FakeProcess:
    def __init__(self, *, args, **_kwargs):
        self.args = args
        self.alive = False
        self.terminated = False

    def start(self):
        self.alive = True
        self.args[4].set()

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        if self.args[5].is_set():
            self.alive = False

    def terminate(self):
        self.terminated = True
        self.alive = False


class FakeContext:
    def Event(self):
        return threading.Event()

    def Queue(self):
        return queue.Queue()

    def Process(self, **kwargs):
        return FakeProcess(**kwargs)


class WakeWordTests(unittest.TestCase):
    def setUp(self):
        self.settings = WakeWordSettings(
            threshold=0.5,
            sample_rate=16_000,
            frame_duration=0.08,
            startup_timeout=1.0,
            microphone_release_timeout=1.0,
        )

    def test_prediction_score_matches_normalized_model_name(self):
        predictions = {
            "other": np.array([0.9]),
            "hey jarvis": np.array([0.1, 0.7]),
        }

        self.assertEqual(
            _prediction_score(predictions, "hey_jarvis"),
            0.7,
        )

    def test_model_download_uses_project_cache_and_onnx(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_directory = Path(temporary_directory)
            settings = WakeWordSettings(
                model_directory=model_directory,
            )

            def create_fake_models(*, model_names, target_directory):
                target = Path(target_directory)
                target.mkdir(parents=True, exist_ok=True)
                (target / "hey_jarvis_v0.1.onnx").touch()
                (target / "melspectrogram.onnx").touch()
                (target / "embedding_model.onnx").touch()

            model = Mock()
            with patch(
                "openwakeword.utils.download_models",
                side_effect=create_fake_models,
            ) as download_mock, patch(
                "openwakeword.model.Model",
                return_value=model,
            ) as model_class:
                result = _create_wake_word_model(settings)

        self.assertIs(result, model)
        download_mock.assert_called_once_with(
            model_names=["hey_jarvis"],
            target_directory=str(model_directory.resolve()),
        )
        self.assertEqual(
            model_class.call_args.kwargs["inference_framework"],
            "onnx",
        )
        self.assertTrue(
            model_class.call_args.kwargs["wakeword_models"][0].endswith(
                "hey_jarvis_v0.1.onnx"
            )
        )

    def test_worker_detects_and_releases_microphone(self):
        block = np.zeros((1280, 1), dtype=np.int16)
        sounddevice = FakeSoundDevice([block, block])
        activated = threading.Event()
        pause = threading.Event()
        paused = threading.Event()
        ready = threading.Event()
        stop = threading.Event()
        errors = queue.Queue()
        model = FakeModel([0.1, 0.8])
        worker = threading.Thread(
            target=_wake_word_worker,
            args=(
                self.settings,
                activated,
                pause,
                paused,
                ready,
                stop,
                errors,
                lambda _settings: model,
                sounddevice,
            ),
        )

        worker.start()
        self.assertTrue(ready.wait(1.0))
        self.assertTrue(activated.wait(1.0))
        self.assertTrue(paused.wait(1.0))
        self.assertTrue(pause.is_set())
        self.assertTrue(errors.empty())
        stop.set()
        pause.clear()
        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(sounddevice.arguments["dtype"], "int16")
        self.assertEqual(sounddevice.arguments["blocksize"], 1280)

    def test_service_controls_sleep_pause_and_stop(self):
        service = WakeWordService(
            settings=self.settings,
            context=FakeContext(),
        )

        service.enter_sleep_mode()
        self.assertTrue(service.is_running)
        self.assertTrue(service.is_sleeping)

        service._activated_event.set()
        service._pause_event.set()
        service._paused_event.set()
        self.assertTrue(service.wait_for_activation(timeout=0.1))
        service.pause()

        service.stop()
        self.assertFalse(service.is_running)

    def test_wait_for_activation_can_time_out(self):
        service = WakeWordService(
            settings=self.settings,
            context=FakeContext(),
        )
        service.start()

        self.assertFalse(service.wait_for_activation(timeout=0.01))
        service.stop()
