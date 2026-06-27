"""OpenWakeWord listener isolated in a background process."""

from __future__ import annotations

import multiprocessing
import queue
from typing import Any

import numpy as np

from config import WakeWordSettings, get_wake_word_settings


class WakeWordError(RuntimeError):
    """Raised when the background wake-word listener cannot operate."""


def _create_wake_word_model(settings: WakeWordSettings) -> Any:
    import openwakeword
    from openwakeword.model import Model

    model_directory = settings.model_directory.resolve()
    openwakeword.utils.download_models(
        model_names=[settings.model],
        target_directory=str(model_directory),
    )
    candidates = sorted(model_directory.glob(f"{settings.model}*.onnx"))
    if not candidates:
        raise WakeWordError(
            f"No ONNX wake-word model was downloaded for {settings.model}."
        )

    return Model(
        wakeword_models=[str(candidates[-1])],
        inference_framework="onnx",
        melspec_model_path=str(model_directory / "melspectrogram.onnx"),
        embedding_model_path=str(model_directory / "embedding_model.onnx"),
    )


def _load_sounddevice() -> Any:
    import sounddevice

    return sounddevice


def _prediction_score(predictions: dict[str, Any], model_name: str) -> float:
    if not predictions:
        return 0.0

    normalized_target = "".join(
        character for character in model_name.lower() if character.isalnum()
    )
    for name, value in predictions.items():
        normalized_name = "".join(
            character for character in str(name).lower()
            if character.isalnum()
        )
        if normalized_name == normalized_target:
            return float(np.asarray(value).reshape(-1)[-1])

    return max(
        float(np.asarray(value).reshape(-1)[-1])
        for value in predictions.values()
    )


def _wake_word_worker(
    settings: WakeWordSettings,
    activated_event: Any,
    pause_event: Any,
    paused_event: Any,
    ready_event: Any,
    stop_event: Any,
    error_queue: Any,
    model_factory: Any = _create_wake_word_model,
    sounddevice_module: Any | None = None,
) -> None:
    """Capture audio until activation while honoring pause and stop signals."""
    try:
        model = model_factory(settings)
        sounddevice = sounddevice_module or _load_sounddevice()
        block_size = max(
            1, int(settings.sample_rate * settings.frame_duration)
        )
        ready_event.set()

        while not stop_event.is_set():
            if pause_event.is_set():
                paused_event.set()
                stop_event.wait(0.05)
                continue

            paused_event.clear()
            with sounddevice.InputStream(
                samplerate=settings.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=block_size,
            ) as stream:
                while not stop_event.is_set() and not pause_event.is_set():
                    data, _overflowed = stream.read(block_size)
                    audio = np.asarray(data, dtype=np.int16).reshape(-1)
                    predictions = model.predict(audio)
                    if (
                        _prediction_score(predictions, settings.model)
                        >= settings.threshold
                    ):
                        pause_event.set()
                        activated_event.set()
                        break

            paused_event.set()
    except Exception as exc:
        try:
            error_queue.put(str(exc))
        finally:
            ready_event.set()
            activated_event.set()
            paused_event.set()


class WakeWordService:
    """Control the wake-word process and coordinate microphone ownership."""

    def __init__(
        self,
        settings: WakeWordSettings | None = None,
        context: Any | None = None,
    ) -> None:
        self.settings = settings or get_wake_word_settings()
        self._context = context or multiprocessing.get_context("spawn")
        self._activated_event = self._context.Event()
        self._pause_event = self._context.Event()
        self._paused_event = self._context.Event()
        self._ready_event = self._context.Event()
        self._stop_event = self._context.Event()
        self._error_queue = self._context.Queue()
        self._process: Any | None = None

    @property
    def is_running(self) -> bool:
        return bool(self._process and self._process.is_alive())

    @property
    def is_sleeping(self) -> bool:
        return self.is_running and not self._pause_event.is_set()

    def start(self) -> None:
        """Start the listener process and wait until its model is ready."""
        if self.is_running:
            return

        self._activated_event.clear()
        self._pause_event.set()
        self._paused_event.clear()
        self._ready_event.clear()
        self._stop_event.clear()
        self._drain_errors()
        self._process = self._context.Process(
            target=_wake_word_worker,
            args=(
                self.settings,
                self._activated_event,
                self._pause_event,
                self._paused_event,
                self._ready_event,
                self._stop_event,
                self._error_queue,
            ),
            name="jarvis-wake-word",
            daemon=True,
        )
        self._process.start()

        if not self._ready_event.wait(self.settings.startup_timeout):
            self.stop()
            raise WakeWordError(
                "OpenWakeWord did not initialize before the startup timeout."
            )
        self._raise_background_error()
        if not self.is_running:
            raise WakeWordError("The OpenWakeWord process stopped unexpectedly.")

    def enter_sleep_mode(self) -> None:
        """Start or resume listening for the configured wake phrase."""
        self.start()
        self._activated_event.clear()
        self._paused_event.clear()
        self._pause_event.clear()

    def pause(self) -> None:
        """Pause detection and wait until the microphone has been released."""
        if not self.is_running:
            return
        self._pause_event.set()
        if not self._paused_event.wait(
            self.settings.microphone_release_timeout
        ):
            raise WakeWordError(
                "Wake-word listener did not release the microphone."
            )
        self._raise_background_error()

    def wait_for_activation(self, timeout: float | None = None) -> bool:
        """Wait for detection and ensure the microphone is free on success."""
        self._raise_background_error()
        detected = self._activated_event.wait(timeout)
        if not detected:
            return False
        if not self._paused_event.wait(
            self.settings.microphone_release_timeout
        ):
            raise WakeWordError(
                "Wake word was detected, but the microphone is still busy."
            )
        self._raise_background_error()
        return True

    def stop(self) -> None:
        """Stop the process; safe to call repeatedly."""
        process = self._process
        if process is None:
            return
        self._stop_event.set()
        self._pause_event.clear()
        if process.is_alive():
            process.join(timeout=3.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        self._process = None

    def _raise_background_error(self) -> None:
        try:
            message = self._error_queue.get_nowait()
        except queue.Empty:
            return
        raise WakeWordError(f"OpenWakeWord failed: {message}")

    def _drain_errors(self) -> None:
        while True:
            try:
                self._error_queue.get_nowait()
            except queue.Empty:
                return
