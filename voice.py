"""Microphone, Whisper speech recognition, and Piper TTS."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from config import VoiceSettings, get_voice_settings
from piper_tts import PiperTTS


class VoiceError(RuntimeError):
    pass


class VoiceTimeout(VoiceError):
    pass


def _create_whisper_model(settings: VoiceSettings) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(
        settings.model,
        device=settings.device,
        compute_type=settings.compute_type,
    )


def _load_sounddevice() -> Any:
    import sounddevice

    return sounddevice


class Voice:
    def __init__(
        self,
        settings: VoiceSettings | None = None,
        model_factory: Callable[[VoiceSettings], Any] = _create_whisper_model,
        tts_factory: Callable[[Path], PiperTTS] = PiperTTS,
        sounddevice_module: Any | None = None,
    ) -> None:
        self.settings = settings or get_voice_settings()
        self._model_factory = model_factory
        self._tts_factory = tts_factory
        self._sounddevice = sounddevice_module
        self._model: Any | None = None
        self._tts: PiperTTS | None = None
        self._speech_thread: threading.Thread | None = None
        self._speech_error: VoiceError | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

    @property
    def is_speaking(self) -> bool:
        thread = self._speech_thread
        return bool(thread and thread.is_alive())

    def listen(self) -> str:
        return self.transcribe(self._record_audio())

    def transcribe(self, audio: np.ndarray) -> str:
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            raise VoiceError("No audio was recorded.")

        try:
            if self._model is None:
                self._model = self._model_factory(self.settings)

            segments, _info = self._model.transcribe(
                audio.astype(np.float32, copy=False),
                language=self.settings.language,
                vad_filter=True,
            )

            text = " ".join(segment.text.strip() for segment in segments).strip()

        except Exception as exc:
            raise VoiceError(f"Whisper transcription failed: {exc}") from exc

        if not text:
            raise VoiceError("Whisper did not recognize any speech.")

        return text

    def speak(self, text: str, *, blocking: bool = True) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Speech text must not be empty.")

        self.interrupt(wait=True)
        if self.is_speaking:
            raise VoiceError("Previous speech output could not be stopped.")
        self._stop_event.clear()
        self._speech_error = None

        thread = threading.Thread(
            target=self._speak_worker,
            args=(text.strip(),),
            name="jarvis-piper-tts",
            daemon=True,
        )
        self._speech_thread = thread
        thread.start()

        if blocking:
            try:
                self.wait_until_done()
            except KeyboardInterrupt:
                self.interrupt(wait=True)
                raise

    def wait_until_done(self) -> None:
        thread = self._speech_thread

        if thread and thread is not threading.current_thread():
            thread.join()

        if self._speech_error:
            error = self._speech_error
            self._speech_error = None
            raise error

    def interrupt(self, *, wait: bool = False) -> None:
        self._stop_event.set()
        with self._lock:
            tts = self._tts
        if tts is not None:
            tts.stop()

        thread = self._speech_thread

        if (
            wait
            and thread
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=2.0)

    def _speak_worker(self, text: str) -> None:
        try:
            tts = self._tts_factory(self.settings.piper_model_path)
            with self._lock:
                self._tts = tts
            if self._stop_event.is_set():
                return
            tts.speak(text)
        except Exception as exc:
            self._speech_error = VoiceError(f"Speech output failed: {exc}")
        finally:
            with self._lock:
                self._tts = None

    def _record_audio(self) -> np.ndarray:
        block_size = max(
            1,
            int(self.settings.sample_rate * self.settings.block_duration),
        )

        silent_blocks_needed = max(
            1,
            int(self.settings.silence_duration / self.settings.block_duration),
        )

        pre_roll: deque[np.ndarray] = deque(maxlen=3)
        frames: list[np.ndarray] = []
        heard_speech = False
        silent_blocks = 0
        started_at = time.monotonic()

        try:
            sounddevice = self._sounddevice or _load_sounddevice()

            with sounddevice.InputStream(
                samplerate=self.settings.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
            ) as stream:
                while True:
                    data, _overflowed = stream.read(block_size)
                    block = np.asarray(data, dtype=np.float32).reshape(-1)

                    level = float(np.sqrt(np.mean(np.square(block))))
                    elapsed = time.monotonic() - started_at

                    if not heard_speech:
                        pre_roll.append(block.copy())

                        if level >= self.settings.silence_threshold:
                            heard_speech = True
                            frames.extend(pre_roll)
                            pre_roll.clear()

                        elif elapsed >= self.settings.start_timeout:
                            raise VoiceTimeout(
                                "No speech detected before the microphone timeout."
                            )

                        continue

                    frames.append(block.copy())

                    if level >= self.settings.silence_threshold:
                        silent_blocks = 0
                    else:
                        silent_blocks += 1

                        if silent_blocks >= silent_blocks_needed:
                            break

                    if elapsed >= self.settings.max_duration:
                        break

        except VoiceError:
            raise

        except Exception as exc:
            raise VoiceError(f"Microphone recording failed: {exc}") from exc

        if not frames:
            raise VoiceError("No audio was recorded.")

        return np.concatenate(frames).astype(np.float32, copy=False)
