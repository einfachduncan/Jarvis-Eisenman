"""Microphone, Whisper speech recognition, and interruptible TTS."""

from __future__ import annotations

import threading
import time
import subprocess
import sys
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from config import VoiceSettings, get_voice_settings


class VoiceError(RuntimeError):
    """Raised when recording, transcription, or speech output fails."""


class VoiceTimeout(VoiceError):
    """Raised when no speech starts before the configured timeout."""


def _create_whisper_model(settings: VoiceSettings) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(
        settings.model,
        device=settings.device,
        compute_type=settings.compute_type,
    )


def _create_tts_engine() -> Any:
    import pyttsx3

    try:
        return pyttsx3.init()
    except Exception:
        if sys.platform == "win32":
            return PowerShellTTSEngine()
        raise


def _load_sounddevice() -> Any:
    import sounddevice

    return sounddevice


class PowerShellTTSEngine:
    """Minimal pyttsx3-compatible fallback using Windows System.Speech."""

    def __init__(self) -> None:
        self._rate = 185
        self._volume = 1.0
        self._text = ""
        self._process: subprocess.Popen[str] | None = None
        self._stopped = False
        self._lock = threading.Lock()

    def setProperty(self, name: str, value: int | float) -> None:
        if name == "rate":
            self._rate = int(value)
        elif name == "volume":
            self._volume = float(value)

    def say(self, text: str) -> None:
        self._text = text

    def runAndWait(self) -> None:
        speech_rate = max(-10, min(10, round((self._rate - 185) / 15)))
        volume = max(0, min(100, round(self._volume * 100)))
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "[Console]::InputEncoding = [Text.Encoding]::UTF8; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {speech_rate}; $s.Volume = {volume}; "
            "$text = [Console]::In.ReadToEnd(); "
            "$s.Speak($text); $s.Dispose()"
        )
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            creationflags=creation_flags,
        )
        with self._lock:
            self._process = process
            stopped = self._stopped
        if stopped:
            process.terminate()

        _stdout, stderr = process.communicate(self._text)
        with self._lock:
            self._process = None
            stopped = self._stopped
        if process.returncode and not stopped:
            message = (stderr or "").strip()
            raise RuntimeError(message or "Windows speech synthesis failed.")

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()


class Voice:
    """Voice input/output with lazy Whisper loading and stoppable speech."""

    def __init__(
        self,
        settings: VoiceSettings | None = None,
        model_factory: Callable[[VoiceSettings], Any] = _create_whisper_model,
        tts_factory: Callable[[], Any] = _create_tts_engine,
        sounddevice_module: Any | None = None,
    ) -> None:
        self.settings = settings or get_voice_settings()
        self._model_factory = model_factory
        self._tts_factory = tts_factory
        self._sounddevice = sounddevice_module
        self._model: Any | None = None
        self._engine: Any | None = None
        self._speech_thread: threading.Thread | None = None
        self._speech_error: VoiceError | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

    @property
    def is_speaking(self) -> bool:
        thread = self._speech_thread
        return bool(thread and thread.is_alive())

    def listen(self) -> str:
        """Record one utterance from the default microphone and transcribe it."""
        return self.transcribe(self._record_audio())

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe 16 kHz mono float audio with faster-whisper."""
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
        """Speak text, optionally in the background."""
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
            name="jarvis-tts",
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
        """Wait for current speech and surface any TTS failure."""
        thread = self._speech_thread
        if thread and thread is not threading.current_thread():
            thread.join()
        if self._speech_error:
            error = self._speech_error
            self._speech_error = None
            raise error

    def interrupt(self, *, wait: bool = False) -> None:
        """Stop current speech safely; repeated calls are harmless."""
        self._stop_event.set()
        with self._lock:
            engine = self._engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

        thread = self._speech_thread
        if (
            wait
            and thread
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=2.0)

    def _speak_worker(self, text: str) -> None:
        engine = None
        try:
            engine = self._tts_factory()
            with self._lock:
                self._engine = engine
            if self._stop_event.is_set():
                return
            engine.setProperty("rate", self.settings.tts_rate)
            engine.setProperty("volume", self.settings.tts_volume)
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            self._speech_error = VoiceError(f"Speech output failed: {exc}")
        finally:
            with self._lock:
                self._engine = None

    def _record_audio(self) -> np.ndarray:
        block_size = max(
            1, int(self.settings.sample_rate * self.settings.block_duration)
        )
        silent_blocks_needed = max(
            1,
            int(
                self.settings.silence_duration
                / self.settings.block_duration
            ),
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
