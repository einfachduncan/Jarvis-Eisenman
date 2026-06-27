"""Interruptible Piper text-to-speech with automatic voice download."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import requests
import sounddevice as sd
import soundfile as sf


VOICE_NAME = "de_DE-thorsten-high"
VOICE_BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
    "de/de_DE/thorsten/high"
)


class PiperTTS:
    def __init__(
        self,
        model_path: str | Path,
        *,
        http_client: Any = requests,
        process_factory: Any = subprocess.Popen,
        sounddevice_module: Any = sd,
        soundfile_module: Any = sf,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.config_path = Path(f"{self.model_path}.json")
        self._http = http_client
        self._process_factory = process_factory
        self._sounddevice = sounddevice_module
        self._soundfile = soundfile_module
        self._process: subprocess.Popen[str] | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        if not text.strip():
            return

        self._stop_event.clear()
        self._ensure_voice_files()
        if self._stop_event.is_set():
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        try:
            process = self._process_factory(
                [
                    sys.executable,
                    "-m",
                    "piper",
                    "--model",
                    str(self.model_path),
                    "-f",
                    str(wav_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            with self._lock:
                self._process = process
                stopped = self._stop_event.is_set()
            if stopped:
                process.terminate()

            _stdout, stderr = process.communicate(text)
            with self._lock:
                self._process = None

            if self._stop_event.is_set():
                return
            if process.returncode:
                raise RuntimeError(
                    (stderr or "").strip()
                    or f"Piper exited with code {process.returncode}."
                )

            audio, sample_rate = self._soundfile.read(wav_path, dtype="float32")
            if self._stop_event.is_set():
                return
            self._sounddevice.play(audio, sample_rate)
            self._sounddevice.wait()
        finally:
            wav_path.unlink(missing_ok=True)

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        try:
            self._sounddevice.stop()
        except Exception:
            pass

    def _ensure_voice_files(self) -> None:
        files = (
            (self.model_path, f"{VOICE_BASE_URL}/{VOICE_NAME}.onnx"),
            (self.config_path, f"{VOICE_BASE_URL}/{VOICE_NAME}.onnx.json"),
        )
        for destination, url in files:
            if self._stop_event.is_set():
                return
            if destination.is_file() and destination.stat().st_size > 0:
                continue
            self._download(url, destination)

    def _download(self, url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = Path(f"{destination}.part")
        try:
            with self._http.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with temporary_path.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if self._stop_event.is_set():
                            return
                        if chunk:
                            output.write(chunk)
            if temporary_path.stat().st_size == 0:
                raise RuntimeError(f"Downloaded Piper file is empty: {url}")
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)
