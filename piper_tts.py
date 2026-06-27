from pathlib import Path
import subprocess
import tempfile

import sounddevice as sd
import soundfile as sf


class PiperTTS:
    def __init__(self, model_path: str = "models/piper/de_DE-thorsten-high.onnx"):
        self.model_path = Path(model_path)

    def speak(self, text: str) -> None:
        if not text.strip():
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Piper model not found: {self.model_path}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        try:
            subprocess.run(
                [
                    "python",
                    "-m",
                    "piper",
                    "--model",
                    str(self.model_path),
                    "--output_file",
                    str(wav_path),
                ],
                input=text,
                text=True,
                check=True,
            )

            audio, sample_rate = sf.read(wav_path)
            sd.play(audio, sample_rate)
            sd.wait()

        finally:
            wav_path.unlink(missing_ok=True)


if __name__ == "__main__":
    PiperTTS().speak("Hallo, wie geht es dir")