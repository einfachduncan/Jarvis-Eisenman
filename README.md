# Javis

Phase 4 provides a persistent, streaming terminal conversation with Google
Gemini, a dedicated JARVIS personality, and optional voice input and output.
Microphone audio is transcribed locally with faster-whisper, and replies are
spoken through the operating system's text-to-speech engine. On Windows,
`pyttsx3` uses SAPI where available and automatically falls back to the local
`.NET System.Speech` engine. OpenWakeWord runs in a separate background process
and activates JARVIS when it hears “Hey Jarvis”. Camera access is deliberately
not included.

## Requirements

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- A working microphone and audio output for voice mode

## Installation

Open PowerShell in this folder and run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open `.env` and add your key:

```dotenv
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash
WHISPER_MODEL=base
WHISPER_LANGUAGE=
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WAKE_WORD_MODEL=hey_jarvis
WAKE_WORD_THRESHOLD=0.5
WAKE_WORD_MODEL_DIR=
```

The `.env` file is ignored by Git. `.env.example` documents the available
settings without containing a secret. The default `base` Whisper model runs on
the CPU using 8-bit computation. Its model files are downloaded the first time
the `voice` command is used. Leave `WHISPER_LANGUAGE` empty for automatic
language detection, or set a language code such as `de` or `en`.
The OpenWakeWord ONNX models are downloaded into the ignored `.models` folder
when sleep mode is first started. `WAKE_WORD_MODEL_DIR` can override that
location.

## Start

```powershell
python main.py
```

Enter a prompt and press Enter. Gemini's response is displayed as it arrives.
The following commands are available:

| Command | Action |
| --- | --- |
| `help` | Show all commands |
| `history` | Show the current conversation |
| `clear` | Start a fresh conversation |
| `voice` | Record one microphone prompt and speak the answer |
| `sleep` | Wait in the background for “Hey Jarvis” |
| `exit` | Close JARVIS |

The slash forms such as `/history` also work. `clear` resets both the displayed
history and the context sent to Gemini. History is intentionally not stored
after the program exits because database persistence belongs to a later phase.

JARVIS's behavior is defined in `prompts/jarvis.txt`.

## Voice mode

Enter `voice`, wait for `Listening...`, and speak normally. Recording ends
after one second of silence or after 30 seconds. The recognized text is shown,
sent through the same ongoing chat, and JARVIS reads the response aloud.

Press `Ctrl+C` while JARVIS is speaking to interrupt only the speech output;
the chat remains open.

Additional tuning is available through `VOICE_SAMPLE_RATE`,
`VOICE_SILENCE_THRESHOLD`, `VOICE_SILENCE_DURATION`, `VOICE_START_TIMEOUT`,
`VOICE_MAX_DURATION`, `VOICE_TTS_RATE`, and `VOICE_TTS_VOLUME`.

## Use from Python

```python
from gemini_client import ChatSession

chat = ChatSession()
print(chat.send("My name is Alex."))
print(chat.send("What is my name?"))
```

Use `chat.stream(prompt)` to iterate over response chunks. The original
`ask(prompt)` helper remains available for independent requests. Invalid input,
missing configuration, API failures, and empty model responses are reported
through `ValueError`, `ConfigurationError`, or `GeminiError`.

Voice functionality is also available separately:

```python
from voice import Voice

voice = Voice()
text = voice.listen()
voice.speak(f"I heard: {text}")
voice.interrupt()
```

`Voice.interrupt()` is safe to call repeatedly. Whisper and the TTS engine are
loaded lazily, only when they are first used.

## Wake word and sleep mode

Enter `sleep` to release the terminal and start the OpenWakeWord background
process. Say “Hey Jarvis”; after detection, the wake-word process closes its
microphone stream before the normal Whisper recorder starts. This prevents both
audio systems from opening the microphone at the same time.

JARVIS records one spoken request, answers it, and returns to normal terminal
mode. Enter `sleep` again when it should wait for another activation. Press
`Ctrl+C` while waiting to cancel sleep mode without closing the chat.

The default activation threshold is `0.5`. It can be tuned with
`WAKE_WORD_THRESHOLD`; higher values reduce false activations but can make the
phrase harder to detect.

## Tests

The test suite does not make real API requests:

```powershell
python -m unittest discover -s tests -v
```
