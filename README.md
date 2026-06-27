# Javis

Phase 3 provides a persistent, streaming terminal conversation with Google
Gemini, a dedicated JARVIS personality, and optional voice input and output.
Microphone audio is transcribed locally with faster-whisper, and replies are
spoken through the operating system's text-to-speech engine. On Windows,
`pyttsx3` uses SAPI where available and automatically falls back to the local
`.NET System.Speech` engine. It deliberately does not include a wake word or
camera access.

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
```

The `.env` file is ignored by Git. `.env.example` documents the available
settings without containing a secret. The default `base` Whisper model runs on
the CPU using 8-bit computation. Its model files are downloaded the first time
the `voice` command is used. Leave `WHISPER_LANGUAGE` empty for automatic
language detection, or set a language code such as `de` or `en`.

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
the chat remains open. There is intentionally no wake-word listener in this
phase.

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

## Tests

The test suite does not make real API requests:

```powershell
python -m unittest discover -s tests -v
```
