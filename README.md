# Javis

Phase 2 provides a persistent, streaming terminal conversation with Google
Gemini and a dedicated JARVIS personality. The conversation remains available
for the duration of the running program. It deliberately does not include
voice input or output, camera access, a database, or a GUI.

## Requirements

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

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
```

The `.env` file is ignored by Git. `.env.example` documents the available
settings without containing a secret.

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
| `exit` | Close JARVIS |

The slash forms such as `/history` also work. `clear` resets both the displayed
history and the context sent to Gemini. History is intentionally not stored
after the program exits because database persistence belongs to a later phase.

JARVIS's behavior is defined in `prompts/jarvis.txt`.

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

## Tests

The test suite does not make real API requests:

```powershell
python -m unittest discover -s tests -v
```
