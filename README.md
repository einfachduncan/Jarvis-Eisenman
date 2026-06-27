# Javis

Phase 1 provides a small terminal chat that sends independent text prompts to
Google Gemini. It deliberately does not include voice input or output, camera
access, a GUI, conversation memory, or PC control.

## Requirements

- Python 3.9 or newer
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

Enter a prompt and press Enter. Use `/exit` to close the program. Each prompt
is a separate request; storing conversation history belongs to a later phase.

## Use from Python

```python
from gemini_client import ask

answer = ask("Explain recursion in one sentence.")
print(answer)
```

`ask(prompt)` raises `ValueError` for an empty prompt, `ConfigurationError`
when the API key is missing, and `GeminiError` if the API request fails or
returns no text.

## Tests

The test suite does not make real API requests:

```powershell
python -m unittest discover -s tests -v
```
