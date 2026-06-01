# AI Study — My AI Engineer Learning Journey

A hands-on learning project documenting my path from Software Test Engineer to AI Engineer.

## Structure

### week-01 — Modern Python for AI (Days 1–5)

A fully packaged, installable CLI chatbot built with modern Python practices.

| Day | Topic | Key deliverable |
|-----|-------|----------------|
| Day 01 | Type hints & dataclasses | `ChatMessage`, `ChatConfig` dataclasses, `build_messages()` |
| Day 02 | Virtual environments & packaging | `src` layout, `pyproject.toml`, `pip install -e .` |
| Day 03 | async/await & httpx | `send_request_to_model()` rewritten as async |
| Day 04 | CLI with Typer | `ai-chat-cli` command with `--model`, `--system-prompt`, `--max-history` flags |
| Day 05 | Refactor & consolidate | Full refactor into `chatbot.py` — dataclasses, type hints, clean separation of concerns |

**Project:** `week-01/ai_chatbot/`

```
src/ai_chatbot/
├── chatbot.py   ← Week 01 final: dataclasses + type hints + async CLI
├── cli.py       ← Day 04: Typer CLI
└── chat.py      ← Day 01–03: incremental work
```

### week-00 — Pre-Roadmap (Days 1–5)

| Folder | What I built |
|--------|-------------|
| `week-00/day-002/` | First steps with Python and local LLMs via Ollama |
| `week-00/day-003/` | Stateful chatbot with conversation memory |
| `week-00/day-004/` | Upgraded chatbot with system prompts |
| `week-00/day-005-review/` | Refactored chatbot — `/api/chat`, sliding window memory, error handling |

### notes/

| File | Description |
|------|-------------|
| `notes/AI-Study-Comprehensive-Notes.docx` | Living study notes — updated after every session |

## Stack

- Python 3.14
- [Ollama](https://ollama.com) — local LLM runner
- Llama3 — local model
- [httpx](https://www.python-httpx.org) — async HTTP client
- [Typer](https://typer.tiangolo.com) — CLI framework

## Installation & Running

```bash
# Clone the repo
git clone https://github.com/Archie-f/ai-study
cd ai-study/week-01/ai_chatbot

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install -e .

# Make sure Ollama is running with llama3
ollama serve
ollama pull llama3

# Run
ai-chat
ai-chat --model llama3 --max-history 6
ai-chat --system-prompt "You are a Linux expert."
```

## What I learned — Week 01

- Type hints are documentation that tools enforce — they catch bugs before runtime
- Dataclasses replace raw dicts with structured, editor-friendly objects
- `src` layout and `pyproject.toml` are the modern Python packaging standard
- `async/await` lets the program do other work while waiting for a slow LLM response
- Typer turns type-annotated function parameters directly into CLI flags
