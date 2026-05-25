# AI Study — My AI Engineer Learning Journey

A hands-on learning project documenting my path from Software Test Engineer to AI Engineer.

## Structure

### week-00 — Pre-Roadmap (Days 1–5 + Week 01 Day 01)

| Folder | What I built |
|--------|-------------|
| `week-00/day-002/` | First steps with Python and local LLMs via Ollama |
| `week-00/day-003/` | Stateful chatbot with conversation memory |
| `week-00/day-004/` | Upgraded chatbot with system prompts |
| `week-00/day-005-review/` | Refactored chatbot — `/api/chat`, sliding window memory, error handling |
| `week-00/day-005-review/day-005-review-and-refine.py` | Week 01 Day 01: added type hints, `ChatMessage` & `ChatConfig` dataclasses, `build_messages()` |

### notes/

| File | Description |
|------|-------------|
| `notes/AI-Study-Comprehensive-Notes.docx` | Living study notes — updated after every session |

## Stack

- Python 3.14
- [Ollama](https://ollama.com) — local LLM runner
- Llama3 — local model

## Running the chatbot

1. Install and start Ollama: `ollama serve`
2. Pull the model: `ollama pull llama3`
3. Run: `python3 week-00/day-005-review/day-005-review-and-refine.py`
