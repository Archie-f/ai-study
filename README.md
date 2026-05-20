# AI Study — My AI Engineer Learning Journey

A hands-on learning project documenting my path from Software Test Engineer to AI Engineer.

## Structure

| Folder | What I built |
|--------|-------------|
| `day-02/` | First steps with Python and local LLMs via Ollama |
| `day-03/` | Stateful chatbot with conversation memory |
| `day-04/` | Upgraded chatbot with system prompts |
| `day-05-review/` | Refactored chatbot — structured messages API, sliding window memory, error handling |

## Stack

- Python 3.10
- [Ollama](https://ollama.com) — local LLM runner
- Llama3 — local model

## Running the chatbot

1. Install and start Ollama: `ollama serve`
2. Pull the model: `ollama pull llama3`
3. Run: `python3 day-05-review/day-05-review-and-refine.py`
