# Ollama Chatbot

![CI](https://github.com/Archie-f/ai-study/actions/workflows/ci.yml/badge.svg?branch=week-03)
![Python](https://img.shields.io/badge/python-3.12-blue)

A command-line chatbot that talks to a local Ollama model.  
Built as part of a structured AI Engineer learning path.

## Features
- Multi-turn conversation with history
- Configurable model, host, and system prompt via `ChatConfig`
- Structured logging (console + optional file)
- Automated test suite with `pytest` and `unittest.mock`

## Requirements
- Python 3.14+
- [Ollama](https://ollama.com) running locally with a model pulled
  (default: `llama3`)

## Installation
```bash
# Clone and enter the week-03 folder
git clone https://github.com/Archie-f/ai-study.git
cd ai-study/week-03

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package and dev dependencies
pip install -e '.[dev]'
```

## Usage
```bash
python -m chatbot
```
You: Hello  
Assistant: Hi! How can I help you today?  
---  
You: quit  
Goodbye!

Exit at any time by typing `quit`, `exit`, `bye`, or `q`.

## Running the Tests
```bash
python -m pytest -v
```
All tests mock the Ollama call — no running server needed.

## Project Structure
```text
week-03/
├── chatbot/
│   ├── __init__.py
│   ├── __main__.py
│   ├── chatbot.py
│   └── config.py
└── tests/
    ├── __init__.py
    └── test_chatbot.py
```