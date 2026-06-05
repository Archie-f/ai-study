# ai-study

![CI](https://github.com/Archie-f/ai-study/actions/workflows/ci.yml/badge.svg)

## What
A week-by-week log of my AI Engineer transition: scripts, notes, and
experiments covering Python for AI, LLM APIs, RAG, agents, and deployment.

## Why
Moving from Senior Test Automation Engineer to AI Engineer. Building in public so the
learning is visible, accountable, and eventually teachable.

## How

**Requirements:** 
* Python 3.11+, 
* an Anthropic API key, 
* a Groq API key, 
* Ollama (optional)

```bash
git clone https://github.com/Archie-f/ai-study.git
cd ai-study
python -m venv .venv && source .venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
cp .env.example .env   # add your API keys
python week-02/hello_llm.py
```

## Weekly Progress

| Week | Topic | Status |
|------|-------|--------|
| Week 01 | Python for AI, packaging, async CLI | Done   |
| Week 02 | LLM APIs, Git workflow, CI | Done   |

## What I Learned
### Week 01
- Type hints and dataclasses — adding Java-style structure to Python without the boilerplate 
- Packaging a CLI tool with pyproject.toml — installable with pip install -e .
- async/await with httpx — non-blocking HTTP calls to a local LLM
- Building a typed CLI with Typer — flags, arguments, and --help out of the box
- Refactoring a working chatbot into a clean, installable package

### Week 02
- Managing API secrets with python-dotenv — never hardcode keys
- Building a unified ask() function that works across Claude, Groq, and Ollama
- The messages[] format shared by all modern LLM APIs
- Git rebase and interactive history cleanup
- GitHub Actions CI — automated linting (``` ruff ```) and tests (``` pytest ```) on every push

