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
* Python 3.12+, 
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

| Week    | Topic                                                                           | Status      |
|---------|---------------------------------------------------------------------------------|-------------|
| Week 01 | Python for AI, packaging, async CLI                                             | Done        |
| Week 02 | LLM APIs, Git workflow, CI                                                      | Done        |
| Week 03 | Chatbot refactor, logging, config, pytest                                       | Done        |
| Week 04 | Prompt Engineering Foundations — few-shot, chain-of-thought, evaluation mindset | In Progress |

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
- GitHub Actions CI — automated linting (`ruff`) and tests (`pytest`) on every push

### Week 03
- Rebuilt the chatbot from scratch, typed, documented, and with proper error handling
- Added structured logging with Python's logging module — levels, handlers, latency timing
- Introduced ChatConfig dataclass — single config object replacing all magic strings
- Added tests directory, wrote tests mocking the Ollama call with unittest.mock.patch
- Defined and used pytest fixtures to share setup across tests without breaking test isolation
- Wrote a professional, scannable README file with CI and Python version badges
- Added ChatConfig class docstring, __main__.py module docstring, completed setup_logging Raises section
- Updated python-version in ci.yml from 3.11 to 3.12 to support the type alias syntax
- Added run() and setup_logging() tests - Expanded coverage to 93%
- Used "builtins.input", "builtins.print", and @pytest.mark.parametrize
- Added "pytest-cov", "mypy", and "ruff" into project.optional-dependencies
- Learned the pre-merge review workflow — mypy, pytest, ruff, README, and CI must all be green before merging to main

### Week 04
- Learned systematic prompting, few-shot prompting and chain-of-thought (CoT)
- Built an evaluation mindset: "Every prompt is a hypothesis. An eval is the experiment that tests it."
- Used Python concepts: zip(), @dataclass, textwrap.dedent(), generator expressions with sum()