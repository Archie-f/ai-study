# AI Study Project — Claude Instructions

## Who I am working with
- **Name:** Akif (Mehmet)
- **Background:** Senior Software Test Engineer, 4+ years Java experience
- **Goal:** Transition to AI Engineer role
- **Study time:** 12–18 hours/week
- **Location:** Norway
- **IDE:** PyCharm on MacBook (Apple Silicon)
- **Python:** 3.14 via Homebrew (`/opt/homebrew/bin/python3`)
- **Local AI:** Ollama with llama3 model

## Project structure
```
ai-study/
├── CLAUDE.md                          ← you are here
├── notes/                             ← study notes folder (also at /Users/archie/PycharmProjects/PythonProject/PythonProject/ai-study/notes)
│   ├── AI-Study-Comprehensive-Notes.docx   ← master notes, updated after each day
│   └── day-XX-takeaway-notes.docx     ← per-day takeaway notes
├── week-00/                           ← pre-roadmap work (days 1–5)
├── week-01/                           ← Week 01: Modern Python for AI
│   └── ai_chatbot/                    ← Week 01 project root
│       ├── pyproject.toml
│       ├── .venv/
│       └── src/ai_chatbot/
│           ├── __init__.py
│           ├── chat.py                ← main chatbot logic (from day-005)
│           └── cli.py                 ← (Day 04, not yet created)
└── .gitignore
```

## GitHub
- Repo: https://github.com/Archie-f/ai-study
- Branching: create `week-XX` branch at start of each week, merge to `main` at end
- Current branch: `week-01`
- Remind Akif to create the week branch if not done yet

## Study roadmap
- **Phase 1 — Python for AI (Weeks 1–4):** current phase
- **Phase 2 — LLM Fundamentals & Local AI (Weeks 5–8)**
- **Phase 3 — RAG & Vector Databases (Weeks 9–14)**
- **Phase 4 — Building AI Applications (Weeks 15–20)**
- **Phase 5 — Job-Ready Skills (Weeks 21–24)**

## Week 01 plan (5 days)
| Day | Topic | Key deliverable |
|-----|-------|----------------|
| Day 1 | Type Hints & Dataclasses | Type-hinted chatbot + ChatMessage/ChatConfig dataclasses |
| Day 2 | Virtual Environments & Packaging | pip install -e . working, src layout |
| Day 3 | Functions, Context Managers & async/await | async send_request using httpx |
| Day 4 | CLI with Typer + pathlib | Working CLI tool with --flags |
| Day 5 | Refactor & Consolidate | Packaged CLI tool pushed to GitHub |

## Notes files
- **Master notes:** `notes/AI-Study-Comprehensive-Notes.docx` — add completed day section at END of each day
- **Per-day takeaway notes:** `notes/day-XX-takeaway-notes.docx` — prepare BEFORE the day, update AFTER
- Day 01 takeaway notes: done ✓ (at `/Users/archie/Documents/Claude/Projects/AI Study/day-01-takeaway-notes.docx`)
- Day 02 takeaway notes: done ✓ (at `notes/day-02-takeaway-notes.docx`)

## Start-of-day checklist (do this at the start of every study session)
1. Read the current day's entry in `week-01-study-plan.docx` (at `/Users/archie/Documents/Claude/Projects/AI Study/`)
2. Check if the `week-XX` branch exists on GitHub — if not, remind Akif to create it
3. Read the existing `day-XX-takeaway-notes.docx` if it exists
4. Read `AI-Study-Comprehensive-Notes.docx` WHERE I LEFT OFF block to know current state

## End-of-day checklist (do this at the end of every study session)
1. Update `day-XX-takeaway-notes.docx` — add real Q&A from the session, fix anything that was wrong
2. Update `AI-Study-Comprehensive-Notes.docx` — add the completed day's section to Section 6 (Daily Study Log)
3. Update the WHERE I LEFT OFF block in `AI-Study-Comprehensive-Notes.docx`
4. Remind Akif to: `git add . && git commit -m "day-XX: description" && git push`

## Java analogies that work well for Akif
- `pyproject.toml` → `pom.xml`
- `venv` → project-scoped Maven local repository
- `__init__.py` → `package-info.java`
- `dataclass` → Java record / POJO
- `pip install -e .` → `mvn install` with symlink
- `src/` layout → `src/main/java/com/yourcompany/`
- `Optional[str]` / `str | None` → `@Nullable String`

## Important preferences
- Always use Java analogies when explaining new Python concepts
- Do NOT add sections to AI-Study-Comprehensive-Notes.docx mid-session — only at end of day
- Prepare takeaway notes BEFORE the session, update them AFTER
- Keep explanations practical and code-focused — Akif learns by doing
