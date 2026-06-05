# Git Workflow

Personal Git conventions for the `ai-study` repository.

---

## Branch Naming

| Pattern | When to use |
|---------|-------------|
| `week-XX` | Weekly study branch (e.g. `week-02`) |
| `feat/short-description` | New feature or script |
| `fix/short-description` | Bug fix |

Always branch off `main`. Merge back to `main` at end of week.

---

## Commit Message Format

This repo follows [Conventional Commits](https://www.conventionalcommits.org/) with a week/day scope:
\<type\>(week-XX/day-YY): \<short imperative description\>

**Types:**

| Type | When to use |
|------|-------------|
| `feat` | New feature or file |
| `fix` | Bug fix |
| `refactor` | Code change with no behaviour change |
| `docs` | Documentation only |
| `test` | Adding or fixing tests |
| `chore` | Config, dependencies, tooling |

**Rules:**
- Imperative mood: "add file" not "added file"
- No period at the end
- Under 72 characters

**Examples:**
```
feat(week-02/day-01): add hello_claude.py with dotenv setup
fix(week-02/day-02): correct stop_reason check in ask()
docs(week-02/day-02): add git-workflow.md
refactor(week-02/day-03): extract ask() into separate module
```
---

## Rebasing

### Staying up to date with main
```bash
git fetch origin
git rebase origin/main
```

### Cleaning up commits before merge (interactive rebase)
```bash
git rebase -i HEAD~N   # N = number of commits to review
```

Use inside your own branch only — never rebase commits already on `main`.

**Commands used in the editor:**

| Command | What it does |
|---------|-------------|
| `pick` | Keep commit as-is |
| `squash` | Merge into commit above, keep message |
| `fixup` | Merge into commit above, discard message |
| `reword` | Keep commit, edit message only |
| `drop` | Delete commit entirely |

After rebasing, push with:
```bash
git push --force-with-lease
```

---

## Pre-Commit Checklist

Run these before every commit:

**1. Check status — nothing unexpected staged:**
```bash
git status
```
- `.env` must NOT appear in the output
- Confirm you are on the correct branch

**2. Review exactly what you are committing:**
```bash
git diff --staged
```
- Read every `+` and `-` line
- No debug prints, no hardcoded secrets, no unfinished code

**3. Commit with a proper message:**
```bash
git add <files>
git commit -m "type(week-XX/day-YY): description"
```

---

## Undo Cheat Sheet

| Situation | Command |
|-----------|---------|
| Undo last commit, keep changes | `git reset HEAD~1` |
| Unstage a file | `git restore --staged <file>` |
| Discard local changes to a file | `git restore <file>` |
| See what changed in a commit | `git show <commit-hash>` |