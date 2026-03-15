---
name: commit-message
description: Write clear, conventional commit messages from a diff or description
---

Write a git commit message following these conventions:

## Format
```
<type>: <short summary in imperative mood>

<optional body — explain WHY, not WHAT>
```

## Types
- **fix**: Bug fix
- **feat**: New feature
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **docs**: Documentation only
- **test**: Adding or updating tests
- **chore**: Build, CI, dependencies, tooling

## Rules
- Subject line: imperative mood ("Add" not "Added"), max 72 chars, no period
- Body: wrap at 72 chars, explain motivation and contrast with previous behavior
- If fixing a bug, mention what was broken and why
- If adding a feature, mention what it enables
- Reference issue numbers when applicable (e.g., "Fixes #42")

## Examples
Good: `fix: prevent duplicate model loads on concurrent requests`
Bad: `fixed bug`, `update code`, `changes`
