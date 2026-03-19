# Contributing to olmlx

Thank you for your interest in contributing! This document outlines how to get started.

## Development Setup

```bash
git clone git@github.com:motsognirr/olmlx.git && cd olmlx
uv sync --no-editable
uv run olmlx  # start server
uv run pytest # run tests
```

## Development Workflow

### TDD

This project uses test-driven development:
1. Write a failing test that describes the desired behavior
2. Implement the minimum code to make the test pass
3. Run `uv run ruff check --fix && uv run ruff format` before committing

### Code Style

- Python 3.11+
- Formatted with `ruff format`
- Linted with `ruff check`
- Type hints where practical

### Before Creating a PR

1. Ensure all tests pass: `uv run pytest`
2. Run formatters: `uv run ruff check --fix && uv run ruff format`
3. Follow the existing code style and patterns
4. Update documentation if adding new features

## Project Structure

See [CLAUDE.md](CLAUDE.md) for detailed project structure and key design decisions.

## Git

- Remote: `git@github.com:motsognirr/olmlx.git`
- Use conventional commit messages

