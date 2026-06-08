---
name: github-manager
description: Manage GitHub repositories, issues, and pull requests using the gh CLI
---

When tasked with managing a GitHub repository, use the `gh` command line tool to perform actions efficiently. Follow these guidelines for common tasks:

## 1. Information Gathering
- **List Issues**: Use `gh issue list` to see open issues. For specific filters (e.g., labels), use `--label "bug"`.
- **Issue Details**: Use `gh issue view <number>` to get the full context of a specific report.
- **PR Status**: Use `gh pr list` and `gh pr view <number>` to track pull requests and their discussions.

## 2. Actionable Tasks
- **Creating Issues**: When reporting a bug or requesting a feature, ensure the title is concise and the body contains:
    - Clear description of the problem/request.
    - Steps to reproduce (for bugs).
    - Expected vs. actual behavior.
- **Managing PRs**: Use `gh pr create` with descriptive titles and linked issues (e.g., "Closes #123").
- **Checking Out Code**: Use `gh pr checkout <number>` to quickly test a contributor's changes locally.

## 3. Best Practices
- Always check for existing issues before creating a new one to avoid duplicates.
- Keep issue descriptions structured and objective.
- When closing an issue, provide a brief explanation of why it was resolved or why it is being closed without a fix.

## Output Format
When summarizing GitHub activity:
- **Action taken**: (e.g., "Queried open issues")
- **Findings**: List the relevant issues/PRs with their IDs and current status.
- **Next Steps**: Recommend the most logical next action based on the findings.
