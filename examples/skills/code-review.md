---
name: code-review
description: Structured code review focusing on correctness, clarity, and maintainability
---

When reviewing code, follow this structured approach:

## 1. Correctness
- Does the code do what it claims to do?
- Are edge cases handled (empty inputs, None values, boundary conditions)?
- Are error paths correct and tested?

## 2. Clarity
- Can you understand the intent without reading comments?
- Are names descriptive and consistent?
- Is the control flow easy to follow?

## 3. Maintainability
- Is there unnecessary complexity or premature abstraction?
- Are there magic numbers or hardcoded values that should be constants?
- Would a future developer understand why this code exists?

## 4. Security
- Is user input validated and sanitized?
- Are there injection risks (SQL, command, XSS)?
- Are secrets or credentials exposed?

## Output format
For each issue found, state:
- **File and line**: where the issue is
- **Severity**: critical / warning / nit
- **Issue**: what's wrong
- **Suggestion**: how to fix it

End with a brief summary of overall code quality.
