---
name: refactor
description: Safe refactoring — improve structure without changing behavior
---

When refactoring code, follow these principles:

## Before starting
- Confirm existing tests pass (or note that tests are missing)
- Understand the current behavior completely before changing structure
- Define the goal: what specific quality are you improving? (readability, performance, testability, removing duplication)

## During refactoring
- Make one type of change at a time — don't mix refactoring with feature work
- Keep each step small and independently verifiable
- Preserve all existing behavior — refactoring means same inputs, same outputs
- If you discover a bug, note it but don't fix it in the same change

## Common patterns
- **Extract function**: when a block of code has a clear single purpose
- **Inline**: when a function/variable adds indirection without clarity
- **Rename**: when a name doesn't reflect current purpose
- **Simplify conditionals**: flatten nested if/else, use early returns
- **Remove dead code**: delete it, don't comment it out — git remembers

## After refactoring
- Run the same tests — they should all still pass
- Verify the diff only contains structural changes, not behavior changes
