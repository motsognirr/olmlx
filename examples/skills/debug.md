---
name: debug
description: Systematic debugging — reproduce, isolate, fix, verify
---

Follow this systematic debugging process:

## 1. Reproduce
- Clarify the exact steps to trigger the bug
- Identify expected vs actual behavior
- Note the environment (OS, versions, config)

## 2. Isolate
- Form a hypothesis about the root cause
- Narrow down: which component, function, or line?
- Use binary search: comment out halves of suspect code to locate the fault
- Check recent changes — what was the last known working state?

## 3. Understand
- Before writing a fix, explain *why* the bug happens
- Trace the data flow from input to the point of failure
- Check if the same pattern exists elsewhere (same bug, different location)

## 4. Fix
- Make the smallest change that fixes the root cause
- Don't patch symptoms — fix the underlying problem
- If a quick fix is needed, mark it clearly and explain the proper fix

## 5. Verify
- Confirm the fix resolves the original reproduction case
- Check that it doesn't break existing behavior
- Suggest a test that would catch this regression
