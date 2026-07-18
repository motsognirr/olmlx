<!--
One `Closes` line per issue so GitHub auto-closes each on merge.
Bundled shorthand (`Closes #1, #2`) only closes the FIRST — repeat the keyword.
Bare references like `(#123)` or `## #123` do NOT close anything.
-->

Closes #

## Summary

<!-- What this changes and why. One short section per issue for multi-issue PRs. -->

## Tests

<!-- New/updated tests, and the suites you ran. TDD: confirm red before green. -->

---

- [ ] `ruff check` + `ruff format` clean
- [ ] Tests pass (`uv run pytest`)
- [ ] Each fixed issue has its own `Closes #N` line above
