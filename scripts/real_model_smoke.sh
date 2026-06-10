#!/usr/bin/env bash
# Real-model smoke runner (#470).
#
# Runs each test file in its own pytest process so a SIGBUS at Metal
# teardown (exit 138 = 128 + SIGBUS, seen after an all-green run) kills
# only that file's process instead of poisoning the whole batch. A file
# counts as passed when pytest exits 0, or when it exits 138 *after*
# printing an all-green summary (the known teardown crash). Anything
# else fails the run.
#
# Usage: scripts/real_model_smoke.sh [test-file ...]
# With no arguments, runs the curated default subset below.
set -u

FILES=("$@")
DEFAULT_RUN=0
if [ ${#FILES[@]} -eq 0 ]; then
  DEFAULT_RUN=1
  FILES=(
    # Text model end-to-end (tiny: Qwen2.5-0.5B-4bit).
    tests/integration/test_real_model.py
    # OpenAI Responses API over a real model (Qwen3-4B-4bit).
    tests/live/test_responses_sdk.py
    # VLM + prompt cache + grammar (gemma-4-26B-A4B-4bit).
    tests/live/test_vlm_cache_grammar.py
    # Speculative: real MTP draft-head download + strict weight load.
    tests/test_mtp_loader.py
  )
fi

# Models the curated default subset's live tests skipif when absent from the
# olmlx store (~/.olmlx/models). Without them present those files skip every
# test, which the summary below (correctly) treats as a failed smoke run, not a
# green one. Pre-pull them so the run exercises real coverage. Pull reuses the
# HF blob cache, so this is cheap once the runner has downloaded them once
# (test_real_model.py and test_mtp_loader.py self-provision and are omitted).
# Only for the default subset — explicit file args manage their own models.
DEFAULT_MODELS=(
  "mlx-community/Qwen3-4B-4bit"
  "mlx-community/gemma-4-26B-A4B-it-4bit"
)
if [ "$DEFAULT_RUN" -eq 1 ]; then
  for m in "${DEFAULT_MODELS[@]}"; do
    echo "::group::pull ${m}"
    uv run olmlx models pull "$m" || echo "warning: pull failed for ${m} (its test will skip)"
    echo "::endgroup::"
  done
fi

overall=0
results=()
for f in "${FILES[@]}"; do
  echo "::group::${f}"
  log=$(mktemp)
  uv run pytest "$f" -m real_model -q 2>&1 | tee "$log"
  code=${PIPESTATUS[0]}
  echo "::endgroup::"
  if [ "$code" -eq 0 ] && grep -qE '[0-9]+ passed' "$log"; then
    results+=("PASS ${f}")
  elif [ "$code" -eq 0 ]; then
    # Exit 0 but nothing actually ran (everything skipped — e.g. the
    # model wasn't downloadable). Silent zero coverage must not read
    # as a green smoke run.
    results+=("FAIL (exit 0 but no tests ran — all skipped?) ${f}")
    overall=1
  elif [ "$code" -eq 138 ] \
      && grep -qE '[0-9]+ passed' "$log" \
      && ! grep -qE '[0-9]+ (failed|error)' "$log"; then
    # Tests were green; the process died afterwards in Metal teardown.
    results+=("PASS (teardown SIGBUS, exit 138) ${f}")
  else
    results+=("FAIL (exit ${code}) ${f}")
    overall=1
  fi
  rm -f "$log"
done

echo
echo "=== real-model smoke summary ==="
printf '%s\n' "${results[@]}"
exit "$overall"
