#!/bin/bash
# Auto-restart loop for extended-bench. The olmlx serve subprocess sometimes
# enters a 500-cascade state after finishing a model. When that happens,
# the orchestrator burns through the rest of the queue with 500s and exits.
# This wrapper detects un-graded models and re-launches with a fresh server.
#
# Skip-existing handles state — already-graded models are not re-attempted.
# Stops once all 23 models have a raw/<safe-name>.json (success or failure).

set -u
OUTPUT_DIR="docs/benchmarks/extended-2026-05"
LOG="extended-bench-run.log"
ATTEMPTS=0
MAX_ATTEMPTS=20

while true; do
    ATTEMPTS=$((ATTEMPTS + 1))
    if [ "$ATTEMPTS" -gt "$MAX_ATTEMPTS" ]; then
        echo "$(date) wrapper: max attempts ($MAX_ATTEMPTS) reached, giving up" | tee -a "$LOG"
        exit 1
    fi
    echo "$(date) wrapper: starting bench attempt $ATTEMPTS" >> "$LOG"
    uv run python scripts/run_extended_bench.py \
        --models-config ~/.olmlx/models.json \
        --spawn-server \
        --enable-code-exec \
        --budget-hours 30 \
        --output "$OUTPUT_DIR/" >> "$LOG" 2>&1

    REMAINING=$(uv run python -c "
import json, glob, re
m = json.load(open('/Users/daniel/.olmlx/models.json'))
def sanitize(s): return re.sub(r'[^a-zA-Z0-9_.-]', '_', s)
done = {p.split('/')[-1].replace('.json','') for p in glob.glob('$OUTPUT_DIR/raw/*.json')}
remaining = [k for k in m if sanitize(k) not in done]
print(len(remaining))
")
    echo "$(date) wrapper: $REMAINING models remaining un-graded" >> "$LOG"
    if [ "$REMAINING" -le 0 ]; then
        echo "$(date) wrapper: all models processed, exiting cleanly" >> "$LOG"
        break
    fi
    # Ensure the server is really down before restarting (pkill in case it lingers).
    pkill -9 -f "olmlx serve" 2>/dev/null
    sleep 5
done
