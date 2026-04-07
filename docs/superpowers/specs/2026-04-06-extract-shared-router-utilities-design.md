# Extract Shared Utilities from Routers

**Issue:** #176
**Date:** 2026-04-06

## Summary

Consolidate two categories of duplicated code across routers and schemas:

1. **`format_error`** — identical NDJSON error formatting in `routers/chat.py` and `routers/generate.py`
2. **`validate_token_limit`** — same settings check duplicated across `schemas/common.py`, `schemas/anthropic.py`, and `schemas/openai.py`

**Not in scope:** `_build_options` stays per-router — the Anthropic and OpenAI versions share only `temperature`/`top_p` and differ in all other fields and field name mappings.

## Design

### 1. `routers/common.py` — `format_error`

New file with a single function:

```python
import json
from datetime import datetime, timezone

def format_error(model: str) -> str:
    return json.dumps({
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "error": "An internal server error occurred during streaming.",
        "done": True,
        "done_reason": "error",
    }) + "\n"
```

**Callers:**
- `routers/chat.py`: replace the `format_error` closure with `format_error(req.model)` from `routers.common`
- `routers/generate.py`: same replacement

### 2. `schemas/common.py` — `validate_token_limit`

New helper function alongside existing `Options` model:

```python
def validate_token_limit(v: int, field_name: str) -> int:
    from olmlx.config import settings
    if v > settings.max_tokens_limit:
        raise ValueError(
            f"{field_name} {v} exceeds configured limit {settings.max_tokens_limit} "
            f"(set OLMLX_MAX_TOKENS_LIMIT to increase)"
        )
    return v
```

**Callers (each keeps its own null/special-value handling):**
- `schemas/common.py` `validate_num_predict`: early-return for `None`/negative, then `return validate_token_limit(v, "num_predict")`
- `schemas/anthropic.py` `validate_max_tokens`: `return validate_token_limit(v, "max_tokens")`
- `schemas/openai.py` `validate_max_tokens`: early-return for `None`, then `return validate_token_limit(v, "value")`

## Testing

- All existing router/schema tests must continue to pass
- Add tests for `format_error` (correct JSON structure, model name propagation)
- Add tests for `validate_token_limit` (passes under limit, raises over limit, error message format)
