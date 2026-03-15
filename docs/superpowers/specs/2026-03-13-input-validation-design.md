# Input Validation for Request Parameters (#12)

## Problem

Request parameters across all API surfaces lack validation. Invalid values (negative max_tokens, temperature > 2, path traversal in model names) are silently accepted, leading to undefined behavior or confusing errors deep in the inference engine.

## Design

### 1. Numeric Parameter Validation (Pydantic `Field`)

All constraints use Pydantic `Field` with `ge`/`le`/`gt`/`lt` bounds. Invalid values return HTTP 422 (Pydantic's default).

#### OpenAI Schemas (`schemas/openai.py`)

Applied to both `OpenAIChatRequest` and `OpenAICompletionRequest`:

| Parameter | Constraint | Notes |
|-----------|-----------|-------|
| `temperature` | `Field(None, ge=0, le=2)` | OpenAI range |
| `top_p` | `Field(None, ge=0, le=1)` | |
| `max_tokens` | `Field(None, ge=1)` | |
| `max_completion_tokens` | `Field(None, ge=1)` | Chat only |
| `n` | `Field(1, ge=1, le=1)` | Reject n>1 since we only return 1 completion |
| `presence_penalty` | `Field(0.0, ge=-2, le=2)` | OpenAI range |
| `frequency_penalty` | `Field(0.0, ge=-2, le=2)` | OpenAI range |

#### Anthropic Schema (`schemas/anthropic.py`)

Applied to `AnthropicMessagesRequest`:

| Parameter | Constraint | Notes |
|-----------|-----------|-------|
| `temperature` | `Field(None, ge=0, le=1)` | Anthropic range is 0-1 |
| `top_p` | `Field(None, ge=0, le=1)` | |
| `top_k` | `Field(None, ge=1)` | |
| `max_tokens` | `Field(4096, ge=1)` | |

#### Ollama ModelOptions (`schemas/common.py`)

| Parameter | Constraint | Notes |
|-----------|-----------|-------|
| `temperature` | `Field(None, ge=0)` | No upper bound (Ollama compat) |
| `top_p` | `Field(None, ge=0, le=1)` | |
| `top_k` | `Field(None, ge=1)` | |
| `min_p` | `Field(None, ge=0, le=1)` | |
| `repeat_last_n` | `Field(None, ge=-1)` | -1 = whole context |
| `num_predict` | `Field(None, ge=-1)` | -1 = infinite |
| `num_ctx` | `Field(None, ge=1)` | |

#### Config (`config.py`)

| Parameter | Constraint | Notes |
|-----------|-----------|-------|
| `port` | `Field(11434, ge=1, le=65535)` | Valid port range |

### 2. Model Name Validation (Registry Level)

Add `validate_model_name(name: str) -> None` in `engine/registry.py`:

- Rejects empty / whitespace-only strings
- Rejects strings over 256 characters
- Rejects names containing `..` (path traversal)
- Raises `ValueError` (mapped to HTTP 400 by global error handler)
- Called at model resolution time (`resolve()` and `add_mapping()`)

### 3. Testing Strategy

Each sub-task includes tests:

- Schema validation: test that valid values pass, boundary values pass, and out-of-range values raise `ValidationError`
- Model name validation: test empty, too long, path traversal, and valid names (including Ollama-style `qwen3:8b` and HF-style `Qwen/Qwen3-8B`)
- Config port: test valid and invalid port values

### 4. Parallelization

Five independent work streams, each touching different files:

1. **OpenAI schemas** — `schemas/openai.py` + tests
2. **Anthropic schema** — `schemas/anthropic.py` + tests
3. **Ollama ModelOptions** — `schemas/common.py` + tests
4. **Config port** — `config.py` + tests
5. **Model name validation** — `engine/registry.py` + tests
