# Input Validation Implementation Plan (#12)

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Pydantic Field constraints to all request parameters across OpenAI, Anthropic, and Ollama schemas, validate port in config, and add model name validation at the registry level.

**Architecture:** Each schema file gets Field constraints on numeric parameters. Registry gets a `validate_model_name()` function called from `resolve()` and `add_mapping()`. All validation uses Pydantic's built-in mechanisms — no custom middleware needed.

**Tech Stack:** Pydantic Field validators, pytest, pydantic ValidationError

**Spec:** `docs/superpowers/specs/2026-03-13-input-validation-design.md`

---

## Task 1: OpenAI Schema Validation

**Files:**
- Modify: `olmlx/schemas/openai.py`
- Modify: `tests/test_schemas.py` (TestOpenAISchemas class)

- [ ] **Step 1: Write failing tests for OpenAI chat request validation**

Add `import pytest` and `from pydantic import ValidationError` to the top of `tests/test_schemas.py` (if not already present). Then add these tests to `TestOpenAISchemas`:

```python

def test_chat_request_temperature_valid_boundary(self):
    req = OpenAIChatRequest(
        model="test",
        messages=[OpenAIChatMessage(role="user", content="hi")],
        temperature=2.0,
    )
    assert req.temperature == 2.0

def test_chat_request_temperature_rejects_negative(self):
    with pytest.raises(ValidationError, match="temperature"):
        OpenAIChatRequest(
            model="test",
            messages=[OpenAIChatMessage(role="user", content="hi")],
            temperature=-0.1,
        )

def test_chat_request_temperature_rejects_above_max(self):
    with pytest.raises(ValidationError, match="temperature"):
        OpenAIChatRequest(
            model="test",
            messages=[OpenAIChatMessage(role="user", content="hi")],
            temperature=2.1,
        )

def test_chat_request_top_p_rejects_above_one(self):
    with pytest.raises(ValidationError, match="top_p"):
        OpenAIChatRequest(
            model="test",
            messages=[OpenAIChatMessage(role="user", content="hi")],
            top_p=1.1,
        )

def test_chat_request_max_tokens_rejects_zero(self):
    with pytest.raises(ValidationError, match="max_tokens"):
        OpenAIChatRequest(
            model="test",
            messages=[OpenAIChatMessage(role="user", content="hi")],
            max_tokens=0,
        )

def test_chat_request_max_completion_tokens_rejects_zero(self):
    with pytest.raises(ValidationError, match="max_completion_tokens"):
        OpenAIChatRequest(
            model="test",
            messages=[OpenAIChatMessage(role="user", content="hi")],
            max_completion_tokens=0,
        )

def test_chat_request_n_rejects_greater_than_one(self):
    with pytest.raises(ValidationError, match="n"):
        OpenAIChatRequest(
            model="test",
            messages=[OpenAIChatMessage(role="user", content="hi")],
            n=2,
        )

def test_chat_request_presence_penalty_rejects_out_of_range(self):
    with pytest.raises(ValidationError, match="presence_penalty"):
        OpenAIChatRequest(
            model="test",
            messages=[OpenAIChatMessage(role="user", content="hi")],
            presence_penalty=2.1,
        )

def test_chat_request_frequency_penalty_rejects_out_of_range(self):
    with pytest.raises(ValidationError, match="frequency_penalty"):
        OpenAIChatRequest(
            model="test",
            messages=[OpenAIChatMessage(role="user", content="hi")],
            frequency_penalty=-2.1,
        )

def test_completion_request_temperature_rejects_negative(self):
    with pytest.raises(ValidationError, match="temperature"):
        OpenAICompletionRequest(model="test", prompt="hi", temperature=-1)

def test_completion_request_n_rejects_greater_than_one(self):
    with pytest.raises(ValidationError, match="n"):
        OpenAICompletionRequest(model="test", prompt="hi", n=2)

def test_completion_request_max_tokens_rejects_zero(self):
    with pytest.raises(ValidationError, match="max_tokens"):
        OpenAICompletionRequest(model="test", prompt="hi", max_tokens=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_schemas.py::TestOpenAISchemas -v -x`
Expected: FAIL — no validation in place yet

- [ ] **Step 3: Add Field constraints to OpenAI schemas**

In `olmlx/schemas/openai.py`, add `Field` to the existing pydantic import (`from pydantic import BaseModel, Field`) and update both request classes:

```python
from pydantic import BaseModel, Field


class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIChatMessage]
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    n: int = Field(1, ge=1, le=1)
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    presence_penalty: float = Field(0.0, ge=-2, le=2)
    frequency_penalty: float = Field(0.0, ge=-2, le=2)
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    seed: int | None = None


class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    n: int = Field(1, ge=1, le=1)
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = Field(None, ge=1)
    presence_penalty: float = Field(0.0, ge=-2, le=2)
    frequency_penalty: float = Field(0.0, ge=-2, le=2)
    seed: int | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schemas.py::TestOpenAISchemas -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add olmlx/schemas/openai.py tests/test_schemas.py
git commit -m "feat: add input validation to OpenAI schemas (#12)"
```

---

## Task 2: Anthropic Schema Validation

**Files:**
- Modify: `olmlx/schemas/anthropic.py`
- Modify: `tests/test_schemas.py` (TestAnthropicSchemas class)

- [ ] **Step 1: Write failing tests for Anthropic request validation**

Ensure `import pytest` and `from pydantic import ValidationError` are at the top of `tests/test_schemas.py` (if not already present). Then add to `TestAnthropicSchemas`:

```python
def test_messages_request_temperature_valid_boundary(self):
    req = AnthropicMessagesRequest(
        model="test",
        messages=[AnthropicMessage(role="user", content="hi")],
        temperature=1.0,
    )
    assert req.temperature == 1.0

def test_messages_request_temperature_rejects_above_one(self):
    with pytest.raises(ValidationError, match="temperature"):
        AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            temperature=1.1,
        )

def test_messages_request_temperature_rejects_negative(self):
    with pytest.raises(ValidationError, match="temperature"):
        AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            temperature=-0.1,
        )

def test_messages_request_top_p_rejects_above_one(self):
    with pytest.raises(ValidationError, match="top_p"):
        AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            top_p=1.1,
        )

def test_messages_request_top_k_rejects_zero(self):
    with pytest.raises(ValidationError, match="top_k"):
        AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            top_k=0,
        )

def test_messages_request_max_tokens_rejects_zero(self):
    with pytest.raises(ValidationError, match="max_tokens"):
        AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            max_tokens=0,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_schemas.py::TestAnthropicSchemas -v -x`
Expected: FAIL

- [ ] **Step 3: Add Field constraints to Anthropic schema**

In `olmlx/schemas/anthropic.py`, add `Field` to the existing pydantic import (`from pydantic import BaseModel, Field`) and update `AnthropicMessagesRequest`:

```python
from pydantic import BaseModel, Field  # Field is new

class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = Field(4096, ge=1)
    stream: bool = False
    temperature: float | None = Field(None, ge=0, le=1)
    top_p: float | None = Field(None, ge=0, le=1)
    top_k: int | None = Field(None, ge=1)
    stop_sequences: list[str] | None = None
    system: str | list[AnthropicContentBlock] | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: dict | None = None
    thinking: AnthropicThinkingParam | None = None
    metadata: dict | None = None

    model_config = {"extra": "allow"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schemas.py::TestAnthropicSchemas -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add olmlx/schemas/anthropic.py tests/test_schemas.py
git commit -m "feat: add input validation to Anthropic schema (#12)"
```

---

## Task 3: Ollama ModelOptions Validation

**Files:**
- Modify: `olmlx/schemas/common.py`
- Modify: `tests/test_schemas.py` (TestCommonSchemas class)

- [ ] **Step 1: Write failing tests for ModelOptions validation**

Ensure `import pytest` and `from pydantic import ValidationError` are at the top of `tests/test_schemas.py` (if not already present). Then add to `TestCommonSchemas`:

```python
def test_model_options_temperature_rejects_negative(self):
    with pytest.raises(ValidationError, match="temperature"):
        ModelOptions(temperature=-0.1)

def test_model_options_temperature_allows_high_values(self):
    opts = ModelOptions(temperature=100.0)
    assert opts.temperature == 100.0

def test_model_options_top_p_rejects_above_one(self):
    with pytest.raises(ValidationError, match="top_p"):
        ModelOptions(top_p=1.1)

def test_model_options_top_p_rejects_negative(self):
    with pytest.raises(ValidationError, match="top_p"):
        ModelOptions(top_p=-0.1)

def test_model_options_top_k_rejects_zero(self):
    with pytest.raises(ValidationError, match="top_k"):
        ModelOptions(top_k=0)

def test_model_options_min_p_rejects_above_one(self):
    with pytest.raises(ValidationError, match="min_p"):
        ModelOptions(min_p=1.1)

def test_model_options_repeat_last_n_allows_negative_one(self):
    opts = ModelOptions(repeat_last_n=-1)
    assert opts.repeat_last_n == -1

def test_model_options_repeat_last_n_rejects_below_negative_one(self):
    with pytest.raises(ValidationError, match="repeat_last_n"):
        ModelOptions(repeat_last_n=-2)

def test_model_options_num_predict_allows_negative_one(self):
    opts = ModelOptions(num_predict=-1)
    assert opts.num_predict == -1

def test_model_options_num_predict_rejects_below_negative_one(self):
    with pytest.raises(ValidationError, match="num_predict"):
        ModelOptions(num_predict=-2)

def test_model_options_num_ctx_rejects_zero(self):
    with pytest.raises(ValidationError, match="num_ctx"):
        ModelOptions(num_ctx=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_schemas.py::TestCommonSchemas -v -x`
Expected: FAIL

- [ ] **Step 3: Add Field constraints to ModelOptions**

In `olmlx/schemas/common.py`, add `Field` to the existing pydantic import:

```python
from pydantic import BaseModel, ConfigDict, Field  # Field is new


class ModelOptions(BaseModel):
    model_config = ConfigDict(extra="allow")
    """Ollama model options / parameters."""

    num_keep: int | None = None
    seed: int | None = None
    num_predict: int | None = Field(None, ge=-1)
    top_k: int | None = Field(None, ge=1)
    top_p: float | None = Field(None, ge=0, le=1)
    min_p: float | None = Field(None, ge=0, le=1)
    tfs_z: float | None = None
    typical_p: float | None = None
    repeat_last_n: int | None = Field(None, ge=-1)
    temperature: float | None = Field(None, ge=0)
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    mirostat: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None
    penalize_newline: bool | None = None
    stop: list[str] | None = None
    numa: bool | None = None
    num_ctx: int | None = Field(None, ge=1)
    num_batch: int | None = None
    num_gpu: int | None = None
    main_gpu: int | None = None
    low_vram: bool | None = None
    vocab_only: bool | None = None
    use_mmap: bool | None = None
    use_mlock: bool | None = None
    num_thread: int | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schemas.py::TestCommonSchemas -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add olmlx/schemas/common.py tests/test_schemas.py
git commit -m "feat: add input validation to Ollama ModelOptions (#12)"
```

---

## Task 4: Config Port Validation

**Files:**
- Modify: `olmlx/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for port validation**

Add to `tests/test_config.py` in `TestSettings`:

```python
def test_port_rejects_zero(self, monkeypatch):
    monkeypatch.setenv("OLMLX_PORT", "0")
    with pytest.raises(ValidationError):
        Settings()

def test_port_rejects_negative(self, monkeypatch):
    monkeypatch.setenv("OLMLX_PORT", "-1")
    with pytest.raises(ValidationError):
        Settings()

def test_port_rejects_above_65535(self, monkeypatch):
    monkeypatch.setenv("OLMLX_PORT", "65536")
    with pytest.raises(ValidationError):
        Settings()

def test_port_accepts_boundary_values(self, monkeypatch):
    monkeypatch.setenv("OLMLX_PORT", "1")
    s = Settings()
    assert s.port == 1

    monkeypatch.setenv("OLMLX_PORT", "65535")
    s = Settings()
    assert s.port == 65535
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py::TestSettings -v -x`
Expected: FAIL — port 0 and 65536 currently accepted

- [ ] **Step 3: Add port validation to config**

In `olmlx/config.py`, change the port field (`Field` is already imported):

```python
port: Annotated[int, Field(ge=1, le=65535)] = 11434
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py::TestSettings -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add olmlx/config.py tests/test_config.py
git commit -m "feat: add port range validation to config (#12)"
```

---

## Task 5: Model Name Validation (Registry Level)

**Files:**
- Modify: `olmlx/engine/registry.py`
- Modify: `tests/test_registry.py`

- [ ] **Step 1: Write failing tests for model name validation**

Add to `tests/test_registry.py` in `TestModelRegistry`:

```python
def test_validate_model_name_rejects_empty(self):
    from olmlx.engine.registry import validate_model_name
    with pytest.raises(ValueError, match="empty"):
        validate_model_name("")

def test_validate_model_name_rejects_whitespace_only(self):
    from olmlx.engine.registry import validate_model_name
    with pytest.raises(ValueError, match="empty"):
        validate_model_name("   ")

def test_validate_model_name_rejects_path_traversal(self):
    from olmlx.engine.registry import validate_model_name
    with pytest.raises(ValueError, match="path traversal"):
        validate_model_name("../etc/passwd")

def test_validate_model_name_rejects_embedded_path_traversal(self):
    from olmlx.engine.registry import validate_model_name
    with pytest.raises(ValueError, match="path traversal"):
        validate_model_name("model/../secret")

def test_validate_model_name_rejects_too_long(self):
    from olmlx.engine.registry import validate_model_name
    with pytest.raises(ValueError, match="256"):
        validate_model_name("a" * 257)

def test_validate_model_name_accepts_ollama_style(self):
    from olmlx.engine.registry import validate_model_name
    validate_model_name("qwen3:8b")  # should not raise

def test_validate_model_name_accepts_hf_style(self):
    from olmlx.engine.registry import validate_model_name
    validate_model_name("Qwen/Qwen3-8B")  # should not raise

def test_validate_model_name_accepts_256_chars(self):
    from olmlx.engine.registry import validate_model_name
    validate_model_name("a" * 256)  # should not raise

def test_resolve_rejects_empty_name(self, registry):
    with pytest.raises(ValueError, match="empty"):
        registry.resolve("")

def test_resolve_rejects_path_traversal(self, registry):
    with pytest.raises(ValueError, match="path traversal"):
        registry.resolve("../etc/passwd")

def test_add_mapping_rejects_empty_name(self, registry):
    with pytest.raises(ValueError, match="empty"):
        registry.add_mapping("", "org/model")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_registry.py -v -x -k "validate_model_name or rejects_empty or rejects_path"`
Expected: FAIL — `validate_model_name` doesn't exist yet

- [ ] **Step 3: Implement validate_model_name and wire it into resolve/add_mapping**

In `olmlx/engine/registry.py`, add the function before the class:

```python
def validate_model_name(name: str) -> None:
    """Validate a model name. Raises ValueError for invalid names."""
    if not name or not name.strip():
        raise ValueError("Model name must not be empty")
    if ".." in name:
        raise ValueError(
            f"Model name {name!r} contains path traversal sequence '..'"
        )
    if len(name) > 256:
        raise ValueError(
            f"Model name must be at most 256 characters, got {len(name)}"
        )
```

Then add `validate_model_name(name)` as the first line of both `resolve()` and `add_mapping()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_registry.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/registry.py tests/test_registry.py
git commit -m "feat: add model name validation at registry level (#12)"
```

---

## Task 6: Final Verification

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS (no regressions)

- [ ] **Step 2: Run linting**

Run: `uv run ruff check --fix && uv run ruff format`
Expected: Clean

- [ ] **Step 3: Final commit if linting made changes**

```bash
git add -u
git commit -m "style: lint and format validation changes (#12)"
```
