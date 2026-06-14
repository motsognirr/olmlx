# Multi-model Panel + Judge Coordinator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a synthetic `type: "panel"` model that answers a request with a per-task-routed panel of models plus a judge that synthesizes one reconciled answer, while behaving as a drop-in tool-calling model (the client still executes tools).

**Architecture:** A new `engine/panel.py` `panel_generate_chat()` mirrors `engine/inference.py:generate_chat()`'s signature and return shapes. It is a *per-turn reconciler* riding the client's existing tool loop: on each stateless HTTP call it routes via a small classifier, runs each panelist through `generate_chat`, then either (a) emits the **deduped union** of the panelists' proposed tool calls as canonical Qwen `<tool_call>` text, or (b) when no panelist wants tools, runs the judge (no tools) to synthesize a fresh merged answer. Because the coordinator returns a `generate_chat`-shaped result whose text is either the judge's prose or `<tool_call>` blocks, both routers' existing `parse_model_output` paths handle it unchanged — they only swap which dispatch function they call. Panels live as `type: "panel"` entries in `models.json`, parsed into a `PanelConfig` and stored in a separate registry map.

**Tech Stack:** Python 3.12, FastAPI, MLX (via existing `generate_chat`), xgrammar (`engine/grammar.py`), pytest with `unittest.mock`.

---

## File Structure

- **Create** `olmlx/engine/panel.py` — pure helpers (`first_user_text`, `route_grammar`, `select_members`, `merge_tool_calls`, `serialize_tool_calls_qwen`, `build_judge_messages`) + async coordinator (`classify`, `_run_panel`, `panel_generate_chat`).
- **Modify** `olmlx/engine/registry.py` — add `PanelConfig` dataclass; parse `type: "panel"` entries in `load()` into `self._panels`; add `is_panel()`, `resolve_panel()`, and a post-load `_validate_panels()`.
- **Modify** `olmlx/routers/openai.py` — branch the dispatch function on `registry.is_panel(req.model)`.
- **Modify** `olmlx/routers/chat.py` — same branch for the Ollama route.
- **Create** `tests/test_panel.py` — coordinator + helper unit tests (mocked `generate_chat`).
- **Create** `tests/test_registry_panel.py` — panel config parsing + validation tests.
- **Modify** `CLAUDE.md` — add the panel invariant note.

### Key data shapes (used across tasks)

- `parse_model_output(text, has_tools, *, thinking_expected=False)` returns `(thinking: str, visible: str, tool_uses: list[dict])`. Each tool_use is `{"name": str, "input": dict, "_span": tuple}` (the `_span` key is internal; helpers must ignore/strip it).
- `generate_chat(...)` non-stream returns `{"text": str, "done": True, "stats": TimingStats, ...}` and may include `"raw_text"`. Streaming returns an async generator yielding `{"text": str}` chunks then a final `{"done": True, "raw_text"?: str, "done_reason"?: str, "stats"?: TimingStats}` chunk.
- `GrammarSpec(kind="json_schema", schema={...})` from `olmlx/engine/grammar.py`.
- Routers re-parse the result text, so the coordinator never assigns tool-call IDs — `_to_openai_tool_calls` (openai) and `_build_tool_calls` (chat) do that downstream.

---

## Task 1: `PanelConfig` dataclass in the registry

**Files:**
- Modify: `olmlx/engine/registry.py` (add near `class ModelConfig`, ~line 306)
- Test: `tests/test_registry_panel.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_registry_panel.py`:

```python
"""Tests for panel-type entries in olmlx.engine.registry."""

import pytest

from olmlx.engine.registry import PanelConfig


class TestPanelConfig:
    def test_from_entry_parses_fields(self):
        entry = {
            "type": "panel",
            "classifier": "qwen3-0.6b",
            "judge": "gpt-oss-20b",
            "routes": {
                "code": ["qwen3-coder", "devstral"],
                "default": ["qwen3", "mistral"],
            },
        }
        pc = PanelConfig.from_entry("my-panel:latest", entry)
        assert pc.name == "my-panel:latest"
        assert pc.classifier == "qwen3-0.6b"
        assert pc.judge == "gpt-oss-20b"
        assert pc.routes["code"] == ["qwen3-coder", "devstral"]

    def test_from_entry_requires_default_route(self):
        entry = {
            "type": "panel",
            "classifier": "c",
            "judge": "j",
            "routes": {"code": ["a"]},
        }
        with pytest.raises(ValueError, match="default"):
            PanelConfig.from_entry("p:latest", entry)

    def test_from_entry_requires_classifier_and_judge(self):
        entry = {"type": "panel", "routes": {"default": ["a"]}}
        with pytest.raises(ValueError, match="classifier"):
            PanelConfig.from_entry("p:latest", entry)

    def test_all_member_names_unions_routes(self):
        pc = PanelConfig.from_entry(
            "p:latest",
            {
                "type": "panel",
                "classifier": "c",
                "judge": "j",
                "routes": {"code": ["a", "b"], "default": ["b", "c2"]},
            },
        )
        assert pc.all_member_names() == {"a", "b", "c2"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_panel.py -v`
Expected: FAIL with `ImportError: cannot import name 'PanelConfig'`.

- [ ] **Step 3: Implement `PanelConfig`**

In `olmlx/engine/registry.py`, add after the `ModelConfig` class definition (the dataclass block ending near line 306; place the new class immediately before `class ModelRegistry`). Ensure `from dataclasses import dataclass, field` and `from typing import Any` are already imported (they are — `ModelConfig` uses them):

```python
@dataclass(frozen=True)
class PanelConfig:
    """A ``type: "panel"`` entry from models.json.

    A panel is not a loadable model: it has no weights. ``classifier``,
    ``judge`` and the per-route member lists all reference other
    models.json entries by name. See engine/panel.py for the coordinator
    that executes a panel.
    """

    name: str
    classifier: str
    judge: str
    routes: dict[str, list[str]]

    @classmethod
    def from_entry(cls, name: str, entry: dict) -> "PanelConfig":
        classifier = entry.get("classifier")
        judge = entry.get("judge")
        routes = entry.get("routes")
        if not classifier or not isinstance(classifier, str):
            raise ValueError(
                f"panel {name!r}: 'classifier' must be a non-empty model name"
            )
        if not judge or not isinstance(judge, str):
            raise ValueError(
                f"panel {name!r}: 'judge' must be a non-empty model name"
            )
        if not isinstance(routes, dict) or not routes:
            raise ValueError(f"panel {name!r}: 'routes' must be a non-empty object")
        for key, members in routes.items():
            if (
                not isinstance(members, list)
                or not members
                or not all(isinstance(m, str) and m for m in members)
            ):
                raise ValueError(
                    f"panel {name!r}: route {key!r} must be a non-empty list "
                    "of model names"
                )
        if "default" not in routes:
            raise ValueError(f"panel {name!r}: 'routes' must contain a 'default' key")
        return cls(name=name, classifier=classifier, judge=judge, routes=dict(routes))

    def all_member_names(self) -> set[str]:
        names: set[str] = set()
        for members in self.routes.values():
            names.update(members)
        return names
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry_panel.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/registry.py tests/test_registry_panel.py
git commit -m "feat(panel): add PanelConfig registry dataclass"
```

---

## Task 2: Registry parses & validates panel entries

**Files:**
- Modify: `olmlx/engine/registry.py` — `__init__` (add `self._panels`), `load()` loop (~line 1252), add `is_panel`/`resolve_panel`/`_validate_panels`
- Test: `tests/test_registry_panel.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry_panel.py`:

```python
import json

from olmlx.engine.registry import ModelRegistry


def _load_registry(tmp_path, monkeypatch, config: dict) -> ModelRegistry:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr("olmlx.engine.registry.settings.models_config", path)
    reg = ModelRegistry()
    reg.load()
    return reg


class TestRegistryPanelLoading:
    def test_panel_entry_loaded_and_resolvable(self, tmp_path, monkeypatch):
        reg = _load_registry(
            tmp_path,
            monkeypatch,
            {
                "qwen3": "Qwen/Qwen3-8B-MLX",
                "small": "org/small",
                "judgem": "org/judge",
                "my-panel": {
                    "type": "panel",
                    "classifier": "small",
                    "judge": "judgem",
                    "routes": {"default": ["qwen3"]},
                },
            },
        )
        assert reg.is_panel("my-panel") is True
        assert reg.is_panel("my-panel:latest") is True
        assert reg.is_panel("qwen3") is False
        pc = reg.resolve_panel("my-panel")
        assert pc.judge == "judgem"
        # A panel name is NOT a normal model.
        assert reg.resolve("my-panel") is None

    def test_panel_with_missing_member_is_dropped(self, tmp_path, monkeypatch):
        reg = _load_registry(
            tmp_path,
            monkeypatch,
            {
                "small": "org/small",
                "judgem": "org/judge",
                "bad-panel": {
                    "type": "panel",
                    "classifier": "small",
                    "judge": "judgem",
                    "routes": {"default": ["does-not-exist"]},
                },
            },
        )
        assert reg.is_panel("bad-panel") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_panel.py::TestRegistryPanelLoading -v`
Expected: FAIL with `AttributeError: 'ModelRegistry' object has no attribute 'is_panel'`.

- [ ] **Step 3: Implement registry parsing + validation**

In `olmlx/engine/registry.py`:

(a) In `ModelRegistry.__init__`, alongside `self._mappings = ...`, add an empty panels dict. Find the `__init__` (search `def __init__` in `class ModelRegistry`) and add:

```python
        self._panels: dict[str, PanelConfig] = {}
```

(b) In `load()`, replace the entry loop (currently at lines 1248-1257):

```python
            self._mappings = {}
            self._raw_unrecognized = {}
            self._dirty_keys = set()
            self._removed_keys = set()
            for k, v in raw.items():
                try:
                    self._mappings[k] = ModelConfig.from_entry(v)
                except (ValueError, TypeError) as exc:
                    logger.warning("Skipping invalid models.json entry %r: %s", k, exc)
                    self._raw_unrecognized[k] = v
```

with:

```python
            self._mappings = {}
            self._panels = {}
            self._raw_unrecognized = {}
            self._dirty_keys = set()
            self._removed_keys = set()
            for k, v in raw.items():
                normalized = self.normalize_name(k)
                if isinstance(v, dict) and v.get("type") == "panel":
                    try:
                        self._panels[normalized] = PanelConfig.from_entry(normalized, v)
                    except (ValueError, TypeError) as exc:
                        logger.warning(
                            "Skipping invalid panel entry %r: %s", k, exc
                        )
                        self._raw_unrecognized[k] = v
                    continue
                try:
                    self._mappings[k] = ModelConfig.from_entry(v)
                except (ValueError, TypeError) as exc:
                    logger.warning("Skipping invalid models.json entry %r: %s", k, exc)
                    self._raw_unrecognized[k] = v
            self._validate_panels()
```

(c) Add these methods to `ModelRegistry` (place them next to `resolve`, after the `resolve` method ~line 1307):

```python
    def is_panel(self, name: str) -> bool:
        """True if *name* resolves to a ``type: "panel"`` entry."""
        return self.normalize_name(name) in self._panels

    def resolve_panel(self, name: str) -> "PanelConfig | None":
        """Resolve *name* to a PanelConfig, or None if it is not a panel."""
        return self._panels.get(self.normalize_name(name))

    def _validate_panels(self) -> None:
        """Drop panels referencing unknown models; warn on judge-in-panel.

        Cross-reference validation runs after all entries are loaded so a
        panel may reference members declared anywhere in the file.
        """
        for name, panel in list(self._panels.items()):
            refs = {panel.classifier, panel.judge} | panel.all_member_names()
            missing = sorted(r for r in refs if self.resolve(r) is None)
            if missing:
                logger.warning(
                    "Dropping panel %r: references unknown models %s",
                    name,
                    missing,
                )
                del self._panels[name]
                continue
            if panel.judge in panel.all_member_names():
                logger.warning(
                    "Panel %r: judge %r is also a panelist; this invites "
                    "self-preference bias.",
                    name,
                    panel.judge,
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry_panel.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/registry.py tests/test_registry_panel.py
git commit -m "feat(panel): load and validate panel entries in registry"
```

---

## Task 3: Pure helpers — request routing inputs

**Files:**
- Create: `olmlx/engine/panel.py`
- Test: `tests/test_panel.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_panel.py`:

```python
"""Tests for olmlx.engine.panel."""

from olmlx.engine.grammar import GrammarSpec
from olmlx.engine.panel import (
    first_user_text,
    route_grammar,
    select_members,
)
from olmlx.engine.registry import PanelConfig


def _panel() -> PanelConfig:
    return PanelConfig.from_entry(
        "p:latest",
        {
            "type": "panel",
            "classifier": "c",
            "judge": "j",
            "routes": {
                "code": ["qwen3-coder", "devstral"],
                "default": ["qwen3", "mistral"],
            },
        },
    )


class TestRoutingHelpers:
    def test_first_user_text_string_content(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi"},
        ]
        assert first_user_text(msgs) == "hello world"

    def test_first_user_text_list_content(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ],
            }
        ]
        assert first_user_text(msgs) == "part one\npart two"

    def test_first_user_text_no_user(self):
        assert first_user_text([{"role": "system", "content": "x"}]) == ""

    def test_route_grammar_enumerates_keys(self):
        spec = route_grammar(_panel())
        assert isinstance(spec, GrammarSpec)
        assert spec.kind == "json_schema"
        enum = spec.schema["properties"]["route"]["enum"]
        assert set(enum) == {"code", "default"}

    def test_select_members_known_route(self):
        assert select_members("code", _panel()) == ["qwen3-coder", "devstral"]

    def test_select_members_unknown_route_falls_back_to_default(self):
        assert select_members("nonsense", _panel()) == ["qwen3", "mistral"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'olmlx.engine.panel'`.

- [ ] **Step 3: Create the module with the helpers**

Create `olmlx/engine/panel.py`:

```python
"""Multi-model panel + judge coordinator (sequential, single-box).

A ``type: "panel"`` model answers a request with a per-task-routed panel
of models plus a judge that synthesizes one reconciled answer, while
behaving as a drop-in tool-calling model: the *client* still executes
tools. The coordinator is a per-turn reconciler riding the client's
existing tool loop — see docs/superpowers/specs/2026-06-14-...-design.md.

INVARIANT: every model call goes through ``generate_chat`` so the
Metal-stream / inference-lock handling is reused. Never touch MLX here.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from olmlx.engine.grammar import GrammarSpec
from olmlx.engine.inference import generate_chat
from olmlx.engine.tool_parser import parse_model_output
from olmlx.utils.timing import TimingStats

if TYPE_CHECKING:
    from olmlx.engine.model_manager import ModelManager
    from olmlx.engine.registry import PanelConfig

logger = logging.getLogger(__name__)


def first_user_text(messages: list[dict]) -> str:
    """Return the first user message's text (stable routing key).

    The conversation's task is fixed by the first user turn, so routing
    on it is deterministically re-derived every stateless HTTP call with
    no stored server state.
    """
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "\n".join(p for p in parts if p)
        return ""
    return ""


def route_grammar(panel: "PanelConfig") -> GrammarSpec:
    """A JSON-schema grammar constraining the classifier to one route key."""
    return GrammarSpec(
        kind="json_schema",
        schema={
            "type": "object",
            "properties": {
                "route": {"type": "string", "enum": sorted(panel.routes)}
            },
            "required": ["route"],
            "additionalProperties": False,
        },
    )


def select_members(route_key: str, panel: "PanelConfig") -> list[str]:
    """Members for *route_key*, falling back to the 'default' route."""
    return panel.routes.get(route_key, panel.routes["default"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_panel.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/panel.py tests/test_panel.py
git commit -m "feat(panel): routing helpers (first_user_text, route_grammar, select_members)"
```

---

## Task 4: Pure helpers — tool-call union & serialization

**Files:**
- Modify: `olmlx/engine/panel.py`
- Test: `tests/test_panel.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_panel.py`:

```python
from olmlx.engine.panel import merge_tool_calls, serialize_tool_calls_qwen
from olmlx.engine.tool_parser import parse_model_output


class TestToolCallUnion:
    def test_merge_dedupes_identical_calls(self):
        # Two panelists; one shared identical call, one unique each.
        per_panelist = [
            [
                {"name": "search", "input": {"q": "x"}, "_span": (0, 1)},
                {"name": "read", "input": {"path": "a"}},
            ],
            [
                {"name": "search", "input": {"q": "x"}},
                {"name": "read", "input": {"path": "b"}},
            ],
        ]
        merged = merge_tool_calls(per_panelist)
        # search{q:x} collapses to one; read{a} and read{b} both kept.
        assert merged == [
            {"name": "search", "input": {"q": "x"}},
            {"name": "read", "input": {"path": "a"}},
            {"name": "read", "input": {"path": "b"}},
        ]
        # _span must be stripped.
        assert all("_span" not in tc for tc in merged)

    def test_merge_empty(self):
        assert merge_tool_calls([[], []]) == []

    def test_serialize_round_trips_through_parser(self):
        merged = [
            {"name": "search", "input": {"q": "x"}},
            {"name": "read", "input": {"path": "a"}},
        ]
        text = serialize_tool_calls_qwen(merged)
        _thinking, _visible, tool_uses = parse_model_output(text, has_tools=True)
        reparsed = [{"name": tu["name"], "input": tu["input"]} for tu in tool_uses]
        assert reparsed == merged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel.py::TestToolCallUnion -v`
Expected: FAIL with `ImportError: cannot import name 'merge_tool_calls'`.

- [ ] **Step 3: Implement the helpers**

Append to `olmlx/engine/panel.py`:

```python
def _tool_key(tool_use: dict) -> str:
    """Stable dedup key: name + canonicalized arguments."""
    args = tool_use.get("input") or {}
    return tool_use["name"] + "\0" + json.dumps(args, sort_keys=True)


def merge_tool_calls(per_panelist: list[list[dict]]) -> list[dict]:
    """Deduped union of every panelist's proposed tool calls.

    Identical ``(name, arguments)`` collapse to one execution; different
    arguments both run. Insertion order is preserved (first panelist
    first). The internal ``_span`` key from ``parse_model_output`` is
    stripped. Tool-call IDs are assigned downstream by the routers.
    """
    merged: list[dict] = []
    seen: set[str] = set()
    for panelist_calls in per_panelist:
        for call in panelist_calls:
            key = _tool_key(call)
            if key in seen:
                continue
            seen.add(key)
            merged.append({"name": call["name"], "input": call.get("input") or {}})
    return merged


def serialize_tool_calls_qwen(tool_uses: list[dict]) -> str:
    """Render tool calls as canonical Qwen ``<tool_call>`` blocks.

    The routers re-parse this text via ``parse_model_output`` (the Qwen
    parser maps ``arguments`` -> ``input``), so the panel's tool turn is
    transparent to both the OpenAI and Ollama routers.
    """
    blocks = []
    for tu in tool_uses:
        payload = json.dumps({"name": tu["name"], "arguments": tu.get("input") or {}})
        blocks.append(f"<tool_call>\n{payload}\n</tool_call>")
    return "\n".join(blocks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_panel.py::TestToolCallUnion -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/panel.py tests/test_panel.py
git commit -m "feat(panel): tool-call union dedup + Qwen serialization"
```

---

## Task 5: Pure helper — judge prompt construction

**Files:**
- Modify: `olmlx/engine/panel.py`
- Test: `tests/test_panel.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_panel.py`:

```python
from olmlx.engine.panel import build_judge_messages


class TestJudgePrompt:
    def test_appends_candidates_as_final_user_turn(self):
        original = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        out = build_judge_messages(original, ["qwen3", "mistral"], ["four", "4"])
        # Original messages preserved as a prefix (not mutated).
        assert out[:2] == original
        assert original[-1]["content"] == "What is 2+2?"  # input untouched
        judge_turn = out[-1]
        assert judge_turn["role"] == "user"
        assert "qwen3" in judge_turn["content"]
        assert "mistral" in judge_turn["content"]
        assert "four" in judge_turn["content"]
        assert "4" in judge_turn["content"]

    def test_handles_empty_candidate_answer(self):
        out = build_judge_messages(
            [{"role": "user", "content": "q"}], ["m1"], [""]
        )
        assert "m1" in out[-1]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel.py::TestJudgePrompt -v`
Expected: FAIL with `ImportError: cannot import name 'build_judge_messages'`.

- [ ] **Step 3: Implement the helper**

Append to `olmlx/engine/panel.py`:

```python
_JUDGE_INSTRUCTION = (
    "You are the judge for a panel of models that all answered the "
    "conversation above. Several candidate answers are listed below. "
    "Synthesize ONE best final answer for the user. Ground every claim in "
    "the tool results already present in this conversation; do not call "
    "tools or invent new facts. Reconcile disagreements and prefer "
    "grounded, specific answers. Output only the final answer."
)


def build_judge_messages(
    original_messages: list[dict],
    member_names: list[str],
    answers: list[str],
) -> list[dict]:
    """Original conversation + a final user turn carrying the candidates.

    ``original_messages`` is not mutated (a new list is returned). The
    judge sees the full conversation — including any tool results the
    client executed — so it can verify groundedness.
    """
    candidates = []
    for name, answer in zip(member_names, answers):
        candidates.append(f"--- Candidate from {name} ---\n{answer.strip()}")
    content = _JUDGE_INSTRUCTION + "\n\n" + "\n\n".join(candidates)
    return [*original_messages, {"role": "user", "content": content}]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_panel.py::TestJudgePrompt -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/panel.py tests/test_panel.py
git commit -m "feat(panel): judge prompt construction"
```

---

## Task 6: Classifier call (`classify`)

**Files:**
- Modify: `olmlx/engine/panel.py`
- Test: `tests/test_panel.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_panel.py`:

```python
import pytest

from olmlx.engine import panel as panel_mod


def _make_panel():
    from olmlx.engine.registry import PanelConfig

    return PanelConfig.from_entry(
        "p:latest",
        {
            "type": "panel",
            "classifier": "c",
            "judge": "j",
            "routes": {"code": ["qa", "qb"], "default": ["da", "db"]},
        },
    )


def _fake_generate_chat_factory(responses: dict):
    """Return an async generate_chat stub keyed by model name -> text."""

    async def _fake(manager, model_name, messages, options=None, tools=None,
                    stream=False, keep_alive=None, max_tokens=512, cache_id="",
                    enable_thinking=None, grammar_spec=None):
        text = responses[model_name]
        return {"text": text, "done": True, "stats": None}

    return _fake


class TestClassify:
    @pytest.mark.asyncio
    async def test_classify_returns_route_members(self, monkeypatch):
        monkeypatch.setattr(
            panel_mod,
            "generate_chat",
            _fake_generate_chat_factory({"c": '{"route": "code"}'}),
        )
        members = await panel_mod.classify(
            manager=None, panel=_make_panel(), user_text="write a function"
        )
        assert members == ["qa", "qb"]

    @pytest.mark.asyncio
    async def test_classify_bad_json_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr(
            panel_mod,
            "generate_chat",
            _fake_generate_chat_factory({"c": "not json at all"}),
        )
        members = await panel_mod.classify(
            manager=None, panel=_make_panel(), user_text="hi"
        )
        assert members == ["da", "db"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel.py::TestClassify -v`
Expected: FAIL with `AttributeError: module 'olmlx.engine.panel' has no attribute 'classify'`.

- [ ] **Step 3: Implement `classify`**

Append to `olmlx/engine/panel.py`:

```python
_CLASSIFIER_SYSTEM = (
    "You are a request router. Classify the user's request into exactly "
    "one category. Respond ONLY with JSON of the form {\"route\": \"<category>\"}."
)


async def classify(
    manager: "ModelManager",
    panel: "PanelConfig",
    user_text: str,
    keep_alive: int | str | None = None,
) -> list[str]:
    """Route the request to a member list via the classifier model.

    The classifier output is grammar-constrained to the route keys; any
    parse failure falls back to the 'default' route.
    """
    categories = ", ".join(sorted(panel.routes))
    messages = [
        {"role": "system", "content": f"{_CLASSIFIER_SYSTEM} Categories: {categories}."},
        {"role": "user", "content": user_text},
    ]
    result = await generate_chat(
        manager,
        panel.classifier,
        messages,
        tools=None,
        stream=False,
        keep_alive=keep_alive,
        max_tokens=32,
        grammar_spec=route_grammar(panel),
    )
    text = (result.get("text") or "").strip()
    try:
        route = json.loads(text).get("route", "default")
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Panel classifier returned non-JSON %r; using default", text)
        route = "default"
    return select_members(route, panel)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_panel.py::TestClassify -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/panel.py tests/test_panel.py
git commit -m "feat(panel): grammar-constrained classifier routing"
```

---

## Task 7: Panel runner (`_run_panel`) — runs panelists, reconciles

**Files:**
- Modify: `olmlx/engine/panel.py`
- Test: `tests/test_panel.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_panel.py`:

```python
class TestRunPanel:
    @pytest.mark.asyncio
    async def test_returns_union_when_any_panelist_wants_tools(self, monkeypatch):
        # da proposes a tool call (Qwen format), db answers in prose.
        responses = {
            "c": '{"route": "default"}',
            "da": '<tool_call>\n{"name": "search", "arguments": {"q": "x"}}\n</tool_call>',
            "db": "I think the answer is 42.",
        }
        monkeypatch.setattr(
            panel_mod, "generate_chat", _fake_generate_chat_factory(responses)
        )
        answers, merged = await panel_mod._run_panel(
            manager=None,
            panel=_make_panel(),
            messages=[{"role": "user", "content": "find x"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
            options=None,
            keep_alive=None,
            max_tokens=128,
            enable_thinking=None,
        )
        assert merged == [{"name": "search", "input": {"q": "x"}}]

    @pytest.mark.asyncio
    async def test_returns_answers_when_no_tools_requested(self, monkeypatch):
        responses = {
            "c": '{"route": "default"}',
            "da": "Answer A",
            "db": "Answer B",
        }
        monkeypatch.setattr(
            panel_mod, "generate_chat", _fake_generate_chat_factory(responses)
        )
        answers, merged = await panel_mod._run_panel(
            manager=None,
            panel=_make_panel(),
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            options=None,
            keep_alive=None,
            max_tokens=128,
            enable_thinking=None,
        )
        assert merged == []
        assert answers == (["da", "db"], ["Answer A", "Answer B"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel.py::TestRunPanel -v`
Expected: FAIL with `AttributeError: module 'olmlx.engine.panel' has no attribute '_run_panel'`.

- [ ] **Step 3: Implement `_run_panel`**

Append to `olmlx/engine/panel.py`:

```python
async def _run_panel(
    manager: "ModelManager",
    panel: "PanelConfig",
    messages: list[dict],
    tools: list[dict] | None,
    options: dict | None,
    keep_alive: int | str | None,
    max_tokens: int,
    enable_thinking: bool | None,
) -> tuple[tuple[list[str], list[str]], list[dict]]:
    """Route, run each panelist once, and reconcile this turn.

    Returns ``((member_names, answers), merged_tool_uses)``. When any
    panelist proposes tool calls, ``merged_tool_uses`` is their deduped
    union and ``answers`` is unused by the caller (the turn is a tool
    turn). When none do, ``merged_tool_uses`` is empty and the caller
    runs the judge over ``answers``.

    Any panelist/classifier failure propagates (fail the request).
    """
    user_text = first_user_text(messages)
    members = await classify(manager, panel, user_text, keep_alive)

    has_tools = bool(tools)
    answers: list[str] = []
    per_panelist_tools: list[list[dict]] = []
    for member in members:
        result = await generate_chat(
            manager,
            member,
            messages,
            options=options,
            tools=tools,
            stream=False,
            keep_alive=keep_alive,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        parse_text = result.get("raw_text") or result.get("text") or ""
        _thinking, visible, tool_uses = parse_model_output(
            parse_text,
            has_tools,
            thinking_expected=bool(result.get("thinking_expected")),
        )
        answers.append(visible)
        per_panelist_tools.append(tool_uses)

    merged = merge_tool_calls(per_panelist_tools)
    return (members, answers), merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_panel.py::TestRunPanel -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/panel.py tests/test_panel.py
git commit -m "feat(panel): _run_panel routes, runs panelists, reconciles turn"
```

---

## Task 8: Coordinator entry — non-streaming (`panel_generate_chat`)

**Files:**
- Modify: `olmlx/engine/panel.py`
- Test: `tests/test_panel.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_panel.py`:

```python
class TestPanelGenerateChatNonStream:
    @pytest.mark.asyncio
    async def test_tool_turn_returns_qwen_blocks(self, monkeypatch):
        responses = {
            "c": '{"route": "default"}',
            "da": '<tool_call>\n{"name": "search", "arguments": {"q": "x"}}\n</tool_call>',
            "db": "prose",
        }
        monkeypatch.setattr(
            panel_mod, "generate_chat", _fake_generate_chat_factory(responses)
        )
        monkeypatch.setattr(
            panel_mod, "_resolve_panel", lambda manager, name: _make_panel()
        )
        result = await panel_mod.panel_generate_chat(
            manager=None,
            model_name="p:latest",
            messages=[{"role": "user", "content": "find x"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
            stream=False,
        )
        # Router parses raw_text -> tool calls.
        _t, _v, tool_uses = parse_model_output(result["raw_text"], has_tools=True)
        assert [tu["name"] for tu in tool_uses] == ["search"]
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_final_turn_returns_judge_answer(self, monkeypatch):
        responses = {
            "c": '{"route": "default"}',
            "da": "Answer A",
            "db": "Answer B",
            "j": "Reconciled final answer.",
        }
        monkeypatch.setattr(
            panel_mod, "generate_chat", _fake_generate_chat_factory(responses)
        )
        monkeypatch.setattr(
            panel_mod, "_resolve_panel", lambda manager, name: _make_panel()
        )
        result = await panel_mod.panel_generate_chat(
            manager=None,
            model_name="p:latest",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            stream=False,
        )
        assert result["text"] == "Reconciled final answer."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel.py::TestPanelGenerateChatNonStream -v`
Expected: FAIL with `AttributeError: module 'olmlx.engine.panel' has no attribute 'panel_generate_chat'`.

- [ ] **Step 3: Implement `_resolve_panel`, the judge call, and the non-stream path**

Append to `olmlx/engine/panel.py`:

```python
def _resolve_panel(manager: "ModelManager", model_name: str) -> "PanelConfig":
    """Resolve *model_name* to its PanelConfig or raise (fail the request)."""
    panel = manager.registry.resolve_panel(model_name)
    if panel is None:
        raise ValueError(f"{model_name!r} is not a configured panel model")
    return panel


async def _judge_answer(
    manager: "ModelManager",
    panel: "PanelConfig",
    messages: list[dict],
    member_names: list[str],
    answers: list[str],
    options: dict | None,
    keep_alive: int | str | None,
    max_tokens: int,
    enable_thinking: bool | None,
    stream: bool,
):
    """Run the judge (no tools) to synthesize the final answer.

    Returns the judge's ``generate_chat`` result verbatim — an async
    generator when ``stream`` else a dict — so the routers stream/format
    it exactly as a single model's output.
    """
    judge_messages = build_judge_messages(messages, member_names, answers)
    return await generate_chat(
        manager,
        panel.judge,
        judge_messages,
        options=options,
        tools=None,  # judge must not redo work
        stream=stream,
        keep_alive=keep_alive,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )


async def panel_generate_chat(
    manager: "ModelManager",
    model_name: str,
    messages: list[dict],
    options: dict | None = None,
    tools: list[dict] | None = None,
    stream: bool = True,
    keep_alive: int | str | None = None,
    max_tokens: int = 512,
    cache_id: str = "",
    enable_thinking: bool | None = None,
    grammar_spec: GrammarSpec | None = None,
) -> AsyncGenerator[dict, None] | dict:
    """Drop-in, ``generate_chat``-compatible entry point for a panel model.

    ``cache_id`` and ``grammar_spec`` are accepted for signature parity
    but not applied to the panel as a whole (the judge/panelists manage
    their own caching).
    """
    panel = _resolve_panel(manager, model_name)
    if stream:
        return _panel_stream(
            manager, panel, messages, options, tools, keep_alive,
            max_tokens, enable_thinking,
        )

    (member_names, answers), merged = await _run_panel(
        manager, panel, messages, tools, options, keep_alive,
        max_tokens, enable_thinking,
    )
    if merged:
        raw = serialize_tool_calls_qwen(merged)
        return {"text": "", "raw_text": raw, "done": True, "stats": TimingStats()}
    return await _judge_answer(
        manager, panel, messages, member_names, answers, options,
        keep_alive, max_tokens, enable_thinking, stream=False,
    )
```

(The `_panel_stream` async generator is implemented in Task 9; add a temporary stub so the module imports — it will be replaced in Task 9:)

```python
async def _panel_stream(*args, **kwargs):  # replaced in Task 9
    raise NotImplementedError
    yield {}  # pragma: no cover  (makes this an async generator)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_panel.py::TestPanelGenerateChatNonStream -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/panel.py tests/test_panel.py
git commit -m "feat(panel): non-streaming coordinator entry point"
```

---

## Task 9: Coordinator entry — streaming (`_panel_stream`)

**Files:**
- Modify: `olmlx/engine/panel.py`
- Test: `tests/test_panel.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_panel.py`:

```python
def _fake_generate_chat_streaming_factory(responses: dict, stream_models: set):
    """generate_chat stub: dict for non-stream models, async-gen for streamed ones."""

    async def _fake(manager, model_name, messages, options=None, tools=None,
                    stream=False, keep_alive=None, max_tokens=512, cache_id="",
                    enable_thinking=None, grammar_spec=None):
        text = responses[model_name]
        if stream and model_name in stream_models:
            async def _gen():
                yield {"text": text}
                yield {"done": True, "done_reason": "stop"}
            return _gen()
        return {"text": text, "done": True, "stats": None}

    return _fake


async def _drain(agen) -> list[dict]:
    return [chunk async for chunk in agen]


class TestPanelGenerateChatStream:
    @pytest.mark.asyncio
    async def test_stream_tool_turn_emits_qwen_text_then_done(self, monkeypatch):
        responses = {
            "c": '{"route": "default"}',
            "da": '<tool_call>\n{"name": "search", "arguments": {"q": "x"}}\n</tool_call>',
            "db": "prose",
        }
        monkeypatch.setattr(
            panel_mod,
            "generate_chat",
            _fake_generate_chat_streaming_factory(responses, stream_models=set()),
        )
        monkeypatch.setattr(panel_mod, "_resolve_panel", lambda m, n: _make_panel())
        agen = await panel_mod.panel_generate_chat(
            manager=None, model_name="p:latest",
            messages=[{"role": "user", "content": "find x"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
            stream=True,
        )
        chunks = await _drain(agen)
        full = "".join(c.get("text", "") for c in chunks)
        _t, _v, tool_uses = parse_model_output(full, has_tools=True)
        assert [tu["name"] for tu in tool_uses] == ["search"]
        assert chunks[-1].get("done") is True

    @pytest.mark.asyncio
    async def test_stream_final_turn_proxies_judge_stream(self, monkeypatch):
        responses = {
            "c": '{"route": "default"}',
            "da": "A", "db": "B",
            "j": "Final synthesized answer.",
        }
        monkeypatch.setattr(
            panel_mod,
            "generate_chat",
            _fake_generate_chat_streaming_factory(responses, stream_models={"j"}),
        )
        monkeypatch.setattr(panel_mod, "_resolve_panel", lambda m, n: _make_panel())
        agen = await panel_mod.panel_generate_chat(
            manager=None, model_name="p:latest",
            messages=[{"role": "user", "content": "hi"}],
            tools=None, stream=True,
        )
        chunks = await _drain(agen)
        full = "".join(c.get("text", "") for c in chunks)
        assert full == "Final synthesized answer."
        assert chunks[-1].get("done") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel.py::TestPanelGenerateChatStream -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Replace the `_panel_stream` stub**

In `olmlx/engine/panel.py`, replace the temporary `_panel_stream` stub from Task 8 with:

```python
async def _panel_stream(
    manager: "ModelManager",
    panel: "PanelConfig",
    messages: list[dict],
    options: dict | None,
    tools: list[dict] | None,
    keep_alive: int | str | None,
    max_tokens: int,
    enable_thinking: bool | None,
) -> AsyncGenerator[dict, None]:
    """Streaming coordinator.

    The panel compute runs *inside* the generator (during iteration) so
    the router's keepalive wrapper stays active. A tool turn yields the
    Qwen blocks as one text chunk; a final turn proxies the judge's token
    stream straight through.
    """
    (member_names, answers), merged = await _run_panel(
        manager, panel, messages, tools, options, keep_alive,
        max_tokens, enable_thinking,
    )
    if merged:
        raw = serialize_tool_calls_qwen(merged)
        yield {"text": raw}
        yield {"text": "", "done": True, "raw_text": raw, "done_reason": "stop"}
        return

    judge_stream = await _judge_answer(
        manager, panel, messages, member_names, answers, options,
        keep_alive, max_tokens, enable_thinking, stream=True,
    )
    async for chunk in judge_stream:
        yield chunk
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_panel.py -v`
Expected: PASS (all panel tests).

- [ ] **Step 5: Commit**

```bash
git add olmlx/engine/panel.py tests/test_panel.py
git commit -m "feat(panel): streaming coordinator entry point"
```

---

## Task 10: Wire the OpenAI router

**Files:**
- Modify: `olmlx/routers/openai.py` (imports + `openai_chat`, lines 310-431)
- Test: `tests/test_panel.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_panel.py`:

```python
class TestRouterDispatch:
    def test_openai_router_imports_panel_dispatch(self):
        from olmlx.routers import openai as openai_router

        assert hasattr(openai_router, "panel_generate_chat")
```

This is a thin guard; the real behavioral coverage is the coordinator tests plus the manual smoke test in Task 12.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel.py::TestRouterDispatch -v`
Expected: FAIL with `AttributeError: module 'olmlx.routers.openai' has no attribute 'panel_generate_chat'`.

- [ ] **Step 3: Add the import and branch**

In `olmlx/routers/openai.py`, add to the imports (next to `from olmlx.engine.inference import generate_chat`):

```python
from olmlx.engine.panel import panel_generate_chat
```

Then in `openai_chat`, immediately after `enable_thinking = resolve_openai_think(...)` (line 374) and before `if req.stream:` (line 376), add:

```python
    registry = request.app.state.registry
    dispatch = panel_generate_chat if registry.is_panel(req.model) else generate_chat
```

Replace both `generate_chat(` calls inside `openai_chat` (the streaming branch at line 377 and the non-streaming branch at line 420) with `dispatch(`. Leave every argument unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_panel.py::TestRouterDispatch -v`
Expected: PASS.

Also run the existing OpenAI router tests to confirm no regression:

Run: `uv run pytest tests/ -k "openai" -q`
Expected: PASS (no new failures vs. baseline).

- [ ] **Step 5: Commit**

```bash
git add olmlx/routers/openai.py tests/test_panel.py
git commit -m "feat(panel): dispatch panel models from the OpenAI router"
```

---

## Task 11: Wire the Ollama router

**Files:**
- Modify: `olmlx/routers/chat.py` (imports + `chat`, lines 163-200 and the non-stream branch)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_panel.py`:

```python
    def test_ollama_router_imports_panel_dispatch(self):
        from olmlx.routers import chat as chat_router

        assert hasattr(chat_router, "panel_generate_chat")
```

(Add this method inside the existing `TestRouterDispatch` class.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_panel.py::TestRouterDispatch::test_ollama_router_imports_panel_dispatch -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Add the import and branch**

In `olmlx/routers/chat.py`, add to the imports (next to `from olmlx.engine.inference import generate_chat`):

```python
from olmlx.engine.panel import panel_generate_chat
```

In the `chat` handler, immediately after `enable_thinking = resolve_think_flag(req.think)` (line 171) add:

```python
    dispatch = (
        panel_generate_chat
        if request.app.state.registry.is_panel(req.model)
        else generate_chat
    )
```

Replace the `generate_chat(` call in the streaming branch (line 178) **and** the corresponding call in the non-streaming branch below it with `dispatch(`. Leave all arguments unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_panel.py::TestRouterDispatch -v`
Expected: PASS.

Run the Ollama chat router tests for regressions:

Run: `uv run pytest tests/ -k "chat and not session and not cli and not tui" -q`
Expected: PASS (no new failures vs. baseline).

- [ ] **Step 5: Commit**

```bash
git add olmlx/routers/chat.py tests/test_panel.py
git commit -m "feat(panel): dispatch panel models from the Ollama router"
```

---

## Task 12: Manual smoke test with stub models

**Files:**
- Read only; no code changes unless a bug surfaces.

- [ ] **Step 1: Create a temporary panel config**

Pick three small models you already have locally (run `uv run olmlx models` to list them) plus a small classifier and a distinct judge. Add a panel entry to your dev `models.json` (`OLMLX_MODELS_CONFIG=models.json`), e.g.:

```jsonc
"smoke-panel": {
  "type": "panel",
  "classifier": "<small-model>",
  "judge": "<distinct-model>",
  "routes": { "default": ["<model-a>", "<model-b>"] }
}
```

Raise `OLMLX_MAX_LOADED_MODELS` to at least 4 (members + judge + classifier) to avoid reload thrash.

- [ ] **Step 2: Start the server**

Run: `OLMLX_MODELS_CONFIG=models.json OLMLX_MAX_LOADED_MODELS=6 uv run olmlx`
Expected: server starts on `http://localhost:11434`, no panel-validation warnings in the log.

- [ ] **Step 3: Non-tool request (judge synthesis path)**

```bash
curl -s localhost:11434/v1/chat/completions -d '{
  "model": "smoke-panel",
  "messages": [{"role": "user", "content": "In one sentence, what is MLX?"}],
  "stream": false
}' | python3 -m json.tool
```

Expected: a single assistant `content` answer; `finish_reason: "stop"`. Server log shows the classifier, both panelists, and the judge each running once.

- [ ] **Step 4: Tool request (union path)**

```bash
curl -s localhost:11434/v1/chat/completions -d '{
  "model": "smoke-panel",
  "stream": false,
  "messages": [{"role": "user", "content": "What files are in the cwd?"}],
  "tools": [{"type": "function", "function": {
     "name": "list_dir", "description": "List a directory",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
}' | python3 -m json.tool
```

Expected: response has `finish_reason: "tool_calls"` and a `tool_calls` array (the deduped union of what the panelists proposed). Feed a fake tool result back as a second request and confirm the loop continues and eventually returns `content`.

- [ ] **Step 5: Commit (only if a fix was needed)**

If steps 3-4 surfaced a bug, write a failing test reproducing it, fix it, then:

```bash
git add -A && git commit -m "fix(panel): <describe the bug found in smoke test>"
```

Otherwise no commit.

---

## Task 13: Documentation

**Files:**
- Modify: `CLAUDE.md` (Non-Obvious Invariants section + Project Structure)

- [ ] **Step 1: Add the project-structure line**

In `CLAUDE.md`, under the `engine/` tree listing, add a line near the other engine modules:

```
│   ├── panel.py        # Multi-model panel + judge coordinator (per-turn reconciler)
```

- [ ] **Step 2: Add the invariant note**

In the "Non-Obvious Invariants" section, add:

```markdown
**Panel coordinator routes through generate_chat** — `engine/panel.py` is a
per-turn reconciler that rides the client's tool loop: it returns a
`generate_chat`-shaped result whose text is either the judge's prose or canonical
Qwen `<tool_call>` blocks, so both routers' existing `parse_model_output` paths
handle it unchanged. Every classifier/panelist/judge call MUST go through
`generate_chat` — never call MLX directly — or the Metal-stream/inference-lock
handling is bypassed. Panels need `max_loaded_models ≥ members + judge + classifier`
to avoid reload thrash.
```

- [ ] **Step 3: Verify markdown renders**

Run: `git diff CLAUDE.md`
Expected: the two additions appear with correct fencing.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): document panel coordinator invariant"
```

---

## Task 14: Full verification & ruff

**Files:** none (verification only).

- [ ] **Step 1: Run the new test modules**

Run: `uv run pytest tests/test_panel.py tests/test_registry_panel.py -v`
Expected: all PASS.

- [ ] **Step 2: Run ruff (required before push)**

Run: `uv run ruff check olmlx/engine/panel.py olmlx/engine/registry.py olmlx/routers/openai.py olmlx/routers/chat.py tests/test_panel.py tests/test_registry_panel.py && uv run ruff format olmlx/engine/panel.py olmlx/engine/registry.py tests/test_panel.py tests/test_registry_panel.py`
Expected: no lint errors; formatting clean.

- [ ] **Step 3: Run the registry + router suites for regressions**

Run: `uv run pytest tests/test_registry.py tests/ -k "openai or (chat and not session and not cli and not tui)" -q`
Expected: PASS (no new failures vs. baseline; note the known full-suite SIGABRT flake — trust targeted suites + CI).

- [ ] **Step 4: Commit any ruff fixups**

```bash
git add -A && git commit -m "style(panel): ruff" || echo "nothing to format"
```

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** entry surface (Tasks 10-11), per-turn reconciler (Tasks 7-9), client-executed tools via Qwen-block round-trip (Tasks 4, 8-9), per-task routing (Tasks 1-2, 6), judge synthesis without tools (Tasks 5, 8), `models.json` config + validation (Tasks 1-2), stop-when-no-tools (Task 7 `merge`), fail-the-request (Tasks 6-7 propagate; Task 8 `_resolve_panel` raises), CLAUDE.md invariant (Task 13).
- **Out of scope (no task, by design):** distributed/parallel execution; adding panels to `/v1/models` listing (follow-up — `list_models` returns `ModelConfig`, panels are a different type).
- **Type consistency:** tool_use dicts use `{"name", "input"}` throughout; `serialize_tool_calls_qwen` writes the `arguments` wire key that `parse_model_output` maps back to `input`. `panel_generate_chat` mirrors `generate_chat`'s exact signature so the router `dispatch = ... if ... else generate_chat` swap is safe.
- **Known caveat to watch in Task 12:** every tool turn re-runs classifier + all N panelists (final turn adds the judge) — sequential under the lock. Expected and accepted; just confirm latency is sane for your panel size.
