"""Live test for mlx-community 'gemma4_unified' text loading.

Loads the REAL ``gemma-4-12B-it-4bit`` checkpoint and verifies that olmlx
loads its language tower via mlx-lm's ``gemma4_text`` module and generates
coherent text.  This repo is an outlier: its top-level model_type is
``gemma4_unified`` and its vision tower is stored under ``vision_embedder.*``,
a layout that loads in neither mlx-lm's ``gemma4_text`` nor mlx-vlm 0.4.4's
``gemma4`` module.  olmlx drops the multimodal weights and loads the language
tower for text inference.

Lives OUTSIDE ``tests/integration/`` on purpose: that package's autouse
``mock_mlx_primitives`` fixture patches ``mlx_lm.load`` etc., so any test there
runs against mocks.  Here only the top-level ``tests/conftest`` applies.

Skipped in CI via ``-m "not real_model"``, and additionally skipped when the
model is not already downloaded so it never triggers a multi-GB download.
"""

import pytest

from olmlx.config import settings

MODEL = "mlx-community/gemma-4-12B-it-4bit"


def _model_dir():
    from olmlx.models.store import _safe_dir_name

    return settings.models_dir / _safe_dir_name(MODEL)


def _model_present() -> bool:
    return (_model_dir() / "config.json").exists()


pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        not _model_present(),
        reason=f"{MODEL} not downloaded in {settings.models_dir}",
    ),
]


def test_loads_language_tower_and_generates():
    import mlx_lm

    from olmlx.engine.model_manager import _load_with_model_type_fallback

    load_path = str(_model_dir())
    model, tokenizer = _load_with_model_type_fallback(mlx_lm, load_path)

    # mlx-lm gemma4_text module, not an mlx-vlm wrapper.
    assert type(model).__module__.endswith("gemma4_text")

    # The full eos set must be threaded in: <eos> (1), <turn|> (106, turn
    # terminator) and <|tool_response> (50, tool-call boundary).  Without 106
    # and 50 generation runs past every turn and degenerates into a repetition
    # loop (the originally reported symptom).
    eos_ids = set(getattr(tokenizer, "eos_token_ids", None) or [])
    assert {1, 106, 50} <= eos_ids

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with exactly: hello world"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    out = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=24, verbose=False)
    assert isinstance(out, str) and out.strip()


def test_generation_terminates_without_repetition_loop():
    """Regression for the runaway loop: with the turn terminator (<turn|>, id
    106) in the stop set, a normal chat turn must terminate on its own well
    before a generous max_tokens, rather than spinning on empty thought
    channels until a length cap.
    """
    import mlx_lm

    from olmlx.engine.model_manager import _load_with_model_type_fallback

    model, tokenizer = _load_with_model_type_fallback(mlx_lm, str(_model_dir()))
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2? Answer in one short sentence."}],
        add_generation_prompt=True,
        tokenize=False,
    )
    out = mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=512, verbose=False
    )
    # A self-terminated turn is far shorter than the cap and is not a tail of
    # repeated thought-channel openers.
    assert len(tokenizer.encode(out)) < 400
    assert out.count("<|channel>thought") <= 1


def test_load_does_not_mutate_config_on_disk():
    """The store's config.json must remain 'gemma4_unified' — the loader reads
    it but never rewrites it (a prior approach corrupted the shared copy).
    """
    import json

    import mlx_lm

    from olmlx.engine.model_manager import _load_with_model_type_fallback

    config_path = _model_dir() / "config.json"
    before = config_path.read_text()
    assert json.loads(before)["model_type"] == "gemma4_unified"

    _load_with_model_type_fallback(mlx_lm, str(_model_dir()))

    assert config_path.read_text() == before


def test_session_tracker_splits_real_gemma4_turn():
    """Feed a real gemma-4 thinking+tool-call turn through ThinkingTracker and
    assert the visible channel is free of channel/tool markup while the raw
    accumulated text still carries the tool call for parsing."""
    import mlx_lm

    from olmlx.chat.session import ThinkingTracker
    from olmlx.engine.model_manager import _load_with_model_type_fallback
    from olmlx.engine.tool_parser import parse_model_output

    model, tokenizer = _load_with_model_type_fallback(mlx_lm, str(_model_dir()))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What's the weather in Paris?"}],
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    raw = mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=256, verbose=False
    )

    tracker = ThinkingTracker(template_has_thinking=True)
    visible_parts = []
    # Feed token-by-token to exercise chunk-boundary handling.
    for ch in raw:
        _td, vd, _te, _ts = tracker.feed(ch)
        if vd:
            visible_parts.append(vd)
    f_think, f_visible, _started = tracker.flush()
    if f_visible:
        visible_parts.append(f_visible)
    visible = "".join(visible_parts)

    assert "<|channel" not in visible
    assert "<channel|" not in visible
    assert "<|tool_call" not in visible
    # Raw text still parses into a tool call.
    _thinking, _vis, tool_uses = parse_model_output(
        tracker.accumulated, has_tools=True, thinking_expected=True
    )
    assert tool_uses and tool_uses[0]["name"] == "get_weather"
