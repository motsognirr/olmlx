"""Anthropic ``thinking.budget_tokens`` enforcement (issue #716).

``budget_tokens`` used to be read, logged, and dropped, so a thinking model
could spend the whole ``max_tokens`` inside ``<think>`` and never answer. It is
now enforced by a request-scoped logits processor that, once ~budget tokens
have been generated with a think block still open, forces the tokenizer's
think-close sequence and then steps aside so the model writes its answer.
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from olmlx.engine.logits_processors import (
    _install_thinking_budget_processor,
    _make_thinking_budget_processor,
    _resolve_think_markers,
)
from olmlx.utils.timing import TimingStats

OPEN = 5
CLOSE = 9
VOCAB = 16


def _logits() -> mx.array:
    return mx.zeros((1, VOCAB))


def _forced_id(out: mx.array) -> int | None:
    """Return the single finite position when *out* forces one token."""
    row = out[0].tolist()
    finite = [i for i, v in enumerate(row) if not math.isinf(v)]
    return finite[0] if len(finite) == 1 else None


def _drive(proc, generated: list[int], *, as_list: bool = False):
    """Call *proc* the way generate_step does: call k sees k generated tokens."""
    history = [1, 2, 3]  # prompt tail
    outs = []
    for k in range(len(generated) + 1):
        toks = history + generated[:k]
        outs.append(proc(toks if as_list else mx.array(toks), _logits()))
    return outs


class TestThinkingBudgetProcessor:
    def test_passthrough_below_budget(self):
        proc = _make_thinking_budget_processor(
            4, (OPEN,), (CLOSE,), initially_open=True
        )
        outs = _drive(proc, [7, 7, 7])
        assert all(_forced_id(o) is None for o in outs)

    def test_forces_close_at_budget_when_thinking_open(self):
        proc = _make_thinking_budget_processor(
            4, (OPEN,), (CLOSE,), initially_open=True
        )
        outs = _drive(proc, [7, 7, 7, 7])
        assert _forced_id(outs[4]) == CLOSE

    def test_inert_after_forcing(self):
        proc = _make_thinking_budget_processor(
            2, (OPEN,), (CLOSE,), initially_open=True
        )
        outs = _drive(proc, [7, 7, CLOSE, 7, 7])
        assert _forced_id(outs[2]) == CLOSE
        assert all(_forced_id(o) is None for o in outs[3:])

    def test_natural_close_before_budget_is_not_forced(self):
        proc = _make_thinking_budget_processor(
            4, (OPEN,), (CLOSE,), initially_open=True
        )
        outs = _drive(proc, [7, CLOSE, 7, 7, 7, 7, 7])
        assert all(_forced_id(o) is None for o in outs)

    def test_model_emitted_open_is_tracked(self):
        """Qwen3-style: the model emits ``<think>`` itself."""
        proc = _make_thinking_budget_processor(
            3, (OPEN,), (CLOSE,), initially_open=False
        )
        outs = _drive(proc, [OPEN, 7, 7])
        assert _forced_id(outs[3]) == CLOSE

    def test_no_thinking_is_never_forced(self):
        proc = _make_thinking_budget_processor(
            2, (OPEN,), (CLOSE,), initially_open=False
        )
        outs = _drive(proc, [7, 7, 7, 7, 7])
        assert all(_forced_id(o) is None for o in outs)

    def test_multi_token_close_is_forced_in_order(self):
        """Gemma-4 style multi-token close (``<channel|>``)."""
        proc = _make_thinking_budget_processor(
            2, (OPEN,), (CLOSE, 10), initially_open=True
        )
        outs = _drive(proc, [7, 7, CLOSE, 10, 7])
        assert _forced_id(outs[2]) == CLOSE
        assert _forced_id(outs[3]) == 10
        assert _forced_id(outs[4]) is None
        assert _forced_id(outs[5]) is None

    def test_list_history(self):
        """Batched path hands processors a Python-list-like history."""
        proc = _make_thinking_budget_processor(
            2, (OPEN,), (CLOSE,), initially_open=True
        )
        outs = _drive(proc, [7, 7], as_list=True)
        assert _forced_id(outs[2]) == CLOSE

    def test_zero_budget_forces_first_token(self):
        proc = _make_thinking_budget_processor(
            0, (OPEN,), (CLOSE,), initially_open=True
        )
        outs = _drive(proc, [])
        assert _forced_id(outs[0]) == CLOSE

    def test_forced_logits_keep_model_forward_in_graph(self):
        """The forced row must still depend on the model's logits, or the
        step's forward (and its KV-cache write) is left an unevaluated lazy
        graph that can escape the generation worker thread."""
        proc = _make_thinking_budget_processor(
            0, (OPEN,), (CLOSE,), initially_open=True
        )
        # A logits row that already masks CLOSE (e.g. an earlier processor)
        # must still produce a finite, sampleable row.
        logits = mx.full((1, VOCAB), -mx.inf)
        out = proc(mx.array([1]), logits)
        assert _forced_id(out) == CLOSE
        lp = out - mx.logsumexp(out, keepdims=True)
        assert not math.isnan(lp[0, CLOSE].item())


class _ToyModel(nn.Module):
    """Always prefers token 7 — i.e. would think forever."""

    def __init__(self):
        super().__init__()
        self.bias = mx.zeros((VOCAB,))

    def make_cache(self):
        return []

    def __call__(self, inputs, cache=None, **kwargs):
        B, L = inputs.shape
        row = mx.zeros((VOCAB,)) + self.bias
        row = mx.where(mx.arange(VOCAB) == 7, 10.0, row)
        return mx.broadcast_to(row, (B, L, VOCAB))


class TestRealGenerateStep:
    """Drive mlx-lm's real ``generate_step`` to lock the call contract
    (one processor call per sampled token, mx.array history)."""

    @pytest.mark.parametrize("close", [(CLOSE,), (CLOSE, 10)])
    def test_close_injected_after_budget(self, close):
        from mlx_lm.generate import generate_step

        proc = _make_thinking_budget_processor(4, (OPEN,), close, initially_open=True)
        out = [
            tok
            for tok, _ in generate_step(
                mx.array([1, 2, 3]),
                _ToyModel(),
                max_tokens=10,
                logits_processors=[proc],
            )
        ]
        assert out[:4] == [7, 7, 7, 7]
        assert tuple(out[4 : 4 + len(close)]) == close
        assert all(t == 7 for t in out[4 + len(close) :])


class TestResolveThinkMarkers:
    def test_mlx_wrapper_attributes(self):
        tok = MagicMock()
        tok.think_start_tokens = (OPEN,)
        tok.think_end_tokens = (CLOSE, 10)
        assert _resolve_think_markers(tok) == ((OPEN,), (CLOSE, 10))

    def test_vocab_fallback(self):
        tok = MagicMock(spec=["get_vocab"])
        tok.get_vocab.return_value = {"<think>": 11, "</think>": 12, "a": 1}
        assert _resolve_think_markers(tok) == ((11,), (12,))

    def test_no_markers(self):
        tok = MagicMock(spec=["get_vocab"])
        tok.get_vocab.return_value = {"a": 1}
        assert _resolve_think_markers(tok) is None

    def test_magicmock_attrs_not_mistaken_for_markers(self):
        tok = MagicMock()
        tok.get_vocab.return_value = {}
        assert _resolve_think_markers(tok) is None


def _lm(**over):
    lm = MagicMock()
    lm.is_distributed = False
    lm.is_speculative = False
    lm.text_tokenizer.think_start_tokens = (OPEN,)
    lm.text_tokenizer.think_end_tokens = (CLOSE,)
    lm.text_tokenizer.decode.return_value = "<think>"
    for k, v in over.items():
        setattr(lm, k, v)
    return lm


class TestInstallThinkingBudget:
    def _install(self, lm, gen_kwargs, budget=10, **kw):
        kw.setdefault("thinking_expected", True)
        kw.setdefault("prompt", "hello")
        kw.setdefault("grammar_active", False)
        return _install_thinking_budget_processor(lm, gen_kwargs, budget, **kw)

    def test_installs_and_preserves_existing(self):
        existing = MagicMock()
        gk = {"logits_processors": [existing]}
        assert self._install(_lm(), gk) is True
        assert gk["logits_processors"][0] is existing
        assert len(gk["logits_processors"]) == 2

    def test_none_budget(self):
        gk = {}
        assert self._install(_lm(), gk, budget=None) is False
        assert "logits_processors" not in gk

    def test_negative_budget(self):
        gk = {}
        assert self._install(_lm(), gk, budget=-1) is False

    @pytest.mark.parametrize("attr", ["is_distributed", "is_speculative"])
    def test_unsupported_model_kinds(self, attr):
        gk = {}
        assert self._install(_lm(**{attr: True}), gk) is False
        assert "logits_processors" not in gk

    def test_grammar_active_skips(self):
        gk = {}
        assert self._install(_lm(), gk, grammar_active=True) is False

    def test_no_markers_skips(self):
        lm = _lm()
        lm.text_tokenizer = MagicMock(spec=["get_vocab"])
        lm.text_tokenizer.get_vocab.return_value = {}
        gk = {}
        assert self._install(lm, gk) is False

    def test_prompt_ending_in_open_starts_open(self):
        """DeepSeek-R1-style templates end the prompt with ``<think>\\n``
        even when the template has no enable_thinking switch."""
        gk = {}
        assert self._install(
            _lm(), gk, budget=1, thinking_expected=False, prompt="x<think>\n"
        )
        proc = gk["logits_processors"][-1]
        proc(mx.array([1]), _logits())
        assert _forced_id(proc(mx.array([1, 7]), _logits())) == CLOSE


class TestGenerateChatWiring:
    @pytest.mark.asyncio
    async def test_budget_installs_processor_downstream(self, mock_manager):
        from olmlx.engine.inference import generate_chat

        mock_manager._loaded["qwen3:latest"].tokenizer.think_start_tokens = (OPEN,)
        mock_manager._loaded["qwen3:latest"].tokenizer.think_end_tokens = (CLOSE,)
        captured: dict = {}

        async def fake_full(lm, prompt, mt, gen_kwargs, *args, **kwargs):
            captured["procs"] = list(gen_kwargs.get("logits_processors", []))
            return {"text": "ok", "done": True, "stats": None}

        with patch("olmlx.engine.inference._full_completion", side_effect=fake_full):
            await generate_chat(
                mock_manager,
                "qwen3:latest",
                [{"role": "user", "content": "hi"}],
                options={"repeat_penalty": 1.0},
                stream=False,
                enable_thinking=True,
                thinking_budget=16,
            )
        assert any(
            getattr(p, "__name__", "") == "thinking_budget_processor"
            for p in captured["procs"]
        )

    @pytest.mark.asyncio
    async def test_no_budget_no_processor(self, mock_manager):
        from olmlx.engine.inference import generate_chat

        mock_manager._loaded["qwen3:latest"].tokenizer.think_start_tokens = (OPEN,)
        mock_manager._loaded["qwen3:latest"].tokenizer.think_end_tokens = (CLOSE,)
        captured: dict = {}

        async def fake_full(lm, prompt, mt, gen_kwargs, *args, **kwargs):
            captured["procs"] = list(gen_kwargs.get("logits_processors", []))
            return {"text": "ok", "done": True, "stats": None}

        with patch("olmlx.engine.inference._full_completion", side_effect=fake_full):
            await generate_chat(
                mock_manager,
                "qwen3:latest",
                [{"role": "user", "content": "hi"}],
                stream=False,
                enable_thinking=True,
            )
        assert not any(
            getattr(p, "__name__", "") == "thinking_budget_processor"
            for p in captured["procs"]
        )


class TestAnthropicRouterForwardsBudget:
    @pytest.mark.asyncio
    async def test_non_streaming(self, app_client):
        with patch(
            "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = {
                "text": "Hi",
                "done": True,
                "stats": TimingStats(eval_count=1),
            }
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "thinking": {"type": "enabled", "budget_tokens": 50},
                },
            )
        assert resp.status_code == 200
        assert mock_gen.call_args.kwargs.get("thinking_budget") == 50

    @pytest.mark.asyncio
    async def test_streaming(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "Hello", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats(eval_count=1)}

            return gen()

        with patch(
            "olmlx.routers.anthropic.generate_chat", side_effect=mock_stream
        ) as mock_gen:
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "stream": True,
                    "thinking": {"type": "enabled", "budget_tokens": 20},
                },
            )
        assert resp.status_code == 200
        assert mock_gen.call_args.kwargs.get("thinking_budget") == 20

    @pytest.mark.asyncio
    async def test_no_thinking_no_budget(self, app_client):
        with patch(
            "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = {
                "text": "Hi",
                "done": True,
                "stats": TimingStats(eval_count=1),
            }
            await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                },
            )
        assert mock_gen.call_args.kwargs.get("thinking_budget") is None
