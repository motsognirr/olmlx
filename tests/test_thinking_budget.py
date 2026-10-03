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

    def test_partial_close_at_boundary_is_completed_not_restarted(self):
        """A multi-token close already begun at the budget boundary must be
        finished from where it stopped, not re-forced from its first token
        (which would emit a duplicated/malformed marker)."""
        proc = _make_thinking_budget_processor(
            3, (OPEN,), (CLOSE, 10), initially_open=True
        )
        outs = _drive(proc, [7, 7, CLOSE, 10, 7])
        assert _forced_id(outs[3]) == 10
        assert _forced_id(outs[4]) is None
        assert _forced_id(outs[5]) is None

    def test_partial_open_at_boundary_is_rechecked(self):
        """A multi-token opener (Gemma-4 ``<|channel>thought``) straddling the
        budget boundary must not make the processor go inert: once the opener
        completes the block is open and must be closed."""
        proc = _make_thinking_budget_processor(
            2, (OPEN, 11), (CLOSE,), initially_open=False
        )
        outs = _drive(proc, [7, OPEN, 11, 7])
        assert _forced_id(outs[2]) is None  # tail ends in a partial opener
        assert _forced_id(outs[3]) == CLOSE  # opener completed -> force close
        assert _forced_id(outs[4]) is None

    def test_partial_open_prefix_that_never_completes_goes_inert(self):
        proc = _make_thinking_budget_processor(
            2, (OPEN, 11), (CLOSE,), initially_open=False
        )
        outs = _drive(proc, [7, OPEN, 7, 7, 7])
        assert [_forced_id(o) for o in outs] == [None] * 6

    def test_batched_token_buffer_history(self):
        """Drive the processor exactly as mlx-lm's ``GenerationBatch._step``
        does: a per-sequence ``TokenBuffer`` seeded with the prompt, updated
        with the previous step's input token, whose fetched 1-D array is the
        history each processor call sees."""
        from mlx_lm.models.cache import TokenBuffer

        proc = _make_thinking_budget_processor(
            3, (OPEN,), (CLOSE,), initially_open=True
        )
        buf = TokenBuffer([1, 2, 3])
        inputs = mx.array([3])  # the last prompt token feeds the first step
        generated = [7, 7, 7]
        outs = []
        for k in range(len(generated) + 1):
            history = buf.update_and_fetch(inputs)
            outs.append(proc(history, _logits()))
            if k < len(generated):
                inputs = mx.array([generated[k]])
        assert [_forced_id(o) for o in outs[:3]] == [None, None, None]
        assert _forced_id(outs[3]) == CLOSE

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

    def test_non_int_marker_ids_degrade_to_not_enforced(self):
        """A marker-shape mismatch from the third-party wrapper must degrade
        to "no markers" (budget not enforced), not raise out of every
        budget-carrying request."""
        tok = MagicMock()
        tok.think_start_tokens = ("<think>",)
        tok.think_end_tokens = (None,)
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

    @pytest.mark.parametrize("attr", ["is_distributed", "is_speculative"])
    def test_unsupported_model_kinds_do_not_warn(self, attr, caplog):
        """Anthropic clients (Claude Code) send a budget on every request, so
        a per-request WARNING for an unenforceable model kind is spam the
        operator can't act on."""
        import logging

        with caplog.at_level(logging.DEBUG, logger="olmlx"):
            self._install(_lm(**{attr: True}), {})
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("not enforced" in r.getMessage() for r in caplog.records)

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

    def test_thinking_expected_without_open_marker_is_not_forced(self):
        """``thinking_expected`` is a request-level flag, not evidence that a
        think block is open in the output. A model that answers directly
        (no opener in the prompt tail or the generation) must not get a
        close marker forced into the middle of its answer."""
        gk = {}
        assert self._install(
            _lm(), gk, budget=1, thinking_expected=True, prompt="hello"
        )
        proc = gk["logits_processors"][-1]
        proc(mx.array([1]), _logits())
        assert _forced_id(proc(mx.array([1, 7]), _logits())) is None

    def test_thinking_expected_model_emitted_open_is_forced(self):
        """Templates that leave the opener to the model (Qwen3 hybrid,
        GLM) still get enforced once the model emits it."""
        gk = {}
        assert self._install(
            _lm(), gk, budget=2, thinking_expected=True, prompt="hello"
        )
        proc = gk["logits_processors"][-1]
        proc(mx.array([1]), _logits())
        proc(mx.array([1, OPEN]), _logits())
        assert _forced_id(proc(mx.array([1, OPEN, 7]), _logits())) == CLOSE

    def test_uninspectable_prompt_falls_back_to_thinking_expected(self):
        """With no string prompt to inspect, keep the request-level flag."""
        gk = {}
        assert self._install(_lm(), gk, budget=1, thinking_expected=True, prompt=None)
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


class TestBudgetMustBeBelowMaxTokens:
    """#743: like Anthropic's API, ``budget_tokens >= max_tokens`` is a 400
    ``invalid_request_error`` — rejected at the schema layer, before any model
    load or generation (streaming and non-streaming alike). olmlx does not add
    Anthropic's 1024 minimum: small budgets are legitimate for local models."""

    def _req(self, **kw):
        from olmlx.schemas.anthropic import AnthropicMessagesRequest

        return AnthropicMessagesRequest(
            model="qwen3",
            messages=[{"role": "user", "content": "hi"}],
            **kw,
        )

    @pytest.mark.parametrize("budget", [100, 101, 10000])
    def test_schema_rejects_budget_at_or_above_max_tokens(self, budget):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="budget_tokens"):
            self._req(
                max_tokens=100,
                thinking={"type": "enabled", "budget_tokens": budget},
            )

    def test_schema_accepts_budget_below_max_tokens(self):
        req = self._req(
            max_tokens=100, thinking={"type": "enabled", "budget_tokens": 99}
        )
        assert req.thinking.budget_tokens == 99

    def test_schema_accepts_small_budget(self):
        """No 1024 minimum (unlike Anthropic) — local models use small budgets."""
        req = self._req(
            max_tokens=4096, thinking={"type": "enabled", "budget_tokens": 16}
        )
        assert req.thinking.budget_tokens == 16

    def test_disabled_thinking_budget_not_checked(self):
        """A budget alongside ``type: disabled`` is inert, so it isn't a 400."""
        req = self._req(
            max_tokens=100, thinking={"type": "disabled", "budget_tokens": 5000}
        )
        assert req.thinking.type == "disabled"

    @pytest.mark.parametrize("thinking_type", ["adaptive", "future-mode"])
    def test_non_enabled_thinking_budget_not_checked(self, thinking_type):
        """Anthropic's budget < max_tokens rule is defined for
        ``type: enabled`` only. Adaptive and unknown (forward-compat) types
        must not be 400'd on it — such a budget can never trigger before
        ``max_tokens`` anyway, so rejecting it would only break clients."""
        req = self._req(
            max_tokens=100,
            thinking={"type": thinking_type, "budget_tokens": 5000},
        )
        assert req.thinking.type == thinking_type

    def test_count_tokens_not_checked(self):
        """count_tokens takes no max_tokens (it defaults to a placeholder 1),
        so a thinking budget there must not be compared against it."""
        from olmlx.schemas.anthropic import AnthropicCountTokensRequest

        req = AnthropicCountTokensRequest(
            model="qwen3",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "enabled", "budget_tokens": 5000},
        )
        assert req.thinking.budget_tokens == 5000

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_router_returns_400_before_generation(self, app_client, stream):
        with patch(
            "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1024,
                    "stream": stream,
                    "thinking": {"type": "enabled", "budget_tokens": 1024},
                },
            )
        assert resp.status_code == 400
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"
        msg = data["error"]["message"]
        assert "budget_tokens" in msg and "max_tokens" in msg
        mock_gen.assert_not_called()
