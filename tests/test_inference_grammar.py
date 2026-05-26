"""Engine-side tests for grammar plumbing (issue #361).

Verifies that ``_install_grammar_processor`` installs a logits processor
onto ``gen_kwargs`` for supported model types, refuses VLM / distributed,
and that ``generate_chat`` forwards ``grammar_spec`` end-to-end.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from olmlx.engine.grammar import GrammarSpec
from olmlx.engine.inference import (
    _install_grammar_processor,
    _resolve_model_vocab_size,
)


def _make_lm(
    *, is_vlm: bool = False, is_distributed: bool = False, vocab_size: int = 1024
):
    """Stand-in LoadedModel exposing only the attributes the helper reads."""
    lm = MagicMock()
    lm.is_vlm = is_vlm
    lm.is_distributed = is_distributed
    lm.text_tokenizer = MagicMock()
    # The grammar module reads ``model.args.vocab_size`` first.
    lm.model.args.vocab_size = vocab_size
    return lm


class TestResolveModelVocabSize:
    def test_reads_from_args(self):
        lm = _make_lm(vocab_size=32000)
        assert _resolve_model_vocab_size(lm) == 32000

    def test_falls_back_to_embed_weight_shape(self):
        lm = MagicMock()
        # No args.vocab_size — clear the auto-MagicMock attribute.
        lm.model.args = None
        # No lm_head either, so fallback to embed_tokens.
        del lm.model.lm_head
        del lm.model.model.lm_head
        lm.model.model.embed_tokens.weight.shape = (12345, 768)
        # Also clear top-level embed_tokens to force the model.model.embed_tokens path.
        del lm.model.embed_tokens
        assert _resolve_model_vocab_size(lm) == 12345

    def test_prefers_nested_lm_head_over_top_embed_tokens(self):
        """If ``model.lm_head`` is missing but ``model.embed_tokens`` is
        present, and ``model.model.lm_head`` exists at the deeper level,
        the helper should still prefer the deeper lm_head. Round-3 review
        flagged the prior owner-first traversal as ordering-sensitive."""
        lm = MagicMock()
        lm.model.args = None
        del lm.model.lm_head  # no top-level lm_head
        lm.model.embed_tokens.weight.shape = (151_643, 4096)  # tokenizer dim
        lm.model.model.lm_head.weight.shape = (151_936, 4096)  # padded out dim
        # Attr-first traversal: lm_head at every depth before any
        # embed_tokens — so the nested lm_head wins.
        assert _resolve_model_vocab_size(lm) == 151_936

    def test_prefers_lm_head_over_embed_tokens(self):
        """For untied or expanded vocab heads the lm_head output dim is
        larger than the embed_tokens input dim. xgrammar sizes the
        bitmask to the supplied vocab_size — if we returned the
        embed_tokens shape we'd undersize the mask and let the tail of
        the vocab through unmasked (review #384, bug 3)."""
        lm = MagicMock()
        lm.model.args = None
        # Top-level lm_head (some architectures wire it this way).
        lm.model.lm_head.weight.shape = (151_936, 4096)  # padded lm_head
        lm.model.model.embed_tokens.weight.shape = (151_643, 4096)  # tokenizer vocab
        # Helper iterates owners in order (model, model.model), and attrs
        # in order (lm_head, embed_tokens), so lm_head wins.
        assert _resolve_model_vocab_size(lm) == 151_936

    def test_returns_none_when_undiscoverable(self):
        lm = MagicMock()
        lm.model.args = None
        # Strip all of the attributes the helper checks at both nesting
        # levels (model.lm_head, model.embed_tokens, model.model.lm_head,
        # model.model.embed_tokens) so the helper returns None.
        for owner in (lm.model, lm.model.model):
            for attr in ("lm_head", "embed_tokens"):
                try:
                    delattr(owner, attr)
                except AttributeError:
                    pass
        assert _resolve_model_vocab_size(lm) is None


class TestInstallGrammarProcessor:
    def test_no_spec_returns_false_and_leaves_kwargs(self):
        lm = _make_lm()
        gen_kwargs: dict = {}
        assert _install_grammar_processor(lm, gen_kwargs, None) is False
        assert "logits_processors" not in gen_kwargs

    def test_vlm_warns_and_skips(self, caplog):
        lm = _make_lm(is_vlm=True)
        gen_kwargs: dict = {}
        with caplog.at_level("WARNING", logger="olmlx.engine.inference"):
            installed = _install_grammar_processor(
                lm, gen_kwargs, GrammarSpec("json_object")
            )
        assert installed is False
        assert "logits_processors" not in gen_kwargs
        assert "VLM" in caplog.text

    def test_has_tools_warns_and_skips(self, caplog):
        """Grammar + tools is broken: the JSON grammar masks tool-call
        tokens, so the model can't emit a tool call (review #384, bug 4)."""
        lm = _make_lm()
        gen_kwargs: dict = {}
        with caplog.at_level("WARNING", logger="olmlx.engine.inference"):
            installed = _install_grammar_processor(
                lm, gen_kwargs, GrammarSpec("json_object"), has_tools=True
            )
        assert installed is False
        assert "logits_processors" not in gen_kwargs
        assert "tool" in caplog.text.lower()

    def test_distributed_warns_and_skips(self, caplog):
        lm = _make_lm(is_distributed=True)
        gen_kwargs: dict = {}
        with caplog.at_level("WARNING", logger="olmlx.engine.inference"):
            installed = _install_grammar_processor(
                lm, gen_kwargs, GrammarSpec("json_object")
            )
        assert installed is False
        assert "logits_processors" not in gen_kwargs
        assert "distributed" in caplog.text

    def test_unresolvable_vocab_warns_and_skips(self, caplog):
        lm = _make_lm()
        # Force the resolver to return None.
        with patch(
            "olmlx.engine.inference._resolve_model_vocab_size", return_value=None
        ):
            with caplog.at_level("WARNING", logger="olmlx.engine.inference"):
                installed = _install_grammar_processor(
                    lm, {}, GrammarSpec("json_object")
                )
        assert installed is False
        assert "vocab_size" in caplog.text

    def test_installs_processor_for_text_model(self):
        lm = _make_lm(vocab_size=1024)
        gen_kwargs: dict = {}
        sentinel = MagicMock(name="grammar_processor")
        with patch(
            "olmlx.engine.inference._make_grammar_processor", return_value=sentinel
        ) as mock_make:
            installed = _install_grammar_processor(
                lm, gen_kwargs, GrammarSpec("json_object")
            )
        assert installed is True
        mock_make.assert_called_once()
        # The new processor must be appended (not replace existing penalty
        # processors — those should still apply alongside).
        assert gen_kwargs["logits_processors"] == [sentinel]

    def test_appends_to_existing_processors(self):
        """A freq/presence penalty processor may already be present; the
        grammar processor must be appended, not replace it."""
        lm = _make_lm(vocab_size=1024)
        existing = MagicMock(name="freq_penalty")
        gen_kwargs: dict = {"logits_processors": [existing]}
        sentinel = MagicMock(name="grammar_processor")
        with patch(
            "olmlx.engine.inference._make_grammar_processor", return_value=sentinel
        ):
            installed = _install_grammar_processor(
                lm, gen_kwargs, GrammarSpec("json_object")
            )
        assert installed is True
        assert gen_kwargs["logits_processors"] == [existing, sentinel]

    def test_unwraps_mlx_lm_tokenizer_wrapper(self):
        """mlx-lm wraps the HF tokenizer in ``TokenizerWrapper`` which
        xgrammar's ``TokenizerInfo.from_huggingface`` rejects with
        'Unsupported tokenizer type'. ``_install_grammar_processor``
        must peel the wrapper via its ``_tokenizer`` attribute before
        handing off to the factory.

        The unwrap is gated on the outer class name (the real wrapper is
        ``mlx_lm.tokenizer_utils.TokenizerWrapper``) — HF fast tokenizers
        also expose ``_tokenizer`` holding the Rust core, and over-eager
        peeling there would hand xgrammar an unsupported type. Uses a
        real class with the matching name to exercise the gate."""

        class TokenizerWrapper:
            def __init__(self, inner):
                self._tokenizer = inner

        lm = _make_lm(vocab_size=1024)
        hf_tokenizer = MagicMock(name="hf_tokenizer")
        wrapper = TokenizerWrapper(hf_tokenizer)
        lm.text_tokenizer = wrapper

        sentinel = MagicMock(name="grammar_processor")
        with patch(
            "olmlx.engine.inference._make_grammar_processor", return_value=sentinel
        ) as mock_make:
            installed = _install_grammar_processor(lm, {}, GrammarSpec("json_object"))

        assert installed is True
        args, _ = mock_make.call_args
        assert args[0] is hf_tokenizer

    def test_does_not_peel_hf_fast_tokenizer_rust_core(self):
        """Regression: an HF fast tokenizer also exposes ``_tokenizer``
        holding the Rust ``tokenizers.Tokenizer`` core. Peeling there
        would hand xgrammar a type it rejects. The gate is on the outer
        class name (``TokenizerWrapper``), not on attribute presence."""

        class PreTrainedTokenizerFast:  # not the real one, just the name
            def __init__(self):
                self._tokenizer = MagicMock(name="rust_core")

        lm = _make_lm(vocab_size=1024)
        bare = PreTrainedTokenizerFast()
        lm.text_tokenizer = bare

        sentinel = MagicMock(name="grammar_processor")
        with patch(
            "olmlx.engine.inference._make_grammar_processor", return_value=sentinel
        ) as mock_make:
            _install_grammar_processor(lm, {}, GrammarSpec("json_object"))

        args, _ = mock_make.call_args
        # The HF tokenizer is forwarded as-is — the Rust core is NOT
        # extracted.
        assert args[0] is bare


class TestGenerateChatForwardsGrammarSpec:
    """Smoke: ``generate_chat`` passes ``grammar_spec`` into the install
    helper, and a non-None spec sets ``grammar_active`` on the downstream
    call. Mock the install helper and the downstream paths to avoid
    real generation."""

    @pytest.mark.asyncio
    async def test_grammar_spec_reaches_install_helper(self, mock_manager):
        from olmlx.engine.inference import generate_chat

        spec = GrammarSpec("json_object")
        # Stub the install helper so we can observe the call; we don't
        # actually need to install anything to verify it was invoked
        # with our spec.
        captured: dict = {}

        def fake_install(lm, gen_kwargs, grammar_spec, *, has_tools=False):
            captured["spec"] = grammar_spec
            captured["has_tools"] = has_tools
            return grammar_spec is not None

        with patch(
            "olmlx.engine.inference._install_grammar_processor",
            side_effect=fake_install,
        ):
            # Stub the downstream completion so we don't actually generate.
            async def fake_full(*args, **kwargs):
                captured["grammar_active"] = kwargs.get("grammar_active")
                return {"text": "{}", "done": True, "stats": None}

            with patch(
                "olmlx.engine.inference._full_completion", side_effect=fake_full
            ):
                await generate_chat(
                    mock_manager,
                    "qwen3:latest",
                    [{"role": "user", "content": "hi"}],
                    options={},
                    stream=False,
                    grammar_spec=spec,
                )

        assert captured["spec"] is spec
        assert captured["grammar_active"] is True
