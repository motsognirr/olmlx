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
        lm.model.model.embed_tokens.weight.shape = (12345, 768)
        assert _resolve_model_vocab_size(lm) == 12345

    def test_returns_none_when_undiscoverable(self):
        lm = MagicMock()
        lm.model.args = None
        # Remove embed_tokens so the fallback fails too.
        del lm.model.model.embed_tokens
        del lm.model.embed_tokens
        # Without a model.model.embed_tokens or model.embed_tokens, helper
        # returns None.
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
        'Unsupported tokenizer type'. ``_install_grammar_processor`` must
        peel the wrapper via its ``_tokenizer`` attribute before handing
        off to the factory."""
        lm = _make_lm(vocab_size=1024)
        hf_tokenizer = MagicMock(name="hf_tokenizer")
        wrapper = MagicMock(name="TokenizerWrapper")
        wrapper._tokenizer = hf_tokenizer
        lm.text_tokenizer = wrapper

        sentinel = MagicMock(name="grammar_processor")
        with patch(
            "olmlx.engine.inference._make_grammar_processor", return_value=sentinel
        ) as mock_make:
            installed = _install_grammar_processor(
                lm, {}, GrammarSpec("json_object")
            )

        assert installed is True
        # The factory must receive the *unwrapped* HF tokenizer, not the
        # mlx-lm TokenizerWrapper.
        args, _ = mock_make.call_args
        assert args[0] is hf_tokenizer

    def test_passes_text_tokenizer_through_when_no_wrapper(self):
        """A bare HF tokenizer (no ``_tokenizer`` attribute) should be
        forwarded as-is — the unwrap is opportunistic, not mandatory."""
        lm = _make_lm(vocab_size=1024)
        bare = MagicMock(name="bare_hf_tokenizer", spec=["vocab_size", "encode"])
        lm.text_tokenizer = bare

        sentinel = MagicMock(name="grammar_processor")
        with patch(
            "olmlx.engine.inference._make_grammar_processor", return_value=sentinel
        ) as mock_make:
            _install_grammar_processor(lm, {}, GrammarSpec("json_object"))

        args, _ = mock_make.call_args
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

        def fake_install(lm, gen_kwargs, grammar_spec):
            captured["spec"] = grammar_spec
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
