"""Regression coverage for branch/error/formatting paths in
``olmlx.engine.inference``.

The module is already high-coverage; these tests target the remaining
no-GPU-reachable branches: lm_head vocab-size resolution fallbacks, KV-cache
byte estimation (turboquant/spectral ratios, MLA layout, sliding-window
introspection, NAS no-op layers, wrapper-arg failure), option/default merging,
KV-quant spec parsing, BOS-aware cache tokenization, message-boundary token
detection, segmented-chat splitting, chat-template-text extraction, token
counting return-shape handling, penalty-processor out-of-range guards,
native-tool-hint edge cases, tool-message-to-response conversion, the gpt-oss
channel filter, and the stop-sequence / finish-reason branches in
``_full_completion``.

Every test is hermetic and deterministic: no network, no real model load, no
GPU. Fakes are built from ``SimpleNamespace``/``MagicMock`` and tiny ``mx``
arrays, so each finishes in well under a second.
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

import olmlx.engine.inference as _inf_mod
from olmlx.engine.inference import (
    _add_native_tool_hint,
    _build_generate_kwargs,
    _convert_tool_messages_to_responses,
    _full_completion,
    _get_chat_template_text,
    _GptOssChannelFilter,
    _make_frequency_penalty_processor,
    _make_presence_penalty_processor,
    _merge_default_options,
    _message_boundary_token_ids,
    _parse_kv_cache_quant,
    _resolve_model_vocab_size,
    count_chat_tokens,
    estimate_kv_cache_bytes,
    tokenize_for_cache,
    tokenize_segmented_chat,
)


# --------------------------------------------------------------------------- #
# _resolve_model_vocab_size                                                    #
# --------------------------------------------------------------------------- #
class TestResolveModelVocabSize:
    """The lm_head bitmask sizing helper for grammar-constrained decoding."""

    def test_uses_args_vocab_size_when_positive_int(self):
        model = SimpleNamespace(args=SimpleNamespace(vocab_size=32000))
        lm = SimpleNamespace(model=model)
        assert _resolve_model_vocab_size(lm) == 32000

    def test_falls_through_args_when_vocab_size_not_positive(self):
        # vocab_size present but <= 0 must be ignored and lm_head weight used.
        lm_head = SimpleNamespace(weight=SimpleNamespace(shape=(40000, 8)))
        inner = SimpleNamespace(args=SimpleNamespace(vocab_size=0), lm_head=lm_head)
        lm = SimpleNamespace(model=inner)
        assert _resolve_model_vocab_size(lm) == 40000

    def test_prefers_lm_head_output_dim_over_embed(self):
        # top-level embed_tokens (input dim), nested lm_head (larger output dim):
        # lm_head must win even though embed_tokens is shallower.
        nested = SimpleNamespace(
            lm_head=SimpleNamespace(weight=SimpleNamespace(shape=(50000, 4)))
        )
        embed = SimpleNamespace(weight=SimpleNamespace(shape=(48000, 4)))
        model = SimpleNamespace(args=None, embed_tokens=embed, model=nested)
        lm = SimpleNamespace(model=model)
        assert _resolve_model_vocab_size(lm) == 50000

    def test_falls_back_to_embed_when_no_lm_head(self):
        embed = SimpleNamespace(weight=SimpleNamespace(shape=(33000, 4)))
        model = SimpleNamespace(args=None, embed_tokens=embed, model=None)
        lm = SimpleNamespace(model=model)
        assert _resolve_model_vocab_size(lm) == 33000

    def test_weight_shape_exception_swallowed_returns_none(self):
        class BadWeight:
            @property
            def shape(self):
                raise RuntimeError("boom")

        model = SimpleNamespace(
            args=None, lm_head=SimpleNamespace(weight=BadWeight()), model=None
        )
        lm = SimpleNamespace(model=model)
        assert _resolve_model_vocab_size(lm) is None

    def test_no_sources_returns_none(self):
        model = SimpleNamespace(args=None, model=None)
        lm = SimpleNamespace(model=model)
        assert _resolve_model_vocab_size(lm) is None


# --------------------------------------------------------------------------- #
# estimate_kv_cache_bytes                                                      #
# --------------------------------------------------------------------------- #
def _uniform_args(**over):
    base = dict(
        num_attention_heads=8,
        num_hidden_layers=4,
        hidden_size=512,
        num_key_value_heads=2,
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestEstimateKvCacheBytes:
    def test_zero_or_negative_tokens_is_zero(self):
        model = SimpleNamespace(args=_uniform_args())
        assert estimate_kv_cache_bytes(model, 0) == 0
        assert estimate_kv_cache_bytes(model, -5) == 0

    def test_uniform_args_fallback(self):
        model = SimpleNamespace(args=_uniform_args())
        # head_dim = 512/8 = 64; raw = layers*2*kv*head_dim*tokens*2
        # = 4*2*2*64*10*2 = 20480; *1.3.
        assert estimate_kv_cache_bytes(model, 10) == int(20480 * 1.3)

    def test_explicit_head_dim_attribute_used(self):
        # head_dim present on args overrides hidden_size//num_heads.
        model = SimpleNamespace(args=_uniform_args(head_dim=128))
        assert estimate_kv_cache_bytes(model, 10) == int(4 * 2 * 2 * 128 * 10 * 2 * 1.3)

    def test_turboquant_ratio_reduces_estimate(self):
        model = SimpleNamespace(args=_uniform_args())
        plain = estimate_kv_cache_bytes(model, 10)
        tq = estimate_kv_cache_bytes(model, 10, kv_cache_quant="turboquant:4")
        assert tq < plain

    def test_spectral_ratio_reduces_estimate(self):
        model = SimpleNamespace(args=_uniform_args())
        plain = estimate_kv_cache_bytes(model, 10)
        sq = estimate_kv_cache_bytes(model, 10, kv_cache_quant="spectral:2")
        assert sq < plain

    def test_mla_model_uses_lora_layout(self):
        args = SimpleNamespace(
            kv_lora_rank=512, qk_rope_head_dim=64, num_hidden_layers=2
        )
        model = SimpleNamespace(args=args)
        # raw = layers*2*(lora+rope)*tokens*2 = 2*2*576*10*2 = 46080; *1.3.
        assert estimate_kv_cache_bytes(model, 10) == int(46080 * 1.3)

    def test_layer_introspection_skips_noop_attention(self):
        real_attn = SimpleNamespace(n_kv_heads=2, head_dim=64)
        layers = [
            SimpleNamespace(self_attn=None),  # no-op layer — no KV cache
            SimpleNamespace(self_attn=real_attn),
        ]
        model = SimpleNamespace(
            args=_uniform_args(), model=SimpleNamespace(layers=layers)
        )
        # Only one attention layer counted: 2*kv*head_dim*tokens*2 = 5120; *1.3.
        assert estimate_kv_cache_bytes(model, 10) == int(5120 * 1.3)

    def test_introspection_alt_kv_head_attr_name(self):
        # Qwen3-Next exposes num_key_value_heads, not n_kv_heads.
        attn = SimpleNamespace(num_key_value_heads=2, head_dim=64)
        model = SimpleNamespace(
            args=_uniform_args(),
            model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]),
        )
        assert estimate_kv_cache_bytes(model, 10) == int(2 * 2 * 64 * 10 * 2 * 1.3)

    def test_sliding_window_caps_effective_tokens(self):
        win = 4
        attn = SimpleNamespace(n_kv_heads=2, head_dim=64, is_sliding=True)
        model = SimpleNamespace(
            args=_uniform_args(sliding_window=win),
            model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]),
        )
        capped = estimate_kv_cache_bytes(model, 1000)
        # Past the window more tokens don't grow the estimate.
        assert estimate_kv_cache_bytes(model, 100) == capped
        # Below the window the estimate scales with token count.
        assert estimate_kv_cache_bytes(model, win - 1) < capped
        # An identical non-sliding layer scales fully with the prompt.
        attn_full = SimpleNamespace(n_kv_heads=2, head_dim=64)
        model_full = SimpleNamespace(
            args=_uniform_args(),
            model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn_full)]),
        )
        assert estimate_kv_cache_bytes(model_full, 1000) > capped

    def test_introspection_falls_back_when_kv_heads_unknown(self):
        # self_attn present but no recognised kv-head attribute → args fallback.
        attn = SimpleNamespace(head_dim=64)
        model = SimpleNamespace(
            args=_uniform_args(),
            model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]),
        )
        assert estimate_kv_cache_bytes(model, 10) == int(20480 * 1.3)

    def test_wrapper_args_without_inner_raises(self):
        # A text_config wrapper whose language_model can't be resolved must
        # raise loudly rather than crash opaquely later.
        args = SimpleNamespace(text_config={"foo": 1})
        model = SimpleNamespace(args=args)  # no language_model
        with pytest.raises(AttributeError, match="text_config wrapper"):
            estimate_kv_cache_bytes(model, 10)

    def test_no_args_anywhere_raises(self):
        model = SimpleNamespace()  # no args, no config, no language_model
        with pytest.raises(AttributeError, match="no 'args' attribute"):
            estimate_kv_cache_bytes(model, 10)


# --------------------------------------------------------------------------- #
# _merge_default_options                                                       #
# --------------------------------------------------------------------------- #
class TestMergeDefaultOptions:
    def test_request_value_wins_per_key(self):
        merged = _merge_default_options({"temperature": 0.7}, {"temperature": 0.2})
        assert merged == {"temperature": 0.2}

    def test_partial_request_keeps_unspecified_defaults(self):
        # The opencode/Qwen3-Coder regression: a request with top_k must NOT
        # discard the model-default temperature.
        merged = _merge_default_options(
            {"temperature": 0.7, "top_p": 0.9}, {"top_k": 40}
        )
        assert merged == {"temperature": 0.7, "top_p": 0.9, "top_k": 40}

    def test_none_request_uses_defaults(self):
        assert _merge_default_options({"temperature": 0.7}, None) == {
            "temperature": 0.7
        }

    def test_empty_request_uses_defaults(self):
        assert _merge_default_options({"temperature": 0.7}, {}) == {"temperature": 0.7}

    def test_none_defaults_treated_as_empty(self):
        assert _merge_default_options(None, {"top_k": 5}) == {"top_k": 5}

    def test_both_none_yields_empty_dict(self):
        assert _merge_default_options(None, None) == {}

    def test_returns_new_dict_does_not_mutate_inputs(self):
        defaults = {"temperature": 0.7}
        request = {"top_k": 5}
        merged = _merge_default_options(defaults, request)
        merged["temperature"] = 999
        assert defaults == {"temperature": 0.7}
        assert request == {"top_k": 5}


# --------------------------------------------------------------------------- #
# _apply_sampling_defaults (Ollama-parity sampling defaults, #646)             #
# --------------------------------------------------------------------------- #
class TestApplySamplingDefaults:
    def test_defaults_applied_when_all_omitted(self, monkeypatch):
        monkeypatch.setattr(_inf_mod.settings, "sampling_defaults_enabled", True)
        monkeypatch.setattr(_inf_mod.settings, "default_temperature", 0.8)
        monkeypatch.setattr(_inf_mod.settings, "default_top_p", 0.9)
        monkeypatch.setattr(_inf_mod.settings, "default_top_k", 40)
        monkeypatch.setattr(_inf_mod.settings, "default_repeat_penalty", 1.1)
        monkeypatch.setattr(_inf_mod.settings, "default_repeat_last_n", 64)
        assert _inf_mod._apply_sampling_defaults({}) == {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "repeat_last_n": 64,
        }

    def test_request_value_overrides_default(self, monkeypatch):
        monkeypatch.setattr(_inf_mod.settings, "sampling_defaults_enabled", True)
        monkeypatch.setattr(_inf_mod.settings, "default_temperature", 0.8)
        monkeypatch.setattr(_inf_mod.settings, "default_repeat_penalty", 1.1)
        merged = _inf_mod._apply_sampling_defaults(
            {"temperature": 0.1, "repeat_penalty": 1.5}
        )
        # Explicit values win; unspecified keys still get the default.
        assert merged["temperature"] == 0.1
        assert merged["repeat_penalty"] == 1.5
        assert merged["top_p"] == _inf_mod.settings.default_top_p

    def test_explicit_greedy_zero_temperature_preserved(self, monkeypatch):
        # A client that deliberately requests greedy (temperature=0) must keep
        # it — 0 is a real value, not "unset".
        monkeypatch.setattr(_inf_mod.settings, "sampling_defaults_enabled", True)
        monkeypatch.setattr(_inf_mod.settings, "default_temperature", 0.8)
        merged = _inf_mod._apply_sampling_defaults({"temperature": 0})
        assert merged["temperature"] == 0

    def test_disabled_returns_input_unchanged(self, monkeypatch):
        monkeypatch.setattr(_inf_mod.settings, "sampling_defaults_enabled", False)
        assert _inf_mod._apply_sampling_defaults({}) == {}
        assert _inf_mod._apply_sampling_defaults({"top_k": 5}) == {"top_k": 5}

    def test_does_not_mutate_input(self, monkeypatch):
        monkeypatch.setattr(_inf_mod.settings, "sampling_defaults_enabled", True)
        req = {"temperature": 0.2}
        _inf_mod._apply_sampling_defaults(req)
        assert req == {"temperature": 0.2}

    def test_builds_sampler_and_repeat_penalty_processor(self, monkeypatch):
        # End-to-end with the real kwargs builder: the defaults must produce a
        # sampler (temperature present) AND a repeat-penalty logits processor —
        # the combination that breaks the #646 degenerate runaway.
        monkeypatch.setattr(_inf_mod.settings, "sampling_defaults_enabled", True)
        merged = _inf_mod._apply_sampling_defaults({})
        kwargs = _build_generate_kwargs(merged, is_vlm=False)
        assert "sampler" in kwargs
        assert kwargs.get("logits_processors")


# --------------------------------------------------------------------------- #
# _parse_kv_cache_quant                                                        #
# --------------------------------------------------------------------------- #
class TestParseKvCacheQuant:
    def test_turboquant_spec(self):
        assert _parse_kv_cache_quant("turboquant:4") == ("turboquant", 4)

    def test_spectral_spec(self):
        assert _parse_kv_cache_quant("spectral:2") == ("spectral", 2)


# --------------------------------------------------------------------------- #
# tokenize_for_cache (BOS heuristic)                                           #
# --------------------------------------------------------------------------- #
class TestTokenizeForCache:
    def test_no_bos_adds_special(self):
        captured = {}

        def encode(text, add_special_tokens):
            captured["add_special_tokens"] = add_special_tokens
            return [1, 2, 3]

        tok = SimpleNamespace(bos_token=None, encode=encode)
        assert tokenize_for_cache(tok, "hello") == [1, 2, 3]
        # bos_token is None → add_special_tokens must be True.
        assert captured["add_special_tokens"] is True

    def test_prompt_already_starts_with_bos_skips_special(self):
        captured = {}

        def encode(text, add_special_tokens):
            captured["add_special_tokens"] = add_special_tokens
            return [9]

        tok = SimpleNamespace(bos_token="<s>", encode=encode)
        tokenize_for_cache(tok, "<s>hi")
        # prompt already starts with bos → do not add special tokens again.
        assert captured["add_special_tokens"] is False

    def test_bos_present_but_prompt_lacks_it_adds_special(self):
        captured = {}

        def encode(text, add_special_tokens):
            captured["add_special_tokens"] = add_special_tokens
            return [9]

        tok = SimpleNamespace(bos_token="<s>", encode=encode)
        tokenize_for_cache(tok, "hi")
        assert captured["add_special_tokens"] is True


# --------------------------------------------------------------------------- #
# _message_boundary_token_ids                                                  #
# --------------------------------------------------------------------------- #
class TestMessageBoundaryTokenIds:
    def test_list_eos_ids_all_added_nones_skipped(self):
        tok = SimpleNamespace(eos_token_id=[1, 2, None, 3])
        assert _message_boundary_token_ids(tok) == {1, 2, 3}

    def test_scalar_eos_plus_known_eom_strings(self):
        def convert(s):
            return {"<|end|>": 100, "<end_of_turn>": 101}.get(s)

        tok = SimpleNamespace(
            eos_token_id=7, convert_tokens_to_ids=convert, unk_token_id=0
        )
        assert _message_boundary_token_ids(tok) == {7, 100, 101}

    def test_unk_and_none_eom_ids_skipped(self):
        def convert(s):
            # one resolves to unk (skip), one resolves to None (skip).
            return {"<|end|>": 0, "<end_of_turn>": None}.get(s)

        tok = SimpleNamespace(
            eos_token_id=7, convert_tokens_to_ids=convert, unk_token_id=0
        )
        assert _message_boundary_token_ids(tok) == {7}

    def test_convert_exception_skipped(self):
        def convert(s):
            raise ValueError("no such token")

        tok = SimpleNamespace(
            eos_token_id=5, convert_tokens_to_ids=convert, unk_token_id=None
        )
        assert _message_boundary_token_ids(tok) == {5}

    def test_no_eos_no_convert_empty(self):
        tok = SimpleNamespace(eos_token_id=None)
        assert _message_boundary_token_ids(tok) == set()


# --------------------------------------------------------------------------- #
# tokenize_segmented_chat                                                      #
# --------------------------------------------------------------------------- #
class TestTokenizeSegmentedChat:
    def test_empty_messages_yields_empty_segments(self):
        tok = SimpleNamespace(eos_token_id=1)
        seg = tokenize_segmented_chat(tok, [], full_tokens=[1, 2, 3])
        assert seg.segments == []

    def test_no_eom_token_single_segment(self):
        tok = SimpleNamespace(eos_token_id=None)
        messages = [{"role": "user", "content": "hi"}]
        seg = tokenize_segmented_chat(tok, messages, full_tokens=[10, 11, 12])
        assert len(seg.segments) == 1
        assert seg.segments[0].tokens == [10, 11, 12]
        assert seg.segments[0].role == "user"

    def test_fewer_boundaries_than_messages_single_segment(self):
        # Template emitted fewer EOMs than messages → can't align → fallback.
        tok = SimpleNamespace(eos_token_id=99)
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        seg = tokenize_segmented_chat(tok, messages, full_tokens=[1, 99, 2, 3])
        assert len(seg.segments) == 1
        assert seg.segments[0].tokens == [1, 99, 2, 3]

    def test_one_eom_per_message_splits_and_flattens(self):
        tok = SimpleNamespace(eos_token_id=99)
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        full = [1, 99, 2, 99, 3]  # trailing 3 absorbed into last segment
        seg = tokenize_segmented_chat(tok, messages, full_tokens=full)
        assert len(seg.segments) == 2
        # flatten() must reconstruct the full token list exactly.
        assert seg.flatten() == full
        assert seg.segments[0].role == "user"
        assert seg.segments[1].role == "assistant"

    def test_extra_boundaries_padded_with_first_role(self):
        # More EOMs than messages (Harmony preamble) → leading extra segment
        # gets messages[0]['role'].
        tok = SimpleNamespace(eos_token_id=99)
        messages = [{"role": "user", "content": "a"}]
        full = [5, 99, 1, 99]
        seg = tokenize_segmented_chat(tok, messages, full_tokens=full)
        assert len(seg.segments) == 2
        assert seg.segments[0].role == "user"
        assert seg.flatten() == full

    def test_full_tokens_omitted_uses_apply_chat_template(self):
        # Mode 2: function tokenizes via apply_chat_template itself.
        tok = MagicMock()
        tok.apply_chat_template.return_value = [1, 99, 2, 99]
        tok.eos_token_id = 99
        tok.convert_tokens_to_ids = lambda s: None
        tok.unk_token_id = None
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        seg = tokenize_segmented_chat(tok, messages)
        assert seg.flatten() == [1, 99, 2, 99]
        assert len(seg.segments) == 2


# --------------------------------------------------------------------------- #
# count_chat_tokens (return-shape handling)                                    #
# --------------------------------------------------------------------------- #
class TestCountChatTokens:
    def test_flat_token_list(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = [1, 2, 3, 4]
        assert count_chat_tokens(tok, [{"role": "user", "content": "hi"}]) == 4

    def test_batch_of_one_nested_list(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = [[1, 2, 3]]
        assert count_chat_tokens(tok, [{"role": "user", "content": "hi"}]) == 3

    def test_mapping_with_input_ids(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = {"input_ids": [7, 8]}
        assert count_chat_tokens(tok, [{"role": "user", "content": "hi"}]) == 2

    def test_mapping_nested_input_ids(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = {"input_ids": [[7, 8, 9]]}
        assert count_chat_tokens(tok, [{"role": "user", "content": "hi"}]) == 3

    def test_mapping_without_input_ids_raises(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = {"attention_mask": [1, 1]}
        with pytest.raises(TypeError, match="without 'input_ids'"):
            count_chat_tokens(tok, [{"role": "user", "content": "hi"}])

    def test_unexpected_return_type_raises(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = "not a token list"
        with pytest.raises(TypeError, match="Unexpected return type"):
            count_chat_tokens(tok, [{"role": "user", "content": "hi"}])


# --------------------------------------------------------------------------- #
# penalty processor out-of-range / incremental paths                          #
# --------------------------------------------------------------------------- #
class TestPenaltyProcessorEdgeCases:
    def test_empty_tokens_noop(self):
        proc = _make_frequency_penalty_processor(1.0)
        logits = mx.array([5.0, 5.0])
        out = proc([], logits)
        assert mx.allclose(out, logits).item()

    def test_zero_penalty_noop(self):
        proc = _make_frequency_penalty_processor(0.0)
        logits = mx.array([5.0, 5.0])
        out = proc([0, 1], logits)
        assert mx.allclose(out, logits).item()

    def test_frequency_out_of_range_seed_token_ignored(self):
        proc = _make_frequency_penalty_processor(1.0)
        # token 9 is out of range (vocab 2); only token 0 penalised.
        result = proc([0, 9], mx.array([5.0, 5.0]))
        assert mx.allclose(result, mx.array([4.0, 5.0])).item()

    def test_frequency_incremental_out_of_range_ignored(self):
        proc = _make_frequency_penalty_processor(1.0)
        proc([0], mx.array([5.0, 5.0]))  # init seeds freq {0:1}
        result = proc([0, 9], mx.array([5.0, 5.0]))  # new token 9 out of range
        assert mx.allclose(result, mx.array([4.0, 5.0])).item()

    def test_frequency_counts_accumulate(self):
        proc = _make_frequency_penalty_processor(1.0)
        proc([0, 0], mx.array([5.0, 5.0]))  # init: token 0 seen twice
        result = proc([0, 0, 0], mx.array([5.0, 5.0]))  # incremental: +1 → 3
        assert mx.allclose(result, mx.array([2.0, 5.0])).item()

    def test_presence_incremental_new_token(self):
        proc = _make_presence_penalty_processor(0.5)
        proc([0], mx.array([10.0, 10.0]))  # init seeds {0}
        result = proc([0, 1], mx.array([10.0, 10.0]))  # token 1 newly seen
        assert mx.allclose(result, mx.array([10.0, 9.5])).item()

    def test_presence_incremental_repeat_token_no_double_penalty(self):
        proc = _make_presence_penalty_processor(0.5)
        proc([0], mx.array([10.0, 10.0]))  # init seeds {0}
        result = proc([0, 0], mx.array([10.0, 10.0]))  # 0 already seen
        assert mx.allclose(result, mx.array([10.0, 10.0])).item()

    def test_presence_incremental_out_of_range_ignored(self):
        proc = _make_presence_penalty_processor(0.5)
        proc([0], mx.array([10.0, 10.0]))
        result = proc([0, 5], mx.array([10.0, 10.0]))  # 5 out of range
        assert mx.allclose(result, mx.array([10.0, 10.0])).item()


# --------------------------------------------------------------------------- #
# _get_chat_template_text                                                      #
# --------------------------------------------------------------------------- #
class TestGetChatTemplateText:
    def test_direct_string_template(self):
        tok = SimpleNamespace(chat_template="{{ messages }}")
        assert _get_chat_template_text(tok) == "{{ messages }}"

    def test_nested_tokenizer_template(self):
        inner = SimpleNamespace(chat_template="nested-tpl")
        tok = SimpleNamespace(chat_template=None, tokenizer=inner)
        assert _get_chat_template_text(tok) == "nested-tpl"

    def test_list_of_named_templates_joined(self):
        tok = SimpleNamespace(
            chat_template=[
                {"name": "default", "template": "AAA"},
                {"name": "tool_use", "template": "BBB"},
                "ignored-non-dict",
            ]
        )
        assert _get_chat_template_text(tok) == "AAA BBB"

    def test_none_everywhere_returns_empty(self):
        tok = SimpleNamespace(chat_template=None, tokenizer=None)
        assert _get_chat_template_text(tok) == ""

    def test_non_string_non_list_returns_empty(self):
        tok = SimpleNamespace(chat_template=12345)
        assert _get_chat_template_text(tok) == ""


# --------------------------------------------------------------------------- #
# _convert_tool_messages_to_responses                                         #
# --------------------------------------------------------------------------- #
class TestConvertToolMessages:
    def test_no_tool_messages_passthrough(self):
        messages = [{"role": "user", "content": "hi"}]
        # Returns the same object unchanged when no tool role is present.
        assert _convert_tool_messages_to_responses(messages) is messages

    def test_tool_response_merged_into_preceding_assistant_with_name(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc1", "function": {"name": "read", "arguments": {}}}
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "file body"},
        ]
        result = _convert_tool_messages_to_responses(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["tool_responses"] == [
            {"name": "read", "response": "file body"}
        ]

    def test_orphan_tool_message_creates_assistant_with_unknown_name(self):
        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": "stranded"},
        ]
        result = _convert_tool_messages_to_responses(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["tool_responses"] == [
            {"name": "unknown", "response": "stranded"}
        ]


# --------------------------------------------------------------------------- #
# _add_native_tool_hint                                                        #
# --------------------------------------------------------------------------- #
class TestAddNativeToolHint:
    def test_no_system_message_passthrough(self):
        messages = [{"role": "user", "content": "use <function=foo>"}]
        assert _add_native_tool_hint(messages) is messages

    def test_empty_messages_passthrough(self):
        assert _add_native_tool_hint([]) == []

    def test_missing_content_key_treated_as_empty(self):
        messages = [{"role": "system"}]
        assert _add_native_tool_hint(messages) == messages

    def test_non_string_content_skipped(self):
        messages = [{"role": "system", "content": [{"type": "text", "text": "x"}]}]
        assert _add_native_tool_hint(messages) is messages

    def test_conflict_pattern_appends_hint(self):
        messages = [{"role": "system", "content": "call <function=Foo>{...}"}]
        result = _add_native_tool_hint(messages)
        assert "native tool call format" in result[0]["content"]
        # Original input not mutated (shallow copy).
        assert messages[0]["content"] == "call <function=Foo>{...}"

    def test_pattern_in_template_suppressed(self):
        # When a conflict pattern is native to the template, no hint added.
        messages = [{"role": "system", "content": "use <|python_tag|>{...}"}]
        result = _add_native_tool_hint(messages, native_template_text="<|python_tag|>")
        assert "native tool call format" not in result[0]["content"]

    def test_idempotent_when_hint_already_present(self):
        from olmlx.engine.inference import _NATIVE_TOOL_HINT

        content = "call <function=Foo> " + _NATIVE_TOOL_HINT
        messages = [{"role": "system", "content": content}]
        # Hint already present → return unchanged (no double-append).
        assert _add_native_tool_hint(messages) is messages


# --------------------------------------------------------------------------- #
# _GptOssChannelFilter                                                         #
# --------------------------------------------------------------------------- #
class TestGptOssChannelFilter:
    def test_final_channel_content_yielded(self):
        filt = _GptOssChannelFilter()
        seq = ["<|channel|>", "final", "<|message|>", "answer"]
        yielded = [t for t in seq if filt.should_yield(t)]
        assert yielded == ["answer"]
        assert filt.get_fallback_texts() == []

    def test_analysis_buffered_and_used_as_fallback_when_no_final(self):
        filt = _GptOssChannelFilter()
        seq = ["<|channel|>", "analysis", "<|message|>", "thinking..."]
        yielded = [t for t in seq if filt.should_yield(t)]
        assert yielded == []  # analysis is not yielded inline
        # No final channel → analysis text is the fallback.
        assert filt.get_fallback_texts() == ["thinking..."]

    def test_no_channel_plain_text_yielded(self):
        filt = _GptOssChannelFilter()
        # Plain non-structural text in init state is passed through.
        assert filt.should_yield("hello") is True

    def test_structural_tokens_never_yielded(self):
        filt = _GptOssChannelFilter()
        assert filt.should_yield("<|start|>") is False
        assert filt.should_yield("<|end|>") is False

    def test_full_text_accumulates_all_tokens(self):
        filt = _GptOssChannelFilter()
        for t in ["<|channel|>", "final", "<|message|>", "hi"]:
            filt.should_yield(t)
        assert filt.get_full_text() == "<|channel|>final<|message|>hi"


# --------------------------------------------------------------------------- #
# _full_completion stop-sequence / finish-reason branches                      #
# --------------------------------------------------------------------------- #
def _patch_lock_and_ref(monkeypatch):
    """Replace the GPU lock/ref context managers so _full_completion runs its
    post-processing branches without touching Metal."""

    @contextlib.asynccontextmanager
    async def fake_locked(*a, **k):
        yield None

    @contextlib.contextmanager
    def fake_ref(*a, **k):
        yield None

    monkeypatch.setattr(_inf_mod, "_inference_locked", fake_locked)
    monkeypatch.setattr(_inf_mod, "_inference_ref", fake_ref)


def _mock_lm():
    lm = MagicMock()
    lm.inference_queue_timeout = 30.0
    lm.sync_mode = None
    return lm


class TestFullCompletionFinishReason:
    async def test_timeout_done_reason_passthrough_no_finish_reason(self, monkeypatch):
        _patch_lock_and_ref(monkeypatch)

        async def fake_inner(*a, **k):
            return {
                "text": "partial output",
                "done": True,
                "stats": MagicMock(),
                "done_reason": "timeout",
            }

        monkeypatch.setattr(_inf_mod, "_full_completion_inner", fake_inner)
        result = await _full_completion(_mock_lm(), "prompt", 50, {}, MagicMock())
        # No stop sequences → text unchanged, done_reason preserved, no
        # finish_reason injected.
        assert result["text"] == "partial output"
        assert result["done_reason"] == "timeout"
        assert "finish_reason" not in result

    async def test_stop_sequence_truncates_and_sets_reason(self, monkeypatch):
        _patch_lock_and_ref(monkeypatch)

        async def fake_inner(*a, **k):
            return {"text": "hello STOP world", "done": True, "stats": MagicMock()}

        monkeypatch.setattr(_inf_mod, "_full_completion_inner", fake_inner)
        result = await _full_completion(
            _mock_lm(), "prompt", 50, {"stop": ["STOP"]}, MagicMock()
        )
        assert result["text"] == "hello "
        assert result["finish_reason"] == "stop"

    async def test_stop_sequence_not_present_leaves_text_untouched(self, monkeypatch):
        _patch_lock_and_ref(monkeypatch)

        async def fake_inner(*a, **k):
            return {"text": "no marker here", "done": True, "stats": MagicMock()}

        monkeypatch.setattr(_inf_mod, "_full_completion_inner", fake_inner)
        result = await _full_completion(
            _mock_lm(), "prompt", 50, {"stop": ["ZZZ"]}, MagicMock()
        )
        assert result["text"] == "no marker here"
        assert "finish_reason" not in result

    async def test_earliest_stop_sequence_wins(self, monkeypatch):
        _patch_lock_and_ref(monkeypatch)

        async def fake_inner(*a, **k):
            return {"text": "aa END bb HALT cc", "done": True, "stats": MagicMock()}

        monkeypatch.setattr(_inf_mod, "_full_completion_inner", fake_inner)
        # HALT is later in the text but listed first: earliest position wins.
        result = await _full_completion(
            _mock_lm(), "prompt", 50, {"stop": ["HALT", "END"]}, MagicMock()
        )
        assert result["text"] == "aa "
        assert result["finish_reason"] == "stop"
