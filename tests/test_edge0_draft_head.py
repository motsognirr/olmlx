"""Tests for the Edge0 MTP draft head: conversion, loading, request gating."""

import json
import types

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from olmlx.engine.edge0 import convert, draft_head
from olmlx.engine.edge0.pack import Edge0PackError
from tests.edge0_factories import make_full_pack, tiny_tokens


def _bf16(values):
    return np.array(mx.array(values).astype(mx.bfloat16).astype(mx.float32))


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    full = make_full_pack(tmp_path_factory.mktemp("edge0"), with_vision=False)
    convert.convert_pack(full.root)
    return full


def _weights(full):
    return mx.load(str(full.root / draft_head.DRAFT_DIRNAME / "model.safetensors"))


class TestConversion:
    def test_pack_conversion_records_the_draft_head(self, converted):
        section = convert.read_edge0_config(converted.root)
        assert section["draft_head"] == draft_head.DRAFT_DIRNAME
        config = json.loads(
            (converted.root / draft_head.DRAFT_DIRNAME / "config.json").read_text()
        )
        assert config["model_type"] == "qwen4_exp_mtp"
        assert config["block_size"] == 4
        assert "ple_storage" not in config["text_config"]
        assert config["quantization"]["bits"] == 8
        assert config["quantization"]["layers.0.mlp.switch_mlp.down_proj"] == {
            "group_size": 64,
            "bits": 2,
            "mode": "affine",
        }

    def test_pack_without_a_draft_head(self, tmp_path):
        full = make_full_pack(tmp_path, with_vision=False, with_draft_head=False)
        convert.convert_pack(full.root)
        assert convert.read_edge0_config(full.root)["draft_head"] is None
        assert not (full.root / draft_head.DRAFT_DIRNAME).exists()

    def test_norms_are_recentred_and_layer_tensors_prefixed(self, converted):
        weights = _weights(converted)
        np.testing.assert_array_equal(
            np.array(weights["layers.0.self_attn.k_norm.weight"].astype(mx.float32)),
            _bf16(converted.draft_dense["self_attn.k_norm.weight"] - 1.0),
        )
        np.testing.assert_array_equal(
            np.array(weights["pre_fc_norm_hidden.weight"].astype(mx.float32)),
            _bf16(converted.draft_dense["pre_fc_norm_hidden.weight"] - 1.0),
        )
        np.testing.assert_array_equal(
            np.array(weights["layers.0.mlp.gate.weight"].astype(mx.float32)),
            _bf16(converted.draft_dense["mlp.gate.weight"]),
        )
        assert "hyper_connection_mixer.hc_norm.weight" in weights
        assert "layers.0.hyper_connection_mixer.hc_norm.weight" not in weights

    def test_q8_matrices_are_repacked_without_offset(self, converted):
        weights = _weights(converted)
        scales = weights["fc_embedding.scales"].astype(mx.float32)
        np.testing.assert_array_equal(
            np.array(weights["fc_embedding.biases"].astype(mx.float32)),
            np.array(scales) * -128.0,
        )
        dense = mx.dequantize(
            weights["fc_embedding.weight"],
            scales,
            scales * -128.0,
            group_size=32,
            bits=8,
        )
        np.testing.assert_allclose(
            np.array(dense),
            converted.draft_dense["fc_embedding.weight"],
            rtol=5e-3,
            atol=1e-7,
        )

    @pytest.mark.parametrize(
        "index,name", [(0, "gate_proj"), (1, "up_proj"), (2, "down_proj")]
    )
    def test_two_bit_experts_are_exact_up_to_scale_rounding(
        self, converted, index, name
    ):
        weights = _weights(converted)
        prefix = f"layers.0.mlp.switch_mlp.{name}"
        scales = weights[prefix + ".scales"].astype(mx.float32)
        np.testing.assert_array_equal(
            np.array(weights[prefix + ".biases"].astype(mx.float32)), -np.array(scales)
        )
        dense = mx.dequantize(
            weights[prefix + ".weight"], scales, -scales, group_size=64, bits=2
        )
        expected = converted.draft_experts[index]
        assert dense.shape == expected.shape
        # Gate rows are the even rows of the interleaved plane, up the odd.
        np.testing.assert_allclose(np.array(dense), expected, rtol=5e-3, atol=1e-7)

    def test_truncated_experts_file(self, tmp_path):
        full = make_full_pack(tmp_path, with_vision=False)
        path = full.root / "mtp" / "experts.bin"
        path.write_bytes(path.read_bytes()[:-1])
        with pytest.raises(Edge0PackError, match="mtp/experts.bin"):
            convert.convert_pack(full.root)


class TestLoading:
    def test_wrapped_model_carries_the_draft_head(self, converted):
        from tests.test_edge0_loader import _load

        wrapped, _, store = _load(converted.root)
        try:
            drafter = wrapped._vlm_draft_head
            assert type(drafter).__name__ == "Qwen4ExpMTPDraftModel"
            experts = drafter.layers[0].mlp.switch_mlp.gate_proj
            assert (experts.bits, experts.group_size) == (2, 64)
            assert isinstance(drafter.layers[0].mlp.gate, nn.Linear)
        finally:
            store.close()

    def test_setting_disables_it(self, converted, monkeypatch):
        from tests.test_edge0_loader import _load

        monkeypatch.setattr("olmlx.config.settings.edge0_draft_head", False)
        wrapped, _, store = _load(converted.root)
        try:
            assert wrapped._vlm_draft_head is None
        finally:
            store.close()

    def test_config_cannot_redirect_the_draft_head(self, tmp_path):
        from tests.test_edge0_loader import _load

        full = make_full_pack(tmp_path, with_vision=False)
        convert.convert_pack(full.root)
        outside = tmp_path / "elsewhere"
        (full.root / draft_head.DRAFT_DIRNAME).rename(outside)
        config = json.loads((full.root / "config.json").read_text())
        config["edge0"]["draft_head"] = "../elsewhere"
        (full.root / "config.json").write_text(json.dumps(config))
        wrapped, _, store = _load(full.root)
        try:
            assert wrapped._vlm_draft_head is None
        finally:
            store.close()

    def test_broken_draft_head_only_costs_the_speedup(self, tmp_path):
        from tests.test_edge0_loader import _load

        full = make_full_pack(tmp_path, with_vision=False)
        convert.convert_pack(full.root)
        (full.root / draft_head.DRAFT_DIRNAME / "model.safetensors").write_bytes(b"x")
        wrapped, _, store = _load(full.root)
        try:
            assert wrapped._vlm_draft_head is None
        finally:
            store.close()

    def test_speculative_generation_runs_and_matches_plain_greedy(
        self, converted, metal_default_device
    ):
        # mlx-vlm's generate loop reads Metal device limits, so this cannot
        # run with the suite's CPU default device.
        from mlx_vlm import stream_generate

        from tests.test_edge0_loader import _load

        wrapped, processor, store = _load(converted.root)
        try:
            prompt = "the cat"

            def run(**extra):
                return [
                    chunk.token
                    for chunk in stream_generate(
                        wrapped,
                        processor,
                        prompt,
                        max_tokens=12,
                        temperature=0.0,
                        **extra,
                    )
                ]

            plain = run()
            drafted = run(
                draft_model=wrapped._vlm_draft_head,
                draft_kind="mtp",
                draft_block_size=3,
            )
            assert len(drafted) == len(plain) > 0
            vocab = len(tiny_tokens()[0])
            assert all(0 <= int(token) < vocab for token in drafted)
            # The draft only proposes: the first token is the target's own.
            assert int(drafted[0]) == int(plain[0])
        finally:
            store.close()


def _lm(drafter=None, **overrides):
    model = types.SimpleNamespace(_vlm_draft_head=drafter)
    fields = dict(model=model, is_vlm=True, is_distributed=False, kv_cache_quant=None)
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


class TestRequestGating:
    def _will(self, lm, gen_kwargs=None, **overrides):
        from olmlx.engine import inference

        args = dict(has_media=False, grammar_active=False, max_tokens=64)
        args.update(overrides)
        return inference._will_draft(lm, gen_kwargs or {}, **args)

    def test_drafts_for_a_plain_text_request(self):
        assert self._will(_lm(nn.Linear(2, 2)))
        assert self._will(_lm(nn.Linear(2, 2)), {"repetition_penalty": 1.1})

    @pytest.mark.parametrize(
        "lm_overrides,call_overrides",
        [
            ({}, {"has_media": True}),
            ({}, {"grammar_active": True}),
            ({}, {"max_tokens": -1}),
            ({}, {"max_tokens": 0}),
            ({"is_vlm": False}, {}),
            ({"is_distributed": True}, {}),
            ({"kv_cache_quant": "turboquant:4"}, {}),
        ],
    )
    def test_does_not_draft(self, lm_overrides, call_overrides):
        assert not self._will(_lm(nn.Linear(2, 2), **lm_overrides), **call_overrides)

    def test_does_not_draft_with_a_logits_processor(self):
        # mlx-vlm's speculative loop applies processors to the first token
        # only, so a thinking budget would silently stop being enforced.
        assert not self._will(
            _lm(nn.Linear(2, 2)), {"logits_processors": [lambda tokens, logits: logits]}
        )

    def test_no_draft_head_no_drafting(self):
        assert not self._will(_lm(None))

    def test_mock_model_cannot_route_here(self):
        from unittest.mock import MagicMock

        assert not self._will(MagicMock())

    def test_prompt_cache_wins_by_default(self, monkeypatch):
        from olmlx.engine import inference

        args = dict(has_media=False, grammar_active=False, max_tokens=64)
        lm = _lm(nn.Linear(2, 2))
        assert inference.settings.edge0_draft_over_prompt_cache is False
        assert not inference._draft_replaces_prompt_cache(lm, {}, **args)
        monkeypatch.setattr(inference.settings, "edge0_draft_over_prompt_cache", True)
        assert inference._draft_replaces_prompt_cache(lm, {}, **args)
        assert not inference._draft_replaces_prompt_cache(_lm(None), {}, **args)

    def test_attach_adds_mlx_vlm_kwargs(self, monkeypatch):
        from olmlx.engine import inference

        monkeypatch.setattr(inference.settings, "edge0_draft_block_size", 3)
        drafter = nn.Linear(2, 2)
        sampler = object()
        kwargs = {"temperature": 0.7, "sampler": sampler}
        assert inference._attach_vlm_draft_head(
            _lm(drafter), kwargs, has_media=False, grammar_active=False, max_tokens=8
        )
        assert kwargs["draft_model"] is drafter
        assert kwargs["draft_kind"] == "mtp"
        assert kwargs["draft_block_size"] == 3
        assert kwargs["sampler"] is sampler

    def test_greedy_request_drops_the_sampler(self):
        from olmlx.engine import inference

        kwargs = {"temperature": 0, "sampler": object()}
        inference._attach_vlm_draft_head(
            _lm(nn.Linear(2, 2)),
            kwargs,
            has_media=False,
            grammar_active=False,
            max_tokens=8,
        )
        assert "sampler" not in kwargs

    @pytest.mark.parametrize("key", ["prompt_cache", "prompt_cache_state"])
    def test_attach_skips_when_a_cache_is_already_attached(self, key):
        from olmlx.engine import inference

        kwargs = {key: object()}
        assert not inference._attach_vlm_draft_head(
            _lm(nn.Linear(2, 2)),
            kwargs,
            has_media=False,
            grammar_active=False,
            max_tokens=8,
        )
        assert "draft_model" not in kwargs
