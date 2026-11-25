"""Tests for olmlx.engine.edge0.loader: a converted pack through the real loader."""

import threading

import mlx.core as mx
import numpy as np
import pytest

from olmlx.engine.edge0 import convert
from olmlx.engine.edge0.latent_moe import LatentExpertStore, LatentMoE
from olmlx.engine.edge0.pack import Edge0PackError
from olmlx.engine.edge0.ple import Fp8NGramEmbedding
from tests.edge0_factories import TINY_HIDDEN, make_full_pack, tiny_tokens
from tests.test_edge0_latent_moe import _reference_forward, _silu


def _load(root, **kwargs):
    """Load a synthetic converted pack the way ``_load_flash_moe_model`` does.

    Calls mlx-vlm's loader function directly rather than ``mlx_vlm.load``:
    the suite's real-model guard blocks that attribute, and this is a tiny
    generated model, not a real one. ``TestLoadVlmWiring`` covers that
    ``load_vlm`` passes the same arguments.
    """
    from mlx_vlm.utils import load as mlx_vlm_load

    from olmlx.engine.edge0.loader import prepare_mlx_vlm_load
    from olmlx.engine.flash.flash_moe_model import wrap_flash_moe

    model, processor = mlx_vlm_load(str(root), lazy=True, **prepare_mlx_vlm_load(root))
    wrapped, store = wrap_flash_moe(
        model,
        root / "flash_moe",
        io_threads=kwargs.get("io_threads", 4),
        cache_budget_experts=kwargs.get("cache_budget_experts", 8),
    )
    return wrapped, processor, store


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    full = make_full_pack(tmp_path_factory.mktemp("edge0"))
    convert.convert_pack(full.root)
    wrapped, processor, store = _load(full.root)
    yield full, wrapped, processor, store
    store.close()


class TestLoad:
    def test_every_layer_gets_latent_experts(self, loaded):
        _, wrapped, _, store = loaded
        assert isinstance(store, LatentExpertStore)
        assert wrapped._weight_store is store
        for layer in wrapped.layers:
            assert isinstance(layer.mlp._flash_moe, LatentMoE)
            assert not hasattr(layer.mlp, "switch_mlp")

    def test_ple_table_is_the_mmap_reader(self, loaded):
        _, wrapped, _, _ = loaded
        embedding = wrapped.layers[1].ple.ple_embedding.ngram_embedding
        assert isinstance(embedding, Fp8NGramEmbedding)

    def test_trunk_is_quantized_where_the_pack_is(self, loaded):
        import mlx.nn as nn

        _, wrapped, _, _ = loaded
        layer = wrapped.layers[0]
        assert isinstance(layer.linear_attn.in_proj_qkv, nn.QuantizedLinear)
        assert layer.linear_attn.in_proj_qkv.bits == 8
        assert isinstance(layer.mlp.gate, nn.Linear)  # BF16 in the pack
        assert isinstance(
            wrapped.language_model.model.embed_tokens, nn.QuantizedEmbedding
        )

    def test_processor_carries_the_pack_tokenizer(self, loaded):
        _, _, processor, _ = loaded
        tokens, _, _ = tiny_tokens()
        tokenizer = processor.tokenizer
        assert tokenizer.encode("<|im_end|>", add_special_tokens=False) == [
            tokens.index("<|im_end|>")
        ]

    def test_forward_produces_finite_logits(self, loaded):
        _, wrapped, _, store = loaded
        tokens, _, _ = tiny_tokens()
        ids = mx.array([[5, 17, 42, 7, 99, 3]])
        out = wrapped.language_model(ids)
        logits = out.logits if hasattr(out, "logits") else out
        mx.eval(logits)
        assert logits.shape == (1, 6, len(tokens))
        assert bool(mx.all(mx.isfinite(logits.astype(mx.float32))).item())
        assert store.stats.snapshot()["cache_misses"] > 0

    def test_forward_from_another_thread(self, loaded):
        # Load and generation run on different asyncio.to_thread workers; under
        # MLX's thread-local streams any weight left lazy at load would raise
        # "There is no Stream(gpu, N) in current thread" here.
        _, wrapped, _, _ = loaded
        result = {}

        def run():
            try:
                out = wrapped.language_model(mx.array([[1, 2, 3]]))
                logits = out.logits if hasattr(out, "logits") else out
                mx.eval(logits)
                result["shape"] = logits.shape
            except Exception as exc:  # noqa: BLE001
                result["error"] = exc

        worker = threading.Thread(target=run)
        worker.start()
        worker.join()
        assert "error" not in result, result.get("error")
        assert result["shape"][:2] == (1, 3)


class TestMoeBlock:
    def test_block_matches_router_plus_latent_plus_shared_reference(self, loaded):
        full, wrapped, _, _ = loaded
        layer = 2
        skeleton = full.skeleton
        rng = np.random.default_rng(11)
        x = rng.standard_normal((1, 4, TINY_HIDDEN)).astype(np.float32)

        previous = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            got = np.array(wrapped.layers[layer].mlp(mx.array(x)).astype(mx.float32))
        finally:
            mx.set_default_device(previous)

        x64 = x.astype(np.float64)
        logits = x64 @ skeleton[f"blk.{layer}.ffn_gate_inp.weight"].astype(np.float64).T
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        inds = np.argsort(-probs, axis=-1, kind="stable")[..., :2]
        scores = np.take_along_axis(probs, inds, axis=-1)
        scores /= scores.sum(-1, keepdims=True)
        routed = _reference_forward(full.latent, layer, x64, inds, scores)

        shared = _silu(
            x64 @ skeleton[f"blk.{layer}.ffn_gate_shexp.weight"].astype(np.float64).T
        ) * (x64 @ skeleton[f"blk.{layer}.ffn_up_shexp.weight"].astype(np.float64).T)
        shared = (
            shared @ skeleton[f"blk.{layer}.ffn_down_shexp.weight"].astype(np.float64).T
        )
        gate = (
            x64
            @ skeleton[f"blk.{layer}.ffn_gate_inp_shexp.weight"].astype(np.float64).T
        )
        expected = routed + shared / (1.0 + np.exp(-gate))

        # The shared expert's Q8_0 scales are rounded to bfloat16 (<0.4% per
        # block); the routed path is exact to float32.
        np.testing.assert_allclose(
            got, expected, rtol=0, atol=6e-3 * np.abs(expected).max()
        )


class TestVerification:
    def test_missing_trunk_tensor_is_reported(self, tmp_path):
        full = make_full_pack(tmp_path, with_vision=False)
        convert.convert_pack(full.root)
        shard = sorted(full.root.glob("model-*.safetensors"))[0]
        weights = dict(mx.load(str(shard)))
        dropped = "language_model.model.layers.0.mlp.gate.weight"
        del weights[dropped]
        # mx.load is lazy: materialize before replacing the file it reads.
        mx.eval(weights)
        rewritten = shard.with_name("rewritten.safetensors")
        mx.save_safetensors(str(rewritten), weights, metadata={"format": "mlx"})
        rewritten.replace(shard)
        with pytest.raises(Edge0PackError, match="mlp.gate.weight"):
            _load(full.root)

    def test_pack_without_vision_loads_text_only(self, tmp_path):
        full = make_full_pack(tmp_path, with_vision=False)
        convert.convert_pack(full.root)
        wrapped, _, store = _load(full.root)
        try:
            out = wrapped.language_model(mx.array([[1, 2]]))
            logits = out.logits if hasattr(out, "logits") else out
            mx.eval(logits)
            assert logits.shape[1] == 2
        finally:
            store.close()

    def test_conversion_from_another_pack_is_refused(self, tmp_path):
        full = make_full_pack(tmp_path, with_vision=False)
        convert.convert_pack(full.root)
        config_path = full.root / "flash_moe" / "flash_moe_config.json"
        import json

        config = json.loads(config_path.read_text())
        config["latent"]["latent"] = 512
        config_path.write_text(json.dumps(config))
        with pytest.raises(Edge0PackError, match="different pack"):
            _load(full.root)


class TestLoadVlmWiring:
    def test_converted_pack_loads_non_strict_with_the_ple_hook(
        self, tmp_path, monkeypatch
    ):
        import mlx_vlm
        from mlx_vlm.models.qwen4_exp import ple_storage

        from olmlx.engine import vlm_load

        full = make_full_pack(tmp_path, with_vision=False)
        convert.convert_pack(full.root)
        calls = {}

        def fake_load(path, **kwargs):
            calls["path"], calls["kwargs"] = path, kwargs
            from olmlx.engine import qwen4_exp_ple
            from olmlx.engine.edge0 import ple as edge0_ple

            calls["hooked"] = getattr(
                ple_storage.QuantizedMMapNGramEmbedding, "_olmlx_ple_dispatch", False
            ) and (
                qwen4_exp_ple._PLE_TABLES.get(edge0_ple.MANIFEST_FORMAT)
                is edge0_ple.Fp8NGramEmbedding
            )
            return mx.zeros(1), object()

        monkeypatch.setattr(mlx_vlm, "load", fake_load)
        monkeypatch.setattr(
            "olmlx.engine.model_manager._materialize_module_buffers", lambda m: None
        )
        vlm_load.load_vlm(str(full.root), lazy=True)
        assert calls["path"] == str(full.root)
        assert calls["kwargs"] == {"strict": False, "lazy": True}
        assert calls["hooked"] is True

    def test_other_models_load_strictly(self, tmp_path, monkeypatch):
        import json

        import mlx_vlm

        from olmlx.engine import vlm_load

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
        calls = {}
        monkeypatch.setattr(
            mlx_vlm, "load", lambda path, **kw: calls.update(kw) or (object(), object())
        )
        monkeypatch.setattr(
            "olmlx.engine.model_manager._materialize_module_buffers", lambda m: None
        )
        vlm_load.load_vlm(str(tmp_path))
        assert "strict" not in calls
