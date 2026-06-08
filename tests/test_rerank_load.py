from unittest.mock import patch

from olmlx.engine.model_manager import LoadedModel, ModelManager


async def test_probe_cache_capabilities_skips_reranker(registry):
    # A reranker has no LLM prompt cache; the probe must early-return BEFORE
    # touching mlx-lm's make_prompt_cache, and leave persistence off. Patching
    # make_prompt_cache and asserting it's never called proves the guard
    # short-circuited the body (the persistence flags default to False, so they
    # alone wouldn't catch a regression that ran the probe).
    mgr = ModelManager(registry)
    lm = LoadedModel(
        name="bge",
        hf_path="bge",
        model=object(),
        tokenizer=object(),
        is_reranker=True,
    )
    with patch("mlx_lm.models.cache.make_prompt_cache") as mock_make:
        await mgr._probe_cache_capabilities(lm)  # must not raise
    mock_make.assert_not_called()
    assert lm.supports_cache_persistence is False
    assert lm.uses_checkpoint_persistence is False


def test_load_model_reranker_branch(registry, monkeypatch, tmp_path):
    # The reranker branch builds the encoder via load_cross_encoder and the
    # tokenizer via transformers.AutoTokenizer, returning the standard
    # (model, tokenizer, is_vlm=False, caps, decoder=None) tuple.
    import olmlx.engine.model_manager as mm

    sentinel_model = object()
    sentinel_tok = object()

    monkeypatch.setattr(
        mm.ModelManager, "_detect_model_kind", lambda self, hf: "reranker"
    )
    monkeypatch.setattr(
        "olmlx.engine.rerank.load_cross_encoder", lambda path: sentinel_model
    )
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda path, **kw: sentinel_tok
    )

    mgr = ModelManager(registry)
    model, tokenizer, is_vlm, caps, decoder = mgr._load_model(str(tmp_path))

    assert model is sentinel_model
    assert tokenizer is sentinel_tok
    assert is_vlm is False
    assert decoder is None
