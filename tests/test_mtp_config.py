from olmlx.engine.registry import _VALID_SPECULATIVE_STRATEGIES


def test_mtp_is_a_valid_strategy():
    assert "mtp" in _VALID_SPECULATIVE_STRATEGIES


from olmlx.engine.mtp.draft_model import MTPConfig

_DENSE_CFG = {
    "block_size": 3,
    "model_type": "qwen3_5_mtp",
    "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    "text_config": {
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "full_attention_interval": 4,
        "tie_word_embeddings": False,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
    },
}

_MOE_CFG = {
    "block_size": 3,
    "model_type": "qwen3_5_mtp",
    "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    "text_config": {
        **_DENSE_CFG["text_config"],
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "shared_expert_intermediate_size": 768,
        "norm_topk_prob": True,
    },
}


def test_mtp_config_parses_dense():
    cfg = MTPConfig.from_dict(_DENSE_CFG)
    assert cfg.block_size == 3
    assert cfg.hidden_size == 5120
    assert cfg.head_dim == 256
    assert cfg.num_key_value_heads == 4
    assert cfg.vocab_size == 248320
    assert cfg.num_experts == 0
    assert cfg.quant_group_size == 64 and cfg.quant_bits == 4


def test_mtp_config_parses_moe():
    cfg = MTPConfig.from_dict(_MOE_CFG)
    assert cfg.num_experts == 128
    assert cfg.num_experts_per_tok == 8
    assert cfg.is_moe is True


def test_mtp_config_to_qwen35_text_args():
    cfg = MTPConfig.from_dict(_DENSE_CFG)
    args = cfg.to_qwen35_text_args()
    assert args.hidden_size == 5120
