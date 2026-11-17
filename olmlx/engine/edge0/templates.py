"""Static config an Edge0 pack does not carry.

An Edge0 pack ships weights and a GGUF header but none of the Hugging Face
side files (``config.json`` processor/tokenizer configs) that mlx-vlm loads a
``qwen4_exp`` model from. The values here are Qwen3.8-Flash-Next's — Edge0
keeps that model's tokenizer, chat template, vision tower and processor
unchanged — and :mod:`olmlx.engine.edge0.convert` fills in everything the
pack does state from its GGUF metadata.
"""

from __future__ import annotations

#: Added tokens llama.cpp types as "control" but the HF tokenizer does not
#: mark special (so ``skip_special_tokens`` keeps them).
NON_SPECIAL_CONTROL_TOKENS = frozenset(
    {
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|fim_pad|>",
        "<|repo_name|>",
        "<|file_sep|>",
    }
)

#: ``(?i:...)`` spelling of the QWEN35 pre-tokenizer split llama.cpp uses.
PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

TOKENIZER_CONFIG = {
    "add_prefix_space": False,
    "audio_bos_token": "<|audio_start|>",
    "audio_eos_token": "<|audio_end|>",
    "audio_token": "<|audio_pad|>",
    "backend": "tokenizers",
    "bos_token": None,
    "clean_up_tokenization_spaces": False,
    "eos_token": "<|im_end|>",
    "errors": "replace",
    "image_token": "<|image_pad|>",
    "model_specific_special_tokens": {
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        "audio_token": "<|audio_pad|>",
        "image_token": "<|image_pad|>",
        "video_token": "<|video_pad|>",
        "vision_bos_token": "<|vision_start|>",
        "vision_eos_token": "<|vision_end|>",
    },
    "pad_token": "<|endoftext|>",
    "pretokenize_regex": PRETOKENIZE_REGEX,
    "processor_class": "Qwen3VLProcessor",
    "split_special_tokens": False,
    "tokenizer_class": "Qwen2Tokenizer",
    "unk_token": None,
    "video_token": "<|video_pad|>",
    "vision_bos_token": "<|vision_start|>",
    "vision_eos_token": "<|vision_end|>",
}

_IMAGE_NORM = {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}

PREPROCESSOR_CONFIG = {
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    **_IMAGE_NORM,
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}

VIDEO_PREPROCESSOR_CONFIG = {
    "size": {"longest_edge": 25165824, "shortest_edge": 4096},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    **_IMAGE_NORM,
    "processor_class": "Qwen3VLProcessor",
    "video_processor_type": "Qwen3VLVideoProcessor",
}

PROCESSOR_CONFIG = {
    "image_processor": {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        **_IMAGE_NORM,
        "image_processor_type": "Qwen3VLImageProcessor",
        "max_pixels": 16777216,
        "merge_size": 2,
        "min_pixels": 65536,
        "patch_size": 16,
        "rescale_factor": 0.00392156862745098,
        "temporal_patch_size": 2,
    },
    "processor_class": "Qwen3VLProcessor",
    "video_processor": {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "fps": 2.0,
        **_IMAGE_NORM,
        "max_frames": 768,
        "max_pixels": 25165824,
        "merge_size": 2,
        "min_frames": 4,
        "min_pixels": 4096,
        "patch_size": 16,
        "rescale_factor": 0.00392156862745098,
        "temporal_patch_size": 2,
        "video_processor_type": "Qwen3VLVideoProcessor",
    },
}

#: Vision tower geometry when the pack ships no projector to read it from.
DEFAULT_VISION_CONFIG = {
    "deepstack_visual_indexes": [],
    "depth": 27,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 1152,
    "in_channels": 3,
    "initializer_range": 0.02,
    "intermediate_size": 4304,
    "model_type": "qwen4_exp",
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": 2560,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}

#: Special-token ids by name; resolved against the pack vocabulary.
SPECIAL_TOKEN_NAMES = {
    "image_token_id": "<|image_pad|>",
    "video_token_id": "<|video_pad|>",
    "vision_start_token_id": "<|vision_start|>",
    "vision_end_token_id": "<|vision_end|>",
}
