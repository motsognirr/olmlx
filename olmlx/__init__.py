import os as _os

__version__ = "0.1.0"

# mlx >= 0.32.3 sends single-token decode attention with head dim 512 and GQA 8
# (Gemma 4's full-attention layers) to a generic vector kernel once the context
# reaches MLX_SDPA_D512_MIN_KL keys (default 1024). On chips without NAX
# (pre-M5) that kernel is several times slower than the unfused path:
# Gemma-4-26B-A4B decode on an M4 Pro lost 17% at 4k context and 46% at 32k.
# A huge threshold keeps the unfused path. mlx reads the variable on every
# attention call, so setting it here (before any generation) is enough, and a
# value the user exported wins.
_os.environ.setdefault("MLX_SDPA_D512_MIN_KL", "1000000000")
