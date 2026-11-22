"""MLX environment defaults applied when olmlx is imported.

mlx >= 0.32.3 routes single-token decode attention with head dim 512 and GQA 8
(Gemma 4's full-attention layers) to a generic vector kernel once the context
reaches ``MLX_SDPA_D512_MIN_KL`` keys (default 1024). On chips without NAX
(pre-M5) that kernel is far slower than the unfused path mlx 0.32.2 used:
Gemma-4-26B-A4B decode on an M4 Pro dropped 17% at 4k context and 46% at 32k,
and a bare SDPA call at 32k took 5.15 ms vs 1.25 ms. olmlx raises the
threshold so the unfused path is used, unless the user set the variable.
mlx reads it on every attention call, so setting it at import is enough.
"""

import os
import subprocess
import sys

_PROBE = "import os, olmlx; print(os.environ.get('MLX_SDPA_D512_MIN_KL'))"


def _probe(env_overrides: dict[str, str | None]) -> str:
    env = dict(os.environ)
    for k, v in env_overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_import_disables_the_d512_vector_kernel_by_default():
    assert _probe({"MLX_SDPA_D512_MIN_KL": None}) == "1000000000"


def test_user_setting_is_kept():
    assert _probe({"MLX_SDPA_D512_MIN_KL": "1024"}) == "1024"
