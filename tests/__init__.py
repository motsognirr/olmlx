"""Test package init — runs before tests/conftest.py imports anything.

OLMLX_TESTS_CPU_DEVICE=1 moves MLX's default device to the CPU backend for
the unit suite. CI sets it because two self-hosted runners share one Mac
with real-inference jobs: concurrent Metal pressure from the unit suite
(64 test files do real mx.* compute) starved the runner heartbeat and got
jobs killed with "runner lost communication" (#596).

This must happen before ``mlx_lm``/``mlx_vlm`` are imported anywhere:
their module-level ``generation_stream`` binds the default device at
import time, and tests/conftest.py imports both. Package ``__init__`` is
the one hook guaranteed to run first.

Tests that exercise Metal kernels explicitly (shardquant kernel parity,
the rope batched-decode parity gate) opt back into the GPU via the
``metal_default_device`` fixture in tests/conftest.py.
"""

import os

if os.environ.get("OLMLX_TESTS_CPU_DEVICE") == "1":
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
