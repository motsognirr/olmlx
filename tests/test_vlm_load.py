"""Conformance test for the mlx-vlm load chokepoint.

All ``mlx_vlm.load`` calls must go through ``olmlx.engine.vlm_load.load_vlm``
so load-time workarounds (engine/gemma4_sanitize_fix.py) are applied
structurally — a call site that bypasses the chokepoint silently loses them.
"""

import re
from pathlib import Path

OLMLX_ROOT = Path(__file__).resolve().parents[1] / "olmlx"


def test_no_direct_mlx_vlm_load_call_sites():
    offenders = []
    for path in OLMLX_ROOT.rglob("*.py"):
        if path.name == "vlm_load.py":
            continue
        text = path.read_text()
        for i, line in enumerate(text.splitlines(), start=1):
            if re.search(r"\bmlx_vlm\s*\.\s*load\s*\(", line):
                offenders.append(f"{path.relative_to(OLMLX_ROOT.parent)}:{i}")
    assert not offenders, (
        "Direct mlx_vlm.load call sites found — route them through "
        "olmlx.engine.vlm_load.load_vlm so load-time workarounds apply: "
        + ", ".join(offenders)
    )
