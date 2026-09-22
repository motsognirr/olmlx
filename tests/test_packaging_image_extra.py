"""mflux image generation lives in the optional [image] extra (#723)."""

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _project():
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]


def test_mflux_not_core():
    core = " ".join(_project()["dependencies"]).lower()
    assert "mflux" not in core, "mflux must stay in the optional [image] extra"


def test_image_extra_declares_bounded_mflux():
    image = _project()["optional-dependencies"]["image"]
    (req,) = [r for r in image if r.lower().startswith("mflux")]
    # olmlx reaches into mflux internals (variant module paths, callbacks), so
    # the extra must carry an upper bound.
    assert "<" in req, f"mflux requirement must be upper-bounded, got {req!r}"
