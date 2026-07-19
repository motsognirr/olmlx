"""Audio/TTS deps live in the [audio] extra, not core (#469).

A default ``uv sync`` must not pull mlx-audio + misaki + spaCy + the
en_core_web_sm wheel (TTS-only) or torchvision (transformers falls back to
its PIL image-processor backend). torch stays installed transitively via
xgrammar and mlx-whisper but must not be pinned by olmlx itself.

The spaCy model ``en_core_web_sm`` is a special case: spaCy models are not on
PyPI (distributed as GitHub-release wheels), so a bare ``en-core-web-sm``
requirement is unresolvable for ``pip install "olmlx[audio]"`` and the
``[tool.uv.sources]`` URL override does not travel in published wheel metadata.
It therefore lives in the ``dev`` dependency-group (so ``uv sync`` still
provisions it) and is NOT in the published ``[audio]`` extra; end users fetch
it once with ``python -m spacy download en_core_web_sm``.
"""

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject():
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="module")
def project(pyproject):
    return pyproject["project"]


def _names(requirements):
    """Normalized distribution names, extras/specifiers stripped."""
    names = []
    for req in requirements:
        name = req.split(";")[0].split("[")[0]
        for sep in ("==", ">=", "<=", "~=", ">", "<", "!="):
            name = name.split(sep)[0]
        names.append(name.strip().lower())
    return names


def test_audio_deps_not_in_core_dependencies(project):
    core = _names(project["dependencies"])
    for dep in ("mlx-audio", "misaki", "en-core-web-sm", "torch", "torchvision"):
        assert dep not in core, f"{dep} must not be a core dependency (#469)"


def test_audio_extra_carries_tts_deps(project):
    audio = _names(project["optional-dependencies"]["audio"])
    for dep in ("mlx-audio", "misaki"):
        assert dep in audio, f"{dep} missing from the [audio] extra"


def test_spacy_model_excluded_from_published_extras(project):
    # en-core-web-sm is unresolvable on PyPI, so it must NOT appear in any
    # published extra — otherwise `pip install "olmlx[audio]"` fails on an
    # unsatisfiable Requires-Dist.
    for extra, reqs in project["optional-dependencies"].items():
        assert "en-core-web-sm" not in _names(reqs), (
            f"en-core-web-sm must not be in the published [{extra}] extra "
            "(not on PyPI — provisioned via the dev group instead)"
        )


def test_spacy_model_provisioned_via_dev_group(pyproject):
    # Kept for dev: `uv sync` must still install the spaCy model via the
    # [tool.uv.sources] URL override, which only applies to the dev group.
    dev = _names(pyproject["dependency-groups"]["dev"])
    assert "en-core-web-sm" in dev, (
        "en-core-web-sm must stay in the dev group so `uv sync` provisions it"
    )


def test_voice_extra_includes_audio_extra(project):
    # `--voice` speaks replies via Kokoro TTS, so installing [voice] alone
    # must also pull the [audio] deps (self-referential extra).
    voice = project["optional-dependencies"]["voice"]
    assert any(
        req.replace(" ", "").lower().startswith("olmlx[audio]") for req in voice
    ), "[voice] extra must include olmlx[audio]"


def test_lockfile_expands_voice_extra_to_audio_deps():
    # The self-reference is only as good as its resolution: assert uv.lock
    # actually expanded olmlx[voice] to the audio deps (a malformed
    # self-reference would resolve to sounddevice alone).
    with (PYPROJECT.parent / "uv.lock").open("rb") as f:
        lock = tomllib.load(f)
    olmlx = next(p for p in lock["package"] if p["name"] == "olmlx")
    extras = olmlx["optional-dependencies"]
    audio = {d["name"] for d in extras["audio"]}
    voice = {d["name"] for d in extras["voice"]}
    assert {"mlx-audio", "misaki"} <= audio
    assert "en-core-web-sm" not in audio
    assert audio | {"sounddevice"} <= voice
