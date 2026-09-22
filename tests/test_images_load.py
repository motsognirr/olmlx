"""Image-model (mflux) load path (#723).

mflux lives in the optional ``[image]`` extra, so every test here installs a
fake ``mflux`` into ``sys.modules`` — the real package is never required.
"""

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from olmlx.engine.model_manager import LoadedModel, ModelManager
from olmlx.engine.registry import ModelRegistry
from olmlx.engine.template_caps import TemplateCaps


class _FakeMfluxConfig:
    def __init__(self, model_name, aliases):
        self.model_name = model_name
        self.aliases = aliases


_AVAILABLE = {
    "qwen-image": _FakeMfluxConfig(
        "Qwen/Qwen-Image-2512", ["qwen-image", "qwen", "qwen-image-2512"]
    ),
    "qwen-image-2.1": _FakeMfluxConfig(
        "Qwen/Qwen-Image-2.1", ["qwen-image-2.1", "qwen-2.1"]
    ),
    "dev": _FakeMfluxConfig("black-forest-labs/FLUX.1-dev", ["dev"]),
}


def _stub_mflux(monkeypatch, *, qwen_image_cls=None, qwen21_cls=None):
    """Install a fake mflux package exposing the modules image_gen imports."""
    mods = {}

    def _mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        mods[name] = m
        monkeypatch.setitem(sys.modules, name, m)
        return m

    _mod("mflux")
    _mod("mflux.models.common.config.model_config", AVAILABLE_MODELS=_AVAILABLE)
    _mod(
        "mflux.models.qwen.variants.txt2img.qwen_image",
        QwenImage=qwen_image_cls or MagicMock(name="QwenImage"),
    )
    _mod(
        "mflux.models.qwen21.variants.txt2img.qwen_image_21",
        QwenImage21=qwen21_cls or MagicMock(name="QwenImage21"),
    )
    return mods


@pytest.fixture
def registry(tmp_path, monkeypatch):
    cfg = {
        "qwen-image:2.1": {
            "type": "image",
            "hf_path": "Qwen/Qwen-Image-2.1",
            "image_quantize": 8,
        },
        "qwen-image:20b": {"type": "image", "hf_path": "Qwen/Qwen-Image-2512"},
        # Declared image, but not an mflux-supported repo — must fail at load.
        "bogus:image": {"type": "image", "hf_path": "Qwen/Qwen-Image-Typo"},
        "qwen3:32b": "Qwen/Qwen3-32B-4bit",
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr("olmlx.engine.registry.settings.models_config", path)
    reg = ModelRegistry()
    reg.load()
    return reg


def _bare_manager(registry, store=None):
    mgr = ModelManager.__new__(ModelManager)
    mgr.registry = registry
    mgr.store = store
    return mgr


class TestModelKindFromEntry:
    """The image kind comes from the *requested* entry, never from hf_path."""

    @staticmethod
    def _manager(registry, monkeypatch):
        mgr = ModelManager(registry, None)
        captured = {}

        def fake_load(hf_path, *args):
            captured["hf_path"] = hf_path
            captured["image_config"] = args[-1] if args else None
            return (MagicMock(), None, False, TemplateCaps(), False, None)

        monkeypatch.setattr(mgr, "_load_model_and_shard", fake_load)
        monkeypatch.setattr(
            mgr,
            "_detect_model_kind",
            MagicMock(side_effect=AssertionError("image kind must not be sniffed")),
        )
        return mgr, captured

    @pytest.mark.asyncio
    async def test_declared_image_entry_skips_detection(self, registry, monkeypatch):
        mgr, captured = self._manager(registry, monkeypatch)
        lm = await mgr.ensure_loaded("qwen-image:2.1")
        assert lm.is_image is True
        assert captured["image_config"].image_quantize == 8
        assert captured["hf_path"] == "Qwen/Qwen-Image-2.1"

    @pytest.mark.asyncio
    async def test_entries_sharing_hf_path_keep_their_own_quantize(
        self, tmp_path, monkeypatch
    ):
        cfg = {
            "qwen-image:q4": {
                "type": "image",
                "hf_path": "Qwen/Qwen-Image-2.1",
                "image_quantize": 4,
            },
            "qwen-image:q8": {
                "type": "image",
                "hf_path": "Qwen/Qwen-Image-2.1",
                "image_quantize": 8,
            },
        }
        path = tmp_path / "models.json"
        path.write_text(json.dumps(cfg))
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", path)
        reg = ModelRegistry()
        reg.load()
        mgr, captured = self._manager(reg, monkeypatch)
        await mgr.ensure_loaded("qwen-image:q8")
        assert captured["image_config"].image_quantize == 8

    def test_detect_never_returns_image(self, registry, tmp_path):
        # Misclassification regression: mflux's ModelConfig.from_name resolves
        # Qwen/Qwen3-32B-4bit to a Qwen-Image base. Detection must never use
        # it — the config.json path decides, and never says "image".
        mgr = _bare_manager(registry)
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"model_type": "qwen3"}))
        with patch("huggingface_hub.hf_hub_download", return_value=str(cfg)):
            assert mgr._detect_model_kind("Qwen/Qwen3-32B-4bit") == "text"
            assert mgr._detect_model_kind("Qwen/Qwen-Image-2.1") != "image"


class TestResolveVariant:
    def test_exact_model_name_resolves(self, monkeypatch):
        from olmlx.engine import image_gen

        _stub_mflux(monkeypatch)
        variant, cfg = image_gen.resolve_image_variant("Qwen/Qwen-Image-2.1")
        assert variant.key == "qwen-image-2.1"
        assert cfg is _AVAILABLE["qwen-image-2.1"]

    def test_qwen_image_alias_is_2512(self, monkeypatch):
        from olmlx.engine import image_gen

        _stub_mflux(monkeypatch)
        variant, cfg = image_gen.resolve_image_variant("Qwen/Qwen-Image-2512")
        assert variant.key == "qwen-image"

    @pytest.mark.parametrize(
        "hf_path",
        [
            "Qwen/Qwen3-32B-4bit",  # substring-matcher hazard
            "Qwen/Qwen-Image-2.1-typo",
            "mlx-community/Qwen-Image-2.1-4bit",  # not an exact mflux name
            "black-forest-labs/FLUX.1-dev",  # mflux model, unsupported variant
        ],
    )
    def test_non_exact_or_unsupported_rejected(self, monkeypatch, hf_path):
        from olmlx.engine import image_gen

        _stub_mflux(monkeypatch)
        with pytest.raises(ValueError, match="not a supported image model"):
            image_gen.resolve_image_variant(hf_path)


class TestLoadModelImage:
    def test_load_dispatches_to_variant_class(self, registry, monkeypatch):
        fake_model = MagicMock()
        cls21 = MagicMock(return_value=fake_model)
        _stub_mflux(monkeypatch, qwen21_cls=cls21)
        mgr = _bare_manager(registry)
        with patch("olmlx.engine.model_manager._materialize_image_model") as mat:
            model, tok, is_vlm, caps, dec = mgr._load_model_image(
                "Qwen/Qwen-Image-2.1", registry.resolve("qwen-image:2.1")
            )
        assert model is fake_model
        assert tok is None and is_vlm is False and dec is None
        assert isinstance(caps, TemplateCaps)
        cls21.assert_called_once()
        kwargs = cls21.call_args.kwargs
        assert kwargs["quantize"] == 8  # from models.json image_quantize
        # mflux owns resolution + download: never a local path.
        assert kwargs["model_path"] is None
        assert kwargs["model_config"] is _AVAILABLE["qwen-image-2.1"]
        mat.assert_called_once_with(fake_model)

    def test_load_20b_uses_qwen_image_class(self, registry, monkeypatch):
        cls = MagicMock(return_value=MagicMock())
        _stub_mflux(monkeypatch, qwen_image_cls=cls)
        mgr = _bare_manager(registry)
        with patch("olmlx.engine.model_manager._materialize_image_model"):
            mgr._load_model_image(
                "Qwen/Qwen-Image-2512", registry.resolve("qwen-image:20b")
            )
        assert cls.call_args.kwargs["quantize"] is None

    def test_declared_but_unsupported_repo_rejected(self, registry, monkeypatch):
        cls = MagicMock()
        _stub_mflux(monkeypatch, qwen_image_cls=cls, qwen21_cls=cls)
        mgr = _bare_manager(registry)
        with pytest.raises(ValueError, match="not a supported image model"):
            mgr._load_model_image(
                "Qwen/Qwen-Image-Typo", registry.resolve("bogus:image")
            )
        cls.assert_not_called()

    def test_missing_mflux_gives_install_hint(self, registry, monkeypatch):
        monkeypatch.setitem(sys.modules, "mflux", None)
        monkeypatch.setitem(
            sys.modules, "mflux.models.common.config.model_config", None
        )
        mgr = _bare_manager(registry)
        with pytest.raises(ValueError, match=r"uv sync --extra image"):
            mgr._load_model_image(
                "Qwen/Qwen-Image-2.1", registry.resolve("qwen-image:2.1")
            )

    def test_drifted_mflux_is_not_reported_as_missing_extra(
        self, registry, monkeypatch
    ):
        # mflux installed, but the variant module is gone (internal drift):
        # reinstalling the extra would not help, so don't say it would.
        _stub_mflux(monkeypatch)
        monkeypatch.setitem(
            sys.modules, "mflux.models.qwen21.variants.txt2img.qwen_image_21", None
        )
        mgr = _bare_manager(registry)
        with patch("importlib.util.find_spec", return_value=object()):
            with pytest.raises(RuntimeError, match="incompatible") as ei:
                mgr._load_model_image(
                    "Qwen/Qwen-Image-2.1", registry.resolve("qwen-image:2.1")
                )
        assert "--extra image" not in str(ei.value)

    def test_load_model_skips_store_download(self, registry, monkeypatch):
        # mflux resolves + downloads image models itself; the olmlx store must
        # not try to snapshot a diffusers-layout repo.
        store = MagicMock()
        mgr = _bare_manager(registry, store=store)
        sentinel = (object(), None, False, TemplateCaps(), None)
        mc = registry.resolve("qwen-image:2.1")
        with patch.object(
            ModelManager, "_load_model_image", return_value=sentinel
        ) as li:
            out = mgr._load_model("Qwen/Qwen-Image-2.1", image_config=mc)
        assert out is sentinel
        li.assert_called_once_with("Qwen/Qwen-Image-2.1", mc)
        store.ensure_downloaded.assert_not_called()


class TestGuards:
    def _image_lm(self):
        return LoadedModel(
            name="qwen-image:2.1",
            hf_path="Qwen/Qwen-Image-2.1",
            model=MagicMock(),
            tokenizer=None,
            is_image=True,
        )

    def test_loadedmodel_is_image_default_false(self):
        lm = LoadedModel(name="x", hf_path="x", model=object(), tokenizer=None)
        assert lm.is_image is False

    def test_batch_ineligible(self):
        from olmlx.engine.inference import _batch_eligible

        lm = self._image_lm()
        lm.batching = True
        assert _batch_eligible(lm, {}, max_tokens=16, images=None, audio=None) is False

    @pytest.mark.asyncio
    async def test_prompt_cache_probe_skipped(self):
        lm = self._image_lm()
        mgr = ModelManager.__new__(ModelManager)
        with patch("mlx_lm.models.cache.make_prompt_cache") as mk:
            await mgr._probe_cache_capabilities(lm)
        mk.assert_not_called()

    def test_adapter_base_rejected(self):
        with pytest.raises(ValueError, match="image"):
            ModelManager._reject_adapter_base(self._image_lm())

    def test_close_skips_grammar_drop(self):
        lm = self._image_lm()
        with patch("olmlx.engine.grammar.drop_for_tokenizer") as drop:
            ModelManager._close_loaded_model(lm)
        drop.assert_not_called()


class TestPullRejectsImageModels:
    @pytest.mark.asyncio
    async def test_pull_image_model_rejected(self, registry, tmp_path, monkeypatch):
        # mflux downloads image models into the HF cache on first use; a store
        # pull would put a second, never-used copy into OLMLX_MODELS_DIR.
        from olmlx.models.store import ModelStore

        monkeypatch.setattr("olmlx.models.store.settings.models_dir", tmp_path / "m")
        store = ModelStore(registry)
        with patch.object(store, "ensure_downloaded") as dl:
            with pytest.raises(ValueError, match="image model"):
                async for _ in store.pull("qwen-image:2.1"):
                    pass
        dl.assert_not_called()
