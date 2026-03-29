"""Tests for olmlx.bench.scenarios."""

import json

import pytest

from olmlx.bench.scenarios import SCENARIOS, Scenario, get_scenarios


class TestScenario:
    def test_to_dict_roundtrip(self):
        s = Scenario(
            name="test",
            description="A test scenario",
            env_overrides={"FOO": "bar"},
        )
        d = s.to_dict()
        restored = Scenario.from_dict(d)
        assert restored.name == s.name
        assert restored.description == s.description
        assert restored.env_overrides == s.env_overrides


class TestScenariosList:
    def test_has_scenarios(self):
        assert len(SCENARIOS) >= 11  # 9 standard + 2 distributed

    def test_all_have_required_fields(self):
        for s in SCENARIOS:
            assert s.name
            assert s.description

    def test_unique_names(self):
        names = [s.name for s in SCENARIOS]
        assert len(names) == len(set(names))

    def test_env_keys_have_olmlx_prefix(self):
        for s in SCENARIOS:
            for key in s.env_overrides:
                assert key.startswith("OLMLX_"), (
                    f"Scenario {s.name!r} has non-OLMLX env key {key!r}"
                )

    def test_baseline_has_no_overrides(self):
        baseline = [s for s in SCENARIOS if s.name == "baseline"]
        assert len(baseline) == 1
        assert baseline[0].env_overrides == {}


class TestGetScenarios:
    def test_none_returns_all(self):
        result = get_scenarios(None)
        assert len(result) == len(SCENARIOS)

    def test_filter_by_name(self):
        result = get_scenarios(["baseline", "no-cache"])
        assert len(result) == 2
        assert result[0].name == "baseline"
        assert result[1].name == "no-cache"

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_scenarios(["nonexistent"])


class TestSkipChecks:
    def test_baseline_never_skips(self, tmp_path):
        baseline = get_scenarios(["baseline"])[0]
        assert not baseline.should_skip(tmp_path)

    def test_flash_skips_without_layout(self, tmp_path):
        flash = get_scenarios(["flash"])[0]
        assert flash.should_skip(tmp_path)

    def test_flash_runs_with_layout(self, tmp_path):
        flash_dir = tmp_path / "flash"
        flash_dir.mkdir()
        (flash_dir / "flash_layout.json").write_text("{}")
        flash = get_scenarios(["flash"])[0]
        assert not flash.should_skip(tmp_path)

    def test_flash_moe_skips_without_moe_config(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 768}))
        moe = get_scenarios(["flash-moe"])[0]
        assert moe.should_skip(tmp_path)

    def test_flash_moe_runs_with_moe_config(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"num_local_experts": 8, "hidden_size": 768})
        )
        moe = get_scenarios(["flash-moe"])[0]
        assert not moe.should_skip(tmp_path)

    def test_flash_moe_detects_routed_experts(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"n_routed_experts": 4}))
        moe = get_scenarios(["flash-moe"])[0]
        assert not moe.should_skip(tmp_path)

    def test_flash_moe_detects_text_config_wrapper(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"text_config": {"num_local_experts": 8}})
        )
        moe = get_scenarios(["flash-moe"])[0]
        assert not moe.should_skip(tmp_path)

    def test_flash_moe_skips_with_non_object_config(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps([1, 2, 3]))
        moe = get_scenarios(["flash-moe"])[0]
        assert moe.should_skip(tmp_path)

    def test_flash_moe_skips_with_invalid_json_config(self, tmp_path):
        (tmp_path / "config.json").write_text("not json")
        moe = get_scenarios(["flash-moe"])[0]
        assert moe.should_skip(tmp_path)


class TestDistributedScenarios:
    def test_distributed_scenarios_exist(self):
        names = [s.name for s in SCENARIOS]
        assert "distributed" in names
        assert "distributed+tq4" in names

    def test_distributed_is_server_mode(self):
        dist = get_scenarios(["distributed"])[0]
        assert dist.server_mode is True

    def test_distributed_tq4_is_server_mode(self):
        dist = get_scenarios(["distributed+tq4"])[0]
        assert dist.server_mode is True

    def test_distributed_has_env_overrides(self):
        dist = get_scenarios(["distributed"])[0]
        assert dist.env_overrides.get("OLMLX_EXPERIMENTAL_DISTRIBUTED") == "true"

    def test_distributed_tq4_has_both_overrides(self):
        dist = get_scenarios(["distributed+tq4"])[0]
        assert dist.env_overrides.get("OLMLX_EXPERIMENTAL_DISTRIBUTED") == "true"
        assert (
            dist.env_overrides.get("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT")
            == "turboquant:4"
        )

    def test_distributed_skips_without_hostfile(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "olmlx.bench.scenarios._DEFAULT_HOSTFILE", tmp_path / "nonexistent.json"
        )
        dist = get_scenarios(["distributed"])[0]
        assert dist.should_skip(tmp_path)

    def test_distributed_skips_with_too_few_hosts(self, tmp_path, monkeypatch):
        hostfile = tmp_path / "hostfile.json"
        hostfile.write_text(json.dumps({"hosts": ["192.168.1.1"], "model": "m"}))
        monkeypatch.setattr("olmlx.bench.scenarios._DEFAULT_HOSTFILE", hostfile)
        dist = get_scenarios(["distributed"])[0]
        assert dist.should_skip(tmp_path)

    def test_distributed_skips_without_model_field(self, tmp_path, monkeypatch):
        hostfile = tmp_path / "hostfile.json"
        hostfile.write_text(json.dumps({"hosts": ["h1", "h2"]}))
        monkeypatch.setattr("olmlx.bench.scenarios._DEFAULT_HOSTFILE", hostfile)
        dist = get_scenarios(["distributed"])[0]
        assert dist.should_skip(tmp_path)

    def test_distributed_skips_with_invalid_json(self, tmp_path, monkeypatch):
        hostfile = tmp_path / "hostfile.json"
        hostfile.write_text("not json")
        monkeypatch.setattr("olmlx.bench.scenarios._DEFAULT_HOSTFILE", hostfile)
        dist = get_scenarios(["distributed"])[0]
        assert dist.should_skip(tmp_path)

    def test_distributed_runs_with_valid_hostfile(self, tmp_path, monkeypatch):
        hostfile = tmp_path / "hostfile.json"
        hostfile.write_text(
            json.dumps({"hosts": ["192.168.1.1", "192.168.1.2"], "model": "some/model"})
        )
        monkeypatch.setattr("olmlx.bench.scenarios._DEFAULT_HOSTFILE", hostfile)
        dist = get_scenarios(["distributed"])[0]
        assert not dist.should_skip(tmp_path)

    def test_server_mode_roundtrip(self):
        s = Scenario(name="test", description="test", server_mode=True)
        d = s.to_dict()
        assert d["server_mode"] is True
        restored = Scenario.from_dict(d)
        assert restored.server_mode is True

    def test_server_mode_default_false(self):
        s = Scenario(name="test", description="test")
        assert s.server_mode is False
        restored = Scenario.from_dict({"name": "test", "description": "test"})
        assert restored.server_mode is False
