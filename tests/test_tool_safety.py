"""Tests for olmlx.chat.tool_safety."""

import pytest

from olmlx.chat.tool_safety import ToolPolicy, ToolSafetyConfig, ToolSafetyPolicy


class TestToolPolicy:
    def test_default_policy_is_confirm(self):
        """Unknown tools require confirmation by default."""
        config = ToolSafetyConfig()
        policy = ToolSafetyPolicy(config)
        assert policy.get_policy("unknown_tool") == ToolPolicy.CONFIRM

    def test_allow_policy_skips_confirmation(self):
        """Allowed tools execute without decider."""
        config = ToolSafetyConfig(tool_policies={"read_file": ToolPolicy.ALLOW})
        policy = ToolSafetyPolicy(config)
        assert policy.get_policy("read_file") == ToolPolicy.ALLOW

    def test_deny_policy_blocks_execution(self):
        """Denied tools are blocked."""
        config = ToolSafetyConfig(tool_policies={"delete_file": ToolPolicy.DENY})
        policy = ToolSafetyPolicy(config)
        assert policy.get_policy("delete_file") == ToolPolicy.DENY

    def test_use_skill_follows_default_policy(self):
        """use_skill is not special to the policy — handled by session instead."""
        config = ToolSafetyConfig(default_policy=ToolPolicy.CONFIRM)
        policy = ToolSafetyPolicy(config)
        assert policy.get_policy("use_skill") == ToolPolicy.CONFIRM

    def test_config_override_default(self):
        """Explicit per-tool config overrides default policy."""
        config = ToolSafetyConfig(
            default_policy=ToolPolicy.ALLOW,
            tool_policies={"write_file": ToolPolicy.CONFIRM},
        )
        policy = ToolSafetyPolicy(config)
        assert policy.get_policy("write_file") == ToolPolicy.CONFIRM
        assert policy.get_policy("other_tool") == ToolPolicy.ALLOW

    def test_config_override_per_tool(self):
        """Per-tool config overrides default for any tool."""
        config = ToolSafetyConfig(
            default_policy=ToolPolicy.CONFIRM,
            tool_policies={"use_skill": ToolPolicy.DENY},
        )
        policy = ToolSafetyPolicy(config)
        assert policy.get_policy("use_skill") == ToolPolicy.DENY


class TestClassifyBatch:
    def test_classify_batch(self):
        """Splits tool list into allow, confirm, deny groups."""
        config = ToolSafetyConfig(
            default_policy=ToolPolicy.CONFIRM,
            tool_policies={
                "read_file": ToolPolicy.ALLOW,
                "delete_file": ToolPolicy.DENY,
            },
        )
        policy = ToolSafetyPolicy(config)
        tool_uses = [
            {"name": "read_file", "input": {}, "id": "1"},
            {"name": "write_file", "input": {}, "id": "2"},
            {"name": "delete_file", "input": {}, "id": "3"},
        ]
        allow, confirm, deny = policy.classify_batch(tool_uses)
        assert [tu["name"] for tu in allow] == ["read_file"]
        assert [tu["name"] for tu in confirm] == ["write_file"]
        assert [tu["name"] for tu in deny] == ["delete_file"]

    def test_classify_batch_empty(self):
        """Empty tool list returns empty groups."""
        policy = ToolSafetyPolicy(ToolSafetyConfig())
        allow, confirm, deny = policy.classify_batch([])
        assert allow == []
        assert confirm == []
        assert deny == []


class TestCheckAndConfirm:
    @pytest.mark.asyncio
    async def test_allow_returns_true(self):
        """Allowed tools return True without calling decider."""
        decider_called = False

        async def decider(name, args):
            nonlocal decider_called
            decider_called = True
            return False

        config = ToolSafetyConfig(tool_policies={"read_file": ToolPolicy.ALLOW})
        policy = ToolSafetyPolicy(config, decider=decider)
        result = await policy.check_and_confirm("read_file", {})
        assert result is True
        assert not decider_called

    @pytest.mark.asyncio
    async def test_deny_returns_false(self):
        """Denied tools return False without calling decider."""
        decider_called = False

        async def decider(name, args):
            nonlocal decider_called
            decider_called = True
            return True

        config = ToolSafetyConfig(tool_policies={"delete_file": ToolPolicy.DENY})
        policy = ToolSafetyPolicy(config, decider=decider)
        result = await policy.check_and_confirm("delete_file", {})
        assert result is False
        assert not decider_called

    @pytest.mark.asyncio
    async def test_confirm_calls_decider_approved(self):
        """Confirm policy calls the async decider and returns its result (True)."""

        async def decider(name, args):
            return True

        config = ToolSafetyConfig(default_policy=ToolPolicy.CONFIRM)
        policy = ToolSafetyPolicy(config, decider=decider)
        result = await policy.check_and_confirm("write_file", {"path": "/tmp/x"})
        assert result is True

    @pytest.mark.asyncio
    async def test_confirm_calls_decider_denied(self):
        """Confirm policy calls the async decider and returns its result (False)."""

        async def decider(name, args):
            return False

        config = ToolSafetyConfig(default_policy=ToolPolicy.CONFIRM)
        policy = ToolSafetyPolicy(config, decider=decider)
        result = await policy.check_and_confirm("write_file", {"path": "/tmp/x"})
        assert result is False

    @pytest.mark.asyncio
    async def test_confirm_no_decider_denies(self):
        """Without a decider, confirm tools are denied."""
        config = ToolSafetyConfig(default_policy=ToolPolicy.CONFIRM)
        policy = ToolSafetyPolicy(config, decider=None)
        result = await policy.check_and_confirm("write_file", {})
        assert result is False
