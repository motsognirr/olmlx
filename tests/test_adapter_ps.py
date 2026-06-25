"""/api/ps surfaces the adapter's base model (issue #362)."""

from copy import copy


class TestPsAdapterBase:
    async def test_adapter_row_reports_adapter_base(self, app_client):
        manager = app_client._transport.app.state.model_manager
        base = manager._loaded["qwen3:latest"]
        adapter = copy(base)
        adapter.name = "qwen3-8b:my-coder-lora"
        adapter.hf_path = "acme/my-coder-lora"
        adapter.adapter_base = "qwen3:latest"
        manager._loaded["qwen3-8b:my-coder-lora"] = adapter
        try:
            resp = await app_client.get("/api/ps")
            assert resp.status_code == 200
            rows = {m["name"]: m for m in resp.json()["models"]}
            assert rows["qwen3-8b:my-coder-lora"]["adapter_base"] == "qwen3:latest"
            # A plain base model reports a null adapter_base.
            assert rows["qwen3:latest"]["adapter_base"] is None
        finally:
            manager._loaded.pop("qwen3-8b:my-coder-lora", None)
