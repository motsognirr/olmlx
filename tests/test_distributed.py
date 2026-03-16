"""Tests for experimental distributed inference."""

import socket
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from olmlx.config import ExperimentalSettings


class TestExperimentalSettings:
    """Tests for ExperimentalSettings configuration."""

    def test_defaults(self, monkeypatch):
        for key in (
            "EXPERIMENTAL_DISTRIBUTED",
            "EXPERIMENTAL_DISTRIBUTED_HOSTFILE",
            "EXPERIMENTAL_DISTRIBUTED_BACKEND",
            "EXPERIMENTAL_DISTRIBUTED_PORT",
            "EXPERIMENTAL_DISTRIBUTED_SIDEBAND_PORT",
        ):
            monkeypatch.delenv(key, raising=False)
        s = ExperimentalSettings()
        assert s.distributed is False
        assert s.distributed_hostfile == Path("~/.olmlx/hostfile.json")
        assert s.distributed_backend == "ring"
        assert s.distributed_port == 32323
        assert s.distributed_sideband_port == 32400

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("EXPERIMENTAL_DISTRIBUTED", "true")
        monkeypatch.setenv("EXPERIMENTAL_DISTRIBUTED_HOSTFILE", "/tmp/hosts.json")
        monkeypatch.setenv("EXPERIMENTAL_DISTRIBUTED_BACKEND", "mpi")
        monkeypatch.setenv("EXPERIMENTAL_DISTRIBUTED_PORT", "40000")
        monkeypatch.setenv("EXPERIMENTAL_DISTRIBUTED_SIDEBAND_PORT", "40100")
        s = ExperimentalSettings()
        assert s.distributed is True
        assert s.distributed_hostfile == Path("/tmp/hosts.json")
        assert s.distributed_backend == "mpi"
        assert s.distributed_port == 40000
        assert s.distributed_sideband_port == 40100

    def test_distributed_false_by_default(self, monkeypatch):
        monkeypatch.delenv("EXPERIMENTAL_DISTRIBUTED", raising=False)
        s = ExperimentalSettings()
        assert s.distributed is False

    def test_separate_from_main_settings(self, monkeypatch):
        """ExperimentalSettings uses EXPERIMENTAL_ prefix, not OLMLX_."""
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DISTRIBUTED", "true")
        monkeypatch.delenv("EXPERIMENTAL_DISTRIBUTED", raising=False)
        s = ExperimentalSettings()
        assert s.distributed is False  # Should NOT read OLMLX_ prefix


class TestInferenceRequest:
    """Tests for InferenceRequest serialization."""

    def test_to_json_roundtrip(self):
        from olmlx.engine.distributed import InferenceRequest

        req = InferenceRequest(
            prompt_tokens=[1, 2, 3, 4],
            max_tokens=100,
            gen_kwargs={"temp": 0.7, "top_p": 0.9},
            action="generate",
        )
        data = req.to_json()
        restored = InferenceRequest.from_json(data)
        assert restored.prompt_tokens == [1, 2, 3, 4]
        assert restored.max_tokens == 100
        assert restored.gen_kwargs == {"temp": 0.7, "top_p": 0.9}
        assert restored.action == "generate"

    def test_shutdown_action(self):
        from olmlx.engine.distributed import InferenceRequest

        req = InferenceRequest(
            prompt_tokens=[],
            max_tokens=0,
            gen_kwargs={},
            action="shutdown",
        )
        data = req.to_json()
        restored = InferenceRequest.from_json(data)
        assert restored.action == "shutdown"

    def test_from_json_bytes(self):
        from olmlx.engine.distributed import InferenceRequest

        req = InferenceRequest(
            prompt_tokens=[10, 20],
            max_tokens=50,
            gen_kwargs={},
            action="generate",
        )
        data = req.to_json()
        # from_json should handle both str and bytes
        restored = InferenceRequest.from_json(data.encode("utf-8"))
        assert restored.prompt_tokens == [10, 20]


class TestSidebandProtocol:
    """Tests for the length-prefixed JSON sideband protocol."""

    def test_send_recv_message(self):
        from olmlx.engine.distributed import _recv_message, _send_message

        # Create a socket pair for testing
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        port = server_sock.getsockname()[1]
        server_sock.listen(1)

        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect(("127.0.0.1", port))
        conn, _ = server_sock.accept()

        try:
            msg = {"hello": "world", "nums": [1, 2, 3]}
            _send_message(client_sock, msg)
            received = _recv_message(conn)
            assert received == msg
        finally:
            client_sock.close()
            conn.close()
            server_sock.close()

    def test_recv_returns_none_on_closed_connection(self):
        from olmlx.engine.distributed import _recv_message

        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        port = server_sock.getsockname()[1]
        server_sock.listen(1)

        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect(("127.0.0.1", port))
        conn, _ = server_sock.accept()

        try:
            client_sock.close()
            result = _recv_message(conn)
            assert result is None
        finally:
            conn.close()
            server_sock.close()


class TestCoordinatorWorkerIntegration:
    """Integration tests for coordinator and worker sideband communication."""

    def test_coordinator_broadcasts_to_worker(self):
        from olmlx.engine.distributed import (
            DistributedCoordinator,
            DistributedWorker,
            InferenceRequest,
        )

        coordinator = DistributedCoordinator(world_size=2, port=0)
        actual_port = coordinator.port

        # Connect a worker in a thread
        received_requests = []

        def worker_fn():
            worker = DistributedWorker(
                coordinator_host="127.0.0.1",
                port=actual_port,
            )
            worker.send_ready()
            req = worker.wait_for_inference()
            received_requests.append(req)
            worker.close()

        t = threading.Thread(target=worker_fn)
        t.start()

        coordinator.wait_for_workers(timeout=5.0)

        coordinator.broadcast_inference(
            prompt_tokens=[1, 2, 3],
            max_tokens=100,
            gen_kwargs={"temp": 0.7},
        )

        t.join(timeout=5.0)
        coordinator.close()

        assert len(received_requests) == 1
        req = received_requests[0]
        assert isinstance(req, InferenceRequest)
        assert req.prompt_tokens == [1, 2, 3]
        assert req.max_tokens == 100
        assert req.gen_kwargs == {"temp": 0.7}
        assert req.action == "generate"

    def test_coordinator_broadcasts_shutdown(self):
        from olmlx.engine.distributed import (
            DistributedCoordinator,
            DistributedWorker,
        )

        coordinator = DistributedCoordinator(world_size=2, port=0)
        actual_port = coordinator.port

        received_requests = []

        def worker_fn():
            worker = DistributedWorker(
                coordinator_host="127.0.0.1",
                port=actual_port,
            )
            worker.send_ready()
            req = worker.wait_for_inference()
            received_requests.append(req)
            worker.close()

        t = threading.Thread(target=worker_fn)
        t.start()

        coordinator.wait_for_workers(timeout=5.0)
        coordinator.broadcast_shutdown()

        t.join(timeout=5.0)
        coordinator.close()

        assert len(received_requests) == 1
        assert received_requests[0] is None

    def test_coordinator_wait_for_workers_timeout(self):
        from olmlx.engine.distributed import DistributedCoordinator

        coordinator = DistributedCoordinator(world_size=3, port=0)
        try:
            with pytest.raises(TimeoutError, match="workers to connect"):
                coordinator.wait_for_workers(timeout=0.5)
        finally:
            coordinator.close()

    def test_coordinator_wait_for_ready_timeout(self):
        """Coordinator times out if worker connects but never sends ready."""
        from olmlx.engine.distributed import DistributedWorker, DistributedCoordinator

        coordinator = DistributedCoordinator(world_size=2, port=0)
        actual_port = coordinator.port

        def worker_fn():
            worker = DistributedWorker(
                coordinator_host="127.0.0.1",
                port=actual_port,
            )
            # Deliberately do NOT send ready
            time.sleep(2)
            worker.close()

        t = threading.Thread(target=worker_fn)
        t.start()

        try:
            with pytest.raises(TimeoutError, match="ready"):
                coordinator.wait_for_workers(timeout=0.5)
        finally:
            coordinator.close()
            t.join(timeout=5.0)

    def test_multiple_broadcasts(self):
        from olmlx.engine.distributed import (
            DistributedCoordinator,
            DistributedWorker,
        )

        coordinator = DistributedCoordinator(world_size=2, port=0)
        actual_port = coordinator.port

        received = []

        def worker_fn():
            worker = DistributedWorker(
                coordinator_host="127.0.0.1",
                port=actual_port,
            )
            worker.send_ready()
            while True:
                req = worker.wait_for_inference()
                if req is None:
                    break
                received.append(req)
            worker.close()

        t = threading.Thread(target=worker_fn)
        t.start()

        coordinator.wait_for_workers(timeout=5.0)

        # Send multiple inference requests
        coordinator.broadcast_inference([1, 2], 50, {"temp": 0.5})
        coordinator.broadcast_inference([3, 4, 5], 100, {"temp": 0.9})
        coordinator.broadcast_shutdown()

        t.join(timeout=5.0)
        coordinator.close()

        assert len(received) == 2
        assert received[0].prompt_tokens == [1, 2]
        assert received[1].prompt_tokens == [3, 4, 5]


class TestShardDetection:
    """Tests for model shard() capability detection."""

    def test_model_with_shard_method(self):
        """Models with shard() should be accepted for distributed inference."""
        model = MagicMock()
        model.shard = MagicMock()
        assert hasattr(model, "shard")

    def test_model_without_shard_method(self):
        """Models without shard() should be detected."""
        model = MagicMock(spec=[])  # No methods
        assert not hasattr(model, "shard")


class TestModelManagerDistributed:
    """Tests for distributed-aware ModelManager."""

    def test_model_manager_accepts_distributed_group(self):
        from olmlx.engine.model_manager import ModelManager
        from olmlx.engine.registry import ModelRegistry

        registry = ModelRegistry()
        group = MagicMock()
        group.rank.return_value = 0
        manager = ModelManager(registry, distributed_group=group)
        assert manager._distributed_group is group

    def test_model_manager_default_no_distributed(self):
        from olmlx.engine.model_manager import ModelManager
        from olmlx.engine.registry import ModelRegistry

        registry = ModelRegistry()
        manager = ModelManager(registry)
        assert manager._distributed_group is None


class TestLoadedModelDistributed:
    """Tests for is_distributed flag on LoadedModel."""

    def test_loaded_model_default_not_distributed(self):
        from olmlx.engine.model_manager import LoadedModel

        lm = LoadedModel(
            name="test",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        assert lm.is_distributed is False

    def test_loaded_model_distributed_flag(self):
        from olmlx.engine.model_manager import LoadedModel

        lm = LoadedModel(
            name="test",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            is_distributed=True,
        )
        assert lm.is_distributed is True


class TestInferenceDistributedBroadcast:
    """Tests for distributed broadcast before inference."""

    def test_set_distributed_coordinator(self):
        from olmlx.engine.inference import (
            set_distributed_coordinator,
        )

        mock_coordinator = MagicMock()
        set_distributed_coordinator(mock_coordinator)

        from olmlx.engine import inference

        assert inference._distributed_coordinator is mock_coordinator

        # Clean up
        set_distributed_coordinator(None)

    def test_no_coordinator_by_default(self):
        from olmlx.engine.inference import _distributed_coordinator

        # After cleanup from previous test or fresh import
        assert _distributed_coordinator is None


class TestModelManagerSharding:
    """Tests for model.shard() being called during load when distributed."""

    @pytest.fixture
    def mock_registry(self):
        registry = MagicMock()
        registry.normalize_name.return_value = "test-model"
        registry.resolve.return_value = "test/model"
        return registry

    def test_shard_called_when_distributed_group_set(self, mock_registry, monkeypatch):
        """When distributed_group is set, model.shard(group) should be called after load."""
        from olmlx.engine.model_manager import ModelManager

        model = MagicMock()
        model.shard = MagicMock()
        tokenizer = MagicMock()
        caps = MagicMock()

        group = MagicMock()

        manager = ModelManager(mock_registry, distributed_group=group)
        # Patch _load_model to return our mock model
        manager._load_model = MagicMock(return_value=(model, tokenizer, False, caps))

        # Call _load_model_and_shard (the internal that wraps _load_model + shard)
        result_model, result_tok, is_vlm, result_caps, is_distributed = (
            manager._load_model_and_shard("test/model")
        )

        model.shard.assert_called_once_with(group)
        assert is_distributed is True

    def test_shard_not_called_when_no_group(self, mock_registry):
        """When no distributed_group, model.shard() should NOT be called."""
        from olmlx.engine.model_manager import ModelManager

        model = MagicMock()
        model.shard = MagicMock()
        tokenizer = MagicMock()
        caps = MagicMock()

        manager = ModelManager(mock_registry)
        manager._load_model = MagicMock(return_value=(model, tokenizer, False, caps))

        result_model, result_tok, is_vlm, result_caps, is_distributed = (
            manager._load_model_and_shard("test/model")
        )

        model.shard.assert_not_called()
        assert is_distributed is False

    def test_shard_raises_for_unsupported_model(self, mock_registry):
        """Models without shard() should raise ValueError when distributed."""
        from olmlx.engine.model_manager import ModelManager

        model = MagicMock(spec=[])  # No shard method
        tokenizer = MagicMock()
        caps = MagicMock()

        group = MagicMock()

        manager = ModelManager(mock_registry, distributed_group=group)
        manager._load_model = MagicMock(return_value=(model, tokenizer, False, caps))

        with pytest.raises(ValueError, match="does not support distributed"):
            manager._load_model_and_shard("test/model")


class TestInferenceBroadcast:
    """Tests for distributed broadcast before stream_generate."""

    def test_broadcast_called_before_streaming(self):
        """When coordinator is set and model is distributed, broadcast should happen."""
        from olmlx.engine.inference import (
            set_distributed_coordinator,
        )

        mock_coordinator = MagicMock()
        set_distributed_coordinator(mock_coordinator)

        try:
            from olmlx.engine.inference import _maybe_broadcast_distributed

            lm = MagicMock()
            lm.is_distributed = True

            _maybe_broadcast_distributed(lm, [1, 2, 3], 100, {"temp": 0.7})

            mock_coordinator.broadcast_inference.assert_called_once_with(
                prompt_tokens=[1, 2, 3],
                max_tokens=100,
                gen_kwargs={"temp": 0.7},
            )
        finally:
            set_distributed_coordinator(None)

    def test_no_broadcast_when_not_distributed(self):
        """When model is not distributed, no broadcast should happen."""
        from olmlx.engine.inference import set_distributed_coordinator

        mock_coordinator = MagicMock()
        set_distributed_coordinator(mock_coordinator)

        try:
            from olmlx.engine.inference import _maybe_broadcast_distributed

            lm = MagicMock()
            lm.is_distributed = False

            _maybe_broadcast_distributed(lm, [1, 2, 3], 100, {})

            mock_coordinator.broadcast_inference.assert_not_called()
        finally:
            set_distributed_coordinator(None)

    def test_no_broadcast_when_no_coordinator(self):
        """When no coordinator is set, no broadcast should happen."""
        from olmlx.engine.inference import (
            _maybe_broadcast_distributed,
            set_distributed_coordinator,
        )

        set_distributed_coordinator(None)

        lm = MagicMock()
        lm.is_distributed = True

        # Should not raise
        _maybe_broadcast_distributed(lm, [1, 2, 3], 100, {})


class TestExperimentalModuleGlobal:
    """Tests that the experimental singleton is available."""

    def test_experimental_singleton_exists(self):
        from olmlx.config import experimental

        assert experimental is not None
        assert isinstance(experimental, ExperimentalSettings)

    def test_experimental_singleton_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("EXPERIMENTAL_DISTRIBUTED", raising=False)
        # Re-instantiate to test defaults
        s = ExperimentalSettings()
        assert s.distributed is False
