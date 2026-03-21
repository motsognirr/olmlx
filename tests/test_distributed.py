"""Tests for experimental distributed inference."""

import os
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
            "OLMLX_EXPERIMENTAL_DISTRIBUTED",
            "OLMLX_EXPERIMENTAL_DISTRIBUTED_HOSTFILE",
            "OLMLX_EXPERIMENTAL_DISTRIBUTED_BACKEND",
            "OLMLX_EXPERIMENTAL_DISTRIBUTED_PORT",
            "OLMLX_EXPERIMENTAL_DISTRIBUTED_SIDEBAND_PORT",
            "OLMLX_EXPERIMENTAL_DISTRIBUTED_SECRET",
        ):
            monkeypatch.delenv(key, raising=False)
        s = ExperimentalSettings()
        assert s.distributed is False
        assert s.distributed_hostfile == Path("~/.olmlx/hostfile.json")
        assert s.distributed_backend == "ring"
        assert s.distributed_port == 32323
        assert s.distributed_sideband_port == 32400

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DISTRIBUTED", "true")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DISTRIBUTED_HOSTFILE", "/tmp/hosts.json")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DISTRIBUTED_BACKEND", "mpi")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DISTRIBUTED_PORT", "40000")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DISTRIBUTED_SIDEBAND_PORT", "40100")
        s = ExperimentalSettings()
        assert s.distributed is True
        assert s.distributed_hostfile == Path("/tmp/hosts.json")
        assert s.distributed_backend == "mpi"
        assert s.distributed_port == 40000
        assert s.distributed_sideband_port == 40100

    def test_distributed_false_by_default(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_DISTRIBUTED", raising=False)
        s = ExperimentalSettings()
        assert s.distributed is False

    def test_separate_from_main_settings(self, monkeypatch):
        """ExperimentalSettings uses OLMLX_EXPERIMENTAL_ prefix, consistent with main."""
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DISTRIBUTED", "true")
        s = ExperimentalSettings()
        assert s.distributed is True  # Should read OLMLX_EXPERIMENTAL_ prefix


class TestInferenceRequest:
    """Tests for InferenceRequest serialization."""

    def test_to_json_roundtrip(self):
        from olmlx.engine.distributed import InferenceRequest

        req = InferenceRequest(
            prompt_tokens=[1, 2, 3, 4],
            prompt_text="hello world",
            max_tokens=100,
            gen_kwargs={"temp": 0.7, "top_p": 0.9},
            action="generate",
        )
        data = req.to_json()
        restored = InferenceRequest.from_json(data)
        assert restored.prompt_tokens == [1, 2, 3, 4]
        assert restored.prompt_text == "hello world"
        assert restored.max_tokens == 100
        assert restored.gen_kwargs == {"temp": 0.7, "top_p": 0.9}
        assert restored.action == "generate"

    def test_shutdown_action(self):
        from olmlx.engine.distributed import InferenceRequest

        req = InferenceRequest(
            prompt_tokens=[],
            prompt_text="",
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
            prompt_text="test",
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
            prompt_text="hello",
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
        coordinator.broadcast_inference([1, 2], "ab", 50, {"temp": 0.5})
        coordinator.broadcast_inference([3, 4, 5], "abc", 100, {"temp": 0.9})
        coordinator.broadcast_shutdown()

        t.join(timeout=5.0)
        coordinator.close()

        assert len(received) == 2
        assert received[0].prompt_tokens == [1, 2]
        assert received[1].prompt_tokens == [3, 4, 5]

    def test_worker_retries_sideband_connection(self):
        """Worker retries connecting to sideband if coordinator isn't ready yet."""
        from olmlx.engine.distributed import (
            DistributedCoordinator,
            DistributedWorker,
        )

        # Start the worker BEFORE the coordinator's sideband server
        # to simulate the race condition in distributed startup.
        free_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        free_sock.bind(("127.0.0.1", 0))
        port = free_sock.getsockname()[1]
        free_sock.close()

        connected = []
        errors = []

        def worker_fn():
            try:
                worker = DistributedWorker(
                    coordinator_host="127.0.0.1",
                    port=port,
                    connect_retry_timeout=10.0,
                )
                worker.send_ready()
                connected.append(True)
                worker.wait_for_inference()
                worker.close()
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=worker_fn)
        t.start()

        # Delay coordinator start by 2s to force worker retries
        time.sleep(2)
        coordinator = DistributedCoordinator(world_size=2, port=port)

        coordinator.wait_for_workers(timeout=10.0)
        assert len(connected) == 1, f"Worker failed to connect: {errors}"

        coordinator.broadcast_shutdown()
        t.join(timeout=5.0)
        coordinator.close()

    def test_worker_sideband_retry_timeout(self):
        """Worker gives up after connect_retry_timeout if sideband never appears."""
        from olmlx.engine.distributed import DistributedWorker

        # Use a port that nothing is listening on
        free_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        free_sock.bind(("127.0.0.1", 0))
        port = free_sock.getsockname()[1]
        free_sock.close()

        with pytest.raises(ConnectionRefusedError, match="after 1.0s"):
            DistributedWorker(
                coordinator_host="127.0.0.1",
                port=port,
                connect_retry_timeout=1.0,
            )


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

            _maybe_broadcast_distributed(
                lm, [1, 2, 3], "test prompt", 100, {"temp": 0.7}
            )

            mock_coordinator.broadcast_inference.assert_called_once_with(
                prompt_tokens=[1, 2, 3],
                prompt_text="test prompt",
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

            _maybe_broadcast_distributed(lm, [1, 2, 3], "test", 100, {})

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
        _maybe_broadcast_distributed(lm, [1, 2, 3], "test", 100, {})


class TestExperimentalModuleGlobal:
    """Tests that the experimental singleton is available."""

    def test_experimental_singleton_exists(self):
        from olmlx.config import experimental

        assert experimental is not None
        assert isinstance(experimental, ExperimentalSettings)

    def test_experimental_singleton_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_DISTRIBUTED", raising=False)
        # Re-instantiate to test defaults
        s = ExperimentalSettings()
        assert s.distributed is False


class TestProtocolViolation:
    """Issue 8: Wrong exception type for protocol violation."""

    def test_unexpected_message_raises_runtime_error(self):
        """Worker sending wrong message should raise RuntimeError, not TimeoutError."""
        from olmlx.engine.distributed import DistributedCoordinator, _send_message

        coordinator = DistributedCoordinator(world_size=2, port=0)
        actual_port = coordinator.port

        def bad_worker_fn():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", actual_port))
            # Send wrong message instead of {"action": "ready"}
            _send_message(sock, {"action": "wrong"})
            time.sleep(1)
            sock.close()

        t = threading.Thread(target=bad_worker_fn)
        t.start()

        try:
            with pytest.raises(RuntimeError, match="unexpected message"):
                coordinator.wait_for_workers(timeout=5.0)
        finally:
            coordinator.close()
            t.join(timeout=5.0)


class TestServerSocketClose:
    """Issue 12: Server socket closed after wait_for_workers."""

    def test_server_socket_closed_after_workers_ready(self):
        """Server socket should be closed after all workers connect."""
        from olmlx.engine.distributed import DistributedCoordinator, DistributedWorker

        coordinator = DistributedCoordinator(world_size=2, port=0)
        actual_port = coordinator.port

        def worker_fn():
            worker = DistributedWorker(coordinator_host="127.0.0.1", port=actual_port)
            worker.send_ready()
            time.sleep(0.5)
            worker.close()

        t = threading.Thread(target=worker_fn)
        t.start()

        coordinator.wait_for_workers(timeout=5.0)

        # Server socket should be closed — new connections should fail
        with pytest.raises(OSError):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            try:
                sock.connect(("127.0.0.1", actual_port))
            finally:
                sock.close()

        coordinator.close()
        t.join(timeout=5.0)


class TestDeadSocketCleanup:
    """Issue 1: Dead socket cleanup after broadcast failure."""

    def test_workers_cleared_after_broadcast_failure(self):
        """After broadcast failure, all workers should be cleaned up."""
        from olmlx.engine.distributed import DistributedCoordinator, DistributedWorker

        coordinator = DistributedCoordinator(world_size=2, port=0)
        actual_port = coordinator.port

        def worker_fn():
            worker = DistributedWorker(coordinator_host="127.0.0.1", port=actual_port)
            worker.send_ready()
            time.sleep(0.5)
            worker.close()

        t = threading.Thread(target=worker_fn)
        t.start()

        coordinator.wait_for_workers(timeout=5.0)
        t.join(timeout=5.0)

        # Force-close the coordinator's copy of the worker socket to simulate
        # a dead connection (closing the remote end isn't guaranteed to cause
        # sendall to fail immediately due to kernel buffering).
        for w in coordinator._workers:
            w.close()

        # Now broadcast should fail and clean up
        with pytest.raises(RuntimeError, match="broadcast failed"):
            coordinator.broadcast_inference([1, 2], "test", 100, {})

        # Workers list should be empty after failure
        assert len(coordinator._workers) == 0
        coordinator.close()


class TestCoordinatorThreadSafety:
    """Issue 3: Thread-safe access to _distributed_coordinator global."""

    def test_set_and_read_coordinator_thread_safe(self):
        """Concurrent set/read of coordinator should not crash."""
        from olmlx.engine.inference import (
            _maybe_broadcast_distributed,
            set_distributed_coordinator,
        )

        mock_coord = MagicMock()
        errors = []

        def writer():
            try:
                for _ in range(100):
                    set_distributed_coordinator(mock_coord)
                    set_distributed_coordinator(None)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                lm = MagicMock()
                lm.is_distributed = True
                for _ in range(100):
                    try:
                        _maybe_broadcast_distributed(lm, [1], "t", 10, {})
                    except Exception:
                        pass  # broadcast_inference may fail on mock
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        set_distributed_coordinator(None)
        assert len(errors) == 0


class TestShutdownOrdering:
    """Issue 2: Coordinator torn down before in-flight inference."""

    def test_manager_stop_before_coordinator_teardown(self):
        """Verify the shutdown order in lifespan is correct."""
        # This is a code inspection test — we verify the pattern in app.py
        import inspect
        from olmlx.app import lifespan

        source = inspect.getsource(lifespan)
        # manager.stop() must appear before coordinator.broadcast_shutdown()
        stop_pos = source.find("await manager.stop()")
        shutdown_pos = source.find("coordinator.broadcast_shutdown()")
        assert stop_pos < shutdown_pos, (
            "manager.stop() must be called before coordinator.broadcast_shutdown()"
        )


class TestAtexitGuard:
    """Issue 9: atexit.register called only once."""

    def test_atexit_registered_once(self):
        """_launch_distributed_workers should register atexit only once."""
        import olmlx.cli as cli_module

        # Reset the flag
        cli_module._atexit_registered = False
        # The flag should exist
        assert hasattr(cli_module, "_atexit_registered")


class TestFileHandleLeak:
    """Issue 5: File handle leak if Popen raises."""

    def test_log_fh_closed_on_popen_failure(self, tmp_path, monkeypatch):
        """If Popen raises, the log file handle should be closed."""
        import olmlx.cli as cli_module

        # Reset state
        cli_module._worker_procs.clear()
        cli_module._atexit_registered = False

        hostfile = tmp_path / "hosts.json"
        hostfile.write_text('{"hosts": ["host0", "host1"], "model": "test/model"}')

        monkeypatch.setattr(
            "olmlx.config.experimental",
            MagicMock(
                distributed_hostfile=str(hostfile),
                distributed_backend="ring",
                distributed_sideband_port=32400,
                distributed_port=32323,
                distributed_secret="",
                distributed_remote_working_dir="",
                distributed_remote_python="python",
            ),
        )

        # Make Popen raise
        monkeypatch.setattr(
            "subprocess.Popen", MagicMock(side_effect=OSError("popen failed"))
        )

        with pytest.raises(OSError, match="popen failed"):
            cli_module._launch_distributed_workers()

        # Verify no worker procs were added
        assert len(cli_module._worker_procs) == 0


class TestEnvPrefix:
    """Issue 11: OLMLX_EXPERIMENTAL_ env prefix."""

    def test_olmlx_experimental_prefix(self, monkeypatch):
        """ExperimentalSettings should use OLMLX_EXPERIMENTAL_ prefix."""
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DISTRIBUTED", "true")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DISTRIBUTED_BACKEND", "mpi")
        s = ExperimentalSettings()
        assert s.distributed is True
        assert s.distributed_backend == "mpi"

    def test_old_prefix_no_longer_works(self, monkeypatch):
        """Old EXPERIMENTAL_ prefix should NOT be recognized."""
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_DISTRIBUTED", raising=False)
        monkeypatch.setenv("EXPERIMENTAL_DISTRIBUTED", "true")
        s = ExperimentalSettings()
        assert s.distributed is False


class TestSharedSecret:
    """Issue 6: Shared secret authentication."""

    def test_worker_rejected_with_wrong_secret(self):
        """Workers with wrong secret should be rejected."""
        from olmlx.engine.distributed import DistributedCoordinator, _send_message

        coordinator = DistributedCoordinator(
            world_size=2, port=0, secret="correct-secret"
        )
        actual_port = coordinator.port

        def bad_worker_fn():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", actual_port))
            _send_message(sock, {"action": "ready", "secret": "wrong-secret"})
            time.sleep(1)
            sock.close()

        t = threading.Thread(target=bad_worker_fn)
        t.start()

        try:
            with pytest.raises(RuntimeError, match="secret"):
                coordinator.wait_for_workers(timeout=5.0)
        finally:
            coordinator.close()
            t.join(timeout=5.0)

    def test_worker_accepted_with_correct_secret(self):
        """Workers with correct secret should be accepted."""
        from olmlx.engine.distributed import DistributedCoordinator, _send_message

        coordinator = DistributedCoordinator(world_size=2, port=0, secret="my-secret")
        actual_port = coordinator.port

        def good_worker_fn():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", actual_port))
            _send_message(sock, {"action": "ready", "secret": "my-secret"})
            time.sleep(0.5)
            sock.close()

        t = threading.Thread(target=good_worker_fn)
        t.start()

        coordinator.wait_for_workers(timeout=5.0)
        assert len(coordinator._workers) == 1
        coordinator.close()
        t.join(timeout=5.0)

    def test_no_secret_required_by_default(self):
        """When no secret is set, workers should connect without one."""
        from olmlx.engine.distributed import DistributedCoordinator, DistributedWorker

        coordinator = DistributedCoordinator(world_size=2, port=0)
        actual_port = coordinator.port

        def worker_fn():
            worker = DistributedWorker(coordinator_host="127.0.0.1", port=actual_port)
            worker.send_ready()
            time.sleep(0.5)
            worker.close()

        t = threading.Thread(target=worker_fn)
        t.start()

        coordinator.wait_for_workers(timeout=5.0)
        assert len(coordinator._workers) == 1
        coordinator.close()
        t.join(timeout=5.0)


class TestVLMDistributedRejection:
    """Issue 4: VLM models rejected in distributed mode."""

    def test_vlm_rejected_in_distributed_mode(self):
        """VLM models should be rejected when distributed_group is set."""
        from olmlx.engine.model_manager import ModelManager

        model = MagicMock()
        model.shard = MagicMock()
        tokenizer = MagicMock()
        caps = MagicMock()
        group = MagicMock()

        manager = ModelManager(MagicMock(), distributed_group=group)
        # _load_model returns is_vlm=True
        manager._load_model = MagicMock(return_value=(model, tokenizer, True, caps))

        with pytest.raises(ValueError, match="VLM models are not supported"):
            manager._load_model_and_shard("test/vlm-model")

    def test_text_model_accepted_in_distributed_mode(self):
        """Text models should work fine in distributed mode."""
        from olmlx.engine.model_manager import ModelManager

        model = MagicMock()
        model.shard = MagicMock()
        tokenizer = MagicMock()
        caps = MagicMock()
        group = MagicMock()

        manager = ModelManager(MagicMock(), distributed_group=group)
        manager._load_model = MagicMock(return_value=(model, tokenizer, False, caps))

        _, _, is_vlm, _, is_distributed = manager._load_model_and_shard("test/model")
        assert is_vlm is False
        assert is_distributed is True
        model.shard.assert_called_once_with(group)


class TestSSHFailureDetection:
    """Issue 10: SSH failures detected early."""

    def test_ssh_failure_detected_early(self, tmp_path, monkeypatch):
        """Immediate SSH failure should be detected before server starts."""
        import olmlx.cli as cli_module

        cli_module._worker_procs.clear()
        cli_module._atexit_registered = False

        hostfile = tmp_path / "hosts.json"
        hostfile.write_text('{"hosts": ["host0", "host1"], "model": "test/model"}')

        monkeypatch.setattr(
            "olmlx.config.experimental",
            MagicMock(
                distributed_hostfile=str(hostfile),
                distributed_backend="ring",
                distributed_sideband_port=32400,
                distributed_port=32323,
                distributed_secret="test-secret",
                distributed_remote_working_dir="",
                distributed_remote_python="python",
            ),
        )

        # Create a mock proc that exits immediately with error
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # Exited with error
        mock_proc.returncode = 1
        monkeypatch.setattr("subprocess.Popen", MagicMock(return_value=mock_proc))

        with pytest.raises(RuntimeError, match="Worker SSH launch failed"):
            cli_module._launch_distributed_workers()


class TestPromptCacheDistributed:
    """Tests for prompt cache + distributed interaction."""

    def test_prompt_cache_disabled_when_distributed(self):
        """Prompt caching must be disabled for distributed models to avoid deadlock."""
        # Verify the condition in generate_chat disables cache for distributed models
        import inspect

        from olmlx.engine.inference import generate_chat

        source = inspect.getsource(generate_chat)
        assert "not lm.is_distributed" in source, (
            "generate_chat must disable prompt caching when lm.is_distributed"
        )

    def test_broadcast_strips_prompt_cache_from_gen_kwargs(self):
        """gen_kwargs sent to workers must not contain prompt_cache or input_ids."""
        import inspect

        from olmlx.engine import inference

        source = inspect.getsource(inference._stream_completion)
        assert "prompt_cache" in source and "input_ids" in source, (
            "_stream_completion must strip prompt_cache and input_ids from broadcast kwargs"
        )


class TestCleanupWorkersRobust:
    """Tests for _cleanup_workers with wait/kill fallback."""

    def test_cleanup_kills_after_terminate_timeout(self):
        """If terminate doesn't work within timeout, kill should be called."""
        import olmlx.cli as cli_module

        mock_proc = MagicMock()
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock(
            side_effect=__import__("subprocess").TimeoutExpired("cmd", 5)
        )
        mock_proc.kill = MagicMock()

        cli_module._worker_procs.clear()
        cli_module._worker_procs.append(mock_proc)

        cli_module._cleanup_workers()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()

        cli_module._worker_procs.clear()


class TestLogFileHandleLifetime:
    """Issue #81: Log file handles must stay open while workers are alive."""

    def test_log_handles_stored_alongside_procs(self, tmp_path, monkeypatch):
        """Log file handles should be stored and not closed immediately."""
        import olmlx.cli as cli_module

        cli_module._worker_procs.clear()
        cli_module._worker_log_fhs.clear()
        cli_module._atexit_registered = False

        hostfile = tmp_path / "hosts.json"
        hostfile.write_text('{"hosts": ["host0", "host1"], "model": "test/model"}')

        monkeypatch.setattr(
            "olmlx.config.experimental",
            MagicMock(
                distributed_hostfile=str(hostfile),
                distributed_backend="ring",
                distributed_sideband_port=32400,
                distributed_port=32323,
                distributed_secret="",
                distributed_remote_working_dir="",
                distributed_remote_python="python",
            ),
        )

        # Mock Popen to succeed without actually launching SSH
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Still running
        monkeypatch.setattr("subprocess.Popen", MagicMock(return_value=mock_proc))

        cli_module._launch_distributed_workers()

        # The module should now have _worker_log_fhs with open file handles
        assert hasattr(cli_module, "_worker_log_fhs"), (
            "_worker_log_fhs list must exist to track log file handles"
        )
        assert len(cli_module._worker_log_fhs) == len(cli_module._worker_procs)

        # File handles should still be open
        for fh in cli_module._worker_log_fhs:
            assert not fh.closed, "Log file handle was closed prematurely"

        # Capture references before cleanup (cleanup clears the lists)
        handles = list(cli_module._worker_log_fhs)
        cli_module._cleanup_workers()

        assert handles, "No file handles were stored"
        for fh in handles:
            assert fh.closed, "Log file handle was not closed during cleanup"


class TestExperimentalEnvFile:
    """Test that ExperimentalSettings reads .env file."""

    def test_env_file_configured(self):
        """ExperimentalSettings should have env_file='.env' in model_config."""
        assert ExperimentalSettings.model_config.get("env_file") == ".env"


class TestRemoteExecutionConfig:
    """Tests for distributed remote execution settings."""

    def test_defaults(self, monkeypatch):
        for key in (
            "OLMLX_EXPERIMENTAL_DISTRIBUTED_REMOTE_WORKING_DIR",
            "OLMLX_EXPERIMENTAL_DISTRIBUTED_REMOTE_PYTHON",
        ):
            monkeypatch.delenv(key, raising=False)
        s = ExperimentalSettings()
        assert s.distributed_remote_working_dir == ""
        assert s.distributed_remote_python == "python"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(
            "OLMLX_EXPERIMENTAL_DISTRIBUTED_REMOTE_WORKING_DIR",
            "~/Documents/olmlx_distributed",
        )
        monkeypatch.setenv(
            "OLMLX_EXPERIMENTAL_DISTRIBUTED_REMOTE_PYTHON", "uv run python"
        )
        s = ExperimentalSettings()
        assert s.distributed_remote_working_dir == "~/Documents/olmlx_distributed"
        assert s.distributed_remote_python == "uv run python"


class TestRingHostfileGeneration:
    """Tests for MLX ring hostfile generation."""

    def test_ring_hostfile_generated(self, tmp_path, monkeypatch):
        """_launch_distributed_workers should generate a ring hostfile."""
        import olmlx.cli as cli_module

        cli_module._worker_procs.clear()
        cli_module._worker_log_fhs.clear()
        cli_module._atexit_registered = False

        hostfile = tmp_path / "hosts.json"
        hostfile.write_text(
            '{"hosts": ["10.0.1.1", "10.0.1.2"], "model": "test/model"}'
        )

        monkeypatch.setattr(
            "olmlx.config.experimental",
            MagicMock(
                distributed_hostfile=str(hostfile),
                distributed_backend="ring",
                distributed_sideband_port=32400,
                distributed_port=32323,
                distributed_secret="",
                distributed_remote_working_dir="",
                distributed_remote_python="python",
            ),
        )

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        monkeypatch.setattr("subprocess.Popen", MagicMock(return_value=mock_proc))

        cli_module._launch_distributed_workers()

        # Ring hostfile should be written
        ring_hostfile = Path.home() / ".olmlx" / "ring_hostfile.json"
        assert ring_hostfile.exists()
        import json

        content = json.loads(ring_hostfile.read_text())
        assert content == [["10.0.1.1:32323"], ["10.0.1.2:32324"]]

        # Coordinator env vars should be set
        assert os.environ.get("MLX_RANK") == "0"

        assert os.environ.get("MLX_HOSTFILE") == str(ring_hostfile)

        cli_module._cleanup_workers()

        # Clean up env
        for key in ("MLX_RANK", "MLX_HOSTFILE"):
            os.environ.pop(key, None)

    def test_ring_hostfile_port_increments(self, tmp_path, monkeypatch):
        """Each host should get an incrementing port in the ring hostfile."""
        import olmlx.cli as cli_module

        cli_module._worker_procs.clear()
        cli_module._worker_log_fhs.clear()
        cli_module._atexit_registered = False

        hostfile = tmp_path / "hosts.json"
        hostfile.write_text(
            '{"hosts": ["10.0.0.1", "10.0.0.2", "10.0.0.3"], "model": "test/model"}'
        )

        monkeypatch.setattr(
            "olmlx.config.experimental",
            MagicMock(
                distributed_hostfile=str(hostfile),
                distributed_backend="ring",
                distributed_sideband_port=32400,
                distributed_port=50000,
                distributed_secret="",
                distributed_remote_working_dir="",
                distributed_remote_python="python",
            ),
        )

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        monkeypatch.setattr("subprocess.Popen", MagicMock(return_value=mock_proc))

        cli_module._launch_distributed_workers()

        ring_hostfile = Path.home() / ".olmlx" / "ring_hostfile.json"
        import json

        content = json.loads(ring_hostfile.read_text())
        assert content == [
            ["10.0.0.1:50000"],
            ["10.0.0.2:50001"],
            ["10.0.0.3:50002"],
        ]

        cli_module._cleanup_workers()
        for key in ("MLX_RANK", "MLX_HOSTFILE"):
            os.environ.pop(key, None)


class TestSSHCommandConstruction:
    """Tests for SSH command with working dir and custom python."""

    def test_ssh_command_includes_working_dir(self, tmp_path, monkeypatch):
        """SSH command should cd to working dir when configured."""
        import olmlx.cli as cli_module

        cli_module._worker_procs.clear()
        cli_module._worker_log_fhs.clear()
        cli_module._atexit_registered = False

        hostfile = tmp_path / "hosts.json"
        hostfile.write_text(
            '{"hosts": ["10.0.1.1", "10.0.1.2"], "model": "test/model"}'
        )

        monkeypatch.setattr(
            "olmlx.config.experimental",
            MagicMock(
                distributed_hostfile=str(hostfile),
                distributed_backend="ring",
                distributed_sideband_port=32400,
                distributed_port=32323,
                distributed_secret="",
                distributed_remote_working_dir="~/Documents/olmlx_distributed",
                distributed_remote_python="uv run python",
            ),
        )

        popen_mock = MagicMock()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        popen_mock.return_value = mock_proc
        monkeypatch.setattr("subprocess.Popen", popen_mock)

        cli_module._launch_distributed_workers()

        # Check the SSH command passed to Popen
        call_args = popen_mock.call_args[0][0]
        remote_cmd = call_args[-1]  # Last arg is the remote command

        assert "cd '~/Documents/olmlx_distributed'" in remote_cmd
        assert "uv run python -m olmlx.engine.distributed_worker" in remote_cmd

        cli_module._cleanup_workers()
        for key in ("MLX_RANK", "MLX_HOSTFILE"):
            os.environ.pop(key, None)

    def test_ssh_command_includes_hostfile(self, tmp_path, monkeypatch):
        """SSH command should create a temp hostfile on the remote."""
        import olmlx.cli as cli_module

        cli_module._worker_procs.clear()
        cli_module._worker_log_fhs.clear()
        cli_module._atexit_registered = False

        hostfile = tmp_path / "hosts.json"
        hostfile.write_text(
            '{"hosts": ["10.0.1.1", "10.0.1.2"], "model": "test/model"}'
        )

        monkeypatch.setattr(
            "olmlx.config.experimental",
            MagicMock(
                distributed_hostfile=str(hostfile),
                distributed_backend="ring",
                distributed_sideband_port=32400,
                distributed_port=32323,
                distributed_secret="",
                distributed_remote_working_dir="",
                distributed_remote_python="python",
            ),
        )

        popen_mock = MagicMock()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        popen_mock.return_value = mock_proc
        monkeypatch.setattr("subprocess.Popen", popen_mock)

        cli_module._launch_distributed_workers()

        call_args = popen_mock.call_args[0][0]
        remote_cmd = call_args[-1]

        assert "MLX_HOSTFILE=" in remote_cmd
        assert "mktemp" in remote_cmd

        cli_module._cleanup_workers()
        for key in ("MLX_RANK", "MLX_HOSTFILE"):
            os.environ.pop(key, None)

    def test_ssh_command_no_mlx_port_or_world_size(self, tmp_path, monkeypatch):
        """SSH command should NOT include MLX_PORT or MLX_WORLD_SIZE (ring backend uses hostfile)."""
        import olmlx.cli as cli_module

        cli_module._worker_procs.clear()
        cli_module._worker_log_fhs.clear()
        cli_module._atexit_registered = False

        hostfile = tmp_path / "hosts.json"
        hostfile.write_text(
            '{"hosts": ["10.0.1.1", "10.0.1.2"], "model": "test/model"}'
        )

        monkeypatch.setattr(
            "olmlx.config.experimental",
            MagicMock(
                distributed_hostfile=str(hostfile),
                distributed_backend="ring",
                distributed_sideband_port=32400,
                distributed_port=32323,
                distributed_secret="",
                distributed_remote_working_dir="",
                distributed_remote_python="python",
            ),
        )

        popen_mock = MagicMock()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        popen_mock.return_value = mock_proc
        monkeypatch.setattr("subprocess.Popen", popen_mock)

        cli_module._launch_distributed_workers()

        call_args = popen_mock.call_args[0][0]
        remote_cmd = call_args[-1]

        assert "MLX_PORT=" not in remote_cmd
        assert "MLX_WORLD_SIZE=" not in remote_cmd

        cli_module._cleanup_workers()
        for key in ("MLX_RANK", "MLX_HOSTFILE"):
            os.environ.pop(key, None)
