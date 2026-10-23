import json
import os
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pytest

from nanobot.desktop.adapter import DesktopAdapter, DesktopError


def _prepared_config(
    path: Path,
    workspace: Path,
    *,
    host: str = "127.0.0.1",
) -> tuple[int, int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as gateway_socket:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as websocket_socket:
            gateway_socket.bind(("127.0.0.1", 0))
            websocket_socket.bind(("127.0.0.1", 0))
            gateway_port = gateway_socket.getsockname()[1]
            websocket_port = websocket_socket.getsockname()[1]
    workspace.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "agents": {"defaults": {"provider": "openai", "model": "gpt-4o-mini", "workspace": str(workspace)}},
                "providers": {"openai": {"apiKey": "desktop-provider-sentinel"}},
                "gateway": {"host": host, "port": gateway_port},
                "channels": {
                    "websocket": {
                        "enabled": True,
                        "host": "127.0.0.1",
                        "port": websocket_port,
                        "tokenIssueSecret": "desktop-webui-sentinel",
                        "websocketRequiresToken": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return gateway_port, websocket_port


def _wait_until_running(adapter: DesktopAdapter, handle: str, timeout: float = 8.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = adapter.handle("status", {"handle": handle})
        if result["state"] == "running":
            return result
        time.sleep(0.05)
    raise AssertionError(f"desktop gateway did not become ready: {result!r}")


def test_hello_requires_an_explicit_existing_config_without_fallback(tmp_path: Path) -> None:
    adapter = DesktopAdapter()

    with pytest.raises(DesktopError) as error:
        adapter.handle("hello", {"config_path": str(tmp_path / "missing.json"), "workspace": None})

    assert error.value.code == "config_unavailable"


def test_hello_is_read_only_and_does_not_return_credentials(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    config_path = tmp_path / "profile" / "config.json"
    _prepared_config(config_path, workspace)
    before = config_path.read_bytes()
    adapter = DesktopAdapter()

    result = adapter.handle(
        "hello",
        {"config_path": str(config_path), "workspace": None},
    )
    rendered = json.dumps(result)

    assert result["protocol_version"] == 1
    assert result["capabilities"] == ["managed-gateway-v1"]
    assert result["config_path"] == str(config_path.resolve())
    assert result["workspace"] == str(workspace.resolve())
    assert "desktop-provider-sentinel" not in rendered
    assert "desktop-webui-sentinel" not in rendered
    assert config_path.read_bytes() == before
    assert not (workspace / ".nanobot").exists()


def test_desktop_start_returns_handle_without_waiting_for_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "profile" / "config.json"
    _prepared_config(config_path, tmp_path / "workspace")
    adapter = DesktopAdapter()
    gateway_pid: int | None = None
    handle: str | None = None
    try:
        adapter.handle("hello", {"config_path": str(config_path), "workspace": None})
        runtime = adapter._runtime
        selection = adapter._selection
        assert runtime is not None
        assert selection is not None

        def readiness_must_be_polled_later(*_args: object, **_kwargs: object) -> tuple[bool, str]:
            pytest.fail("start must not perform readiness probes")

        monkeypatch.setattr(adapter, "_ready", readiness_must_be_polled_later)
        began = time.monotonic()
        started = adapter.handle("start", {})
        elapsed = time.monotonic() - began
        assert elapsed < 3.0
        assert started["state"] == "starting"
        handle_value = started["handle"]
        assert isinstance(handle_value, str)
        handle = handle_value
        identity = runtime.desktop_identity(selection.options, handle=handle)
        assert identity is not None
        gateway_pid = identity.pid

        reconciled = adapter.handle("start", {"handle": handle})
        assert reconciled["state"] == "starting"
        assert reconciled["handle"] == handle

        stopped = adapter.handle("stop", {"handle": handle})
        assert stopped["state"] == "stopped"
        child = runtime._owned_process
        assert child is not None
        assert child.wait(timeout=5) is not None
        assert not runtime._is_pid_running(gateway_pid)
    finally:
        runtime = adapter._runtime
        selection = adapter._selection
        if runtime is not None and selection is not None:
            try:
                current = runtime._read_state()
                if current is not None:
                    identity = runtime.desktop_identity(selection.options)
                    if identity is not None and (gateway_pid is None or identity.pid == gateway_pid):
                        runtime.stop_desktop(selection.options, handle or identity.handle, timeout_s=5)
                child = runtime._owned_process
                if child is not None and gateway_pid in {None, child.pid}:
                    child.wait(timeout=5)
            except Exception:
                pass
        adapter.close()


def test_desktop_created_gateway_rejects_port_proof_mismatch_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.gateway.runtime import GatewayClientLease

    config_path = tmp_path / "profile" / "config.json"
    gateway_port, websocket_port = _prepared_config(config_path, tmp_path / "workspace")
    adapter = DesktopAdapter()
    handle: str | None = None
    original_state: dict[str, Any] | None = None
    try:
        adapter.handle("hello", {"config_path": str(config_path), "workspace": None})
        runtime = adapter._runtime
        selection = adapter._selection
        assert runtime is not None
        assert selection is not None
        started = adapter.handle("start", {})
        handle_value = started["handle"]
        assert isinstance(handle_value, str)
        handle = handle_value
        assert started["state"] == "starting"
        running = _wait_until_running(adapter, handle)
        assert running["state"] == "running"

        original_state = runtime._read_state()
        assert original_state is not None
        assert original_state.get("port") == gateway_port
        assert original_state.get("desktop_management_port") == gateway_port
        assert original_state.get("desktop_websocket_ports") == [websocket_port]
        lease_before = GatewayClientLease(runtime, kind="port-proof-check").snapshot()

        for field, altered in (
            ("port", gateway_port % 65535 + 1),
            ("port", True),
            ("port", str(gateway_port)),
            ("desktop_management_port", gateway_port % 65535 + 1),
            ("desktop_management_port", True),
            ("desktop_management_port", 65536),
            ("desktop_websocket_ports", [websocket_port % 65535 + 1]),
            ("desktop_websocket_ports", [True]),
        ):
            candidate = dict(original_state)
            candidate[field] = altered
            runtime._write_state(candidate)
            sent_signals: list[int] = []

            def refuse_signal(_pidfd: int, sent_signal: int) -> None:
                sent_signals.append(sent_signal)
                raise OSError("unexpected desktop signal")

            with monkeypatch.context() as patcher:
                patcher.setattr(runtime, "_pidfd_send_signal", refuse_signal)
                with pytest.raises(DesktopError) as status_error:
                    adapter.handle("status", {"handle": handle})
                assert status_error.value.code == "gateway_selector_conflict"

                start_result = adapter.handle("start", {})
                assert start_result["state"] == "unavailable"
                assert start_result["code"] == "gateway_selector_conflict"

                stop_result = runtime.stop_desktop(selection.options, handle, timeout_s=1)
                assert not stop_result.ok
                assert stop_result.message == "gateway_selector_conflict"

            assert sent_signals == []
            assert runtime._read_state() == candidate
            assert GatewayClientLease(runtime, kind="port-proof-check").snapshot() == lease_before

        missing_owner_ports = dict(original_state)
        missing_owner_ports.pop("desktop_management_port")
        missing_owner_ports.pop("desktop_websocket_ports")
        runtime._write_state(missing_owner_ports)
        with pytest.raises(DesktopError) as missing_error:
            adapter.handle("status", {"handle": handle})
        assert missing_error.value.code == "gateway_identity_unavailable"
        assert runtime._read_state() == missing_owner_ports
        assert GatewayClientLease(runtime, kind="port-proof-check").snapshot() == lease_before

        runtime._write_state(original_state)
        stopped = adapter.handle("stop", {"handle": handle})
        assert stopped["state"] == "stopped"
        child = runtime._owned_process
        assert child is not None
        assert child.wait(timeout=5) == 0
    finally:
        runtime = adapter._runtime
        selection = adapter._selection
        if runtime is not None and selection is not None:
            if original_state is not None:
                runtime._write_state(original_state)
            try:
                identity = runtime.desktop_identity(selection.options, handle=handle)
                if identity is not None:
                    runtime.stop_desktop(selection.options, handle or identity.handle, timeout_s=5)
                child = runtime._owned_process
                if child is not None and child.poll() is None:
                    child.wait(timeout=5)
            except Exception:
                pass
        adapter.close()


def test_desktop_start_keeps_admission_handle_if_state_is_replaced_before_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.gateway.runtime import (
        GatewayDesktopLifecycleError,
        GatewayStartOptions,
        RuntimeResult,
    )

    config_path = tmp_path / "profile" / "config.json"
    _prepared_config(config_path, tmp_path / "workspace")
    adapter = DesktopAdapter()
    replacement: subprocess.Popen[bytes] | None = None
    runtime = None
    selection = None
    admitted_state: dict[str, object] | None = None
    admitted_handle: str | None = None
    try:
        adapter.handle("hello", {"config_path": str(config_path), "workspace": None})
        runtime = adapter._runtime
        selection = adapter._selection
        assert runtime is not None
        assert selection is not None
        original_start = runtime.start_desktop
        captured: list[str] = []

        def replace_after_admission(
            options: GatewayStartOptions,
            *,
            preflight: Callable[[], None],
            lock_timeout_s: float = 2.0,
        ) -> RuntimeResult:
            nonlocal replacement, admitted_state, admitted_handle
            admission = original_start(
                options,
                preflight=preflight,
                lock_timeout_s=lock_timeout_s,
            )
            state = runtime._read_state()
            assert state is not None
            admitted_state = state
            identity = getattr(admission, "identity", None)
            if identity is None:
                identity = runtime.desktop_identity(options)
            assert identity is not None
            admitted_handle = identity.handle
            captured.append(identity.handle)
            replacement = subprocess.Popen(
                [sys.executable, "-I", "-c", "import time; time.sleep(60)"],
                start_new_session=True,
            )
            runtime._write_state({"pid": replacement.pid, "marker": "replacement"})
            return admission

        monkeypatch.setattr(runtime, "start_desktop", replace_after_admission)
        started = adapter.handle("start", {})
        assert started["state"] == "starting"
        assert started["handle"] == captured[0]
        assert started["handle"] == admitted_handle

        signal_attempts: list[int] = []
        with monkeypatch.context() as signal_patch:
            signal_patch.setattr(
                runtime,
                "_pidfd_send_signal",
                lambda _pidfd, sent_signal: signal_attempts.append(sent_signal),
            )
            refused = adapter.handle("stop", {"handle": admitted_handle})
        assert refused["state"] == "unavailable"
        assert refused["code"] == "gateway_identity_replaced"
        assert signal_attempts == []
        assert replacement is not None and replacement.poll() is None
    finally:
        if runtime is not None:
            state = runtime._read_state()
            if state is not None and state.get("marker") == "replacement" and admitted_state is not None:
                runtime._write_state(admitted_state)
            if admitted_state is not None and admitted_handle is not None:
                try:
                    adapter.handle("stop", {"handle": admitted_handle})
                    child = runtime._owned_process
                    if child is not None:
                        child.wait(timeout=5)
                except (DesktopError, GatewayDesktopLifecycleError, subprocess.TimeoutExpired):
                    pass
        if replacement is not None:
            replacement.terminate()
            replacement.wait(timeout=5)
        adapter.close()


def test_hello_rejects_missing_environment_references_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    config_path = tmp_path / "profile" / "config.json"
    _prepared_config(config_path, workspace)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "desktop-provider-sentinel", "${NANOBOT_DESKTOP_MISSING_SENTINEL}"
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("NANOBOT_DESKTOP_MISSING_SENTINEL", raising=False)
    before = config_path.read_bytes()
    adapter = DesktopAdapter()

    with pytest.raises(DesktopError) as error:
        adapter.handle("hello", {"config_path": str(config_path), "workspace": None})

    assert error.value.code == "config_invalid"
    assert "NANOBOT_DESKTOP_MISSING_SENTINEL" not in str(error.value)
    assert config_path.read_bytes() == before
    assert not (workspace / ".nanobot").exists()
    assert adapter._selection is None


def test_hello_resolves_environment_references_before_provider_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nanobot.desktop.adapter as adapter_module

    workspace = tmp_path / "workspace"
    config_path = tmp_path / "profile" / "config.json"
    _prepared_config(config_path, workspace)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "desktop-provider-sentinel", "${NANOBOT_DESKTOP_PROVIDER_SENTINEL}"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NANOBOT_DESKTOP_PROVIDER_SENTINEL", "resolved-provider-secret")
    observed: list[str] = []
    monkeypatch.setattr(
        adapter_module,
        "validate_provider_setup",
        lambda config: observed.append(config.providers.openai.api_key),
    )
    monkeypatch.setattr(adapter_module, "runtime_preflight", lambda *_args, **_kwargs: None)
    before = config_path.read_bytes()
    adapter = DesktopAdapter()

    result = adapter.handle("hello", {"config_path": str(config_path), "workspace": None})

    assert result["protocol_version"] == 1
    assert observed == ["resolved-provider-secret", "resolved-provider-secret"]
    assert config_path.read_bytes() == before
    assert "resolved-provider-secret" not in json.dumps(result)
    adapter.close()


def test_child_rejects_missing_environment_references_before_gateway_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import nanobot.desktop.gateway_child as child

    workspace = tmp_path / "workspace"
    config_path = tmp_path / "profile" / "config.json"
    _prepared_config(config_path, workspace)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "desktop-provider-sentinel", "${NANOBOT_DESKTOP_CHILD_MISSING_SENTINEL}"
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("NANOBOT_DESKTOP_CHILD_MISSING_SENTINEL", raising=False)
    starts: list[bool] = []
    monkeypatch.setattr(child, "_run_gateway", lambda *_args, **_kwargs: starts.append(True))
    before = config_path.read_bytes()

    result = child.main(["--config", str(config_path)])

    assert result == 1
    assert starts == []
    assert "NANOBOT_DESKTOP_CHILD_MISSING_SENTINEL" not in capsys.readouterr().err
    assert config_path.read_bytes() == before
    assert not (workspace / ".nanobot").exists()


def test_child_resolves_environment_references_before_provider_and_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nanobot.desktop.gateway_child as child

    workspace = tmp_path / "workspace"
    config_path = tmp_path / "profile" / "config.json"
    _prepared_config(config_path, workspace)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "desktop-provider-sentinel", "${NANOBOT_DESKTOP_CHILD_PROVIDER_SENTINEL}"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "NANOBOT_DESKTOP_CHILD_PROVIDER_SENTINEL", "child-resolved-provider-secret"
    )
    observed: list[str] = []
    monkeypatch.setattr(
        child,
        "validate_provider_setup",
        lambda config: observed.append(config.providers.openai.api_key),
    )
    monkeypatch.setattr(child, "runtime_preflight", lambda *_args, **_kwargs: None)
    started: list[str] = []
    monkeypatch.setattr(
        child,
        "_run_gateway",
        lambda config, **_kwargs: started.append(config.providers.openai.api_key),
    )
    before = config_path.read_bytes()

    result = child.main(["--config", str(config_path)])

    assert result == 0
    assert observed == ["child-resolved-provider-secret"]
    assert started == ["child-resolved-provider-secret"]
    assert config_path.read_bytes() == before


def test_desktop_lifecycle_controls_one_isolated_child_and_confirms_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    config_path = tmp_path / "profile" / "config.json"
    _prepared_config(config_path, workspace)
    adapter = DesktopAdapter()
    unrelated: subprocess.Popen[bytes] | None = None
    original_state: dict[str, object] | None = None
    original_signal_verifier = None
    runtime = None
    gateway_pid: int | None = None
    try:
        adapter.handle("hello", {"config_path": str(config_path), "workspace": None})
        runtime = adapter._runtime
        selection = adapter._selection
        assert runtime is not None
        assert selection is not None
        stale_state = {
            "pid": 1_000_000_000,
            "port": selection.options.port,
            "workspace": selection.options.workspace,
            "config_path": selection.options.config_path,
            "command": runtime._build_child_command(selection.options),
            "launch_mode": "background",
            "stable_identity": "stale-process-birth",
        }
        runtime._write_state(stale_state)
        started = adapter.handle("start", {})
        assert started["state"] == "starting"
        handle = started["handle"]
        assert isinstance(handle, str)
        _wait_until_running(adapter, handle)
        identity = runtime.desktop_identity(selection.options, handle=handle)
        assert identity is not None
        gateway_pid = identity.pid

        state = runtime._read_state()
        assert state is not None
        command = state["command"]
        assert isinstance(command, list)
        assert command[:4] == [
            sys.executable,
            "-I",
            "-m",
            "nanobot.desktop.gateway_child",
        ]
        actual_cmdline = [
            part.decode() for part in Path(f"/proc/{gateway_pid}/cmdline").read_bytes().split(b"\0") if part
        ]
        assert actual_cmdline == command

        status = adapter.handle("status", {"handle": handle})
        assert status["state"] == "running"
        bootstrap = adapter.handle("bootstrap", {"handle": handle})
        assert bootstrap["origin"] == started["origin"]
        assert bootstrap["secret"] == "desktop-webui-sentinel"
        assert isinstance(bootstrap["ws_url"], str)

        reconciled = adapter.handle("start", {"handle": handle})
        assert reconciled["state"] == "starting"
        assert reconciled["handle"] == handle
        _wait_until_running(adapter, handle)

        claimed_state = runtime._read_state()
        assert claimed_state is not None
        runtime.paths.state_path.unlink()
        missing_claim_signals: list[int] = []
        with monkeypatch.context() as claim_patch:
            claim_patch.setattr(
                runtime,
                "_pidfd_send_signal",
                lambda _pidfd, sent_signal: missing_claim_signals.append(sent_signal),
            )
            missing_claim = adapter.handle("stop", {"handle": handle})
        assert missing_claim["state"] == "unavailable"
        assert missing_claim["code"] == "gateway_identity_conflict"
        assert missing_claim_signals == []
        assert runtime._is_pid_running(gateway_pid)
        runtime._write_state(claimed_state)

        # Hold the verified child alive through a deterministic grace timeout.
        # The only requested signal is SIGTERM and the selected state survives.
        timeout_signals: list[int] = []
        with monkeypatch.context() as timeout_patch:
            timeout_patch.setattr(
                runtime,
                "_pidfd_send_signal",
                lambda _pidfd, sent_signal: timeout_signals.append(sent_signal),
            )
            timeout_patch.setattr(runtime, "_wait_for_pidfd_exit", lambda _fd, _timeout: False)
            timeout_result = runtime.stop_desktop(
                selection.options,
                handle,
                timeout_s=0,
            )
        assert not timeout_result.ok
        assert timeout_result.message == "gateway_stop_timeout"
        assert timeout_signals == [signal.SIGTERM]
        assert runtime._read_state() == claimed_state
        assert runtime._is_pid_running(gateway_pid)

        # Replace the state at the final signal boundary. pidfd must remain
        # bound to the original gateway, and the replacement must be untouched.
        unrelated = subprocess.Popen(
            [sys.executable, "-I", "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        original_state = runtime._read_state()
        assert original_state is not None
        original_signal_verifier = runtime._verify_desktop_signal_target
        replacement = {
            "pid": unrelated.pid,
            "marker": "replacement-state",
        }

        def replace_at_signal_boundary(*_args: object, **_kwargs: object) -> bool:
            runtime._write_state(replacement)
            return False

        monkeypatch.setattr(runtime, "_verify_desktop_signal_target", replace_at_signal_boundary)
        refused_stop = adapter.handle("stop", {"handle": handle})
        assert refused_stop["state"] == "unavailable"
        assert refused_stop["code"] == "gateway_identity_conflict"
        assert runtime._read_state() == replacement
        assert runtime._is_pid_running(gateway_pid)
        assert unrelated.poll() is None

        runtime._write_state(original_state)
        monkeypatch.setattr(runtime, "_verify_desktop_signal_target", original_signal_verifier)
        stopped = adapter.handle("stop", {"handle": handle})
        assert stopped["state"] == "stopped"
        assert not runtime._is_pid_running(gateway_pid)
        assert not runtime.paths.state_path.exists()
        child = runtime._owned_process
        assert child is not None
        assert child.wait(timeout=5) == 0

        # A captured handle is reconciliation-only after exit; it cannot spawn.
        absent = adapter.handle("start", {"handle": handle})
        assert absent["state"] == "stopped"
        assert not runtime.paths.state_path.exists()
    finally:
        if runtime is not None and original_signal_verifier is not None:
            monkeypatch.setattr(runtime, "_verify_desktop_signal_target", original_signal_verifier)
        if original_state is not None and runtime is not None:
            state = runtime._read_state()
            if state is not None and state.get("marker") == "replacement-state":
                runtime._write_state(original_state)
        if gateway_pid is not None and runtime is not None:
            try:
                current = runtime._read_state()
                if current is not None and current.get("pid") == gateway_pid:
                    handle_value = started.get("handle") if "started" in locals() else None
                    if isinstance(handle_value, str):
                        adapter.handle("stop", {"handle": handle_value})
            except Exception:
                pass
        if unrelated is not None:
            unrelated.terminate()
            unrelated.wait(timeout=5)
        adapter.close()


def test_desktop_adopts_normal_on_demand_gateway_and_promotes_its_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nanobot.desktop.adapter as adapter_module
    import nanobot.gateway.runtime as gateway_runtime_module
    from nanobot.gateway.runtime import (
        GatewayClientLease,
        GatewayDesktopLifecycleError,
        GatewayInstance,
        GatewayRuntime,
        GatewayStartOptions,
        RuntimeResult,
    )

    workspace = tmp_path / "workspace"
    config_path = tmp_path / "profile" / "config.json"
    gateway_port, websocket_port = _prepared_config(config_path, workspace)
    instance = GatewayInstance.resolve(config_path=config_path, workspace=None)
    normal_runtime = GatewayRuntime(paths=instance.paths, python_executable=sys.executable)
    client_lease = GatewayClientLease(normal_runtime, kind="desktop-adoption-fixture")
    normal_child: subprocess.Popen[bytes] | None = None
    adapter = DesktopAdapter()
    try:
        client_lease.acquire()
        launch = client_lease.ensure_on_demand_gateway(
            instance.start_options(port=gateway_port),
        )
        assert launch.ok
        normal_child = normal_runtime._owned_process
        assert normal_child is not None

        adapter.handle("hello", {"config_path": str(config_path), "workspace": None})
        runtime = adapter._runtime
        selection = adapter._selection
        assert runtime is not None
        assert selection is not None
        assert selection.workspace_selector is None
        assert selection.workspace == workspace.resolve()
        assert selection.instance.paths == instance.paths
        assert runtime.paths.state_path == normal_runtime.paths.state_path

        deadline = time.monotonic() + 20.0
        before_identity = None
        while time.monotonic() < deadline:
            try:
                before_identity = runtime.desktop_identity(selection.options)
            except GatewayDesktopLifecycleError as exc:
                assert exc.code == "gateway_identity_unavailable"
                before_identity = None
            if before_identity is not None and before_identity.gateway_id:
                break
            if normal_child.poll() is not None:
                break
            time.sleep(0.05)
        assert before_identity is not None
        assert before_identity.gateway_id
        assert before_identity.backend_version
        assert before_identity.capability == "managed-gateway-v1"
        assert before_identity.python_executable == sys.executable

        normal_state = runtime._read_state()
        assert normal_state is not None
        assert normal_state.get("desktop_management_port") == gateway_port
        assert normal_state.get("desktop_websocket_ports") == [websocket_port]
        assert normal_child.pid is not None
        assert before_identity.birth_identity == normal_state.get("identity")
        assert Path(f"/proc/{normal_child.pid}").stat().st_uid == os.geteuid()
        assert os.getpgid(normal_child.pid) == normal_child.pid
        assert os.getsid(normal_child.pid) == normal_child.pid
        assert Path(f"/proc/{normal_child.pid}/comm").read_text(encoding="ascii").strip() == (
            "nanobot-gateway"
        )
        assert [
            part.decode()
            for part in Path(f"/proc/{normal_child.pid}/cmdline").read_bytes().split(b"\0")
            if part
        ] == ["nanobot-gateway"]

        def assert_refused_without_state_or_signal(
            candidate_state: dict[str, Any],
            expected_code: str,
            *,
            start_operation: Callable[[], object],
            selected_runtime: GatewayRuntime,
            selected_options: GatewayStartOptions,
            stop_code: str | None = None,
        ) -> None:
            runtime._write_state(candidate_state)
            lease_before = GatewayClientLease(
                runtime,
                kind="desktop-adoption-check",
            ).snapshot()
            sent_signals: list[int] = []
            with monkeypatch.context() as signal_patch:
                signal_patch.setattr(
                    selected_runtime,
                    "_pidfd_send_signal",
                    lambda _pidfd, sent_signal: sent_signals.append(sent_signal),
                )
                start_result = start_operation()
                if isinstance(start_result, dict):
                    assert start_result["state"] == "unavailable"
                    assert start_result["code"] == expected_code
                else:
                    assert isinstance(start_result, RuntimeResult)
                    assert not start_result.ok
                    assert start_result.message == expected_code
                stop_result = selected_runtime.stop_desktop(
                    selected_options,
                    before_identity.handle,
                )
            assert not stop_result.ok
            assert stop_result.message == (stop_code or expected_code)
            assert sent_signals == []
            assert runtime._read_state() == candidate_state
            assert GatewayClientLease(
                runtime,
                kind="desktop-adoption-check",
            ).snapshot() == lease_before
            assert normal_child.poll() is None

        foreground_state = dict(normal_state)
        foreground_state["launch_mode"] = "foreground"
        assert_refused_without_state_or_signal(
            foreground_state,
            "gateway_identity_conflict",
            start_operation=lambda: adapter.handle("start", {}),
            selected_runtime=runtime,
            selected_options=selection.options,
        )

        with monkeypatch.context() as group_patch:
            group_patch.setattr(
                gateway_runtime_module.os,
                "getpgid",
                lambda pid: pid + 1,
            )
            assert_refused_without_state_or_signal(
                normal_state,
                "gateway_identity_conflict",
                start_operation=lambda: adapter.handle("start", {}),
                selected_runtime=runtime,
                selected_options=selection.options,
            )

        # A role title is only a Linux proctitle hint. It cannot stand in for
        # fresh owner publication or authorize either adoption or stop.
        legacy_state = dict(normal_state)
        for key in (
            "desktop_gateway_id",
            "desktop_backend_version",
            "desktop_capability",
            "desktop_python_executable",
        ):
            legacy_state.pop(key, None)
        assert_refused_without_state_or_signal(
            legacy_state,
            "gateway_identity_unavailable",
            start_operation=lambda: adapter.handle("start", {}),
            selected_runtime=runtime,
            selected_options=selection.options,
        )

        legacy_without_port_proof = dict(normal_state)
        legacy_without_port_proof.pop("desktop_management_port", None)
        legacy_without_port_proof.pop("desktop_websocket_ports", None)
        assert_refused_without_state_or_signal(
            legacy_without_port_proof,
            "gateway_identity_unavailable",
            start_operation=lambda: adapter.handle("start", {}),
            selected_runtime=runtime,
            selected_options=selection.options,
        )

        missing_capability_state = dict(normal_state)
        missing_capability_state.pop("desktop_capability", None)
        assert_refused_without_state_or_signal(
            missing_capability_state,
            "gateway_identity_unavailable",
            start_operation=lambda: adapter.handle("start", {}),
            selected_runtime=runtime,
            selected_options=selection.options,
        )

        wrong_selector = replace(
            selection.options,
            workspace=str((tmp_path / "different-workspace").resolve()),
        )
        assert_refused_without_state_or_signal(
            normal_state,
            "gateway_selector_conflict",
            start_operation=lambda: runtime.start_desktop(
                wrong_selector,
                preflight=lambda: adapter._check_preflight(selection),
            ),
            selected_runtime=runtime,
            selected_options=wrong_selector,
        )

        wrong_port = replace(
            selection.options,
            port=selection.options.port % 65535 + 1,
        )
        assert_refused_without_state_or_signal(
            normal_state,
            "gateway_selector_conflict",
            start_operation=lambda: runtime.start_desktop(
                wrong_port,
                preflight=lambda: adapter._check_preflight(selection),
            ),
            selected_runtime=runtime,
            selected_options=wrong_port,
        )

        for field, altered in (
            ("port", gateway_port % 65535 + 1),
            ("desktop_management_port", gateway_port % 65535 + 1),
            ("desktop_websocket_ports", [websocket_port % 65535 + 1]),
        ):
            mismatched_ports = dict(normal_state)
            mismatched_ports[field] = altered
            assert_refused_without_state_or_signal(
                mismatched_ports,
                "gateway_selector_conflict",
                start_operation=lambda: adapter.handle("start", {}),
                selected_runtime=runtime,
                selected_options=selection.options,
            )

        wrong_websocket_port = replace(
            selection.options,
            desktop_listener_ports=(websocket_port % 65535 + 1,),
        )
        assert_refused_without_state_or_signal(
            normal_state,
            "gateway_selector_conflict",
            start_operation=lambda: runtime.start_desktop(
                wrong_websocket_port,
                preflight=lambda: adapter._check_preflight(selection),
            ),
            selected_runtime=runtime,
            selected_options=wrong_websocket_port,
        )

        runtime._write_state(normal_state)

        alternate_python = tmp_path / "alternate-venv" / "bin" / "python"
        alternate_python.parent.mkdir(parents=True)
        alternate_python.symlink_to(Path(sys.executable).resolve())
        assert Path(sys.executable).samefile(alternate_python)

        # The executable inode alone cannot distinguish venv selectors that
        # resolve to the same system interpreter; the published path must match.
        wrong_interpreter_state = dict(normal_state)
        wrong_interpreter_state["desktop_python_executable"] = str(alternate_python)
        assert_refused_without_state_or_signal(
            wrong_interpreter_state,
            "gateway_identity_unavailable",
            start_operation=lambda: adapter.handle("start", {}),
            selected_runtime=runtime,
            selected_options=selection.options,
        )
        runtime._write_state(normal_state)

        # Simulate the alternate venv as the caller's selected interpreter and
        # a matching persisted command. /proc/exe samefile and the role title
        # both pass; independent current-process publication still rejects it.
        alternate_state = dict(normal_state)
        normal_command = alternate_state.get("command")
        assert isinstance(normal_command, list)
        alternate_state["command"] = [str(alternate_python), *normal_command[1:]]
        runtime._write_state(alternate_state)
        alternate_runtime = GatewayRuntime(
            paths=instance.paths,
            python_executable=str(alternate_python),
        )
        assert_refused_without_state_or_signal(
            alternate_state,
            "gateway_identity_unavailable",
            start_operation=lambda: alternate_runtime.start_desktop(
                selection.options,
                preflight=lambda: adapter._check_preflight(selection),
            ),
            selected_runtime=alternate_runtime,
            selected_options=selection.options,
            stop_code="gateway_selector_conflict",
        )
        runtime._write_state(normal_state)

        initial_lease = GatewayClientLease(runtime, kind="desktop-adoption-check").snapshot()
        assert initial_lease.auto_stop is True

        original_request_json = adapter_module._request_json
        for route, status_code, injected_payload, expected_code in (
            ("/health", 200, {"ready": False}, "gateway_unavailable"),
            (
                "/webui/terminal",
                200,
                {"protocolVersion": 1, "gatewayId": "unrelated-gateway"},
                "gateway_identity_mismatch",
            ),
            ("/webui/terminal", 401, {"error": "unauthorized"}, "webui_auth_failed"),
            ("/assets", 503, {"error": "unavailable"}, "webui_assets_unavailable"),
        ):
            def degraded_request(
                host: str,
                port: int,
                path: str,
                **kwargs: Any,
            ) -> tuple[int, object]:
                if path == route:
                    return status_code, injected_payload
                return original_request_json(host, port, path, **kwargs)

            with monkeypatch.context() as readiness_patch:
                readiness_patch.setattr(adapter_module, "_request_json", degraded_request)
                if route == "/assets":
                    def fail_asset_probe(*_args: Any, **_kwargs: Any) -> None:
                        raise DesktopError("webui_assets_unavailable")

                    readiness_patch.setattr(adapter_module, "_probe_assets", fail_asset_probe)
                rejected = adapter.handle("status", {"handle": before_identity.handle})
            assert rejected["state"] == "unavailable"
            assert rejected["code"] == expected_code
            assert rejected["handle"] == before_identity.handle
            assert runtime._read_state() == normal_state
            assert normal_child.poll() is None
            same_identity = runtime.desktop_identity(
                selection.options,
                handle=before_identity.handle,
            )
            assert same_identity is not None
            assert same_identity.pid == before_identity.pid

        result = adapter.handle("start", {})
        assert result["state"] == "starting"
        handle = result["handle"]
        assert isinstance(handle, str)
        assert handle == before_identity.handle
        _wait_until_running(adapter, handle)
        after_identity = runtime.desktop_identity(selection.options, handle=handle)
        assert after_identity is not None
        assert after_identity.pid == before_identity.pid
        assert after_identity.birth_identity == before_identity.birth_identity
        assert after_identity.handle == before_identity.handle
        after_lease = GatewayClientLease(runtime, kind="desktop-adoption-check").snapshot()
        assert after_lease.auto_stop is False
        assert normal_child.poll() is None

        # Simulate adapter loss and a fresh host-side adapter reconnecting to
        # the persistent gateway without replaying or replacing the process.
        adapter.close()
        adapter = DesktopAdapter()
        adapter.handle("hello", {"config_path": str(config_path), "workspace": None})
        reattached = adapter.handle("start", {})
        assert reattached["state"] == "starting"
        assert reattached["handle"] == handle
        _wait_until_running(adapter, handle)
        assert adapter._runtime is not None
        assert adapter._selection is not None
        reattached_identity = adapter._runtime.desktop_identity(
            adapter._selection.options,
            handle=handle,
        )
        assert reattached_identity is not None
        assert reattached_identity.pid == before_identity.pid
        assert normal_child.poll() is None

        stopped = adapter.handle("stop", {"handle": handle})
        assert stopped["state"] == "stopped"
        assert normal_child.wait(timeout=5) == 0
    finally:
        if normal_child is not None and normal_child.poll() is None:
            state = normal_runtime._read_state()
            if state is not None:
                normal_runtime.stop(timeout_s=5)
            if normal_child.poll() is None:
                normal_child.terminate()
            normal_child.wait(timeout=5)
        client_lease.release(wait_for_stop=False)
        adapter.close()


def test_simultaneous_desktop_acquisitions_start_one_gateway(tmp_path: Path) -> None:
    from nanobot.gateway.runtime import GatewayClientLease

    config_path = tmp_path / "profile" / "config.json"
    workspace = tmp_path / "workspace"
    _prepared_config(config_path, workspace)
    adapters = [DesktopAdapter(), DesktopAdapter()]
    try:
        for adapter in adapters:
            adapter.handle("hello", {"config_path": str(config_path), "workspace": None})

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(adapter.handle, "start", {}) for adapter in adapters]
            results = [future.result(timeout=35) for future in futures]

        assert all(result["state"] == "starting" for result in results)

        handles = [result["handle"] for result in results]
        assert all(isinstance(handle, str) for handle in handles)
        assert handles[0] == handles[1]
        for adapter, handle in zip(adapters, handles, strict=True):
            assert isinstance(handle, str)
            _wait_until_running(adapter, handle)
        identities = [
            adapter._runtime.desktop_identity(adapter._selection.options, handle=handle)
            for adapter, handle in zip(adapters, handles, strict=True)
            if adapter._runtime is not None and adapter._selection is not None
        ]
        assert len(identities) == 2
        assert identities[0] is not None
        assert identities[1] is not None
        assert identities[0].pid == identities[1].pid

        child_pids: list[int] = []
        for proc_entry in Path("/proc").iterdir():
            if not proc_entry.name.isdigit():
                continue
            try:
                command = (proc_entry / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if b"nanobot.desktop.gateway_child" in command and os.fsencode(
                str(config_path)
            ) in command:
                child_pids.append(int(proc_entry.name))
        assert child_pids == [identities[0].pid]

        for adapter, handle in zip(adapters, handles, strict=True):
            lease = GatewayClientLease(
                adapter._runtime,
                kind="desktop-concurrent-check",
            ).snapshot()
            assert lease.auto_stop is False
            assert lease.clients >= 1
        stopped = adapters[0].handle("stop", {"handle": handles[0]})
        assert stopped["state"] == "stopped"
        for adapter in adapters:
            runtime = adapter._runtime
            if runtime is not None and runtime._owned_process is not None:
                assert runtime._owned_process.wait(timeout=5) == 0
    finally:
        for adapter in adapters:
            runtime = adapter._runtime
            selection = adapter._selection
            if runtime is not None and selection is not None:
                try:
                    identity = runtime.desktop_identity(selection.options)
                    if identity is not None:
                        runtime.stop_desktop(selection.options, identity.handle, timeout_s=5)
                except Exception:
                    pass
                child = runtime._owned_process
                if child is not None and child.poll() is None:
                    child.wait(timeout=5)
            adapter.close()


@pytest.mark.parametrize("listener", ["gateway", "websocket"])
def test_desktop_refuses_foreign_listener_without_starting(
    tmp_path: Path,
    listener: str,
) -> None:
    gateway_port, websocket_port = _prepared_config(
        tmp_path / "profile" / "config.json",
        tmp_path / "workspace",
    )
    config_path = tmp_path / "profile" / "config.json"
    adapter = DesktopAdapter()
    adapter.handle("hello", {"config_path": str(config_path), "workspace": None})
    runtime = adapter._runtime
    selection = adapter._selection
    assert runtime is not None
    assert selection is not None
    port = gateway_port if listener == "gateway" else websocket_port
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as foreign_listener:
            foreign_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            foreign_listener.bind(("127.0.0.1", port))
            foreign_listener.listen(1)
            result = adapter.handle("start", {})
        assert result["state"] == "unavailable"
        assert result["code"] == "gateway_listener_conflict"
        assert result["handle"] is None
        assert not runtime.paths.state_path.exists()
        lease = runtime._desktop_lease
        assert lease is None
        assert runtime._owned_process is None
    finally:
        adapter.close()


def test_hello_refuses_non_loopback_gateway_configuration(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    config_path = tmp_path / "profile" / "config.json"
    _prepared_config(config_path, workspace, host="0.0.0.0")
    adapter = DesktopAdapter()

    with pytest.raises(DesktopError) as error:
        adapter.handle("hello", {"config_path": str(config_path), "workspace": None})

    assert error.value.code == "loopback_required"

