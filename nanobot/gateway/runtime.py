"""Gateway-specific configuration for the shared background process runtime."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import http.client
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generator, Literal, cast

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from nanobot.config.paths import get_data_dir
from nanobot.process_runtime import (
    ManagedProcessRuntime,
    ProcessResult,
    ProcessRuntimePaths,
    ProcessStartOptions,
    ProcessStatus,
    _as_int,
    _as_str,
    process_identity_record,
    process_is_running,
)


@dataclass(frozen=True)
class GatewayStartOptions(ProcessStartOptions):
    """Gateway process options, including the compatibility log destination."""

    log_file: str | None = None
    desktop_managed: bool = False
    desktop_listener_ports: tuple[int, ...] = ()

GatewayLaunchMode = Literal["foreground", "background", "unknown"]
GatewayLifetime = Literal["explicit", "on_demand"]
_DESKTOP_HANDLE_VERSION = 1
_DESKTOP_CAPABILITY = "managed-gateway-v1"
_DESKTOP_ADMISSION_LISTENER_PORTS = "desktop_admission_websocket_ports"
_DESKTOP_OWNER_MANAGEMENT_PORT = "desktop_management_port"
_DESKTOP_OWNER_LISTENER_PORTS = "desktop_websocket_ports"


def _is_valid_desktop_port(value: object) -> bool:
    return type(value) is int and 1 <= value <= 65535


def _gateway_health_ready(host: str, port: int, *, timeout_s: float = 0.4) -> bool:
    """Read readiness from the management listener without using proxy settings."""
    connect_host = "127.0.0.1" if host in {"", "0.0.0.0"} else "::1" if host == "::" else host
    connection = http.client.HTTPConnection(connect_host, port, timeout=timeout_s)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        body = response.read(1024)
    except (OSError, http.client.HTTPException, TimeoutError):
        return False
    finally:
        connection.close()
    if response.status != 200:
        return False
    try:
        raw_payload = cast(object, json.loads(body.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(raw_payload, dict):
        return False
    payload = cast(dict[str, object], raw_payload)
    return (
        payload.get("status") == "ok"
        and payload.get("ready") is not False
    )


def _default_config_path() -> Path:
    return (Path.home() / ".nanobot" / "config.json").resolve(strict=False)


@dataclass(frozen=True)
class GatewayStatus(ProcessStatus):
    """Observable lifecycle state for one shared local gateway."""

    launch_mode: GatewayLaunchMode = "unknown"
    lifetime: GatewayLifetime = "explicit"
    clients: int = 0
    ready: bool | None = None


@dataclass(frozen=True)
class GatewayLeaseSnapshot:
    """Live local clients and the gateway lifetime they imply."""

    auto_stop: bool
    clients: int


@dataclass(frozen=True)
class RuntimeResult(ProcessResult):
    """Result of a gateway lifecycle operation."""

    status: GatewayStatus
    promoted: bool = False


@dataclass(frozen=True)
class DesktopGatewayIdentity:
    """Captured process identity for the private desktop lifecycle adapter."""

    pid: int
    birth_identity: str
    process_group: int
    handle: str
    gateway_id: str | None = None
    backend_version: str | None = None
    capability: str | None = None
    python_executable: str | None = None
    starting: bool = False


@dataclass(frozen=True)
class DesktopAdmissionResult(RuntimeResult):
    """Lifecycle admission result bound to the process captured under its locks."""

    identity: DesktopGatewayIdentity | None = None


class GatewayDesktopLifecycleError(RuntimeError):
    """Safe lifecycle refusal code for desktop-owned gateway operations."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class _DesktopHandlePayload:
    pid: int
    birth: str
    group: int
    selectors: str


class GatewayAlreadyRunningError(RuntimeError):
    """Raised when a foreground gateway tries to replace a live instance."""

    def __init__(self, status: GatewayStatus) -> None:
        super().__init__("gateway_already_running")
        self.status = status


def build_gateway_command(python_executable: str, options: ProcessStartOptions) -> list[str]:
    """Build a foreground gateway command for process supervisors."""
    if isinstance(options, GatewayStartOptions) and options.desktop_managed:
        if not options.config_path:
            raise GatewayDesktopLifecycleError("config_unavailable")
        command = [
            python_executable,
            "-I",
            "-m",
            "nanobot.desktop.gateway_child",
            "--config",
            options.config_path,
        ]
        if options.workspace:
            command.extend(["--workspace", options.workspace])
        return command
    command = [
        python_executable,
        "-m",
        "nanobot",
        "gateway",
        "--foreground",
        "--port",
        str(options.port),
    ]
    if options.verbose:
        command.append("--verbose")
    if options.workspace:
        command.extend(["--workspace", options.workspace])
    if options.config_path:
        command.extend(["--config", options.config_path])
    log_file = getattr(options, "log_file", None)
    if isinstance(log_file, str) and log_file:
        command.extend(["--log-file", log_file])
    return command


@dataclass(frozen=True)
class GatewayRuntimePaths(ProcessRuntimePaths):
    """Filesystem layout for one gateway runtime instance."""

    @classmethod
    def for_instance(
        cls,
        *,
        data_dir: Path | None = None,
        workspace: str | None = None,
        config_path: str | None = None,
    ) -> "GatewayRuntimePaths":
        base = data_dir or get_data_dir()
        suffix = _instance_suffix(workspace=workspace, config_path=config_path)
        run_dir = base / "run"
        logs_dir = base / "logs"
        stem = "gateway" if suffix is None else f"gateway.{suffix}"
        return cls(
            run_dir=run_dir,
            logs_dir=logs_dir,
            state_path=run_dir / f"{stem}.json",
            log_path=logs_dir / f"{stem}.log",
        )


@dataclass(frozen=True)
class GatewayInstance:
    """One stable local gateway identity and its child-process selectors."""

    config_path: Path
    workspace: str | None
    paths: GatewayRuntimePaths

    @classmethod
    def resolve(
        cls,
        *,
        config_path: str | Path,
        workspace: str | None = None,
    ) -> "GatewayInstance":
        resolved_config = Path(config_path).expanduser().resolve(strict=False)
        resolved_workspace = (
            str(Path(workspace).expanduser().resolve(strict=False)) if workspace else None
        )
        # The released default instance used gateway.json. Keep that identity stable
        # across upgrades while still namespacing explicit configs and workspaces.
        config_selector = (
            None if resolved_config == _default_config_path() else str(resolved_config)
        )
        return cls(
            config_path=resolved_config,
            workspace=resolved_workspace,
            paths=GatewayRuntimePaths.for_instance(
                data_dir=resolved_config.parent,
                workspace=resolved_workspace,
                config_path=config_selector,
            ),
        )

    def start_options(
        self,
        *,
        port: int,
        verbose: bool = False,
        log_file: str | None = None,
        desktop_managed: bool = False,
        desktop_listener_ports: tuple[int, ...] = (),
    ) -> GatewayStartOptions:
        return GatewayStartOptions(
            port=port,
            verbose=verbose,
            workspace=self.workspace,
            config_path=(
                None
                if not desktop_managed and self.config_path == _default_config_path()
                else str(self.config_path)
            ),
            log_file=log_file,
            desktop_managed=desktop_managed,
            desktop_listener_ports=desktop_listener_ports,
        )


class GatewayRuntime(ManagedProcessRuntime[ProcessStartOptions]):
    """Manage a background ``nanobot gateway`` process."""

    service_name = "gateway"

    def __init__(
        self,
        *,
        paths: GatewayRuntimePaths | None = None,
        platform_name: str | None = None,
        python_executable: str | None = None,
        popen: Callable[..., Any] = subprocess.Popen,
        subprocess_run: Callable[..., Any] = subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(
            paths=paths or GatewayRuntimePaths.for_instance(),
            platform_name=platform_name,
            python_executable=python_executable,
            popen=popen,
            subprocess_run=subprocess_run,
            sleep=sleep,
        )
        self._desktop_lease: GatewayClientLease | None = None

    def _build_child_command(self, options: ProcessStartOptions) -> list[str]:
        return build_gateway_command(self.python_executable, options)

    def _transition_lock(self) -> FileLock:
        """Serialize long lifecycle transitions without blocking child cleanup."""
        return FileLock(f"{self.paths.state_path}.transition.lock")

    def start_background(self, options: ProcessStartOptions) -> RuntimeResult:
        """Start the gateway detached from the current terminal."""
        lease = GatewayClientLease(self, kind="gateway-background")
        while True:
            lease.wait_for_shutdown()
            with self._transition_lock(), self._lifecycle_lock():
                promoted = lease._try_mark_persistent_locked()
                if promoted is None:
                    continue
                result = self._start_background(options)
                return RuntimeResult(result.ok, result.message, result.status, promoted)

    def start_desktop(
        self,
        options: GatewayStartOptions,
        *,
        preflight: Callable[[], None],
        lock_timeout_s: float = 2.0,
    ) -> DesktopAdmissionResult:
        """Atomically adopt or start a persistent gateway for the desktop adapter."""
        if not options.desktop_managed or not options.config_path:
            raise GatewayDesktopLifecycleError("config_unavailable")
        if self.platform_name != "Linux":
            return self._desktop_admission_failure("gateway_platform_unsupported")
        if not self._desktop_selected_ports_valid(options):
            return self._desktop_admission_failure("gateway_selector_conflict")

        lease = self._desktop_lease or GatewayClientLease(self, kind="desktop")
        deadline = time.monotonic() + max(lock_timeout_s, 0.0)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._desktop_admission_failure("gateway_busy")
            pending_shutdown = False
            try:
                with self._desktop_lock(self._transition_lock(), remaining):
                    with self._desktop_lock(self._lifecycle_lock(), remaining):
                        start_new = False
                        with self._desktop_lock(lease.lock, remaining):
                            clients = lease._live_state()
                            if clients.get("stopping"):
                                pending_shutdown = True
                            else:
                                preflight()
                                state = self._read_state()
                                if state is not None:
                                    identity: DesktopGatewayIdentity | None = None
                                    try:
                                        identity = self._classify_desktop_state(state, options)
                                    except GatewayDesktopLifecycleError as exc:
                                        stale_pid = _as_int(state.get("pid"))
                                        if (
                                            exc.code != "gateway_stopped"
                                            or stale_pid is None
                                            or stale_pid <= 0
                                            or self._is_pid_running(stale_pid)
                                        ):
                                            return self._desktop_admission_failure(exc.code)
                                        # Explicit start may recover a verified-absent
                                        # selected instance. Handle-based starts remain
                                        # reconcile-only and never reach this path.
                                        self._clear_state()
                                        state = None
                                    if state is not None:
                                        assert identity is not None
                                        if not self._desktop_identity_compatible(identity):
                                            owns_pending_start = (
                                                identity.starting
                                                and (
                                                    (
                                                        self._desktop_lease is lease
                                                        and lease._acquired
                                                    )
                                                    or self._has_desktop_owner_client(clients)
                                                )
                                            )
                                            if not owns_pending_start:
                                                return self._desktop_admission_failure(
                                                    "gateway_identity_unavailable"
                                                )
                                        promoted = bool(clients.get("auto_stop"))
                                        clients["auto_stop"] = False
                                        lease._register(clients)
                                        self._desktop_lease = lease
                                        return DesktopAdmissionResult(
                                            True,
                                            "gateway_desktop_adopted",
                                            self._desktop_status(state, clients),
                                            promoted,
                                            identity,
                                        )

                                if self._desktop_listener_conflict(options):
                                    return self._desktop_admission_failure(
                                        "gateway_listener_conflict"
                                    )
                                clients["auto_stop"] = False
                                lease._register(clients)
                                self._desktop_lease = lease
                                start_new = True
                        if start_new:
                            # _start_background() calls status(), which takes a
                            # fresh client lease lock. Release this lock after
                            # atomically registering the persistent client.
                            result = self._start_background(options)
                            if not result.ok:
                                with self._desktop_lock(lease.lock, remaining):
                                    clients = lease._live_state()
                                    lease._clients(clients).pop(lease.token, None)
                                    lease._acquired = False
                                    lease._write_or_clear(clients)
                                self._desktop_lease = None
                                return self._desktop_admission_failure(
                                    "gateway_start_failed",
                                    status=result.status,
                                )
                            state = self._read_state()
                            if state is None:
                                return self._desktop_admission_failure(
                                    "gateway_identity_unavailable",
                                    status=result.status,
                                )
                            try:
                                identity = self._classify_desktop_state(state, options)
                            except GatewayDesktopLifecycleError as exc:
                                return self._desktop_admission_failure(
                                    exc.code,
                                    status=result.status,
                                )
                            if not self._desktop_identity_compatible(identity) and not identity.starting:
                                return self._desktop_admission_failure(
                                    "gateway_identity_unavailable",
                                    status=result.status,
                                )
                            return DesktopAdmissionResult(
                                True,
                                "gateway_desktop_started",
                                result.status,
                                False,
                                identity,
                            )
            except GatewayDesktopLifecycleError as exc:
                return self._desktop_admission_failure(exc.code)
            if pending_shutdown:
                if time.monotonic() >= deadline:
                    return self._desktop_admission_failure("gateway_busy")
                self._sleep(min(0.05, max(deadline - time.monotonic(), 0.0)))

    def adopt_desktop_handle(
        self,
        options: GatewayStartOptions,
        handle: str,
        *,
        lock_timeout_s: float = 2.0,
    ) -> DesktopAdmissionResult:
        """Reconcile a captured desktop process and reacquire its persistent lease."""
        payload = self._decode_desktop_handle(handle, options)
        lease = self._desktop_lease or GatewayClientLease(self, kind="desktop")
        remaining = max(lock_timeout_s, 0.0)
        try:
            with self._desktop_lock(self._transition_lock(), remaining):
                with self._desktop_lock(self._lifecycle_lock(), remaining):
                    with self._desktop_lock(lease.lock, remaining):
                        state = self._read_state()
                        if state is None:
                            code = (
                                "gateway_identity_conflict"
                                if self._is_pid_running(payload.pid)
                                else "gateway_stopped"
                            )
                            return self._desktop_admission_failure(code)
                        try:
                            identity = self._classify_desktop_state(
                                state,
                                options,
                                expected=payload,
                            )
                        except GatewayDesktopLifecycleError as exc:
                            return self._desktop_admission_failure(exc.code)
                        if identity.handle != handle:
                            return self._desktop_admission_failure("gateway_identity_conflict")
                        if not self._desktop_identity_compatible(identity) and not identity.starting:
                            return self._desktop_admission_failure(
                                "gateway_identity_unavailable"
                            )
                        clients = lease._live_state()
                        if clients.get("stopping"):
                            return self._desktop_admission_failure("gateway_busy")
                        promoted = bool(clients.get("auto_stop"))
                        clients["auto_stop"] = False
                        lease._register(clients)
                        self._desktop_lease = lease
                        return DesktopAdmissionResult(
                            True,
                            "gateway_desktop_adopted",
                            self._desktop_status(state, clients),
                            promoted,
                            identity,
                        )
        except GatewayDesktopLifecycleError as exc:
            return self._desktop_admission_failure(exc.code)
        except FileLockTimeout:
            return self._desktop_admission_failure("gateway_busy")

    def desktop_identity(
        self,
        options: GatewayStartOptions,
        *,
        handle: str | None = None,
        lock_timeout_s: float = 2.0,
    ) -> DesktopGatewayIdentity | None:
        """Verify the selected managed process, returning None only when it is absent."""
        if not options.desktop_managed or not options.config_path:
            raise GatewayDesktopLifecycleError("config_unavailable")
        payload = self._decode_desktop_handle(handle, options) if handle is not None else None
        try:
            with self._desktop_lock(self._transition_lock(), lock_timeout_s):
                with self._desktop_lock(self._lifecycle_lock(), lock_timeout_s):
                    state = self._read_state()
                    if state is None:
                        if payload is None:
                            return None
                        if self._is_pid_running(payload.pid):
                            raise GatewayDesktopLifecycleError("gateway_identity_conflict")
                        return None

                    state_pid = _as_int(state.get("pid"))
                    if state_pid is None:
                        if payload is None:
                            raise GatewayDesktopLifecycleError("gateway_identity_conflict")
                        if self._is_pid_running(payload.pid):
                            raise GatewayDesktopLifecycleError("gateway_identity_conflict")
                        return None
                    if payload is not None:
                        if state_pid != payload.pid:
                            raise GatewayDesktopLifecycleError("gateway_identity_replaced")
                        if self._state_birth_identity(state) != payload.birth:
                            raise GatewayDesktopLifecycleError("gateway_identity_conflict")
                    if not self._selectors_match(state, options):
                        raise GatewayDesktopLifecycleError("gateway_selector_conflict")
                    self._verify_desktop_port_proof(state, options)
                    if not self._is_pid_running(state_pid):
                        return None
                    return self._classify_desktop_state(state, options, expected=payload)
        except FileLockTimeout as exc:
            raise GatewayDesktopLifecycleError("gateway_busy") from exc

    def release_desktop_lease(self) -> None:
        """Release this adapter process's lease while retaining persistent lifetime."""
        lease = self._desktop_lease
        if lease is None:
            return
        lease.release(wait_for_stop=False)
        self._desktop_lease = None

    def stop_desktop(
        self,
        options: GatewayStartOptions,
        handle: str,
        *,
        timeout_s: float = 20.0,
        lock_timeout_s: float = 1.0,
    ) -> RuntimeResult:
        """Gracefully stop only the captured gateway and wait for its actual exit."""
        if self.platform_name != "Linux":
            return self._desktop_failure("gateway_platform_unsupported")
        deadline = time.monotonic() + max(timeout_s, 0.0)
        pidfd: int | None = None
        try:
            payload = self._decode_desktop_handle(handle, options)
            with self._desktop_lock(self._transition_lock(), lock_timeout_s):
                with self._desktop_lock(self._lifecycle_lock(), lock_timeout_s):
                    state = self._read_state()
                    if state is None:
                        if self._is_pid_running(payload.pid):
                            return self._desktop_failure("gateway_identity_conflict")
                        return self._desktop_stopped()
                    state_pid = _as_int(state.get("pid"))
                    if state_pid != payload.pid:
                        return self._desktop_failure("gateway_identity_replaced")
                    if self._state_birth_identity(state) != payload.birth:
                        return self._desktop_failure("gateway_identity_conflict")
                    if not self._selectors_match(state, options):
                        return self._desktop_failure("gateway_selector_conflict")
                    try:
                        self._verify_desktop_port_proof(state, options)
                    except GatewayDesktopLifecycleError as exc:
                        return self._desktop_failure(exc.code)
                    recorded_command = state.get("command")
                    if not isinstance(recorded_command, list):
                        return self._desktop_failure("gateway_identity_conflict")
                    command_values = cast(list[object], recorded_command)
                    if not all(isinstance(item, str) for item in command_values):
                        return self._desktop_failure("gateway_identity_conflict")
                    if cast(list[str], command_values) not in self._desktop_expected_commands(options):
                        return self._desktop_failure("gateway_identity_conflict")
                    if not self._is_pid_running(payload.pid):
                        self._clear_state()
                        return self._desktop_stopped()
                    identity = self._classify_desktop_state(state, options, expected=payload)
                    if not self._desktop_identity_compatible(identity) and not identity.starting:
                        return self._desktop_failure("gateway_identity_unavailable")
                    try:
                        pidfd = self._open_pidfd(identity.pid)
                    except ProcessLookupError:
                        if self._is_pid_running(identity.pid):
                            return self._desktop_failure("gateway_identity_conflict")
                        self._clear_desktop_state_if_current(state, identity)
                        return self._desktop_stopped()
                    except OSError:
                        return self._desktop_failure("gateway_stop_unavailable")
                    if not self._verify_desktop_signal_target(identity, options):
                        return self._desktop_failure("gateway_identity_conflict")
                    try:
                        self._pidfd_send_signal(pidfd, signal.SIGTERM)
                    except OSError:
                        return self._desktop_failure("gateway_signal_failed")

                if not self._wait_for_pidfd_exit(
                    pidfd, max(deadline - time.monotonic(), 0.0),
                ):
                    return self._desktop_failure("gateway_stop_timeout")

                with self._desktop_lock(self._lifecycle_lock(), lock_timeout_s):
                    state = self._read_state()
                    if state is not None:
                        state_pid = _as_int(state.get("pid"))
                        state_identity = self._state_birth_identity(state)
                        if state_pid != payload.pid or state_identity != payload.birth:
                            return self._desktop_failure("gateway_identity_replaced")
                        if not self._selectors_match(state, options):
                            return self._desktop_failure("gateway_selector_conflict")
                        try:
                            self._verify_desktop_port_proof(state, options)
                        except GatewayDesktopLifecycleError as exc:
                            return self._desktop_failure(exc.code)
                        self._clear_state()
                return self._desktop_stopped()
        except GatewayDesktopLifecycleError as exc:
            return self._desktop_failure(exc.code)
        except FileLockTimeout:
            return self._desktop_failure("gateway_busy")
        finally:
            if pidfd is not None:
                os.close(pidfd)

    @staticmethod
    @contextmanager
    def _desktop_lock(lock: FileLock, timeout_s: float) -> Generator[None]:
        try:
            lock.acquire(timeout=max(timeout_s, 0.0))
        except FileLockTimeout as exc:
            raise GatewayDesktopLifecycleError("gateway_busy") from exc
        try:
            yield
        finally:
            lock.release()

    def _classify_desktop_state(
        self,
        state: dict[str, Any],
        options: GatewayStartOptions,
        *,
        expected: _DesktopHandlePayload | None = None,
    ) -> DesktopGatewayIdentity:
        pid = _as_int(state.get("pid"))
        if pid is None or pid <= 0 or state.get("launch_mode") != "background":
            raise GatewayDesktopLifecycleError("gateway_identity_conflict")
        if not self._selectors_match(state, options):
            raise GatewayDesktopLifecycleError("gateway_selector_conflict")
        recorded_command = state.get("command")
        if (
            not isinstance(recorded_command, list)
            or not recorded_command
        ):
            raise GatewayDesktopLifecycleError("gateway_identity_conflict")
        command_values = cast(list[object], recorded_command)
        if not all(isinstance(value, str) for value in command_values):
            raise GatewayDesktopLifecycleError("gateway_identity_conflict")
        command = cast(list[str], command_values)
        if command not in self._desktop_expected_commands(options):
            raise GatewayDesktopLifecycleError("gateway_identity_conflict")
        if not self._verify_desktop_process(pid, command):
            if not self._is_pid_running(pid):
                raise GatewayDesktopLifecycleError("gateway_stopped")
            raise GatewayDesktopLifecycleError("gateway_identity_conflict")
        actual_ports = self._verify_desktop_port_proof(state, options)

        birth_identity = self._state_birth_identity(state)
        current_identity = self.process_identity(pid)
        if (
            birth_identity is None
            or not isinstance(current_identity, str)
            or current_identity != birth_identity
        ):
            raise GatewayDesktopLifecycleError("gateway_identity_conflict")
        try:
            process_group = os.getpgid(pid)
        except OSError as exc:
            raise GatewayDesktopLifecycleError("gateway_identity_conflict") from exc
        if expected is not None and (
            expected.pid != pid
            or expected.birth != birth_identity
            or expected.group != process_group
        ):
            raise GatewayDesktopLifecycleError("gateway_identity_conflict")

        selector_hash = self._desktop_selector_hash(options)
        if expected is not None and expected.selectors != selector_hash:
            raise GatewayDesktopLifecycleError("gateway_selector_conflict")
        handle = self._make_desktop_handle(
            pid,
            birth_identity,
            process_group,
            selector_hash,
        )
        gateway_id = state.get("desktop_gateway_id")
        backend_version = state.get("desktop_backend_version")
        capability = state.get("desktop_capability")
        python_executable = state.get("desktop_python_executable")
        owner_values = (gateway_id, backend_version, capability, python_executable)
        if actual_ports and not (
            isinstance(gateway_id, str)
            and bool(gateway_id)
            and isinstance(backend_version, str)
            and bool(backend_version)
            and isinstance(capability, str)
            and bool(capability)
            and isinstance(python_executable, str)
            and bool(python_executable)
        ):
            raise GatewayDesktopLifecycleError("gateway_identity_unavailable")
        if not actual_ports and any(value is not None for value in owner_values):
            raise GatewayDesktopLifecycleError("gateway_identity_unavailable")
        starting = (
            not actual_ports
            and all(value is None for value in owner_values)
            and command == build_gateway_command(self.python_executable, options)
        )
        return DesktopGatewayIdentity(
            pid=pid,
            birth_identity=birth_identity,
            process_group=process_group,
            handle=handle,
            gateway_id=gateway_id if isinstance(gateway_id, str) else None,
            backend_version=backend_version if isinstance(backend_version, str) else None,
            capability=capability if isinstance(capability, str) else None,
            python_executable=(
                python_executable if isinstance(python_executable, str) else None
            ),
            starting=starting,
        )

    def _selectors_match(self, state: dict[str, Any], options: GatewayStartOptions) -> bool:
        if (
            not _is_valid_desktop_port(options.port)
            or type(state.get("port")) is not int
            or state.get("port") != options.port
            or not self._desktop_selected_ports_valid(options)
        ):
            return False
        selected_config = Path(options.config_path or "").expanduser().resolve(strict=False)
        stored_config = state.get("config_path")
        if stored_config is None:
            state_config = _default_config_path()
        elif isinstance(stored_config, str) and stored_config:
            state_config = Path(stored_config).expanduser().resolve(strict=False)
        else:
            return False
        if state_config != selected_config:
            return False

        stored_workspace = state.get("workspace")
        if stored_workspace is None:
            canonical_workspace = None
        elif isinstance(stored_workspace, str) and stored_workspace:
            canonical_workspace = str(Path(stored_workspace).expanduser().resolve(strict=False))
        else:
            return False
        return canonical_workspace == options.workspace

    @staticmethod
    def _desktop_selected_ports_valid(options: GatewayStartOptions) -> bool:
        ports = options.desktop_listener_ports
        return (
            type(ports) is tuple
            and bool(ports)
            and all(_is_valid_desktop_port(port) for port in ports)
            and len(set(ports)) == len(ports)
        )

    def _verify_desktop_port_proof(
        self,
        state: dict[str, Any],
        options: GatewayStartOptions,
    ) -> bool:
        """Require actual owner ports, or exact admission-only ports while starting."""
        if not self._desktop_selected_ports_valid(options):
            raise GatewayDesktopLifecycleError("gateway_selector_conflict")
        if _DESKTOP_OWNER_MANAGEMENT_PORT in state or _DESKTOP_OWNER_LISTENER_PORTS in state:
            management_port = state.get(_DESKTOP_OWNER_MANAGEMENT_PORT)
            listener_ports = state.get(_DESKTOP_OWNER_LISTENER_PORTS)
            if not _is_valid_desktop_port(management_port) or not isinstance(
                listener_ports, list
            ):
                raise GatewayDesktopLifecycleError("gateway_selector_conflict")
            listener_port_values = cast(list[object], listener_ports)
            if not all(_is_valid_desktop_port(port) for port in listener_port_values):
                raise GatewayDesktopLifecycleError("gateway_selector_conflict")
            if (
                management_port != options.port
                or tuple(listener_port_values) != options.desktop_listener_ports
            ):
                raise GatewayDesktopLifecycleError("gateway_selector_conflict")
            return True

        provisional_ports = state.get(_DESKTOP_ADMISSION_LISTENER_PORTS)
        if (
            isinstance(provisional_ports, list)
            and all(
                _is_valid_desktop_port(port)
                for port in cast(list[object], provisional_ports)
            )
            and tuple(cast(list[object], provisional_ports)) == options.desktop_listener_ports
        ):
            return False
        if _DESKTOP_ADMISSION_LISTENER_PORTS in state:
            raise GatewayDesktopLifecycleError("gateway_selector_conflict")
        raise GatewayDesktopLifecycleError("gateway_identity_unavailable")

    def _desktop_expected_commands(self, options: GatewayStartOptions) -> list[list[str]]:
        commands = [build_gateway_command(self.python_executable, options)]
        selected_config = Path(options.config_path or "").expanduser().resolve(strict=False)
        regular_config = (
            None if selected_config == _default_config_path() else str(selected_config)
        )
        for verbose in (False, True):
            regular = GatewayStartOptions(
                port=options.port,
                verbose=verbose,
                workspace=options.workspace,
                config_path=regular_config,
            )
            commands.append(build_gateway_command(self.python_executable, regular))
        return commands

    def _verify_desktop_process(self, pid: int, expected_command: list[str]) -> bool:
        if not self._is_pid_running(pid):
            return False
        if self.platform_name != "Linux":
            return False
        proc = Path(f"/proc/{pid}")
        try:
            if proc.stat().st_uid != os.geteuid():
                return False
            raw_command = (proc / "cmdline").read_bytes()
            actual_command = [os.fsdecode(part) for part in raw_command.split(b"\0") if part]
            executable = proc / "exe"
            if not os.path.samefile(executable, expected_command[0]):
                return False
            ordinary_gateway = (
                len(expected_command) >= 5
                and expected_command[1:4] == ["-m", "nanobot", "gateway"]
                and expected_command[4] == "--foreground"
            )
            title = (proc / "comm").read_text(encoding="ascii").strip()
            actual_matches = actual_command == expected_command or (
                ordinary_gateway
                and actual_command == ["nanobot-gateway"]
                and title == "nanobot-gateway"
            )
            process_group = os.getpgid(pid)
            session_id = os.getsid(pid)
        except (OSError, ValueError):
            return False
        return (
            actual_matches
            and os.path.samefile(executable, self.python_executable)
            and process_group == pid
            and session_id == pid
        )

    def _verify_desktop_signal_target(
        self,
        identity: DesktopGatewayIdentity,
        options: GatewayStartOptions,
    ) -> bool:
        state = self._read_state()
        if state is None:
            return False
        try:
            current = self._classify_desktop_state(
                state,
                options,
                expected=self._decode_desktop_handle(identity.handle, options),
            )
        except GatewayDesktopLifecycleError:
            return False
        return current.handle == identity.handle

    def _state_birth_identity(self, state: dict[str, Any]) -> str | None:
        stable = state.get("stable_identity")
        value = stable if stable is not None else state.get("identity")
        return value if isinstance(value, str) and value else None

    def _desktop_selector_hash(self, options: GatewayStartOptions) -> str:
        selectors = {
            "config": str(Path(options.config_path or "").expanduser().resolve(strict=False)),
            "workspace": options.workspace,
            "port": options.port,
            "listener_ports": list(options.desktop_listener_ports),
            "python": self.python_executable,
            "state": str(self.paths.state_path.resolve(strict=False)),
        }
        raw = json.dumps(selectors, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _make_desktop_handle(
        pid: int,
        birth_identity: str,
        process_group: int,
        selectors: str,
    ) -> str:
        basis: dict[str, object] = {
            "v": _DESKTOP_HANDLE_VERSION,
            "pid": pid,
            "birth": birth_identity,
            "group": process_group,
            "selectors": selectors,
        }
        fingerprint = hashlib.sha256(
            json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        encoded = json.dumps(
            {**basis, "fingerprint": fingerprint},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")

    def _decode_desktop_handle(
        self,
        handle: str | None,
        options: GatewayStartOptions,
    ) -> _DesktopHandlePayload:
        if not isinstance(handle, str) or not handle or len(handle) > 2048:
            raise GatewayDesktopLifecycleError("gateway_handle_invalid")
        try:
            padded = handle + "=" * (-len(handle) % 4)
            raw = base64.b64decode(padded, altchars=b"-_", validate=True)
            value: object = json.loads(raw.decode("utf-8"))
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise GatewayDesktopLifecycleError("gateway_handle_invalid") from None
        if not isinstance(value, dict):
            raise GatewayDesktopLifecycleError("gateway_handle_invalid")
        payload = cast(dict[str, object], value)
        if set(payload) != {"v", "pid", "birth", "group", "selectors", "fingerprint"}:
            raise GatewayDesktopLifecycleError("gateway_handle_invalid")
        pid = payload.get("pid")
        birth = payload.get("birth")
        group = payload.get("group")
        selectors = payload.get("selectors")
        fingerprint = payload.get("fingerprint")
        if (
            type(payload.get("v")) is not int
            or payload.get("v") != _DESKTOP_HANDLE_VERSION
            or type(pid) is not int
            or pid <= 0
            or not isinstance(birth, str)
            or not birth
            or type(group) is not int
            or group <= 0
            or not isinstance(selectors, str)
            or len(selectors) != 64
            or not isinstance(fingerprint, str)
            or len(fingerprint) != 64
        ):
            raise GatewayDesktopLifecycleError("gateway_handle_invalid")
        basis = {
            "v": _DESKTOP_HANDLE_VERSION,
            "pid": pid,
            "birth": birth,
            "group": group,
            "selectors": selectors,
        }
        expected_fingerprint = hashlib.sha256(
            json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if fingerprint != expected_fingerprint:
            raise GatewayDesktopLifecycleError("gateway_handle_invalid")
        if selectors != self._desktop_selector_hash(options):
            raise GatewayDesktopLifecycleError("gateway_selector_conflict")
        return _DesktopHandlePayload(
            pid=pid,
            birth=birth,
            group=group,
            selectors=selectors,
        )

    def _desktop_identity_compatible(self, identity: DesktopGatewayIdentity) -> bool:
        from nanobot import __version__

        return (
            bool(identity.gateway_id)
            and identity.backend_version == __version__
            and identity.capability == _DESKTOP_CAPABILITY
            and identity.python_executable == self.python_executable
        )

    @staticmethod
    def _desktop_listener_conflict(options: GatewayStartOptions) -> bool:
        hosts = ["127.0.0.1", "::1"]
        for port in set((options.port, *options.desktop_listener_ports)):
            if type(port) is not int or not 1 <= port <= 65535:
                return True
            for host in hosts:
                try:
                    with socket.create_connection((host, port), timeout=0.1):
                        return True
                except OSError:
                    continue
        return False

    @staticmethod
    def _has_desktop_owner_client(clients: dict[str, object]) -> bool:
        records = clients.get("clients")
        if not isinstance(records, dict):
            return False
        for record in cast(dict[str, object], records).values():
            if isinstance(record, dict) and cast(dict[str, object], record).get("kind") == "desktop":
                return True
        return False

    def _desktop_status(
        self,
        state: dict[str, Any],
        clients: dict[str, object],
    ) -> GatewayStatus:
        pid = _as_int(state.get("pid"))
        command = state.get("command")
        command_tuple = (
            tuple(value for value in cast(list[object], command) if isinstance(value, str))
            if isinstance(command, list)
            else ()
        )
        raw_mode = state.get("launch_mode")
        launch_mode: GatewayLaunchMode = (
            raw_mode if raw_mode in {"foreground", "background"} else "unknown"
        )
        client_records = clients.get("clients")
        client_count = len(cast(dict[str, object], client_records)) if isinstance(
            client_records, dict
        ) else 0
        return GatewayStatus(
            running=pid is not None,
            pid=pid,
            state_path=self.paths.state_path,
            log_path=self.paths.log_path,
            started_at=_as_str(state.get("started_at")),
            port=_as_int(state.get("port")),
            command=command_tuple,
            reason="running" if pid is not None else "not_started",
            launch_mode=launch_mode,
            lifetime="on_demand" if clients.get("auto_stop") is True else "explicit",
            clients=client_count,
            ready=None,
        )

    def _desktop_failure(
        self,
        code: str,
        *,
        status: ProcessStatus | None = None,
    ) -> RuntimeResult:
        gateway_status = status if isinstance(status, GatewayStatus) else GatewayStatus(
            running=False,
            pid=None,
            state_path=self.paths.state_path,
            log_path=self.paths.log_path,
            reason=code,
        )
        return RuntimeResult(False, code, gateway_status)

    def _desktop_admission_failure(
        self,
        code: str,
        *,
        status: ProcessStatus | None = None,
    ) -> DesktopAdmissionResult:
        result = self._desktop_failure(code, status=status)
        return DesktopAdmissionResult(
            result.ok,
            result.message,
            result.status,
            result.promoted,
            None,
        )

    def _desktop_stopped(self) -> RuntimeResult:
        lease = GatewayClientLease(self, kind="desktop-stop")
        try:
            with self._desktop_lock(lease.lock, 1.0):
                lease._clear_locked()
        except GatewayDesktopLifecycleError:
            return self._desktop_failure("gateway_busy")
        if self._desktop_lease is not None:
            self._desktop_lease._acquired = False
            self._desktop_lease = None
        return RuntimeResult(
            True,
            "gateway_stopped",
            GatewayStatus(
                running=False,
                pid=None,
                state_path=self.paths.state_path,
                log_path=self.paths.log_path,
                reason="stopped",
            ),
        )

    def _clear_desktop_state_if_current(
        self,
        state: dict[str, Any],
        identity: DesktopGatewayIdentity,
    ) -> None:
        current = self._read_state()
        if (
            current is not None
            and _as_int(current.get("pid")) == identity.pid
            and self._state_birth_identity(current) == identity.birth_identity
            and current.get("command") == state.get("command")
        ):
            self._clear_state()

    def start_on_demand(self, options: ProcessStartOptions) -> RuntimeResult:
        """Atomically reuse a gateway or start one owned by local client leases."""
        lease = GatewayClientLease(self, kind="gateway-start")
        while True:
            lease.wait_for_shutdown()
            with self._transition_lock(), self._lifecycle_lock():
                if lease._shutdown_pending_locked():
                    continue
                status = self.status()
                if status.running:
                    return RuntimeResult(False, "gateway_already_running", status)
                lease._mark_ephemeral_locked()
                return self._start_background(options)

    def _start_background(self, options: ProcessStartOptions) -> RuntimeResult:
        result = super()._start_background(options)
        if not result.ok:
            return self._result(result)
        state = self._read_state()
        if state and result.status.pid == state.get("pid"):
            state["launch_mode"] = "background"
            state["pending_pid_handoff"] = True
            if (
                isinstance(options, GatewayStartOptions)
                and options.desktop_managed
                and self._desktop_selected_ports_valid(options)
            ):
                state[_DESKTOP_ADMISSION_LISTENER_PORTS] = list(
                    options.desktop_listener_ports
                )
            self._write_state(state)
        return RuntimeResult(True, result.message, self.status())

    def stop(self, *, timeout_s: int = 20) -> RuntimeResult:
        """Stop the gateway recorded by this runtime."""
        with self._transition_lock():
            result = self._stop(timeout_s=timeout_s)
            with self._lifecycle_lock():
                if result.ok or result.message in {
                    "gateway_not_running",
                    "gateway_state_stale",
                }:
                    GatewayClientLease(self, kind="gateway-stop")._clear_locked()
            return self._result(result)

    def status(self, *, reason: str | None = None) -> GatewayStatus:
        """Return process, launch, and client lifetime state in one snapshot."""
        process = super().status(reason=reason)
        state = self._read_state() if process.running else None
        raw_mode = state.get("launch_mode") if state else None
        launch_mode: GatewayLaunchMode = (
            raw_mode if raw_mode in {"foreground", "background"} else "unknown"
        )
        lease = GatewayClientLease(self, kind="gateway-status").snapshot()
        ready: bool | None = None
        health_host = state.get("health_host") if state else None
        if (
            process.running
            and process.pid != os.getpid()
            and isinstance(health_host, str)
            and process.port is not None
        ):
            ready = _gateway_health_ready(health_host, process.port)
        status_reason = process.reason
        if ready is False and reason is None and status_reason == "running":
            status_reason = "websocket_unavailable"
        return GatewayStatus(
            running=process.running,
            pid=process.pid,
            state_path=process.state_path,
            log_path=process.log_path,
            started_at=process.started_at,
            port=process.port,
            command=process.command,
            reason=status_reason,
            launch_mode=launch_mode,
            lifetime="on_demand" if lease.auto_stop else "explicit",
            clients=lease.clients,
            ready=ready,
        )

    def publish_health_host(self, host: str) -> None:
        """Record the management bind host for out-of-process readiness diagnostics."""
        with self._lifecycle_lock():
            state = self._read_state()
            if not state or not self._record_matches_process(state, os.getpid()):
                return
            state["health_host"] = host
            self._write_state(state)

    def publish_desktop_gateway_identity(
        self,
        gateway_id: str,
        backend_version: str,
        capability: str,
        python_executable: str,
        management_port: int,
        websocket_ports: tuple[int, ...],
    ) -> None:
        """Publish non-secret WebSocket identity for independently verified adoption."""
        if (
            not gateway_id
            or len(gateway_id) > 128
            or not backend_version
            or len(backend_version) > 64
            or capability != _DESKTOP_CAPABILITY
            or not Path(python_executable).is_absolute()
            or len(python_executable) > 4096
            or not _is_valid_desktop_port(management_port)
            or type(websocket_ports) is not tuple
            or not websocket_ports
            or not all(_is_valid_desktop_port(port) for port in websocket_ports)
            or len(set(websocket_ports)) != len(websocket_ports)
        ):
            return
        with self._lifecycle_lock():
            state = self._read_state()
            if (
                not state
                or not self._record_matches_process(state, os.getpid())
                or type(state.get("port")) is not int
                or state.get("port") != management_port
            ):
                return
            state["desktop_gateway_id"] = gateway_id
            state["desktop_backend_version"] = backend_version
            state["desktop_capability"] = capability
            state["desktop_python_executable"] = python_executable
            state[_DESKTOP_OWNER_MANAGEMENT_PORT] = management_port
            state[_DESKTOP_OWNER_LISTENER_PORTS] = list(websocket_ports)
            state.pop(_DESKTOP_ADMISSION_LISTENER_PORTS, None)
            self._write_state(state)

    @contextmanager
    def foreground_instance(self, options: ProcessStartOptions) -> Generator[None]:
        """Publish this foreground gateway while it is available to local clients."""
        self._claim_current_process(options)
        try:
            yield
        finally:
            self._release_current_process()

    def _claim_current_process(self, options: ProcessStartOptions) -> GatewayLaunchMode:
        lease = GatewayClientLease(self, kind="gateway-foreground")
        pid = os.getpid()
        while True:
            lease.wait_for_shutdown()
            with self._transition_lock(), self._lifecycle_lock():
                current = self.status()
                state = self._read_state() or {}
                pid_handoff = (
                    self.platform_name == "Windows"
                    and current.running
                    and current.pid != pid
                    and current.pid == os.getppid()
                    and state.get("pid") == current.pid
                    and state.get("launch_mode") == "background"
                    and state.get("pending_pid_handoff") is True
                )
                if current.running and current.pid != pid and not pid_handoff:
                    raise GatewayAlreadyRunningError(current)
                if lease._shutdown_pending_locked():
                    continue
                launch_mode: GatewayLaunchMode = (
                    "background"
                    if state.get("launch_mode") == "background"
                    and (state.get("pid") == pid or pid_handoff)
                    else "foreground"
                )
                state.update(
                    {
                        "pid": pid,
                        "started_at": datetime.now(UTC).isoformat(),
                        "platform": self.platform_name,
                        "port": options.port,
                        "workspace": options.workspace,
                        "config_path": options.config_path,
                        "command": self._build_child_command(options),
                        "log_path": str(self.paths.log_path),
                        "launch_mode": launch_mode,
                    }
                )
                state.pop("pending_pid_handoff", None)
                state.pop("stable_identity", None)
                state.update(self.process_identity_record(pid))
                self._write_state(state)
                if launch_mode == "foreground":
                    lease._try_mark_persistent_locked()
                return launch_mode

    def _release_current_process(self) -> None:
        with self._lifecycle_lock():
            state = self._read_state()
            if state and self._record_matches_process(state, os.getpid()):
                self._clear_state()
            GatewayClientLease(
                self,
                kind="gateway-exit",
            )._finish_shutdown_locked()

    def restart(self, options: ProcessStartOptions, *, timeout_s: int = 20) -> RuntimeResult:
        """Restart an existing gateway without creating a new persistent instance."""
        with self._transition_lock():
            with self._lifecycle_lock():
                status = self.status()
                if not status.running:
                    return RuntimeResult(False, "gateway_not_running", status)
                if status.launch_mode == "foreground":
                    return RuntimeResult(
                        False,
                        "gateway_foreground_restart_required",
                        status,
                    )
            stop_result = self._stop(timeout_s=timeout_s)
            if not stop_result.ok:
                return self._result(stop_result)
            with self._lifecycle_lock():
                return self._start_background(options)

    def _result(self, result: ProcessResult) -> RuntimeResult:
        status = result.status
        gateway_status = status if isinstance(status, GatewayStatus) else self.status()
        return RuntimeResult(result.ok, result.message, gateway_status)


class GatewayClientLease:
    """Reference-count an on-demand gateway shared by local interactive clients."""

    def __init__(
        self,
        runtime: GatewayRuntime,
        *,
        kind: str,
        pid: int | None = None,
        token: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.kind = kind
        self.pid = pid or os.getpid()
        self.token = token or uuid.uuid4().hex
        state_path = runtime.paths.state_path
        self.state_path = state_path.with_name(
            f"{state_path.stem}.clients{state_path.suffix}"
        )
        self.transition_lock = FileLock(f"{state_path}.transition.lock")
        self.lifecycle_lock = FileLock(f"{state_path}.lock")
        self.lock = FileLock(f"{self.state_path}.lock")
        self._acquired = False

    def acquire(self) -> None:
        """Register this client before it starts or attaches to the gateway."""
        while True:
            self.wait_for_shutdown()
            with self.transition_lock, self.lifecycle_lock, self.lock:
                state = self._live_state()
                if state.get("stopping"):
                    continue
                self._register(state)
                return

    def ensure_on_demand_gateway(self, options: ProcessStartOptions) -> RuntimeResult:
        """Atomically reuse a gateway or start one owned by local client leases."""
        if not self._acquired:
            raise RuntimeError("gateway client lease must be acquired before startup")
        return self.runtime.start_on_demand(options)

    def mark_ephemeral(self) -> None:
        """Mark a gateway started by a client for last-client shutdown."""
        with self.transition_lock, self.lifecycle_lock:
            self._mark_ephemeral_locked()

    def _mark_ephemeral_locked(self) -> None:
        with self.lock:
            state = self._live_state()
            state["auto_stop"] = True
            self._write_state(state)

    def mark_persistent(self) -> bool:
        """Keep an explicitly backgrounded gateway alive; return whether it was promoted."""
        while True:
            self.wait_for_shutdown()
            with self.transition_lock, self.lifecycle_lock:
                promoted = self._try_mark_persistent_locked()
                if promoted is not None:
                    return promoted

    def _try_mark_persistent_locked(self) -> bool | None:
        with self.lock:
            state = self._live_state()
            if state.get("stopping"):
                return None
            promoted = bool(state.get("auto_stop"))
            state["auto_stop"] = False
            self._write_or_clear(state)
            return promoted

    def clear(self) -> None:
        """Forget leases after an explicit gateway stop."""
        with self.transition_lock, self.lifecycle_lock:
            self._clear_locked()

    def _clear_locked(self) -> None:
        with self.lock:
            self.state_path.unlink(missing_ok=True)

    def snapshot(self) -> GatewayLeaseSnapshot:
        """Prune dead clients and return current lifetime state."""
        with self.lock:
            state = self._live_state()
            self._write_or_clear(state)
            return GatewayLeaseSnapshot(
                auto_stop=bool(state.get("auto_stop")),
                clients=len(self._clients(state)),
            )

    def begin_orphan_shutdown(self) -> bool:
        """Commit shutdown only while an on-demand gateway still has no clients."""
        with self.transition_lock, self.lifecycle_lock, self.lock:
            state = self._live_state()
            if not bool(state.get("auto_stop")) or self._clients(state):
                self._write_or_clear(state)
                return False
            state["stopping"] = True
            self._write_state(state)
            return True

    def release(self, *, timeout_s: int = 20, wait_for_stop: bool = True) -> bool:
        """Release this client, optionally leaving last-client shutdown to the monitor."""
        if not self._acquired:
            return False
        while True:
            self.wait_for_shutdown()
            with self.transition_lock:
                with self.lifecycle_lock, self.lock:
                    state = self._live_state()
                    if state.get("stopping"):
                        continue
                    clients = self._clients(state)
                    clients.pop(self.token, None)
                    self._acquired = False
                    should_stop = not clients and bool(state.get("auto_stop"))
                    self._write_or_clear(state)
                if not should_stop or not wait_for_stop:
                    return False
                result = self.runtime._stop(timeout_s=timeout_s)
                stopped = result.ok or result.message in {
                    "gateway_not_running",
                    "gateway_state_stale",
                }
                with self.lifecycle_lock:
                    if stopped:
                        self._clear_locked()
                    else:
                        self._mark_ephemeral_locked()
                return stopped

    def wait_for_shutdown(self, *, timeout_s: float = 20) -> None:
        """Wait until a committed orphan shutdown can no longer accept clients."""
        deadline = time.monotonic() + timeout_s
        while True:
            with self.lifecycle_lock:
                with self.lock:
                    state = self._live_state()
                    if not state.get("stopping"):
                        return
                if not self.runtime.status().running:
                    self._finish_shutdown_locked()
                    return
            if time.monotonic() >= deadline:
                raise RuntimeError("gateway is still shutting down; try again shortly")
            time.sleep(0.05)

    def _shutdown_pending_locked(self) -> bool:
        with self.lock:
            return bool(self._live_state().get("stopping"))

    def _finish_shutdown_locked(self) -> None:
        with self.lock:
            state = self._live_state()
            state.pop("stopping", None)
            if not self._clients(state):
                self.state_path.unlink(missing_ok=True)
            else:
                self._write_state(state)

    def _register(self, state: dict[str, object]) -> None:
        clients = self._clients(state)
        record: dict[str, object] = {
            "pid": self.pid,
            "kind": self.kind,
        }
        record.update(process_identity_record(self._process_identity(self.pid), lease=True))
        clients[self.token] = record
        self._write_state(state)
        self._acquired = True

    def _live_state(self) -> dict[str, object]:
        state = self._read_state()
        clients = self._clients(state)
        stale: list[str] = []
        for token, value in clients.items():
            if not isinstance(value, dict):
                stale.append(token)
                continue
            record = cast(dict[str, object], value)
            pid = record.get("pid")
            identity = record.get("stable_identity")
            if identity is None:
                identity = record.get("identity")
            if not isinstance(pid, int) or not self._process_is_running(pid):
                stale.append(token)
                continue
            if self._process_identity_match(identity, pid) == "mismatch":
                stale.append(token)
        for token in stale:
            clients.pop(token, None)
        return state

    def _process_identity(self, pid: int) -> str | int | None:
        resolver = getattr(self.runtime, "process_identity", None)
        value = resolver(pid) if callable(resolver) else None
        return value if isinstance(value, (str, int)) else None

    def _process_identity_match(
        self,
        recorded: object,
        pid: int,
    ) -> Literal["match", "mismatch", "unknown"]:
        matcher = getattr(self.runtime, "process_identity_match", None)
        if callable(matcher):
            result = matcher(recorded, pid)
            if result in {"match", "mismatch", "unknown"}:
                return cast(Literal["match", "mismatch", "unknown"], result)
        if recorded is None:
            return "match"
        current = self._process_identity(pid)
        if current is None:
            return "unknown"
        return "match" if recorded == current else "mismatch"

    def _process_is_running(self, pid: int) -> bool:
        checker = getattr(self.runtime, "process_is_running", None)
        return bool(checker(pid)) if callable(checker) else process_is_running(pid)

    @staticmethod
    def _clients(state: dict[str, object]) -> dict[str, object]:
        value = state.get("clients")
        if isinstance(value, dict):
            return cast(dict[str, object], value)
        clients: dict[str, object] = {}
        state["clients"] = clients
        return clients

    def _read_state(self) -> dict[str, object]:
        try:
            payload: object = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return {"auto_stop": False, "clients": {}}
        if isinstance(payload, dict):
            return cast(dict[str, object], payload)
        return {"auto_stop": False, "clients": {}}

    def _write_or_clear(self, state: dict[str, object]) -> None:
        clients = state.get("clients")
        if not clients and not bool(state.get("auto_stop")):
            self.state_path.unlink(missing_ok=True)
            return
        self._write_state(state)

    def _write_state(self, state: dict[str, object]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f"{self.state_path.name}.",
            suffix=".tmp",
            dir=self.state_path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.state_path)
        finally:
            temporary.unlink(missing_ok=True)


async def monitor_gateway_clients(
    lease: GatewayClientLease,
    shutdown_event: asyncio.Event,
    *,
    poll_interval_s: float = 1.0,
) -> bool:
    """Stop waiting when an on-demand gateway loses every live client."""
    while not shutdown_event.is_set():
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=poll_interval_s)
        except TimeoutError:
            if lease.begin_orphan_shutdown():
                shutdown_event.set()
                return True
    return False


def _instance_suffix(*, workspace: str | None, config_path: str | None) -> str | None:
    raw = "|".join(value for value in (workspace, config_path) if value)
    if not raw:
        return None
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
