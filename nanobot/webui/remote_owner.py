"""Run the remote owner with the gateway when no independent owner is ready."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import suppress
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.remote.client import OwnerRequestError, OwnerUnavailableError, RemoteAccessClient
from nanobot.remote.composition import owner_socket_path, remote_access_root
from nanobot.remote.store import RemoteAccessStore, StoreError

if TYPE_CHECKING:
    from nanobot.config.schema import RemoteAccessConfig
    from nanobot.webui.settings_services import WebUISettingsConfig

_START_TIMEOUT_S = 5.0
_RETRY_INTERVAL_S = 5.0


class GatewayRemoteOwner:
    """Keep one remote owner available for this instance while remote access is enabled."""

    def __init__(self, config: RemoteAccessConfig, settings: WebUISettingsConfig) -> None:
        self._settings = settings
        self._enabled = config.enabled
        self._running = False
        self._listener_port = config.listener_port
        self._state_dir = remote_access_root(config, instance_root=settings.path.parent)
        self._client = RemoteAccessClient(
            owner_socket_path(config, instance_root=settings.path.parent),
            timeout=config.owner_timeout_seconds,
        )
        self._lock = threading.Lock()
        self._child: subprocess.Popen[bytes] | None = None
        self._last_error: str | None = None

    def is_enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        with self._lock:
            self._running = True
            self._reconcile()

    async def watch(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=_RETRY_INTERVAL_S)
            except TimeoutError:
                if self._enabled:
                    await asyncio.to_thread(self._check_owner)

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._stop_child()

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            if not self._running:
                raise OwnerUnavailableError("Remote Access is unavailable while the gateway stops.")
            changed_owner = False
            try:
                previous = self._prepare_owner(enabled)
                changed_owner = previous is not None and previous != enabled
                if changed_owner:
                    self._set_owner_enabled(enabled)
                self._settings.update(
                    lambda config: setattr(config.remote_access, "enabled", enabled)
                )
            except Exception:
                if changed_owner:
                    try:
                        self._set_owner_enabled(not enabled)
                    except (OSError, StoreError, OwnerRequestError, OwnerUnavailableError):
                        logger.exception("Could not restore the previous Remote Access state.")
                if not self._enabled:
                    self._stop_child()
                raise
            self._enabled = enabled
            self._last_error = None
            if not enabled:
                self._stop_child()

    def _check_owner(self) -> None:
        with self._lock:
            if not self._running or not self._enabled:
                return
            if self._last_error is None:
                try:
                    self._client.ping()
                    return
                except (OwnerRequestError, OwnerUnavailableError):
                    pass
            self._reconcile()

    def _reconcile(self) -> None:
        try:
            previous = self._prepare_owner(self._enabled)
            if previous is not None and previous != self._enabled:
                self._set_owner_enabled(self._enabled)
        except (OSError, StoreError, OwnerRequestError, OwnerUnavailableError) as exc:
            if str(exc) != self._last_error:
                self._last_error = str(exc)
                logger.warning("Remote Access is unavailable. {}", exc)
        else:
            self._last_error = None

    def _prepare_owner(self, enabled: bool) -> bool | None:
        if not enabled and not self._client.socket_path.exists():
            return None
        try:
            status = self._client.status()
        except OwnerUnavailableError as exc:
            if not isinstance(exc.__cause__, (FileNotFoundError, ConnectionRefusedError)):
                raise
            if not enabled:
                return None
            self._start_child()
            status = self._client.status()
        if enabled and self._listener_port is not None:
            self._check_listener(self._listener_port)
        return status.get("enabled") is True

    def _set_owner_enabled(self, enabled: bool) -> None:
        client = RemoteAccessClient(
            self._client.socket_path,
            timeout=self._client.timeout,
            admin_token=RemoteAccessStore(self._state_dir).admin_token(),
        )
        client.set_enabled(enabled)

    def _check_listener(self, port: int) -> None:
        connection = HTTPConnection("127.0.0.1", port, timeout=self._client.timeout)
        try:
            connection.request("GET", "/remote/v1/health")
            response = connection.getresponse()
            if response.status == 200 and response.getheader("Server", "").startswith(
                "nanobot-remote"
            ):
                return
        except (OSError, HTTPException):
            pass
        finally:
            connection.close()
        raise OwnerUnavailableError(f"The remote owner listener on port {port} is unavailable.")

    def _start_child(self) -> None:
        self._stop_child()
        with tempfile.TemporaryDirectory(prefix="nanobot-remote-owner-") as directory:
            ready = Path(directory) / "ready"
            command = [
                sys.executable, "-m", "nanobot.remote.owner",
                "--state-dir", str(self._state_dir),
                "--socket", str(self._client.socket_path),
                "--ready-file", str(ready),
                "--parent-stdin",
            ]
            if self._listener_port is not None:
                command.extend(["--listener-port", str(self._listener_port)])
            child = subprocess.Popen(command, stdin=subprocess.PIPE)
            self._child = child
            deadline = time.monotonic() + _START_TIMEOUT_S
            while child.poll() is None and time.monotonic() < deadline:
                if ready.is_file():
                    return
                time.sleep(0.05)
            self._stop_child()
        raise OwnerUnavailableError("The remote owner failed to start. Check the gateway log.")

    def _stop_child(self) -> None:
        child = self._child
        self._child = None
        if child is None:
            return
        if child.stdin is not None:
            with suppress(OSError):
                child.stdin.close()
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
