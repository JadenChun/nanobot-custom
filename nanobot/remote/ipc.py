"""Private Unix-socket IPC for the independent remote-access owner (M07)."""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any, cast

from nanobot.remote.pairing import PairingError, PairingService
from nanobot.remote.store import RemoteAccessStore, StoreError

MAX_IPC_LINE_BYTES = 64 * 1024
IPC_SOCKET_NAME = "owner.sock"


class IPCDeniedError(PermissionError):
    """The peer or request role is not admitted."""


class OwnerIPCServer:
    """Owner-only Unix socket with peer-UID and role checks."""

    def __init__(
        self,
        store: RemoteAccessStore,
        socket_path: Path | None = None,
        pairing: PairingService | None = None,
    ) -> None:
        self.store = store
        self.pairing = pairing
        self.socket_path = Path(socket_path) if socket_path else store.root / IPC_SOCKET_NAME
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started_at: float | None = None

    @property
    def running(self) -> bool:
        return self._socket is not None

    def start(self) -> None:
        if self._socket is not None:
            return
        self.store.ensure_root()
        if self.socket_path.exists():
            if self._probe_existing():
                raise OSError("another owner is already listening on the IPC socket")
            self.socket_path.unlink()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        server.listen(16)
        server.settimeout(0.5)
        self._socket = server
        self._stop.clear()
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._accept_loop, name="remote-owner-ipc", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        server = self._socket
        self._socket = None
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=3)
        try:
            if self.socket_path.exists():
                self.socket_path.unlink()
        except OSError:
            pass

    def _probe_existing(self) -> bool:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.2)
        try:
            probe.connect(str(self.socket_path))
            return True
        except OSError:
            return False
        finally:
            probe.close()

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            server = self._socket
            if server is None:
                return
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            threading.Thread(
                target=self._handle_connection,
                args=(connection,),
                name="remote-owner-ipc-conn",
                daemon=True,
            ).start()

    def _handle_connection(self, connection: socket.socket) -> None:
        connection.settimeout(5.0)
        try:
            self._check_peer(connection)
            raw = self._read_line(connection)
            if raw is None:
                return
            try:
                request = json.loads(raw)
            except json.JSONDecodeError:
                self._write(connection, {"ok": False, "error": {"code": "bad_json", "message": "request is not JSON"}})
                return
            if not isinstance(request, dict):
                self._write(connection, {"ok": False, "error": {"code": "bad_request", "message": "request must be an object"}})
                return
            response = self._dispatch(cast(dict[str, Any], request))
            self._write(connection, response)
        except IPCDeniedError as exc:
            self._write(connection, {"ok": False, "error": {"code": "denied", "message": str(exc)}})
        except (OSError, StoreError) as exc:
            self._write(connection, {"ok": False, "error": {"code": "unavailable", "message": str(exc)}})
        finally:
            try:
                connection.close()
            except OSError:
                pass

    def _check_peer(self, connection: socket.socket) -> None:
        if not hasattr(socket, "SO_PEERCRED"):
            raise IPCDeniedError("peer credentials are unavailable on this platform")
        raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        _pid, uid, _gid = struct.unpack("3i", raw)
        if uid != os.getuid():
            raise IPCDeniedError("IPC peer is not the selected OS user")

    def _read_line(self, connection: socket.socket) -> str | None:
        chunks: list[bytes] = []
        total = 0
        while total <= MAX_IPC_LINE_BYTES:
            chunk = connection.recv(min(4096, MAX_IPC_LINE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if b"\n" in chunk:
                break
        if total > MAX_IPC_LINE_BYTES:
            raise IPCDeniedError("IPC request exceeds the bounded line size")
        data = b"".join(chunks).split(b"\n", 1)[0]
        if not data:
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise IPCDeniedError("IPC request is not UTF-8") from exc

    def _write(self, connection: socket.socket, response: dict[str, Any]) -> None:
        try:
            payload = json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n"
            connection.sendall(payload)
        except OSError:
            pass

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        raw_method = request.get("method")
        raw_role = request.get("role")
        raw_params = request.get("params", {})
        method = raw_method if isinstance(raw_method, str) else None
        role = raw_role if isinstance(raw_role, str) else None
        params: dict[str, Any] = cast(dict[str, Any], raw_params) if isinstance(raw_params, dict) else {}
        if method is None or not isinstance(raw_params, dict):
            return {"ok": False, "error": {"code": "bad_request", "message": "method and params are required"}}
        if role not in {"gateway", "admin"}:
            return {"ok": False, "error": {"code": "denied", "message": "role is not admitted"}}
        if role == "admin" and not self.store.check_admin_token(_string_param(params, "admin_token") or _string_param(request, "admin_token")):
            return {"ok": False, "error": {"code": "denied", "message": "admin token is not valid"}}
        if method == "ping":
            return {"ok": True, "result": {"version": 1, "started_at": self._started_at}}
        if method == "status":
            return {"ok": True, "result": self.store.snapshot()}
        if method == "validate":
            installation_id = _string_param(params, "installation_id")
            refresh_token = _string_param(params, "refresh_token")
            if installation_id is None or refresh_token is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "installation_id and refresh_token are required"}}
            device = self.store.validate_refresh(installation_id, refresh_token)
            if device is None:
                return {"ok": True, "result": {"valid": False}}
            return {
                "ok": True,
                "result": {
                    "valid": True,
                    "installation_id": device.installation_id,
                    "scopes": list(device.scopes),
                    "label": device.label,
                },
            }
        if method == "revoke":
            if role not in {"gateway", "admin"}:
                return {"ok": False, "error": {"code": "denied", "message": "revocation is not admitted"}}
            installation_id = _string_param(params, "installation_id")
            if installation_id is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "installation_id is required"}}
            try:
                device = self.store.revoke_device(installation_id)
            except StoreError as exc:
                return {"ok": False, "error": {"code": "not_found", "message": str(exc)}}
            return {"ok": True, "result": {"installation_id": device.installation_id, "revoked": device.revoked}}
        if role == "gateway":
            return {"ok": False, "error": {"code": "denied", "message": "gateway role cannot use local administration"}}
        if method == "list_devices":
            return {"ok": True, "result": self.store.snapshot()["devices"]}
        if method == "pair_create":
            if self.pairing is None:
                return {"ok": False, "error": {"code": "unavailable", "message": "pairing service is not running"}}
            origin = _string_param(params, "origin")
            label = _string_param(params, "label")
            chat_consent = _bool_param(params, "chat_consent")
            media_consent = _bool_param(params, "media_consent", default=False)
            if origin is None or label is None or chat_consent is None or media_consent is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "origin, label, chat_consent and media_consent are required"}}
            try:
                offer = self.pairing.create_challenge(
                    origin=origin,
                    label=label,
                    chat_consent=chat_consent,
                    media_consent=media_consent,
                )
            except (PairingError, ValueError) as exc:
                return {"ok": False, "error": {"code": "pairing_rejected", "message": str(exc)}}
            return {"ok": True, "result": {**offer.__dict__, "scopes": list(offer.scopes)}}
        if method == "pair_approve":
            if self.pairing is None:
                return {"ok": False, "error": {"code": "unavailable", "message": "pairing service is not running"}}
            pairing_id = _string_param(params, "pairing_id")
            code = _string_param(params, "comparison_code")
            scopes = _string_list_param(params, "scopes")
            if pairing_id is None or code is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "pairing_id and comparison_code are required"}}
            try:
                approved = self.pairing.approve(
                    pairing_id=pairing_id,
                    comparison_code=code,
                    scopes=tuple(scopes) if scopes is not None else None,
                )
            except (PairingError, ValueError) as exc:
                return {"ok": False, "error": {"code": "pairing_rejected", "message": str(exc)}}
            return {"ok": True, "result": {"pairing_id": approved.pairing_id, "installation_id": approved.installation_id, "scopes": list(approved.scopes)}}
        if method == "pair_cancel":
            if self.pairing is None:
                return {"ok": False, "error": {"code": "unavailable", "message": "pairing service is not running"}}
            pairing_id = _string_param(params, "pairing_id")
            if pairing_id is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "pairing_id is required"}}
            return {"ok": True, "result": {"cancelled": self.pairing.cancel(pairing_id)}}
        if method == "set_enabled":
            enabled = _bool_param(params, "enabled")
            if enabled is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "enabled must be boolean"}}
            return {"ok": True, "result": self.store.set_enabled(enabled).enabled}
        if method == "add_device":
            installation_id = _string_param(params, "installation_id")
            label = _string_param(params, "label")
            scopes = _string_list_param(params, "scopes")
            refresh_token = _string_param(params, "refresh_token")
            if installation_id is None or label is None or refresh_token is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "device fields are required"}}
            if scopes is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "scopes must be a string list"}}
            device = self.store.add_device(
                installation_id=installation_id,
                label=label,
                scopes=tuple(scopes),
                refresh_token=refresh_token,
            )
            return {"ok": True, "result": {"installation_id": device.installation_id}}
        if method == "rotate_refresh":
            installation_id = _string_param(params, "installation_id")
            refresh_token = _string_param(params, "refresh_token")
            if installation_id is None or refresh_token is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "installation_id and refresh_token are required"}}
            device = self.store.rotate_refresh(installation_id, refresh_token)
            return {"ok": True, "result": {"installation_id": device.installation_id}}
        return {"ok": False, "error": {"code": "unknown_method", "message": "unknown owner method"}}



def _string_param(params: dict[str, Any], key: str) -> str | None:
    value = params.get(key)
    return value if isinstance(value, str) else None


def _bool_param(params: dict[str, Any], key: str, *, default: bool | None = None) -> bool | None:
    value = params.get(key, default)
    return value if isinstance(value, bool) else None


def _string_list_param(params: dict[str, Any], key: str) -> list[str] | None:
    value = params.get(key)
    if not isinstance(value, list):
        return None
    values = cast(list[Any], value)
    if not all(isinstance(item, str) for item in values):
        return None
    return [item for item in values if isinstance(item, str)]
