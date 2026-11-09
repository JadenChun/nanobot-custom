"""Private Unix-socket IPC for the independent remote-access owner (M07)."""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any, cast

from nanobot.remote.auth import AuthRejectedError, TokenService
from nanobot.remote.delegation import (
    DelegatedAdminOperation,
    DelegationError,
    OwnerDelegationService,
    current_digest,
)
from nanobot.remote.pairing import (
    PairingBusyError,
    PairingCodeInvalidError,
    PairingConsumedError,
    PairingDeniedError,
    PairingDisabledError,
    PairingError,
    PairingExpiredError,
    PairingNotFoundError,
    PairingOwnerRequiredError,
    PairingPending,
    PairingRateLimitedError,
    PairingRejectedError,
    PairingResult,
    PairingService,
)
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
        tokens: TokenService | None = None,
        delegation: OwnerDelegationService | None = None,
    ) -> None:
        self.store = store
        self.pairing = pairing
        self.tokens = tokens
        self.delegation = delegation
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
        if role not in {"gateway", "admin", "pairing"}:
            return {"ok": False, "error": {"code": "denied", "message": "role is not admitted"}}
        if role == "admin" and not self.store.check_admin_token(_string_param(params, "admin_token") or _string_param(request, "admin_token")):
            return {"ok": False, "error": {"code": "denied", "message": "admin token is not valid"}}
        if method == "pair_collect":
            if self.pairing is None:
                return {"ok": False, "error": {"code": "unavailable", "message": "pairing service is not running"}}
            pairing_id = _string_param(params, "pairing_id")
            nonce = _string_param(params, "nonce")
            collector_secret = _string_param(params, "collector_secret")
            installation_id = _string_param(params, "installation_id")
            label = _string_param(params, "label")
            if None in (pairing_id, nonce, collector_secret, installation_id, label):
                return {"ok": False, "error": {"code": "bad_request", "message": "pairing fields are required"}}
            try:
                pending = self.pairing.collect(
                    pairing_id=cast(str, pairing_id),
                    nonce=cast(str, nonce),
                    collector_secret=cast(str, collector_secret),
                    installation_id=cast(str, installation_id),
                    label=cast(str, label),
                    platform=_string_param(params, "platform"),
                    model=_string_param(params, "model"),
                    device_public_key_der=_base64_param(params, "device_public_key"),
                )
            except PairingError as exc:
                return {"ok": False, "error": {"code": _pairing_code(exc), "message": str(exc)}}
            return {"ok": True, "result": _pending_payload(pending)}
        if method == "pair_collect_code":
            if self.pairing is None:
                return {"ok": False, "error": {"code": "unavailable", "message": "pairing service is not running"}}
            origin = _string_param(params, "origin")
            pairing_code = _string_param(params, "pairing_code") or _string_param(
                params, "comparison_code"
            )
            installation_id = _string_param(params, "installation_id")
            label = _string_param(params, "label")
            if None in (origin, pairing_code, installation_id, label):
                return {"ok": False, "error": {"code": "bad_request", "message": "pairing fields are required"}}
            try:
                pending = self.pairing.collect_by_code(
                    origin=cast(str, origin),
                    pairing_code=cast(str, pairing_code),
                    installation_id=cast(str, installation_id),
                    label=cast(str, label),
                    platform=_string_param(params, "platform"),
                    model=_string_param(params, "model"),
                    device_public_key_der=_base64_param(params, "device_public_key"),
                )
            except PairingError as exc:
                return {"ok": False, "error": {"code": _pairing_code(exc), "message": str(exc)}}
            return {"ok": True, "result": _pending_payload(pending)}
        if method == "pair_poll":
            if self.pairing is None:
                return {"ok": False, "error": {"code": "unavailable", "message": "pairing service is not running"}}
            pairing_id = _string_param(params, "pairing_id")
            nonce = _string_param(params, "nonce")
            collector_secret = _string_param(params, "collector_secret")
            if None in (pairing_id, nonce, collector_secret):
                return {"ok": False, "error": {"code": "bad_request", "message": "pairing fields are required"}}
            try:
                result = self.pairing.poll(
                    pairing_id=cast(str, pairing_id),
                    nonce=cast(str, nonce),
                    collector_secret=cast(str, collector_secret),
                )
            except PairingError as exc:
                return {"ok": False, "error": {"code": _pairing_code(exc), "message": str(exc)}}
            if isinstance(result, PairingResult):
                return {
                    "ok": True,
                    "result": {
                        "status": "approved",
                        "pairing_id": result.pairing_id,
                        "installation_id": result.installation_id,
                        "access_token": result.access_token,
                        "refresh_token": result.refresh_token,
                        "expires_in": result.expires_in,
                        "refresh_expires_in": result.refresh_expires_in,
                        "scopes": list(result.scopes),
                    },
                }
            assert isinstance(result, PairingPending)
            return {"ok": True, "result": _pending_payload(result)}
        if method == "pair_recover_begin":
            if self.pairing is None:
                return {"ok": False, "error": {"code": "unavailable", "message": "pairing service is not running"}}
            installation_id = _string_param(params, "installation_id")
            device_key = _base64_param(params, "device_public_key")
            if installation_id is None or device_key is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "recovery fields are required"}}
            try:
                claims = self.pairing.begin_device_recovery(
                    installation_id=installation_id,
                    device_public_key_der=device_key,
                )
            except PairingError as exc:
                return {"ok": False, "error": {"code": _pairing_code(exc), "message": str(exc)}}
            return {"ok": True, "result": {"proof_challenge": claims}}
        if method == "pair_recover_complete":
            if self.pairing is None:
                return {"ok": False, "error": {"code": "unavailable", "message": "pairing service is not running"}}
            installation_id = _string_param(params, "installation_id")
            raw_claims = params.get("claims")
            signature = _base64_param(params, "signature")
            if installation_id is None or not isinstance(raw_claims, dict) or signature is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "proof fields are required"}}
            from nanobot.remote.device_proof import DeviceProofClaims, DeviceProofError

            try:
                claims = DeviceProofClaims.from_payload(cast(dict[str, object], raw_claims))
                pair = self.pairing.complete_device_recovery(
                    installation_id=installation_id,
                    claims=claims,
                    signature_der=signature,
                )
            except DeviceProofError as exc:
                return {"ok": False, "error": {"code": "pairing_rejected", "message": exc.reason}}
            except PairingError as exc:
                return {"ok": False, "error": {"code": _pairing_code(exc), "message": str(exc)}}
            return {
                "ok": True,
                "result": {
                    "status": "approved",
                    "installation_id": installation_id,
                    "access_token": pair.access_token,
                    "refresh_token": pair.refresh_token,
                    "expires_in": pair.expires_in,
                    "refresh_expires_in": pair.refresh_expires_in,
                    "scopes": list(pair.scopes),
                },
            }
        if method == "pair_prove":
            if self.pairing is None:
                return {"ok": False, "error": {"code": "unavailable", "message": "pairing service is not running"}}
            pairing_id = _string_param(params, "pairing_id")
            raw_claims = params.get("claims")
            signature = _base64_param(params, "signature")
            if pairing_id is None or not isinstance(raw_claims, dict) or signature is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "proof fields are required"}}
            from nanobot.remote.device_proof import DeviceProofClaims, DeviceProofError

            try:
                claims = DeviceProofClaims.from_payload(cast(dict[str, object], raw_claims))
                pair = self.pairing.complete_device_proof(
                    pairing_id=pairing_id,
                    claims=claims,
                    signature_der=signature,
                )
            except DeviceProofError as exc:
                return {"ok": False, "error": {"code": "pairing_rejected", "message": exc.reason}}
            except PairingError as exc:
                return {"ok": False, "error": {"code": _pairing_code(exc), "message": str(exc)}}
            return {
                "ok": True,
                "result": {
                    "status": "approved",
                    "pairing_id": pairing_id,
                    "installation_id": claims.installation_id,
                    "access_token": pair.access_token,
                    "refresh_token": pair.refresh_token,
                    "expires_in": pair.expires_in,
                    "refresh_expires_in": pair.refresh_expires_in,
                    "scopes": list(pair.scopes),
                },
            }
        if method == "ping":
            return {"ok": True, "result": {"version": 1, "started_at": self._started_at}}
        if method == "status":
            return {"ok": True, "result": self.store.snapshot()}
        if method == "refresh_access":
            if self.tokens is None:
                return {"ok": False, "error": {"code": "unavailable", "message": "token service is not running"}}
            installation_id = _string_param(params, "installation_id")
            refresh_token = _string_param(params, "refresh_token")
            if installation_id is None or refresh_token is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "installation_id and refresh_token are required"}}
            try:
                pair = self.tokens.refresh(installation_id, refresh_token)
            except AuthRejectedError as exc:
                device = self.store.get_device(installation_id)
                code = "device_revoked" if device is None or device.revoked else "invalid_refresh"
                return {"ok": False, "error": {"code": code, "message": str(exc)}}
            return {
                "ok": True,
                "result": {
                    "access_token": pair.access_token,
                    "refresh_token": pair.refresh_token,
                    "expires_in": pair.expires_in,
                    "refresh_expires_in": pair.refresh_expires_in,
                    "scopes": list(pair.scopes),
                },
            }
        if method == "validate_access":
            if self.tokens is None:
                return {"ok": False, "error": {"code": "unavailable", "message": "token service is not running"}}
            access_token = _string_param(params, "access_token")
            if access_token is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "access_token is required"}}
            try:
                grant = self.tokens.validate_access(access_token)
            except AuthRejectedError as exc:
                return {"ok": True, "result": {"valid": False, "message": str(exc)}}
            revision = self.store.snapshot()["revision"]
            return {
                "ok": True,
                "result": {
                    "valid": True,
                    "installation_id": grant.installation_id,
                    "scopes": list(grant.scopes),
                    "expires_at": grant.expires_at,
                    "generation": revision,
                },
            }
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
            actor = _string_param(params, "actor_installation_id")
            try:
                self.store.record_audit(
                    installation_id=installation_id,
                    operation="device.revoke",
                    decision="attempted",
                    actor_installation_id=actor,
                )
            except StoreError as exc:
                return {
                    "ok": False,
                    "error": {"code": "unavailable", "message": str(exc)},
                }
            try:
                device = self.store.revoke_device(installation_id)
            except StoreError as exc:
                try:
                    self.store.record_audit(
                        installation_id=installation_id,
                        operation="device.revoke",
                        decision="denied",
                        actor_installation_id=actor,
                    )
                except StoreError:
                    pass
                return {"ok": False, "error": {"code": "not_found", "message": str(exc)}}
            try:
                self.store.record_audit(
                    installation_id=device.installation_id,
                    operation="device.revoke",
                    decision="applied",
                    actor_installation_id=actor,
                )
            except StoreError as exc:
                return {
                    "ok": False,
                    "error": {"code": "unavailable", "message": str(exc)},
                }
            return {"ok": True, "result": {"installation_id": device.installation_id, "revoked": device.revoked}}
        if method == "audit_event":
            if role not in {"gateway", "admin"}:
                return {"ok": False, "error": {"code": "denied", "message": "role cannot audit"}}
            installation_id = _string_param(params, "installation_id")
            operation = _string_param(params, "operation")
            decision = _string_param(params, "decision")
            if installation_id is None or operation is None or decision is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "audit fields are required"}}
            try:
                entry = self.store.record_audit(
                    installation_id=installation_id,
                    operation=operation,
                    decision=decision,
                    request_id=_string_param(params, "request_id"),
                    actor_installation_id=_string_param(params, "actor_installation_id"),
                )
            except StoreError as exc:
                return {"ok": False, "error": {"code": "unavailable", "message": str(exc)}}
            return {"ok": True, "result": entry}
        if method in {
            "delegate_authorize",
            "delegate_consume",
            "delegate_execute",
            "device_list",
            "device_rename",
            "device_revoke",
            "device_self_revoke",
            "entry_status",
        }:
            return self._dispatch_delegation(role, method, params)
        if role in {"gateway", "pairing"}:
            return {"ok": False, "error": {"code": "denied", "message": "role cannot use local administration"}}
        if method == "list_devices":
            return {"ok": True, "result": self.store.snapshot()["devices"]}
        if method == "pair_create":
            if self.pairing is None:
                return {"ok": False, "error": {"code": "unavailable", "message": "pairing service is not running"}}
            origin = _string_param(params, "origin")
            label = _string_param(params, "label")
            full_control_consent = _bool_param(params, "full_control_consent", default=False)
            shared_history_consent = _bool_param(
                params, "shared_history_consent", default=False
            )
            scopes = _string_list_param(params, "scopes")
            if origin is None or label is None or full_control_consent is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "pairing fields are required"}}
            if full_control_consent is not True:
                return {
                    "ok": False,
                    "error": {
                        "code": "owner_required",
                        "message": "owner pairing requires full-control consent",
                    },
                }
            try:
                offer = self.pairing.create_challenge(
                    origin=origin,
                    label=label,
                    chat_consent=True,
                    media_consent=True,
                    scopes=tuple(scopes) if scopes is not None else None,
                    full_control_consent=True,
                    shared_history_consent=bool(shared_history_consent),
                    owner_only=True,
                    transport=_string_param(params, "transport") or "tailscale",
                )
            except PairingError as exc:
                return {"ok": False, "error": {"code": _pairing_code(exc), "message": str(exc)}}
            except ValueError as exc:
                return {"ok": False, "error": {"code": "pairing_rejected", "message": str(exc)}}
            return {
                "ok": True,
                "result": {
                    **offer.__dict__,
                    "scopes": list(offer.scopes),
                    "expires_in": max(0, offer.expires_at - int(time.time())),
                    "qr_link": offer.qr_link,
                    "state": "offer_created",
                },
            }
        if method in {"pair_pending", "pair_status"}:
            if self.pairing is None:
                return {"ok": False, "error": {"code": "unavailable", "message": "pairing service is not running"}}
            pairing_id = _string_param(params, "pairing_id")
            try:
                if pairing_id is None:
                    return {
                        "ok": True,
                        "result": {"pending": [_pending_payload(item) for item in self.pairing.pending()]},
                    }
                return {
                    "ok": True,
                    "result": _pending_payload(self.pairing.status(pairing_id)),
                }
            except PairingError as exc:
                return {"ok": False, "error": {"code": _pairing_code(exc), "message": str(exc)}}
        if method == "pair_renew":
            if self.pairing is None:
                return {"ok": False, "error": {"code": "unavailable", "message": "pairing service is not running"}}
            pairing_id = _string_param(params, "pairing_id")
            if pairing_id is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "pairing_id is required"}}
            try:
                pending = self.pairing.renew(pairing_id)
            except PairingError as exc:
                return {"ok": False, "error": {"code": _pairing_code(exc), "message": str(exc)}}
            return {"ok": True, "result": _pending_payload(pending)}
        if method == "pair_approve":
            if self.pairing is None:
                return {"ok": False, "error": {"code": "unavailable", "message": "pairing service is not running"}}
            pairing_id = _string_param(params, "pairing_id")
            code = _string_param(params, "pairing_code") or _string_param(
                params, "comparison_code"
            )
            scopes = _string_list_param(params, "scopes")
            if pairing_id is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "pairing_id is required"}}
            try:
                approved = self.pairing.approve(
                    pairing_id=pairing_id,
                    comparison_code=code,
                    scopes=tuple(scopes) if scopes is not None else None,
                )
            except PairingError as exc:
                return {"ok": False, "error": {"code": _pairing_code(exc), "message": str(exc)}}
            except ValueError as exc:
                return {"ok": False, "error": {"code": "pairing_rejected", "message": str(exc)}}
            self.store.record_audit(
                installation_id=approved.installation_id,
                operation="pairing.approve",
                decision="applied",
                request_id=approved.pairing_id,
            )
            return {"ok": True, "result": {"pairing_id": approved.pairing_id, "installation_id": approved.installation_id, "scopes": list(approved.scopes), "state": "approved"}}
        if method == "pair_deny":
            if self.pairing is None:
                return {"ok": False, "error": {"code": "unavailable", "message": "pairing service is not running"}}
            pairing_id = _string_param(params, "pairing_id")
            if pairing_id is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "pairing_id is required"}}
            try:
                pending = self.pairing.deny(pairing_id)
            except PairingError as exc:
                return {"ok": False, "error": {"code": _pairing_code(exc), "message": str(exc)}}
            if pending.installation_id:
                self.store.record_audit(
                    installation_id=pending.installation_id,
                    operation="pairing.deny",
                    decision="denied",
                    request_id=pending.pairing_id,
                )
            return {"ok": True, "result": _pending_payload(pending)}
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

    def _dispatch_delegation(
        self,
        role: str | None,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if self.delegation is None:
            return {"ok": False, "error": {"code": "unavailable", "message": "delegation is not running"}}
        if role not in {"gateway", "admin"}:
            return {"ok": False, "error": {"code": "denied", "message": "role cannot delegate"}}
        access_token = _string_param(params, "access_token")
        if method == "delegate_authorize":
            if access_token is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "access_token is required"}}
            operation = _string_param(params, "operation")
            request_id = _string_param(params, "request_id")
            digest = _string_param(params, "digest")
            if operation is None or request_id is None or digest is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "operation, request_id and digest are required"}}
            try:
                context, admission, payload = self.delegation.authorize(
                    access_token=access_token,
                    operation=operation,
                    request_id=request_id,
                    digest=digest,
                    arguments=params.get("arguments"),
                )
            except DelegationError as exc:
                return {"ok": False, "error": {"code": exc.code, "message": str(exc)}}
            return {
                "ok": True,
                "result": {
                    "admission": {
                        "handle": admission.handle,
                        "installation_id": admission.installation_id,
                        "operation": admission.operation.value,
                        "digest": admission.digest,
                        "record_revision": admission.record_revision,
                        "expires_at": admission.expires_at,
                    },
                    "context": _context_payload(context),
                    "arguments": payload,
                },
            }
        if method == "delegate_consume":
            handle = _string_param(params, "handle")
            installation_id = _string_param(params, "installation_id")
            operation = _string_param(params, "operation")
            digest = _string_param(params, "digest")
            if None in (handle, installation_id, operation, digest):
                return {"ok": False, "error": {"code": "bad_request", "message": "admission fields are required"}}
            try:
                context = self.delegation.consume(
                    handle=cast(str, handle),
                    installation_id=cast(str, installation_id),
                    operation=cast(str, operation),
                    digest=cast(str, digest),
                )
            except DelegationError as exc:
                return {"ok": False, "error": {"code": exc.code, "message": str(exc)}}
            return {"ok": True, "result": {"context": _context_payload(context)}}
        if method == "delegate_execute":
            if access_token is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "access_token is required"}}
            operation = _string_param(params, "operation")
            request_id = _string_param(params, "request_id")
            if operation is None or request_id is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "operation and request_id are required"}}
            try:
                context = self.delegation.authenticate(access_token)
                raw_arguments = params.get("arguments")
                arguments = (
                    cast(dict[str, Any], raw_arguments)
                    if isinstance(raw_arguments, dict)
                    else {}
                )
                payload = self.delegation.authorize(
                    access_token=access_token,
                    operation=operation,
                    request_id=request_id,
                    digest=current_digest(
                        operation,
                        request_id,
                        arguments,
                        context.installation_id,
                    ),
                    arguments=arguments,
                )[2]
                result = self._execute_delegated(context, operation, payload)
            except DelegationError as exc:
                return {"ok": False, "error": {"code": exc.code, "message": str(exc)}}
            return {"ok": True, "result": result}
        if access_token is None:
            return {"ok": False, "error": {"code": "bad_request", "message": "access_token is required"}}
        try:
            context = self.delegation.authenticate(access_token)
        except DelegationError as exc:
            return {"ok": False, "error": {"code": exc.code, "message": str(exc)}}
        if method == "device_list":
            if not context.is_owner:
                return {"ok": False, "error": {"code": "owner_required", "message": "owner capability is required"}}
            return {"ok": True, "result": {"devices": _safe_devices(self.store.snapshot()["devices"])}}
        if method == "device_rename":
            target = _string_param(params, "target_installation_id")
            label = _string_param(params, "label")
            if target is None or label is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "target and label are required"}}
            try:
                result = self._require_owner_operation(context, DelegatedAdminOperation.DEVICE_RENAME)
                del result
                return {
                    "ok": True,
                    "result": self.delegation.rename_device(
                        target,
                        label,
                        actor_installation_id=context.installation_id,
                    ),
                }
            except DelegationError as exc:
                return {"ok": False, "error": {"code": exc.code, "message": str(exc)}}
        if method == "device_revoke":
            target = _string_param(params, "target_installation_id")
            if target is None:
                return {"ok": False, "error": {"code": "bad_request", "message": "target is required"}}
            try:
                self._require_owner_operation(context, DelegatedAdminOperation.DEVICE_REVOKE)
                return {
                    "ok": True,
                    "result": self.delegation.revoke_device(
                        target,
                        actor_installation_id=context.installation_id,
                    ),
                }
            except DelegationError as exc:
                return {"ok": False, "error": {"code": exc.code, "message": str(exc)}}
        if method == "device_self_revoke":
            try:
                return {"ok": True, "result": self.delegation.revoke_device(context.installation_id)}
            except DelegationError as exc:
                return {"ok": False, "error": {"code": exc.code, "message": str(exc)}}
        if method == "entry_status":
            if not context.is_owner:
                return {"ok": False, "error": {"code": "owner_required", "message": "owner capability is required"}}
            return {"ok": True, "result": {"entries": self.store.snapshot()["owned_entries"]}}
        return {"ok": False, "error": {"code": "unknown_method", "message": "unknown owner method"}}

    @staticmethod
    def _require_owner_operation(
        context: Any,
        operation: DelegatedAdminOperation,
    ) -> None:
        if not context.is_owner:
            raise DelegationError("owner_required", "owner capability is required")

    def _execute_delegated(
        self,
        context: Any,
        operation: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if operation == DelegatedAdminOperation.DEVICE_LIST.value:
            if not context.is_owner:
                raise DelegationError("owner_required", "owner capability is required")
            return {"devices": _safe_devices(self.store.snapshot()["devices"])}
        if operation == DelegatedAdminOperation.DEVICE_RENAME.value:
            self._require_owner_operation(context, DelegatedAdminOperation.DEVICE_RENAME)
            return self.delegation.rename_device(
                payload["target_installation_id"],
                payload["label"],
                actor_installation_id=context.installation_id,
            ) if self.delegation else {}
        if operation == DelegatedAdminOperation.DEVICE_REVOKE.value:
            self._require_owner_operation(context, DelegatedAdminOperation.DEVICE_REVOKE)
            return self.delegation.revoke_device(
                payload["target_installation_id"],
                actor_installation_id=context.installation_id,
            ) if self.delegation else {}
        if operation == DelegatedAdminOperation.DEVICE_SELF_REVOKE.value:
            return self.delegation.revoke_device(
                context.installation_id,
                actor_installation_id=context.installation_id,
            ) if self.delegation else {}
        if operation == DelegatedAdminOperation.ENTRY_STATUS.value:
            if not context.is_owner:
                raise DelegationError("owner_required", "owner capability is required")
            return {"entries": self.store.snapshot()["owned_entries"]}
        raise DelegationError(
            "unimplemented_operation",
            "this delegated operation is handled by the gateway permission adapter",
            status=501,
        )


def _context_payload(context: Any) -> dict[str, Any]:
    return {
        "installation_id": context.installation_id,
        "principal": context.principal,
        "scopes": list(context.scopes),
        "record_revision": context.record_revision,
        "expires_at": context.expires_at,
    }


def _safe_devices(devices: object) -> list[dict[str, Any]]:
    if not isinstance(devices, dict):
        return []
    safe: list[dict[str, Any]] = []
    for device in cast(dict[str, Any], devices).values():
        if not isinstance(device, dict):
            continue
        record = cast(dict[str, Any], device)
        safe.append(
            {
                "installation_id": record.get("installation_id", ""),
                "label": record.get("label", ""),
                "scopes": record.get("scopes", []),
                "created_at": record.get("created_at", 0),
                "last_seen_at": record.get("last_seen_at", 0),
                "revoked_at": record.get("revoked_at"),
                "status": "revoked" if record.get("revoked_at") else "active",
            }
        )
    return safe


def _string_param(params: dict[str, Any], key: str) -> str | None:
    value = params.get(key)
    return value if isinstance(value, str) else None


def _base64_param(params: dict[str, Any], key: str) -> bytes | None:
    value = params.get(key)
    if not isinstance(value, str) or not value:
        return None
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        return None


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


def _pairing_code(exc: PairingError) -> str:
    if isinstance(exc, PairingRateLimitedError):
        return "pairing_rate_limited"
    if isinstance(exc, PairingNotFoundError):
        return "pairing_not_found"
    if isinstance(exc, PairingExpiredError):
        return "pairing_expired"
    if isinstance(exc, PairingConsumedError):
        return "pairing_consumed"
    if isinstance(exc, PairingBusyError):
        return "pairing_busy"
    if isinstance(exc, PairingDisabledError):
        return "remote_access_disabled"
    if isinstance(exc, PairingOwnerRequiredError):
        return "owner_required"
    if isinstance(exc, PairingCodeInvalidError):
        return "pairing_code_invalid"
    if isinstance(exc, PairingDeniedError):
        return "pairing_denied"
    if isinstance(exc, PairingRejectedError):
        return "pairing_rejected"
    return "pairing_error"


def _pending_payload(pending: PairingPending) -> dict[str, Any]:
    device = {
        "installation_id": pending.installation_id,
        "label": pending.label,
        "platform": pending.platform,
        "model": pending.model,
    }
    return {
        "pairing_id": pending.pairing_id,
        "state": pending.state,
        "status": "pending" if pending.state == "pending_approval" else pending.state,
        "expires_at": pending.expires_at,
        "expires_in": max(0, pending.expires_at - int(time.time())),
        "requested_at": pending.requested_at,
        "installation_id": pending.installation_id,
        "label": pending.label,
        "platform": pending.platform,
        "model": pending.model,
        "device": device,
        "origin": pending.origin,
        "transport": pending.transport,
        "scopes": list(pending.scopes),
        # Kept for older internal callers; the desktop-facing name is pairing_code.
        "pairing_code": pending.pairing_code,
        "comparison_code": pending.comparison_code,
        "created_at": pending.created_at,
        "renewable": pending.renewable,
        "device_fingerprint": pending.device_fingerprint,
        "device_generation": pending.device_generation,
        "proof_challenge": pending.proof_challenge,
        "nonce": pending.nonce,
        "collector_secret": pending.collector_secret,
        "challenge": (
            f"{pending.pairing_id}.{pending.nonce}.{pending.collector_secret}"
            if pending.pairing_id and pending.nonce and pending.collector_secret
            else ""
        ),
    }
