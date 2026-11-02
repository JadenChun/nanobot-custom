"""Client for the owner-only Unix-socket IPC (M07)."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, cast

MAX_IPC_RESPONSE_BYTES = 1024 * 1024


class OwnerUnavailableError(RuntimeError):
    """The independent owner cannot be reached within the bounded timeout."""


class OwnerRequestError(RuntimeError):
    """The owner rejected or failed a request."""

    def __init__(self, message: str, *, code: str = "owner_error") -> None:
        super().__init__(message)
        self.code = code


class RemoteAccessClient:
    def __init__(
        self,
        socket_path: Path | str,
        *,
        timeout: float = 2.0,
        admin_token: str | None = None,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.timeout = timeout
        self.admin_token = admin_token

    def ping(self) -> dict[str, Any]:
        return self._call("ping", {}, role="gateway")

    def status(self) -> dict[str, Any]:
        return self._call("status", {}, role="gateway")

    def refresh_access(self, installation_id: str, refresh_token: str) -> dict[str, Any]:
        return self._call(
            "refresh_access",
            {"installation_id": installation_id, "refresh_token": refresh_token},
            role="gateway",
        )

    def validate_access(self, access_token: str) -> dict[str, Any]:
        return self._call(
            "validate_access", {"access_token": access_token}, role="gateway"
        )

    def validate(self, installation_id: str, refresh_token: str) -> dict[str, Any]:
        return self._call(
            "validate",
            {"installation_id": installation_id, "refresh_token": refresh_token},
            role="gateway",
        )

    def revoke(
        self,
        installation_id: str,
        *,
        actor_installation_id: str | None = None,
    ) -> dict[str, Any]:
        return self._call(
            "revoke",
            {
                "installation_id": installation_id,
                "actor_installation_id": actor_installation_id,
            },
            role="gateway",
        )

    def list_devices(self) -> list[dict[str, Any]]:
        return self._call("list_devices", {}, role="admin", require_admin=True)

    def set_enabled(self, enabled: bool) -> bool:
        return self._call(
            "set_enabled", {"enabled": enabled}, role="admin", require_admin=True
        )

    def add_device(
        self,
        installation_id: str,
        label: str,
        scopes: list[str],
        refresh_token: str,
    ) -> dict[str, Any]:
        return self._call(
            "add_device",
            {
                "installation_id": installation_id,
                "label": label,
                "scopes": scopes,
                "refresh_token": refresh_token,
            },
            role="admin",
            require_admin=True,
        )

    def collect_pairing(
        self,
        *,
        pairing_id: str,
        nonce: str,
        collector_secret: str,
        installation_id: str,
        label: str,
    ) -> dict[str, Any]:
        return self._call(
            "pair_collect",
            {
                "pairing_id": pairing_id,
                "nonce": nonce,
                "collector_secret": collector_secret,
                "installation_id": installation_id,
                "label": label,
            },
            role="pairing",
        )

    def poll_pairing(
        self,
        *,
        pairing_id: str,
        nonce: str,
        collector_secret: str,
    ) -> dict[str, Any]:
        return self._call(
            "pair_poll",
            {
                "pairing_id": pairing_id,
                "nonce": nonce,
                "collector_secret": collector_secret,
            },
            role="pairing",
        )

    def create_pairing(
        self,
        *,
        origin: str,
        label: str,
        chat_consent: bool,
        media_consent: bool = False,
        full_control_consent: bool = False,
        shared_history_consent: bool = False,
    ) -> dict[str, Any]:
        return self._call(
            "pair_create",
            {
                "origin": origin,
                "label": label,
                "chat_consent": chat_consent,
                "media_consent": media_consent,
                "full_control_consent": full_control_consent,
                "shared_history_consent": shared_history_consent,
            },
            role="admin",
            require_admin=True,
        )

    def approve_pairing(
        self,
        *,
        pairing_id: str,
        comparison_code: str,
        scopes: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._call(
            "pair_approve",
            {"pairing_id": pairing_id, "comparison_code": comparison_code, "scopes": scopes},
            role="admin",
            require_admin=True,
        )

    def cancel_pairing(self, pairing_id: str) -> bool:
        result = self._call(
            "pair_cancel", {"pairing_id": pairing_id}, role="admin", require_admin=True
        )
        return bool(result.get("cancelled"))

    def delegate_authorize(
        self,
        *,
        access_token: str,
        operation: str,
        request_id: str,
        digest: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._call(
            "delegate_authorize",
            {
                "access_token": access_token,
                "operation": operation,
                "request_id": request_id,
                "digest": digest,
                "arguments": arguments or {},
            },
            role="gateway",
        )

    def delegate_consume(
        self,
        *,
        handle: str,
        installation_id: str,
        operation: str,
        digest: str,
    ) -> dict[str, Any]:
        return self._call(
            "delegate_consume",
            {
                "handle": handle,
                "installation_id": installation_id,
                "operation": operation,
                "digest": digest,
            },
            role="gateway",
        )

    def audit_event(
        self,
        *,
        installation_id: str,
        operation: str,
        decision: str,
        request_id: str | None = None,
        actor_installation_id: str | None = None,
    ) -> dict[str, Any]:
        return self._call(
            "audit_event",
            {
                "installation_id": installation_id,
                "operation": operation,
                "decision": decision,
                "request_id": request_id,
                "actor_installation_id": actor_installation_id,
            },
            role="gateway",
        )

    def rotate_refresh(self, installation_id: str, refresh_token: str) -> dict[str, Any]:
        return self._call(
            "rotate_refresh",
            {"installation_id": installation_id, "refresh_token": refresh_token},
            role="admin",
            require_admin=True,
        )

    def _call(
        self,
        method: str,
        params: dict[str, Any],
        *,
        role: str,
        require_admin: bool = False,
    ) -> Any:
        if require_admin and not self.admin_token:
            raise OwnerRequestError("admin token is required")
        request: dict[str, Any] = {"method": method, "params": params, "role": role}
        if require_admin:
            admin_token = self.admin_token
            if not admin_token:
                raise OwnerRequestError("admin token is required")
            request["admin_token"] = admin_token
        payload = json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout)
        try:
            connection.connect(str(self.socket_path))
            connection.sendall(payload)
            raw = self._recv_line(connection)
        except (OSError, TimeoutError) as exc:
            raise OwnerUnavailableError("remote-access owner is unavailable") from exc
        finally:
            connection.close()
        try:
            response = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OwnerRequestError("owner returned malformed JSON") from exc
        if not isinstance(response, dict):
            raise OwnerRequestError("owner returned an invalid response")
        payload_response = cast(dict[str, Any], response)
        if not isinstance(payload_response.get("ok"), bool):
            raise OwnerRequestError("owner returned an invalid response")
        if not payload_response["ok"]:
            raw_error = payload_response.get("error")
            error = cast(dict[str, Any], raw_error) if isinstance(raw_error, dict) else {}
            raise OwnerRequestError(
                str(error.get("message", "owner request failed")),
                code=str(error.get("code", "owner_error")),
            )
        return payload_response.get("result")

    def _recv_line(self, connection: socket.socket) -> str:
        chunks: list[bytes] = []
        total = 0
        while total <= MAX_IPC_RESPONSE_BYTES:
            chunk = connection.recv(min(4096, MAX_IPC_RESPONSE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if b"\n" in chunk:
                break
        if total > MAX_IPC_RESPONSE_BYTES:
            raise OwnerRequestError("owner response exceeds the bounded size")
        data = b"".join(chunks).split(b"\n", 1)[0]
        if not data:
            raise OwnerRequestError("owner closed the connection without a response")
        return data.decode("utf-8")
