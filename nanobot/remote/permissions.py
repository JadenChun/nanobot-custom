"""Gateway-side owner permission/delegation adapter (M30/M37).

The independent owner process authenticates the device and mints a typed
delegation admission. This adapter consumes that admission and then hands the
operation to the existing workspace permission controller, which still owns
source/session/task/revision checks and the actual grant/Full effects. No
owner handle, native reference or receipt is exposed to the phone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, cast

from nanobot.remote.client import OwnerRequestError, RemoteAccessClient
from nanobot.remote.delegation import DelegatedAdminOperation, current_digest

# Standard WebUI actions that require an owner delegation admission.
ACTION_OPERATIONS: dict[str, DelegatedAdminOperation] = {
    "workspace.grant.upsert": DelegatedAdminOperation.GRANT_COMMIT,
    "workspace.access.prepare_full": DelegatedAdminOperation.FULL_PROPOSAL,
    "workspace.access.mode.set": DelegatedAdminOperation.FULL_COMMIT,
    "workspace.access.add_folder": DelegatedAdminOperation.GRANT_COMMIT,
    "workspace.access.allow_once": DelegatedAdminOperation.GRANT_COMMIT,
    "workspace.access.resolve": DelegatedAdminOperation.GRANT_COMMIT,
    "workspace.access.continue": DelegatedAdminOperation.GRANT_RELEASE,
    "workspace.access.cancel": DelegatedAdminOperation.GRANT_RELEASE,
    "workspace.folder.edit": DelegatedAdminOperation.GRANT_COMMIT,
    "repository.remember": DelegatedAdminOperation.GRANT_COMMIT,
    "repository.forget": DelegatedAdminOperation.GRANT_COMMIT,
    "repository.discover": DelegatedAdminOperation.GRANT_COMMIT,
}


class PermissionAdapterError(RuntimeError):
    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class OwnerPermissionContext:
    installation_id: str
    principal: str
    scopes: tuple[str, ...]
    operation: DelegatedAdminOperation


def _payload_digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _arguments_for(
    operation: DelegatedAdminOperation,
    payload: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    if operation in {
        DelegatedAdminOperation.GRANT_COMMIT,
        DelegatedAdminOperation.GRANT_RELEASE,
        DelegatedAdminOperation.FULL_COMMIT,
        DelegatedAdminOperation.FULL_RELEASE,
    }:
        return {
            "proposal_id": request_id,
            "payload_digest": _payload_digest(payload),
        }
    if operation is DelegatedAdminOperation.GRANT_PROPOSAL:
        extent = "write" if payload.get("file_access") == "read_write" else "read"
        return {
            "chat_id": str(payload.get("chat_id", "")),
            "target": str(payload.get("path", "")),
            "extent": extent,
        }
    if operation is DelegatedAdminOperation.FULL_PROPOSAL:
        return {"chat_id": str(payload.get("chat_id", ""))}
    return {}


class OwnerPermissionAdapter:
    """Consume owner admissions and bind them to the existing permission owner."""

    def __init__(
        self,
        client: RemoteAccessClient,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.client = client
        self._clock = clock

    def authorize(
        self,
        *,
        access_token: str,
        installation_id: str,
        action: str,
        payload: dict[str, Any],
    ) -> OwnerPermissionContext:
        operation = ACTION_OPERATIONS.get(action)
        if operation is None:
            raise PermissionAdapterError(400, "unsupported_owner_permission_action")
        request_id = _payload_digest(payload)[:32]
        arguments = _arguments_for(operation, payload, request_id)
        digest = current_digest(operation.value, request_id, arguments, installation_id)
        try:
            authorized = self.client.delegate_authorize(
                access_token=access_token,
                operation=operation.value,
                request_id=request_id,
                digest=digest,
                arguments=arguments,
            )
        except OwnerRequestError as exc:
            raise _adapter_error(exc.code) from exc
        raw_admission = authorized.get("admission")
        raw_context = authorized.get("context")
        if not isinstance(raw_admission, dict) or not isinstance(raw_context, dict):
            raise PermissionAdapterError(502, "owner_admission_invalid")
        admission = cast(dict[str, Any], raw_admission)
        context = cast(dict[str, Any], raw_context)
        handle = admission.get("handle")
        if not isinstance(handle, str) or not handle:
            raise PermissionAdapterError(502, "owner_admission_invalid")
        if context.get("installation_id") != installation_id:
            raise PermissionAdapterError(403, "installation_mismatch")
        try:
            consumed = self.client.delegate_consume(
                handle=handle,
                installation_id=installation_id,
                operation=operation.value,
                digest=digest,
            )
        except OwnerRequestError as exc:
            raise _adapter_error(exc.code) from exc
        raw_current = consumed.get("context")
        if not isinstance(raw_current, dict):
            raise PermissionAdapterError(403, "authority_changed")
        current = cast(dict[str, Any], raw_current)
        if current.get("installation_id") != installation_id:
            raise PermissionAdapterError(403, "authority_changed")
        raw_scopes = current.get("scopes")
        scopes: tuple[str, ...] = ()
        if isinstance(raw_scopes, (list, tuple)):
            typed_scopes = cast(list[Any] | tuple[Any, ...], raw_scopes)
            scopes = tuple(str(scope) for scope in typed_scopes)
        if "owner" not in scopes:
            raise PermissionAdapterError(403, "owner_required")
        return OwnerPermissionContext(
            installation_id=installation_id,
            principal=str(current.get("principal", f"owner:{installation_id}")),
            scopes=scopes,
            operation=operation,
        )


def _adapter_error(code: str) -> PermissionAdapterError:
    if code in {"device_revoked", "invalid_access", "missing_access", "owner_required"}:
        return PermissionAdapterError(403, code)
    if code in {"admission_mismatch", "authority_changed", "digest_mismatch"}:
        return PermissionAdapterError(409, code)
    if code in {"admission_missing", "admission_expired"}:
        return PermissionAdapterError(409, "authority_changed")
    return PermissionAdapterError(503, "owner_permission_unavailable")


def owner_permission_adapter_error(exc: Exception) -> tuple[int, str] | None:
    if isinstance(exc, PermissionAdapterError):
        return exc.status, exc.code
    return None
