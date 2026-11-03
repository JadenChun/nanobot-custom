"""Test-only adapter for exercising an allowlisted mobile mutation directly.

This mirrors the authenticated WebSocket admission checks while keeping the
production gateway surface free of test-only helpers.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

from nanobot.remote.operations import OperationRegistryError, load_registry
from nanobot.webui.inbound_commands import _mobile_mutation_allowed


async def mobile_mutation_payload(
    handler: Any,
    identity: Any,
    action: str,
    payload: dict[str, Any],
    *,
    access_token: str | None = None,
) -> tuple[int, dict[str, Any]]:
    try:
        spec = load_registry().mutations.get(action)
    except OperationRegistryError:
        spec = None
    if spec is None or not spec.enabled:
        return 403, {"error": "operation_not_allowed"}
    connection = SimpleNamespace(
        mobile_identity=identity,
        mobile_access_token=access_token,
    )
    if not getattr(identity, "is_owner", False) or not _mobile_mutation_allowed(
        connection, action,
    ):
        return 403, {"error": "owner_required"}
    if spec.max_request_bytes is not None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if len(encoded) > spec.max_request_bytes:
            return 413, {"error": "request_too_large"}
    response = await handler.dispatch_webui_mutation(connection, action, payload)
    try:
        body: Any = json.loads(response.body) if response.body else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 502, {"error": "invalid_response"}
    if not isinstance(body, dict):
        return 502, {"error": "invalid_response"}
    return response.status_code, cast(dict[str, Any], body)
