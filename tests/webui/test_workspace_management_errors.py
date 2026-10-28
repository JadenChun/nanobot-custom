"""Workspace management errors keep public recovery guidance truthful."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request

from nanobot.security.folder_grants import (
    NativeProcessesActiveError,
    NativeProcessesUnknownError,
)
from nanobot.webui.ws_http import GatewayHTTPHandler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (NativeProcessesActiveError("native_processes_active"), b"native_processes_active"),
        (NativeProcessesUnknownError("native_processes_unknown"), b"native_processes_unknown"),
    ],
)
async def test_folder_management_reports_live_and_unknown_admission_separately(error, expected):
    handler = object.__new__(GatewayHTTPHandler)
    handler.config = SimpleNamespace(host="127.0.0.1")

    class WorkspaceController:
        def upsert_grant_with_receipt(self, _payload):
            raise error

    handler.workspaces = WorkspaceController()
    request = Request(
        "/api/workspaces/grants/upsert",
        Headers([("Host", "127.0.0.1:8765")]),
    )
    setattr(request, "_nanobot_webui_mutation_request", True)
    setattr(request, "_nanobot_trusted_proxy_authenticated", True)
    setattr(
        request,
        "_nanobot_webui_mutation_payload",
        {
            "expected_revision": 0,
            "path": "/tmp/project",
            "file_access": "read_only",
            "commands": False,
        },
    )

    response = await handler._handle_workspace_management(
        SimpleNamespace(remote_address=("127.0.0.1", 54321)),
        request,
        "/api/workspaces/grants/upsert",
    )

    assert response.status_code == 409
    assert response.body == expected
