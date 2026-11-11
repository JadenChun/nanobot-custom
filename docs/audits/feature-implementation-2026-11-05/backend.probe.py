"""Exploratory feature-contract probes; red assertions are the audit evidence."""
from __future__ import annotations

import json

from websockets.datastructures import Headers
from websockets.http11 import Request

from nanobot.agent.tools.loader import ToolLoader
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.self import MyTool
from nanobot.cron.gateway import validate_gateway_job
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule
from nanobot.webui.transcript import append_transcript_object
from tests.tools.test_tool_loader import admitted_sync as admitted_sync
from tests.tools.test_tool_loader import loader_context as loader_context
from tests.webui.test_generated_assets import _artifact, _event, _media_file
from tests.webui.test_generated_assets import assets_client as assets_client


def test_advertised_goal_tools_are_available(loader_context):
    registry = ToolRegistry()
    names = ToolLoader().load(loader_context, registry)
    assert {"create_goal", "update_goal"} <= set(names), sorted(names)


def test_enabled_runtime_inspection_is_available(loader_context):
    assert MyTool.enabled(loader_context)
    registry = ToolRegistry()
    ToolLoader().load(loader_context, registry)
    assert registry.has("my"), registry.tool_names


async def test_all_sessions_library_includes_session_201(assets_client):
    _project, instance, services, token = assets_client
    for index in range(201):
        path = _media_file(instance / "media", f"audit-{index:03d}.png")
        append_transcript_object(
            f"websocket:audit-{index:03d}",
            {
                "event": "message", "kind": "progress", "turn_id": "turn-1",
                "tool_events": [_event(call_id=f"call-{index}", result={
                    "artifacts": [_artifact(path, artifact_id=f"audit-{index}")],
                })],
            },
        )
    response = await services.endpoint.process_request(
        None,
        Request("/api/generated-assets?limit=500", Headers({"Authorization": f"Bearer {token}"})),
        is_allowed=lambda _client: True,
    )
    assert response.status_code == 200, response.body
    payload = json.loads(response.body)
    assert payload["total"] == 201, (payload["total"], len(payload["groups"]), payload["next"])


async def test_run_now_uses_gateway_configured_service(assets_client):
    _project, instance, services, token = assets_client
    service = CronService(instance / "cron" / "jobs.json")
    service.configure_gateway_execution(validate_gateway_job, lambda: {"enabled": True})
    job = service.add_job(
        "audit-run-now", CronSchedule(kind="every", every_ms=60000),
        "Return a local result", deliver=True, channel="telegram", to="42",
    )
    services.http.cron_service = service
    response = await services.http._handle_webui_automation_action(
        Request(f"/api/webui/automations/run?id={job.id}", Headers({"Authorization": f"Bearer {token}"})),
        "run",
    )
    assert response.status_code == 200, (response.status_code, response.body)
