"""Focused regressions for typed cron mutation callers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request

from nanobot.agent.tools.base import ToolResult
from nanobot.agent.tools.cron import CronTool
from nanobot.cli.gateway_runtime import (
    _register_system_job_with_current_tokens,
    _remove_system_job_with_current_tokens,
)
from nanobot.cron.definition_store import DefinitionStoreConflictError
from nanobot.cron.service import CronDefinitionTokenRequiredError, CronService
from nanobot.cron.types import CronJob, CronPayload, CronSchedule
from nanobot.webui.http_utils import http_json_response
from nanobot.webui.ws_http import GatewayHTTPHandler


def _bound_job() -> CronJob:
    return CronJob(
        id="job",
        name="job",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        payload=CronPayload(
            message="check",
            session_key="websocket:chat",
            origin_channel="websocket",
            origin_chat_id="chat",
        ),
        job_instance_id="instance-1",
        definition_revision="revision-1",
    )


class _CronMutationStub:
    def __init__(self, job: CronJob, *, error: Exception | None = None) -> None:
        self.job = job
        self.error = error
        self.calls: list[tuple[str, str, str]] = []

    def get_job(self, job_id: str) -> CronJob | None:
        return self.job if job_id == self.job.id else None

    def remove_job(
        self,
        job_id: str,
        *,
        job_instance_id: str | None = None,
        definition_revision: str | None = None,
    ) -> str:
        if self.error is not None:
            raise self.error
        self.calls.append((job_id, job_instance_id or "", definition_revision or ""))
        return "removed"

    def enable_job(
        self,
        job_id: str,
        *,
        enabled: bool,
        job_instance_id: str | None = None,
        definition_revision: str | None = None,
    ) -> CronJob:
        self.calls.append((job_id, job_instance_id or "", definition_revision or ""))
        return self.job

    def update_job(
        self,
        job_id: str,
        *,
        name: str | None = None,
        job_instance_id: str | None = None,
        definition_revision: str | None = None,
        **_kwargs: object,
    ) -> CronJob:
        self.calls.append((job_id, job_instance_id or "", definition_revision or ""))
        return self.job


class _BatchCronMutationStub:
    def __init__(self, jobs: list[CronJob], current_jobs: list[CronJob]) -> None:
        self.jobs = jobs
        self.current_jobs = {job.id: job for job in current_jobs}
        self.removed: list[str] = []

    def list_bound_cron_jobs_for_session(
        self,
        _session_key: str,
        *,
        include_disabled: bool = True,
    ) -> list[CronJob]:
        return self.jobs

    def get_job(self, job_id: str) -> CronJob | None:
        return self.current_jobs.get(job_id)

    def remove_job(
        self,
        job_id: str,
        *,
        job_instance_id: str | None = None,
        definition_revision: str | None = None,
    ) -> str:
        self.removed.append(job_id)
        return "removed"


def test_cron_tool_remove_fetches_and_passes_exact_tokens() -> None:
    service = _CronMutationStub(_bound_job())
    result = CronTool(service)._remove_job("job")

    assert result == "Removed job job"
    assert service.calls == [("job", "instance-1", "revision-1")]


def test_cron_tool_remove_returns_clear_conflict_error() -> None:
    service = _CronMutationStub(
        _bound_job(),
        error=DefinitionStoreConflictError("cron definition revision does not match"),
    )
    result = CronTool(service)._remove_job("job")

    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "cron definition conflict" in result
    assert "Refresh the job list" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["enable", "disable", "delete"])
async def test_webui_automation_mutations_pass_exact_tokens(action: str) -> None:
    service = _CronMutationStub(_bound_job())
    handler = SimpleNamespace(
        check_api_token=lambda _request: True,
        cron_service=service,
        local_trigger_store=None,
        _handle_webui_automations=lambda _request: http_json_response({"ok": True}),
    )

    response = await GatewayHTTPHandler._handle_webui_automation_action(
        handler,
        Request(f"/api/webui/automations/{action}?id=job", Headers()),
        action,
    )

    assert response.status_code == 200
    assert service.calls == [("job", "instance-1", "revision-1")]


@pytest.mark.asyncio
async def test_webui_automation_update_passes_exact_tokens() -> None:
    service = _CronMutationStub(_bound_job())
    handler = SimpleNamespace(
        check_api_token=lambda _request: True,
        cron_service=service,
        local_trigger_store=None,
        _handle_webui_automations=lambda _request: http_json_response({"ok": True}),
    )
    request = Request("/api/webui/automations/update", Headers())
    request._nanobot_webui_mutation_payload = {"id": "job", "values": {"name": "changed"}}

    response = await GatewayHTTPHandler._handle_webui_automation_action(
        handler,
        request,
        "update",
    )

    assert response.status_code == 200
    assert service.calls == [("job", "instance-1", "revision-1")]


@pytest.mark.asyncio
async def test_webui_automation_mutation_maps_stale_definition_to_conflict() -> None:
    service = _CronMutationStub(
        _bound_job(),
        error=DefinitionStoreConflictError("cron definition revision does not match"),
    )
    handler = SimpleNamespace(
        check_api_token=lambda _request: True,
        cron_service=service,
        local_trigger_store=None,
        _handle_webui_automations=lambda _request: http_json_response({"ok": True}),
    )

    response = await GatewayHTTPHandler._handle_webui_automation_action(
        handler,
        Request("/api/webui/automations/delete?id=job", Headers()),
        "delete",
    )

    assert response.status_code == 409
    assert b"definition conflict" in response.body
    assert b"Refresh and retry" in response.body


@pytest.mark.asyncio
async def test_session_delete_maps_missing_tokens_to_conflict() -> None:
    job = _bound_job()
    service = _CronMutationStub(
        job,
        error=CronDefinitionTokenRequiredError("definition_revision is required"),
    )
    session_manager = SimpleNamespace(delete_session=lambda _key: pytest.fail("must not delete"))
    handler = SimpleNamespace(
        check_api_token=lambda _request: True,
        session_manager=session_manager,
        cron_service=service,
        local_trigger_store=None,
        workspaces=SimpleNamespace(discard_draft_scope=lambda _key: False),
    )
    service.list_bound_cron_jobs_for_session = lambda _key, include_disabled=True: [job]

    response = GatewayHTTPHandler._handle_session_delete(
        handler,
        Request("/api/sessions/websocket:chat/delete?delete_automations=true", Headers()),
        "websocket:chat",
    )

    assert response.status_code == 409
    assert b"automation deletion conflict" in response.body


@pytest.mark.asyncio
async def test_session_delete_preflights_all_jobs_before_any_delete() -> None:
    first = _bound_job()
    second = _bound_job()
    second.id = "job-2"
    second.definition_revision = "revision-2-old"
    current_second = _bound_job()
    current_second.id = second.id
    current_second.definition_revision = "revision-2-new"
    service = _BatchCronMutationStub([first, second], [first, current_second])
    session_manager = SimpleNamespace(delete_session=lambda _key: pytest.fail("must not delete"))
    handler = SimpleNamespace(
        check_api_token=lambda _request: True,
        session_manager=session_manager,
        cron_service=service,
        local_trigger_store=None,
        workspaces=SimpleNamespace(discard_draft_scope=lambda _key: False),
    )

    response = GatewayHTTPHandler._handle_session_delete(
        handler,
        Request("/api/sessions/websocket:chat/delete?delete_automations=true", Headers()),
        "websocket:chat",
    )

    assert response.status_code == 409
    assert b"job-2" in response.body
    assert service.removed == []


@pytest.mark.parametrize("job_id", ["dream", "heartbeat"])
def test_gateway_system_reconciliation_updates_and_removes_with_tokens(
    tmp_path, job_id: str
) -> None:
    service = CronService(tmp_path / f"{job_id}.json")
    original = CronJob(
        id=job_id,
        name=job_id,
        schedule=CronSchedule(kind="every", every_ms=60_000),
        payload=CronPayload(kind="system_event"),
    )
    created = service.register_system_job(original)
    updated = CronJob(
        id=job_id,
        name=job_id,
        schedule=CronSchedule(kind="every", every_ms=120_000),
        payload=CronPayload(kind="system_event"),
    )

    reconciled = _register_system_job_with_current_tokens(service, updated)

    assert reconciled.schedule.every_ms == 120_000
    assert reconciled.job_instance_id == created.job_instance_id
    assert reconciled.definition_revision != created.definition_revision
    assert _remove_system_job_with_current_tokens(service, job_id)
    assert service.get_job(job_id) is None
