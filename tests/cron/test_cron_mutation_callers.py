"""Focused regressions for typed cron mutation callers."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request

from nanobot.agent.tools.base import ToolResult
from nanobot.agent.tools.cron import CronTool
from nanobot.cron.definition_store import DefinitionStoreConflictError
from nanobot.cron.service import (
    CronDefinitionConflictError,
    CronDefinitionTokenRequiredError,
    CronService,
)
from nanobot.cron.types import CronJob, CronPayload, CronSchedule
from nanobot.triggers.local_types import LocalTrigger
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
def test_explicit_system_definition_mutation_updates_and_removes_with_tokens(
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
        job_instance_id=created.job_instance_id,
        definition_revision=created.definition_revision,
    )

    reconciled = service.register_system_job(updated)

    assert reconciled.schedule.every_ms == 120_000
    assert reconciled.job_instance_id == created.job_instance_id
    assert reconciled.definition_revision != created.definition_revision
    assert service.remove_system_job(
        job_id,
        job_instance_id=reconciled.job_instance_id,
        definition_revision=reconciled.definition_revision,
    )
    assert service.get_job(job_id) is None


def _cascade_handler(monkeypatch, service, triggers, calls, cleanup_mask: int = 0):
    def cleanup(kind: str, key: str) -> bool:
        calls.append((kind, key))
        return bool(cleanup_mask & {"draft": 1, "session": 2, "transcript": 4}[kind])

    monkeypatch.setattr(
        "nanobot.webui.ws_http.delete_webui_thread", lambda key: cleanup("transcript", key)
    )
    return SimpleNamespace(
        check_api_token=lambda _request: True,
        session_manager=SimpleNamespace(delete_session=lambda key: cleanup("session", key)),
        cron_service=service,
        local_trigger_store=triggers,
        workspaces=SimpleNamespace(discard_draft_scope=lambda key: cleanup("draft", key)),
    )


@pytest.mark.parametrize("cron_count", [None, 0, 2])
@pytest.mark.parametrize("has_local", [False, True])
@pytest.mark.parametrize("cleanup_mask", range(8))
def test_session_delete_preserves_optional_service_and_cleanup_order(
    monkeypatch, cron_count: int | None, has_local: bool, cleanup_mask: int
) -> None:
    calls: list[tuple[object, ...]] = []
    jobs = [
        _bound_job(),
        replace(_bound_job(), id="job-2", job_instance_id="instance-2",
                definition_revision="revision-2"),
    ][:cron_count or 0]

    class Service:
        def list_bound_cron_jobs_for_session(self, key: str, *, include_disabled: bool):
            calls.append(("list_cron", key, include_disabled))
            return jobs

        def get_job(self, job_id: str) -> CronJob | None:
            calls.append(("get", job_id))
            return next((job for job in jobs if job.id == job_id), None)

        def remove_job(self, job_id: str, *, job_instance_id: str, definition_revision: str):
            calls.append(("remove", job_id, job_instance_id, definition_revision))
            return "removed"

    local = LocalTrigger("local-1", "local", True, "websocket", "chat", "websocket:chat")

    def list_local(key: str, *, include_disabled: bool):
        calls.append(("list_local", key, include_disabled))
        return [local]

    service = Service() if cron_count is not None else None
    triggers = SimpleNamespace(
        list_for_session=list_local,
        delete=lambda trigger_id: calls.append(("trigger", trigger_id)),
    ) if has_local else None
    handler = _cascade_handler(monkeypatch, service, triggers, calls, cleanup_mask)

    response = GatewayHTTPHandler._handle_session_delete(
        handler,
        Request("/api/sessions/websocket:chat/delete?delete_automations=true", Headers()),
        "websocket:chat",
    )

    expected: list[tuple[object, ...]] = []
    if service is not None:
        expected.append(("list_cron", "websocket:chat", True))
    if has_local:
        expected.append(("list_local", "websocket:chat", True))
    expected.extend(("get", job.id) for job in jobs)
    expected.extend(("remove", job.id, job.job_instance_id, job.definition_revision) for job in jobs)
    if has_local:
        expected.append(("trigger", "local-1"))
    expected.extend((kind, "websocket:chat") for kind in ("draft", "session", "transcript"))
    assert response.status_code == 200
    assert json.loads(response.body) == {"deleted": bool(cleanup_mask)}
    assert calls == expected


@pytest.mark.parametrize("scenario", [
    "preflight_missing", "preflight_stale_instance", "preflight_stale_revision",
    "preflight_missing_instance_token", "preflight_missing_revision_token",
    "preflight_token_error", "preflight_store_error", "preflight_service_error",
    "preflight_unexpected", "remove_missing", "remove_protected", "remove_token_error",
    "remove_store_error", "remove_service_error", "remove_unexpected",
])
def test_session_delete_preserves_preflight_and_cas_failure_effects(monkeypatch, scenario: str):
    calls: list[tuple[object, ...]] = []
    first = _bound_job()
    second = replace(first, id="job-2", job_instance_id="instance-2", definition_revision="revision-2")
    if scenario == "preflight_missing_instance_token":
        second.job_instance_id = ""
    if scenario == "preflight_missing_revision_token":
        second.definition_revision = ""
    current = second
    if scenario == "preflight_stale_instance":
        current = replace(second, job_instance_id="changed-instance")
    if scenario == "preflight_stale_revision":
        current = replace(second, definition_revision="changed-revision")
    failure_types = {
        "token_error": CronDefinitionTokenRequiredError,
        "store_error": DefinitionStoreConflictError,
        "service_error": CronDefinitionConflictError,
        "unexpected": RuntimeError,
    }
    failure = next((kind("controlled failure") for suffix, kind in failure_types.items()
                    if scenario.endswith(suffix)), None)

    class Service:
        def list_bound_cron_jobs_for_session(self, key: str, *, include_disabled: bool):
            calls.append(("list_cron", key, include_disabled))
            return [first, second]

        def get_job(self, job_id: str) -> CronJob | None:
            calls.append(("get", job_id))
            if job_id == first.id:
                return first
            if scenario.startswith("preflight_") and failure is not None:
                raise failure
            return None if scenario == "preflight_missing" else current

        def remove_job(self, job_id: str, *, job_instance_id: str, definition_revision: str):
            calls.append(("remove", job_id, job_instance_id, definition_revision))
            if job_id == first.id:
                return "removed"
            if failure is not None:
                raise failure
            return "protected" if scenario == "remove_protected" else "not_found"

    local = LocalTrigger("local-1", "local", True, "websocket", "chat", "websocket:chat")

    def list_local(key: str, *, include_disabled: bool):
        calls.append(("list_local", key, include_disabled))
        return [local]

    triggers = SimpleNamespace(
        list_for_session=list_local,
        delete=lambda trigger_id: calls.append(("trigger", trigger_id)),
    )
    handler = _cascade_handler(monkeypatch, Service(), triggers, calls)
    request = Request("/api/sessions/websocket:chat/delete?delete_automations=true", Headers())
    if scenario.endswith("unexpected"):
        with pytest.raises(RuntimeError) as caught:
            GatewayHTTPHandler._handle_session_delete(handler, request, "websocket:chat")
        assert caught.value is failure
    else:
        response = GatewayHTTPHandler._handle_session_delete(handler, request, "websocket:chat")
        if scenario == "remove_protected":
            message = "automation job-2 is protected"
        else:
            reason = "controlled failure" if failure is not None else "it changed before deletion"
            message = f"automation deletion conflict for job-2: {reason}. Refresh the session and retry."
        assert response.status_code == 409
        assert response.body == message.encode("utf-8")
    expected = [
        ("list_cron", "websocket:chat", True), ("list_local", "websocket:chat", True),
        ("get", "job"), ("get", "job-2"),
    ]
    if scenario.startswith("remove_"):
        expected.extend(("remove", job.id, job.job_instance_id, job.definition_revision)
                        for job in (first, second))
    assert calls == expected
