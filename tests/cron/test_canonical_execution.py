"""Offline, one-shot claimed legacy-fresh turn through durable text settlement."""

from __future__ import annotations

import asyncio
from collections import deque
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import DeliveryTarget, RunStatus, TurnRequest, TurnResult
from nanobot.config.schema import Config
from nanobot.cron.action_store import CronActionStore
from nanobot.cron.canonical_execution import execute_claimed_fresh_text
from nanobot.cron.completion import CronCompletionProjector
from nanobot.cron.definition_store import (
    CronDefinitionAction,
    CronDefinitionStore,
    DefinitionStoreValidationError,
)
from nanobot.cron.delivery_runtime import CronDeliveryRuntime
from nanobot.cron.delivery_store import CronDeliveryOutboxStore
from nanobot.cron.occurrence_store import (
    CronOccurrenceStore,
    OccurrenceExecutionBusyError,
    OccurrenceExecutionStartedError,
    OccurrenceOwnershipError,
)
from nanobot.cron.service import CronService, _job_definition
from nanobot.cron.types import CronDestination, CronSchedule
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse


class ScriptedProvider(LLMProvider):
    def __init__(self, responses):
        super().__init__(provider_name="synthetic")
        self.responses = deque(responses)
        self.calls = []
        self.generation = GenerationSettings(max_tokens=256, temperature=0.1)

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls.append((messages, tools, model))
        assert self.responses, "unexpected provider request"
        return self.responses.popleft()

    def get_default_model(self):
        return "synthetic-model"


@pytest.fixture
def engine_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr("subprocess.Popen", Mock(side_effect=AssertionError("process execution")))
    for name in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection",
                 "socket.getaddrinfo"):
        monkeypatch.setattr(name, Mock(side_effect=AssertionError("network")))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config()
    config.bind_source_path(tmp_path / "instance" / "config.json")
    config.agents.defaults.workspace = str(workspace)
    config.agents.defaults.model = "synthetic-model"
    config.agents.defaults.model_preset = None
    config.tools.restrict_to_workspace = True
    return config


def _service_claim(tmp_path: Path, *, legacy: bool = True, primary: bool = True):
    root = tmp_path / "cron"
    namespace = "cron:canonical-execution-test"
    definitions = CronDefinitionStore(CronActionStore(root, namespace=namespace))
    occurrences = CronOccurrenceStore(root / "occurrences", namespace=namespace)
    projector = CronCompletionProjector(occurrences, root / "runs")
    outbox = CronDeliveryOutboxStore(root / "outbox", namespace=namespace)
    service = CronService(
        root / "jobs.json", definition_store=definitions,
        occurrence_store=occurrences, completion_projector=projector,
        delivery_runtime=CronDeliveryRuntime(outbox, occurrence_store=occurrences),
    )
    job = service.add_job(
        "report", CronSchedule(kind="every", every_ms=60_000), "compute report",
        deliver=True, channel="telegram" if primary else None,
        to="primary" if primary else None,
        additional_destinations=[CronDestination("discord", "secondary")],
    )
    if legacy:
        job.payload.execution_mode = "legacy_fresh"
        job.payload.session_key = f"cron:{job.id}"
        job.payload.origin_channel = "telegram" if primary else "cli"
        job.payload.origin_chat_id = "primary" if primary else "direct"
        service._definition_action(CronDefinitionAction.update(
            _job_definition(job, job_instance_id=job.job_instance_id),
            expected_job_instance_id=job.job_instance_id,
            expected_definition_revision=job.definition_revision,
        ))
        job = service.get_job(job.id)
        assert job is not None
    claim = service.claim_scheduled_occurrence(
        job.id, 1_700_000_000_000, owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )
    return service, claim


async def test_owned_claim_request_builder_is_pure_detached_and_does_not_consume_claim(tmp_path):
    from nanobot.cron import canonical_execution

    service, claim = _service_claim(tmp_path)
    before = service.occurrence_store.get(claim.occurrence_id)
    policy = deepcopy(claim.destination_policy)
    files = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*")
             if path.is_file()}
    expected = canonical_execution._request_for_claim(claim)

    request = canonical_execution.build_claimed_fresh_request(claim)

    assert request == expected and request is not expected
    request.metadata["detached"] = True
    assert "detached" not in expected.metadata and claim.destination_policy == policy
    assert service.occurrence_store.get(claim.occurrence_id) == before
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*")
            if path.is_file()} == files
    with service.occurrence_store.acquire_execution_fence(claim) as fence:
        service.occurrence_store.begin_execution(claim, fence)


def test_owned_claim_request_builder_preserves_private_failure_instance_and_single_call(monkeypatch):
    from nanobot.cron import canonical_execution

    claim = object()
    calls = []
    failure = ValueError("frozen route failure")

    def reject(value):
        calls.append(value)
        raise failure

    monkeypatch.setattr(canonical_execution, "_request_for_claim", reject)
    with pytest.raises(ValueError) as error:
        canonical_execution.build_claimed_fresh_request(claim)

    assert error.value is failure and calls == [claim]


async def test_real_runner_claim_computes_once_and_prepares_durable_outbox(engine_config) -> None:
    service, claim = _service_claim(engine_config.workspace_path.parent)
    provider = ScriptedProvider([
        LLMResponse(content="fresh report"), LLMResponse(content="I remember the scheduled report"),
    ])
    agent = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        old = agent.sessions.get_or_create(f"cron:{claim.identity.job_id}")
        old.add_run_message("old", "assistant", "old secret")
        agent.sessions.save(old)
        agent.bus.publish_outbound = AsyncMock(side_effect=AssertionError("ordinary send"))
        agent.bus.publish_event = AsyncMock(side_effect=AssertionError("ordinary event"))

        result, projection, outbox = await execute_claimed_fresh_text(
            service, agent, claim, visible_sessions=agent.sessions,
        )

        assert result.run_id == projection.run_id == claim.run_id
        assert result.status is RunStatus.COMPLETED
        assert "old secret" not in str(provider.calls[0][0])
        assert [msg["content"] for msg in agent.sessions.get_or_create(
            f"cron:{claim.identity.job_id}").get_run_messages(claim.run_id)] == [
            "compute report", "fresh report",
        ]
        assert projection.completion["result"] == "fresh report"
        visible = agent.sessions.get_or_create("telegram:primary")
        assert visible.messages == []
        assert [item["run_id"] for item in visible.recent_scheduled_runs()] == [claim.run_id]
        assert claim.run_id in visible.get_model_history()[0]["content"]
        assert [(item.channel, item.to) for item in outbox.destinations] == [
            ("telegram", "primary"), ("discord", "secondary"),
        ]
        agent.bus.publish_outbound.assert_not_awaited()
        agent.bus.publish_event.assert_not_awaited()
        agent.bus.publish_outbound = AsyncMock()
        agent.bus.publish_event = AsyncMock()
        await agent.execute_turn(TurnRequest(
            content="what happened?", route=DeliveryTarget("telegram", "primary"),
        ))
        assert any(claim.run_id in str(message.get("content", ""))
                   for message in provider.calls[1][0])
        with pytest.raises(OccurrenceExecutionStartedError):
            await execute_claimed_fresh_text(service, agent, claim)
        assert len(provider.calls) == 2
    finally:
        await agent.aclose()


async def test_concurrent_same_claim_attempt_is_rejected_before_second_model(tmp_path: Path) -> None:
    service, claim = _service_claim(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()

    class Agent:
        async def execute_silent_cron_turn(self, request):
            entered.set()
            await release.wait()
            return TurnResult(run_id=request.run_id, content="once")

    agent = Agent()
    first = asyncio.create_task(execute_claimed_fresh_text(service, agent, claim))
    await entered.wait()
    try:
        with pytest.raises(OccurrenceExecutionBusyError):
            service.occurrence_store.reclaim_after_reconciliation(
                claim.occurrence_id,
                owner="recovery",
                expected_owner=claim.owner,
                expected_generation=claim.generation,
                resolution="confirmed_not_started",
            )
        assert service.occurrence_store.get(claim.occurrence_id).generation == claim.generation
        retried_claim = service.claim_scheduled_occurrence(
            claim.identity.job_id, claim.identity.scheduled_fire_time_ms, owner=claim.owner,
            expected_job_instance_id=claim.job_snapshot["jobInstanceId"],
            expected_definition_revision=claim.identity.job_revision,
        )
        assert retried_claim is claim
        with pytest.raises(OccurrenceExecutionBusyError):
            await execute_claimed_fresh_text(service, agent, retried_claim)
    finally:
        release.set()
    await first


async def test_reconciliation_is_busy_through_settlement_outbox_and_visible_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.cron import visible_link
    from nanobot.session.manager import SessionManager

    service, claim = _service_claim(tmp_path)
    agent = AsyncMock()
    agent.execute_silent_cron_turn.return_value = TurnResult(run_id=claim.run_id, content="once")
    phases: list[tuple[str, str]] = []

    def prove_busy(phase: str) -> None:
        state = service.occurrence_store.get(claim.occurrence_id).state
        with pytest.raises(OccurrenceExecutionBusyError):
            service.occurrence_store.reclaim_after_reconciliation(
                claim.occurrence_id,
                owner=f"recovery-{phase}",
                expected_owner=claim.owner,
                expected_generation=claim.generation,
                resolution="confirmed_compensated",
            )
        phases.append((phase, state))

    original_complete = service.complete_occurrence
    original_prepare = service.prepare_completed_delivery
    original_link = visible_link.link_completed_text_result

    def complete_with_barrier(*args, **kwargs):
        prove_busy("settlement")
        return original_complete(*args, **kwargs)

    def prepare_with_barrier(occurrence_id):
        prove_busy("outbox")
        return original_prepare(occurrence_id)

    def link_with_barrier(*args, **kwargs):
        prove_busy("visible-link")
        return original_link(*args, **kwargs)

    monkeypatch.setattr(service, "complete_occurrence", complete_with_barrier)
    monkeypatch.setattr(service, "prepare_completed_delivery", prepare_with_barrier)
    monkeypatch.setattr(visible_link, "link_completed_text_result", link_with_barrier)
    await execute_claimed_fresh_text(
        service,
        agent,
        claim,
        visible_sessions=SessionManager(tmp_path / "workspace"),
    )

    assert phases == [
        ("settlement", "claimed"),
        ("outbox", "completed"),
        ("visible-link", "completed"),
    ]
    assert service.occurrence_store.get(claim.occurrence_id).generation == claim.generation


async def test_additional_only_delivery_keeps_first_actual_destination(tmp_path: Path) -> None:
    service, claim = _service_claim(tmp_path, primary=False)
    agent = AsyncMock()
    agent.execute_silent_cron_turn.return_value = TurnResult(run_id=claim.run_id, content="report")
    _result, _projection, outbox = await execute_claimed_fresh_text(service, agent, claim)
    request = agent.execute_silent_cron_turn.await_args.args[0]
    assert request.route.channel == "cli" and request.route.chat_id == "direct"
    assert [(target.channel, target.chat_id) for target in
            request.scheduled_link.additional_destinations] == [("discord", "secondary")]
    assert [(item.channel, item.to) for item in outbox.destinations] == [("discord", "secondary")]


async def test_missing_agent_is_rejected_without_consuming_claim(tmp_path: Path) -> None:
    service, claim = _service_claim(tmp_path)
    with pytest.raises(TypeError, match="silent canonical agent"):
        await execute_claimed_fresh_text(service, object(), claim)
    agent = AsyncMock()
    agent.execute_silent_cron_turn.return_value = TurnResult(run_id=claim.run_id, content="ok")
    await execute_claimed_fresh_text(service, agent, claim)
    agent.execute_silent_cron_turn.assert_awaited_once()


async def test_stale_or_mutated_claim_fails_before_model(tmp_path: Path) -> None:
    service, claim = _service_claim(tmp_path)
    agent = AsyncMock()
    with pytest.raises(OccurrenceOwnershipError):
        await execute_claimed_fresh_text(service, agent, deepcopy(claim))
    claim.job_snapshot["payload"]["message"] = "tampered"
    with pytest.raises(DefinitionStoreValidationError, match="definitionRevision"):
        await execute_claimed_fresh_text(service, agent, claim)
    agent.execute_silent_cron_turn.assert_not_awaited()


async def test_unsupported_mode_or_model_result_keeps_claim_for_reconciliation(tmp_path: Path) -> None:
    modern_service, modern_claim = _service_claim(tmp_path / "modern", legacy=False)
    agent = AsyncMock()
    with pytest.raises(ValueError, match="legacy_fresh"):
        await execute_claimed_fresh_text(modern_service, agent, modern_claim)
    agent.execute_silent_cron_turn.assert_not_awaited()
    assert modern_service.occurrence_store.get(modern_claim.occurrence_id).state == "claimed"

    service, claim = _service_claim(tmp_path / "legacy")
    agent.execute_silent_cron_turn.return_value = TurnResult(
        run_id=claim.run_id, status=RunStatus.APPROVAL_REQUIRED, content="needs approval",
    )
    with pytest.raises(ValueError, match="completed"):
        await execute_claimed_fresh_text(service, agent, claim)
    assert service.occurrence_store.get(claim.occurrence_id).state == "claimed"
    with pytest.raises(OccurrenceExecutionStartedError):
        await execute_claimed_fresh_text(service, agent, claim)
    agent.execute_silent_cron_turn.assert_awaited_once()


async def test_failure_after_model_starts_does_not_replay_and_remains_claimed(tmp_path: Path) -> None:
    service, claim = _service_claim(tmp_path)
    agent = AsyncMock()
    agent.execute_silent_cron_turn.side_effect = RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        await execute_claimed_fresh_text(service, agent, claim)
    assert service.occurrence_store.get(claim.occurrence_id).state == "claimed"
    with pytest.raises(OccurrenceExecutionStartedError):
        await execute_claimed_fresh_text(service, agent, claim)
    agent.execute_silent_cron_turn.assert_awaited_once()


async def test_cancelled_model_cannot_replay_claim(tmp_path: Path) -> None:
    service, claim = _service_claim(tmp_path)
    started = asyncio.Event()

    class Agent:
        async def execute_silent_cron_turn(self, _request):
            started.set()
            await asyncio.Event().wait()

    agent = Agent()
    task = asyncio.create_task(execute_claimed_fresh_text(service, agent, claim))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert service.occurrence_store.get(claim.occurrence_id).state == "claimed"
    with pytest.raises(OccurrenceExecutionStartedError):
        await execute_claimed_fresh_text(service, agent, claim)


async def test_reconciled_successor_fences_stale_claim_before_model(tmp_path: Path) -> None:
    service, claim = _service_claim(tmp_path)
    assert service.occurrence_store is not None
    successor = service.occurrence_store.reclaim_after_reconciliation(
        claim.occurrence_id, owner="successor", expected_owner=claim.owner,
        expected_generation=claim.generation, resolution="confirmed_not_started",
    )
    agent = AsyncMock()
    with pytest.raises(OccurrenceOwnershipError, match="generation"):
        await execute_claimed_fresh_text(service, agent, claim)
    agent.execute_silent_cron_turn.assert_not_awaited()
    assert successor.generation == claim.generation + 1


async def test_outbox_failure_recovers_from_completed_ledger_not_model(tmp_path: Path, monkeypatch) -> None:
    service, claim = _service_claim(tmp_path)
    agent = AsyncMock()
    agent.execute_silent_cron_turn.return_value = TurnResult(run_id=claim.run_id, content="once")
    original = service.prepare_completed_delivery
    def fail_preparation(_occurrence_id):
        raise OSError("outbox")

    monkeypatch.setattr(service, "prepare_completed_delivery", fail_preparation)
    with pytest.raises(OSError, match="outbox"):
        await execute_claimed_fresh_text(service, agent, claim)
    monkeypatch.setattr(service, "prepare_completed_delivery", original)
    assert service.project_occurrence_completion(claim.occurrence_id).run_id == claim.run_id
    assert service.prepare_completed_delivery(claim.occurrence_id).messages == (
        {"content": "once", "media": []},
    )
    with pytest.raises(OccurrenceExecutionStartedError):
        await execute_claimed_fresh_text(service, agent, claim)
    agent.execute_silent_cron_turn.assert_awaited_once()


async def test_visible_link_failure_recovers_without_recomputing(tmp_path: Path, monkeypatch) -> None:
    from nanobot.cron.visible_link import link_completed_text_result
    from nanobot.session.manager import SessionManager

    service, claim = _service_claim(tmp_path)
    sessions = SessionManager(tmp_path / "workspace")
    agent = AsyncMock()
    agent.execute_silent_cron_turn.return_value = TurnResult(run_id=claim.run_id, content="done")
    original = sessions.upsert_session_metadata
    monkeypatch.setattr(sessions, "upsert_session_metadata", Mock(side_effect=OSError("visible write failed")))
    with pytest.raises(OSError, match="visible write failed"):
        await execute_claimed_fresh_text(
            service, agent, claim, visible_sessions=sessions,
        )
    assert service.project_occurrence_completion(claim.occurrence_id).run_id == claim.run_id
    monkeypatch.setattr(sessions, "upsert_session_metadata", original)
    assert link_completed_text_result(service, sessions, claim.occurrence_id) == "telegram:primary"
    assert len(SessionManager(tmp_path / "workspace").get_or_create(
        "telegram:primary").recent_scheduled_runs()) == 1
    with pytest.raises(OccurrenceExecutionStartedError):
        await execute_claimed_fresh_text(service, agent, claim, visible_sessions=sessions)
    agent.execute_silent_cron_turn.assert_awaited_once()
