"""Retained utility and current-owner fail-closed memory regressions."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nanobot.agent.memory import Consolidator, MemoryStore
from nanobot.agent.memory_recovery import MemoryReceiptRecoveryRequired
from nanobot.agent.memory_recovery_coordinator import (
    OfflineMemoryRecoveryError,
    reconcile_pending_memory_receipt_offline,
)
from nanobot.agent.memory_recovery_receipt_finalizer import (
    OfflineReceiptFinalizationError,
    finalize_pending_memory_receipt_offline,
)
from nanobot.agent.turn import TurnResult
from nanobot.cron.action_store import CronActionStore
from nanobot.cron.canonical_execution import execute_claimed_fresh_text
from nanobot.cron.completion import CronCompletionProjector
from nanobot.cron.definition_store import CronDefinitionAction, CronDefinitionStore
from nanobot.cron.delivery_runtime import CronDeliveryRuntime
from nanobot.cron.delivery_store import CronDeliveryOutboxStore
from nanobot.cron.occurrence_store import CronOccurrenceStore, OccurrenceExecutionStartedError
from nanobot.cron.service import CronService, _job_definition
from nanobot.cron.types import CronSchedule
from nanobot.providers.base import LLMResponse
from nanobot.providers.request_budget import FinalizedRequestBudgetExceeded
from nanobot.session.manager import SessionManager, SessionWriteConflict
from nanobot.utils.prompt_budget import (
    AsyncPortableFileLock,
    PromptBudget,
    reduce_messages_to_budget,
)
from tests.agent.p6_memory_contract_fixtures import (
    SummaryProvider,
    concrete_provider,
    file_snapshot,
    fitting_prefix,
    provider_limits,
    receipt_layout,
    summary_runtime,
)


@pytest.mark.asyncio
async def test_async_lock_cancellation_does_not_leak(tmp_path: Path) -> None:
    path = tmp_path / "lock"
    owner = AsyncPortableFileLock(path)
    await owner.__aenter__()
    waiter = asyncio.create_task(AsyncPortableFileLock(path).__aenter__())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await owner.__aexit__(None, None, None)

    replacement = AsyncPortableFileLock(path)
    await asyncio.wait_for(replacement.__aenter__(), timeout=1)
    await replacement.__aexit__(None, None, None)


def test_reducer_keeps_late_system_and_removes_invalid_tool_group() -> None:
    class Counter:
        @staticmethod
        def estimate_prompt_tokens(messages, _tools, _model):
            return len(messages), "test"

    messages = [
        {"role": "system", "content": "first"},
        {"role": "user", "content": "question"},
        {"role": "system", "content": "late policy"},
        {
            "role": "assistant",
            "content": "visible",
            "tool_calls": [{"id": "one"}, {"id": "one"}],
        },
        {"role": "tool", "tool_call_id": "one", "content": "orphaned"},
    ]
    reduced = reduce_messages_to_budget(
        messages,
        Counter(),
        "test",
        [],
        PromptBudget(total_tokens=100, safety_buffer=0),
    )
    assert [item["content"] for item in reduced[:2]] == ["first", "late policy"]
    assert {item.get("content") for item in reduced} >= {"question", "visible"}
    assert not any(item.get("role") == "tool" for item in reduced)
    assert not any(item.get("tool_calls") for item in reduced)


def test_session_generation_rejects_stale_append_and_merges_identical_messages(tmp_path: Path) -> None:
    first_manager = SessionManager(tmp_path)
    first = first_manager.get_or_create("cli:revision")
    first.add_message("user", "same")
    first_manager.save(first)

    stale_manager = SessionManager(tmp_path)
    stale = stale_manager.get_or_create("cli:revision")
    first.add_message("user", "same")
    stale.add_message("user", "same")
    first_manager.save(first)
    stale_manager.save(stale)
    merged = SessionManager(tmp_path).get_or_create("cli:revision")
    assert [item["content"] for item in merged.messages] == ["same", "same", "same"]

    stale.clear()
    latest = first_manager.get_or_create("cli:revision")
    latest.add_message("assistant", "new generation")
    first_manager.save(latest)
    with pytest.raises(SessionWriteConflict):
        stale_manager.save(stale)
    assert "same" in [item["content"] for item in SessionManager(tmp_path).get_or_create("cli:revision").messages]


@pytest.mark.asyncio
async def test_direct_provider_transport_is_budget_guarded():
    provider, calls = concrete_provider(content="ok")
    oversized = [{"role": "user", "content": "x" * 20_000}]
    with pytest.raises(FinalizedRequestBudgetExceeded):
        await provider.chat_once(messages=oversized, max_tokens=1, provider_context=provider_limits(2048))
    assert calls == []
    result = await provider.chat_once(messages=[{"role": "user", "content": "fit"}],
                                     max_tokens=1, provider_context=provider_limits(2048))
    assert result.content == "ok"
    assert calls == ["chat"]
    assert oversized == [{"role": "user", "content": "x" * 20_000}]


@pytest.mark.asyncio
async def test_offline_receipt_digest_conflict_preserves_new_memory_and_evidence(tmp_path):
    workspace, sessions, source, pending = receipt_layout(tmp_path, base=b"newer", end=1, output="obsolete")
    memory = workspace / "memory" / "MEMORY.md"
    memory.write_bytes(b"newest")
    from tests.state_stat import content_stat

    raw, receipt_stat, source_raw = pending.read_bytes(), content_stat(pending.stat()), source.read_bytes()
    for _ in range(2):
        with pytest.raises(OfflineMemoryRecoveryError):
            reconcile_pending_memory_receipt_offline(workspace, sessions_root=sessions)
        with pytest.raises(OfflineReceiptFinalizationError):
            finalize_pending_memory_receipt_offline(workspace, sessions_root=sessions)
        assert memory.read_text() == "newest"
        assert memory.read_bytes() == b"newest"
        header = json.loads(source.read_bytes().splitlines()[0])
        assert header["last_consolidated"] == 0
        assert header["last_archived"] == 0
        assert source.read_bytes() == source_raw
        assert pending.read_bytes() == raw and content_stat(pending.stat()) == receipt_stat
        with pytest.raises(MemoryReceiptRecoveryRequired):
            MemoryStore(workspace, sessions_root=sessions)
    # Base-digest validation can follow conversion; its durable residue is not rolled back.
    assert (workspace / "memory" / ".recovery").is_dir()


@pytest.mark.asyncio
async def test_governor_selects_largest_fitting_closed_source_prefix(tmp_path):
    rows = [
        {"role": "user", "content": "first"},
        *({"role": "assistant", "content": "x"} for _ in range(189)),
        {"role": "user", "content": "next"},
        *({"role": "assistant", "content": "x"} for _ in range(20)),
    ]
    chunk, after = await fitting_prefix(tmp_path / "source", rows, 190)
    assert (0, after.checkpoint["boundary"]) == (0, 190)
    assert chunk == rows[:190]
    assert len(after.events) == 211


def _cron_service(root):
    namespace = "cron:q1-no-resurrection"
    definitions = CronDefinitionStore(CronActionStore(root, namespace=namespace))
    occurrences = CronOccurrenceStore(root / "occurrences", namespace=namespace)
    projector = CronCompletionProjector(occurrences, root / "runs")
    outbox = CronDeliveryOutboxStore(root / "outbox", namespace=namespace)
    return CronService(
        root / "jobs.json", definition_store=definitions, occurrence_store=occurrences,
        completion_projector=projector,
        delivery_runtime=CronDeliveryRuntime(outbox, occurrence_store=occurrences),
    )


@pytest.mark.asyncio
async def test_cron_completion_merges_runtime_state_without_resurrecting_removed_job(tmp_path):
    root = tmp_path / "cron"
    service = _cron_service(root)
    job = service.add_job("race", CronSchedule(kind="every", every_ms=1000), "run")
    job.payload.execution_mode = "legacy_fresh"
    job.payload.session_key = f"cron:{job.id}"
    job.payload.origin_channel = "cli"
    job.payload.origin_chat_id = "direct"
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
    entered, release = asyncio.Event(), asyncio.Event()

    class PausingAgent:
        calls = 0

        async def execute_silent_cron_turn(self, request):
            self.calls += 1
            entered.set()
            await release.wait()
            return TurnResult(run_id=claim.run_id, content="done")

    agent = PausingAgent()
    execution = asyncio.create_task(execute_claimed_fresh_text(service, agent, claim))
    await asyncio.wait_for(entered.wait(), timeout=1)
    external = _cron_service(root)
    try:
        current = external.get_job(job.id)
        assert current is not None
        assert external.remove_job(job.id, expected_job_instance_id=current.job_instance_id,
                                   expected_definition_revision=current.definition_revision) == "removed"
        external.add_job("added", CronSchedule(kind="every", every_ms=1000), "add")
    finally:
        release.set()
    result, projection, outbox = await asyncio.wait_for(execution, timeout=2)
    assert [job.name for job in external.list_jobs(include_disabled=True)] == ["added"]
    reopened = _cron_service(root)
    assert [job.name for job in reopened.list_jobs(include_disabled=True)] == ["added"]
    assert reopened.occurrence_store.get(claim.occurrence_id).state == "completed"
    assert projection.completion["result"] == result.content == "done"
    assert outbox.run_id == claim.run_id
    assert agent.calls == 1
    with pytest.raises(OccurrenceExecutionStartedError):
        await execute_claimed_fresh_text(service, agent, claim)
    assert agent.calls == 1


@pytest.mark.asyncio
async def test_storage_free_summaries_keep_memory_reads_live_without_global_serialization(tmp_path):
    class PausingProvider(SummaryProvider):
        def __init__(self):
            super().__init__([LLMResponse(content="S1"), LLMResponse(content="S2")])
            self.entered, self.release = asyncio.Event(), asyncio.Event()

        async def chat(self, messages, **kwargs):
            call = self.calls
            response = await super().chat(messages, **kwargs)
            if call == 0:
                self.entered.set()
                await self.release.wait()
            return response

    store = MemoryStore(tmp_path, sessions_root=tmp_path / "sessions")
    store.memory_file.write_bytes(b"M0")
    before = file_snapshot(tmp_path)
    provider = PausingProvider()
    runtime = summary_runtime(provider)
    sources = [[{"role": "user", "content": content}] for content in ("first", "second")]
    first = asyncio.create_task(Consolidator.summarize_source_chunk(sources[0], None,
                                   runtime=runtime, max_output_tokens=256))
    await asyncio.wait_for(provider.entered.wait(), timeout=1)
    try:
        second_result = await asyncio.wait_for(Consolidator.summarize_source_chunk(sources[1], None,
                                 runtime=runtime, max_output_tokens=256), timeout=1)
        observed = await asyncio.wait_for(asyncio.to_thread(store.read_memory), timeout=0.5)
    finally:
        provider.release.set()
    first_result = await asyncio.wait_for(first, timeout=1)
    assert first_result.summary == "S1" and first_result.status == "complete" and first_result.calls == 1
    assert second_result.summary == "S2" and second_result.status == "complete" and second_result.calls == 1
    assert provider.calls == 2
    assert [json.loads(prompt[1]["content"]) for prompt in provider.prompts] == [
        {"previous_summary": None, "events": sources[0]},
        {"previous_summary": None, "events": sources[1]},
    ]
    assert "M0" not in str(provider.prompts) and "M1" not in str(provider.prompts)
    assert observed == "M0"
    assert store.read_memory() == "M0"
    assert file_snapshot(tmp_path) == before


def test_scheduled_ring_preserves_approval_state_and_legacy_defaults(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    visible = manager.get_or_create("telegram:approval")
    visible.add_scheduled_run(
        job_id="job-approval", run_id="run-blocked", instruction="blocked instruction",
        result="approval needed", detail_session_key="cron:job-approval",
        approval_granted=False, status="approval_required", stop_reason="approval_required",
    )
    visible.add_scheduled_run(
        job_id="job-approved", run_id="run-approved", instruction="approved instruction",
        result="completed result", detail_session_key="cron:job-approved",
        approval_granted=True, status="completed",
    )
    manager.save(visible)
    loaded = SessionManager(tmp_path).get_or_create("telegram:approval")
    ring = loaded.recent_scheduled_runs()
    assert ring[0]["approval_granted"] is False
    assert ring[0]["status"] == "approval_required"
    assert ring[0]["stop_reason"] == "approval_required"
    assert ring[1]["approval_granted"] is True
    projection = loaded.get_model_history()[0]["content"]
    assert "status=approval_required" in projection
    assert "approval_granted=false" in projection
    assert "status=completed" in projection
    assert "approval_granted=true" in projection

    legacy = SessionManager(tmp_path).get_or_create("telegram:legacy-approval")
    legacy.metadata[legacy.SCHEDULED_RING_KEY] = [{
        "job_id": "legacy-job",
        "run_id": "legacy-run",
        "detail_ref": {"session_key": "cron:legacy-job", "run_id": "legacy-run"},
        "instruction": "legacy instruction",
        "result": "legacy result",
        "status": "completed",
    }]
    SessionManager(tmp_path).save(legacy)
    legacy_loaded = SessionManager(tmp_path).get_or_create("telegram:legacy-approval")
    assert legacy_loaded.recent_scheduled_runs()[0].get("approval_granted", False) is False
    legacy_projection = legacy_loaded.get_model_history()[0]["content"]
    assert "approval_granted=false" in legacy_projection
