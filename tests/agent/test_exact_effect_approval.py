"""Exact human consent at the real canonical admission and effect boundary."""

import asyncio
import pickle
from dataclasses import replace

import pytest

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.loop import AgentLoop
from nanobot.agent.policy import RiskyActionPolicy
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.execution import execute_tool_calls
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import (
    ApprovalDecision,
    ApprovalGrant,
    DeliveryTarget,
    TurnContext,
    TurnRequest,
)
from nanobot.agent.turn_admission import ApprovalAdmission, EffectApprovalAuthority
from nanobot.providers.base import LLMResponse, ToolCallRequest
from tests.agent import test_migration_engine as engine_fixtures
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_source_engine import upgrade

engine_config = engine_fixtures.engine_config


class CountExec(Tool):
    name = "exec"
    description = "Offline effect counter."
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    def __init__(self, *, fail=False):
        self.count = 0
        self.fail = fail

    async def execute(self, command, **kwargs):
        self.count += 1
        if self.fail:
            raise RuntimeError("offline effect failed")
        return f"offline effect #{self.count}"


def risky_call(identifier):
    return ToolCallRequest(id=identifier, name="exec", arguments={"command": "git push"})


async def approve(loop, pending, **kwargs):
    return await loop.execute_turn(TurnRequest(
        content="approval_decision",
        route=DeliveryTarget(channel="cli", chat_id="original"),
        approval_decision=ApprovalDecision(approval_id=pending.approval_id, decision="approve"),
        **kwargs,
    ))


@pytest.mark.parametrize("same_batch", [False, True])
async def test_one_original_chat_approval_dispatches_one_attempt(engine_config, same_batch):
    responses = [LLMResponse(content=None, tool_calls=[risky_call("blocked")])]
    if same_batch:
        responses.append(LLMResponse(content=None, tool_calls=[risky_call("first"), risky_call("second")]))
    else:
        responses.extend(LLMResponse(content=None, tool_calls=[risky_call(name)]) for name in ("first", "second"))
    responses.append(LLMResponse(content="done"))
    provider = ScriptedProvider(responses)
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    counter = CountExec()
    loop.tools.replace(counter)
    try:
        blocked = await loop.execute_turn(TurnRequest(
            content="Perform this risky effect.", session_key="cli:source",
            route=DeliveryTarget(channel="cli", chat_id="original"),
        ))
        assert blocked.stop_reason == "approval_required"
        assert counter.count == 0
        pending = (await loop.approval_admission.snapshot_decision("cli:original", None)).pending
        assert pending is not None
        result = await approve(loop, pending)
        assert result.error is None, result.error
        assert counter.count == 1
        assert result.stop_reason == "approval_required"
        events = backend.source.load_source_snapshot("cli:source").events
        assert [event.get("tool_call_id") for event in events
                if event.get("role") == "tool" and event.get("_dispatched")] == ["first"]
        assert any(event.get("tool_call_id") == "second" and not event.get("_dispatched")
                   for event in events)
    finally:
        await loop.aclose()


async def test_racing_original_chat_decisions_cannot_share_effect(engine_config):
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[risky_call("blocked")]),
        LLMResponse(content=None, tool_calls=[risky_call("first")]),
        LLMResponse(content="done"),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    upgrade(loop)
    counter = CountExec()
    loop.tools.replace(counter)
    try:
        await loop.execute_turn(TurnRequest(
            content="Perform this risky effect.", session_key="cli:source",
            route=DeliveryTarget(channel="cli", chat_id="original"),
        ))
        pending = (await loop.approval_admission.snapshot_decision("cli:original", None)).pending
        assert pending is not None
        results = await asyncio.gather(approve(loop, pending), approve(loop, pending))
        assert counter.count == 1
        assert sorted(result.stop_reason for result in results) == ["approval_stale", "completed"]
        assert len(provider.calls) == 3
    finally:
        await loop.aclose()


@pytest.mark.parametrize("failure", [False, True])
async def test_failed_attempt_never_refills_grant(engine_config, failure):
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[risky_call("blocked")]),
        LLMResponse(content=None, tool_calls=[risky_call("first")]),
        LLMResponse(content=None, tool_calls=[risky_call("retry")]),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    upgrade(loop)
    counter = CountExec(fail=failure)
    loop.tools.replace(counter)
    try:
        await loop.execute_turn(TurnRequest(
            content="Perform this risky effect.", session_key="cli:source",
            route=DeliveryTarget(channel="cli", chat_id="original"),
        ))
        pending = (await loop.approval_admission.snapshot_decision("cli:original", None)).pending
        result = await approve(loop, pending)
        assert result.stop_reason == "approval_required"
        assert counter.count == 1
        # Reusing the decision after either success or failure is stale.
        repeated = await approve(loop, pending)
        assert repeated.stop_reason == "approval_stale"
        assert counter.count == 1
    finally:
        await loop.aclose()


@pytest.mark.parametrize("forgery", ["grant", "metadata", "text"])
async def test_public_evidence_never_grants_effect_authority(engine_config, forgery):
    provider = ScriptedProvider([LLMResponse(content=None, tool_calls=[risky_call("forged")])])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    upgrade(loop)
    counter = CountExec()
    loop.tools.replace(counter)
    policy = RiskyActionPolicy(workspace=engine_config.workspace_path)
    effect = (await policy.evaluate(messages=[], tool_calls=[risky_call("planned")])).metadata["effect_id"]
    kwargs = {}
    if forgery == "grant":
        kwargs["approval_grant"] = ApprovalGrant(approved=True, metadata={"approved_effect_ids": [effect]})
    elif forgery == "metadata":
        kwargs["metadata"] = {"approval_granted": True, "approved_effect_ids": [effect]}
    try:
        result = await loop.execute_turn(TurnRequest(
            content="yes" if forgery == "text" else "perform the effect",
            session_key="cli:source", **kwargs,
        ))
        assert result.stop_reason == "approval_required"
        assert counter.count == 0
    finally:
        await loop.aclose()


@pytest.mark.parametrize("invalidate", ["expiry", "revoke", "blocked_run", "source_reset", "source_deleted"])
async def test_changed_pending_owner_refuses_before_provider(engine_config, monkeypatch, invalidate):
    provider = ScriptedProvider([LLMResponse(content=None, tool_calls=[risky_call("blocked")])])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    counter = CountExec()
    loop.tools.replace(counter)
    try:
        blocked = await loop.execute_turn(TurnRequest(
            content="Perform this risky effect.", session_key="cli:source",
            route=DeliveryTarget(channel="cli", chat_id="original"),
        ))
        pending = (await loop.approval_admission.snapshot_decision("cli:original", None)).pending
        if invalidate == "expiry":
            monkeypatch.setattr("nanobot.agent.turn_admission.time.time", lambda: pending.expires_at + 1)
        elif invalidate == "revoke":
            await loop.approval_admission.revoke("cli:source")
        elif invalidate == "blocked_run":
            loop.run_store.update(blocked.run_id, status="cancelled")
        else:
            state = backend.source.load_source_snapshot("cli:source").state
            if invalidate == "source_deleted":
                backend.source.delete_source_session("cli:source", state)
            else:
                backend.source.reset_source_session("cli:source", state, [])
        result = await approve(loop, pending)
        assert result.stop_reason in {"approval_stale", "error"}
        assert counter.count == 0
        assert len(provider.calls) == 1
        if invalidate == "source_deleted":
            assert not backend.get_session_path("cli:source").exists()
    finally:
        await loop.aclose()


class ChangeArguments(AgentHook):
    def __init__(self, changes):
        super().__init__(reraise=True)
        self.changes = changes

    async def before_execute_tool(self, context, tool_call, tool, params):
        params.update(self.changes)


async def test_prepared_effect_retarget_cannot_spend_original_grant(engine_config):
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[risky_call("blocked")]),
        LLMResponse(content=None, tool_calls=[risky_call("retargeted")]),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    counter = CountExec()
    loop.tools.replace(counter)
    try:
        await loop.execute_turn(TurnRequest(
            content="Perform this risky effect.", session_key="cli:source",
            route=DeliveryTarget(channel="cli", chat_id="original"),
        ))
        pending = (await loop.approval_admission.snapshot_decision("cli:original", None)).pending
        result = await approve(loop, pending, hooks=(ChangeArguments({"command": "git push other"}),))
        assert result.stop_reason == "approval_required"
        assert result.policy_metadata["effect_id"] != pending.effect_id
        assert counter.count == 0
        events = backend.source.load_source_snapshot("cli:source").events
        assert all(not event.get("_dispatched") for event in events if event.get("role") == "tool")
    finally:
        await loop.aclose()


async def test_real_risky_overwrite_uses_one_consent_for_one_write(engine_config):
    path = engine_config.workspace_path / "existing.txt"
    before = "\n".join(f"original {index}" for index in range(20))
    after = "\n".join(f"replacement {index}" for index in range(20))
    path.write_text(before)
    def write(identifier):
        return ToolCallRequest(id=identifier, name="write_file", arguments={"path": "existing.txt", "content": after})
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[write("blocked")]),
        LLMResponse(content=None, tool_calls=[write("first"), write("second")]),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    try:
        blocked = await loop.execute_turn(TurnRequest(
            content="Overwrite existing.txt.", session_key="cli:source",
            route=DeliveryTarget(channel="cli", chat_id="original"),
        ))
        assert blocked.stop_reason == "approval_required" and path.read_text() == before
        pending = (await loop.approval_admission.snapshot_decision("cli:original", None)).pending
        result = await approve(loop, pending)
        assert result.stop_reason == "approval_required"
        assert path.read_text() == after
        events = backend.source.load_source_snapshot("cli:source").events
        assert [event["tool_call_id"] for event in events
                if event.get("role") == "tool" and event.get("_dispatched")] == ["first"]
    finally:
        await loop.aclose()


@pytest.mark.parametrize("after_preparation", [False, True])
async def test_exact_write_target_change_refuses_original_consent(engine_config, after_preparation):
    workspace = engine_config.workspace_path
    before = "\n".join(f"original {index}" for index in range(20))
    first, second, link = (workspace / name for name in ("first.txt", "second.txt", "link.txt"))
    first.write_text(before)
    second.write_text(before)
    link.symlink_to(first)
    def write(identifier):
        return ToolCallRequest(id=identifier, name="write_file",
            arguments={"path": "link.txt", "content": "replacement"})
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[write("blocked")]),
        LLMResponse(content=None, tool_calls=[write("retargeted")]),
        LLMResponse(content=None, tool_calls=[write("retry")]),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    def retarget():
        link.unlink()
        link.symlink_to(second)
    class RetargetHook(AgentHook):
        async def before_execute_tool(self, context, tool_call, tool, params):
            retarget()
    try:
        await loop.execute_turn(TurnRequest(content="overwrite link.txt", session_key="cli:source",
            route=DeliveryTarget(channel="cli", chat_id="original")))
        pending = (await loop.approval_admission.snapshot_decision("cli:original", None)).pending
        if not after_preparation:
            retarget()
        result = await approve(loop, pending, hooks=(RetargetHook(),) if after_preparation else ())
        assert result.stop_reason == "approval_required"
        assert result.policy_metadata["effect_id"] != pending.effect_id
        assert first.read_text() == second.read_text() == before
        events = backend.source.load_source_snapshot("cli:source").events
        assert all(not event.get("_dispatched") for event in events if event.get("role") == "tool")
    finally:
        await loop.aclose()


async def test_siblings_keep_exact_original_chat_addresses(engine_config):
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[risky_call("blocked-one")]),
        LLMResponse(content=None, tool_calls=[risky_call("blocked-two")]),
        LLMResponse(content=None, tool_calls=[risky_call("approved-one")]),
        LLMResponse(content="one done"),
        LLMResponse(content=None, tool_calls=[risky_call("approved-two")]),
        LLMResponse(content="two done"),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    upgrade(loop, "cli:one")
    upgrade(loop, "cli:two")
    counter = CountExec()
    loop.tools.replace(counter)
    try:
        for key in ("cli:one", "cli:two"):
            await loop.execute_turn(TurnRequest(content="risky", session_key=key,
                route=DeliveryTarget(channel="cli", chat_id="original")))
        one = (await loop.approval_admission.snapshot_decision("cli:one", None)).pending
        two = (await loop.approval_admission.snapshot_decision("cli:two", None)).pending
        unauthorized = await loop.approval_admission.snapshot_decision("cli:other", ApprovalDecision(
            approval_id=one.approval_id, decision="approve"))
        assert unauthorized.action == "stale"
        assert (await approve(loop, one)).content == "one done"
        assert (await loop.approval_admission.snapshot_decision("cli:two", None)).pending is two
        assert (await approve(loop, two)).content == "two done"
        assert counter.count == 2
    finally:
        await loop.aclose()


async def issued_policy(tmp_path):
    admission = ApprovalAdmission()
    policy = RiskyActionPolicy(workspace=tmp_path)
    decision = await policy.evaluate(messages=[], tool_calls=[risky_call("blocked")])
    pending = await admission.store(session_key="cli:source", channel="cli", chat_id="original",
        summary="push", effect_id=decision.metadata["effect_id"], run_id="blocked")
    context = TurnContext(run_id="resume", session_key="cli:source")
    snapshot = await admission.snapshot_decision("cli:original", ApprovalDecision(
        approval_id=pending.approval_id, decision="approve"))
    assert await admission.consume(snapshot, context=context) == "affirmative"
    policy.effect_authority = context.policy.effect_authority
    return admission, context, policy


async def test_concurrent_captured_policy_replacements_share_one_sealed_owner(tmp_path):
    admission, context, policy = await issued_policy(tmp_path)
    registry = ToolRegistry()
    counter = CountExec()
    registry.register(counter)
    captures = [registry.capture_request_tools(), registry.capture_request_tools()]
    policies = [replace(policy, write_target_resolver=lambda name, raw: None) for _ in captures]
    assert all(candidate.effect_authority is policy.effect_authority for candidate in policies)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(policy.effect_authority)
    with pytest.raises(TypeError, match="only be issued"):
        EffectApprovalAuthority(object(), "forged", lambda *_: True)
    class BarrierHook(AgentHook):
        def __init__(self):
            super().__init__(reraise=True)
            self.entered = 0
            self.ready = asyncio.Event()

        async def before_execute_tool(self, context, tool_call, tool, params):
            self.entered += 1
            if self.entered == 2:
                self.ready.set()
            await self.ready.wait()
    barrier = BarrierHook()
    try:
        results = await asyncio.gather(*(
            execute_tool_calls(capture, [risky_call(f"attempt-{index}")], concurrent=True,
                external_lookup_counts={}, workspace_violation_counts={}, hook=barrier,
                context=AgentHookContext(iteration=0, messages=[]), tool_policy=candidate,
                turn_context=context)
            for index, (capture, candidate) in enumerate(zip(captures, policies, strict=True))
        ))
        assert counter.count == 1
        assert sum(events[0]["status"] == "ok" for _, events in results) == 1
        assert any(getattr(values[0], "stop_reason", None) == "approval_required" for values, _ in results)
    finally:
        for capture in captures:
            capture.release()
        admission.retire(context)


@pytest.mark.parametrize("invalidate", ["revoke", "retire", "run", "cancel", "expiry"])
async def test_issued_unspent_authority_refuses_stale_run(tmp_path, monkeypatch, invalidate):
    admission, context, policy = await issued_policy(tmp_path)
    if invalidate == "revoke":
        await admission.revoke("cli:source")
    elif invalidate == "retire":
        admission.retire(context)
    elif invalidate == "run":
        context.run_id = "replacement-run"
    elif invalidate == "cancel":
        context.cancelled = True
    else:
        monkeypatch.setattr("nanobot.agent.turn_admission.time.time", lambda: float("inf"))
    decision = await policy.evaluate(messages=[], tool_calls=[risky_call("refused")])
    assert decision.stop_reason == "approval_required"


async def test_public_fingerprints_and_booleans_cannot_manufacture_policy_authority(tmp_path):
    _, _, original = await issued_policy(tmp_path)
    policy = RiskyActionPolicy(workspace=tmp_path, approval_granted=True,
        approved_effect_ids=frozenset({original.effect_authority.effect_id}))
    assert (await policy.evaluate(messages=[], tool_calls=[risky_call("forged")])).stop_reason == "approval_required"


async def test_cancelled_dispatch_does_not_restore_authority(tmp_path):
    admission, context, policy = await issued_policy(tmp_path)
    class CancelExec(CountExec):
        async def execute(self, command, **kwargs):
            self.count += 1
            raise asyncio.CancelledError()
    counter = CancelExec()
    registry = ToolRegistry()
    registry.register(counter)
    with pytest.raises(asyncio.CancelledError):
        await execute_tool_calls(registry, [risky_call("cancelled")], concurrent=False,
            external_lookup_counts={}, workspace_violation_counts={}, hook=AgentHook(),
            context=AgentHookContext(iteration=0, messages=[]), tool_policy=policy,
            turn_context=context)
    assert counter.count == 1
    assert (await policy.evaluate(messages=[], tool_calls=[risky_call("again")])).stop_reason == "approval_required"
    admission.retire(context)


async def test_snapshot_expiring_while_waiting_cannot_issue_authority(tmp_path, monkeypatch):
    admission = ApprovalAdmission()
    policy = RiskyActionPolicy(workspace=tmp_path)
    decision = await policy.evaluate(messages=[], tool_calls=[risky_call("blocked")])
    pending = await admission.store(session_key="cli:source", channel="cli", chat_id="original",
        summary="push", effect_id=decision.metadata["effect_id"], run_id="blocked")
    snapshot = await admission.snapshot_decision("cli:original", ApprovalDecision(
        approval_id=pending.approval_id, decision="approve"))
    monkeypatch.setattr("nanobot.agent.turn_admission.time.time", lambda: pending.expires_at + 1)
    context = TurnContext(run_id="resume", session_key="cli:source")
    assert await admission.consume(snapshot, context=context) == "stale"
    assert context.policy.effect_authority is None


async def test_restart_does_not_reconstruct_saved_approval(engine_config):
    first_provider = ScriptedProvider([LLMResponse(content=None, tool_calls=[risky_call("blocked")])])
    first = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=first_provider)
    upgrade(first)
    try:
        await first.execute_turn(TurnRequest(content="risky", session_key="cli:source",
            route=DeliveryTarget(channel="cli", chat_id="original")))
        pending = (await first.approval_admission.snapshot_decision("cli:source", None)).pending
    finally:
        await first.aclose()
    provider = ScriptedProvider([])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        result = await approve(loop, pending, session_key="cli:source")
        assert result.stop_reason == "approval_stale"
        assert provider.calls == []
    finally:
        await loop.aclose()


async def test_saved_pending_copy_cannot_issue_authority(tmp_path):
    admission = ApprovalAdmission()
    policy = RiskyActionPolicy(workspace=tmp_path)
    decision = await policy.evaluate(messages=[], tool_calls=[risky_call("blocked")])
    pending = await admission.store(session_key="cli:source", channel="cli", chat_id="original",
        summary="push", effect_id=decision.metadata["effect_id"], run_id="blocked")
    snapshot = await admission.snapshot_decision("cli:original", ApprovalDecision(
        approval_id=pending.approval_id, decision="approve"))
    context = TurnContext(run_id="resume", session_key="cli:source")
    assert await admission.consume(replace(snapshot, pending=replace(pending)), context=context) == "stale"
    assert context.policy.effect_authority is None
    assert await admission.consume(snapshot, context=context) == "affirmative"
    admission.retire(context)


async def test_preparation_failure_does_not_spend_effect_attempt(tmp_path, monkeypatch):
    admission, context, policy = await issued_policy(tmp_path)
    registry = ToolRegistry()
    counter = CountExec()
    registry.register(counter)
    prepare = registry.prepare_call
    fail = True
    def prepare_once(name, arguments):
        nonlocal fail
        tool, params, error = prepare(name, arguments)
        if fail:
            fail = False
            return tool, params, "Error: preparation refused"
        return tool, params, error
    monkeypatch.setattr(registry, "prepare_call", prepare_once)
    try:
        for index in range(2):
            _, events = await execute_tool_calls(registry, [risky_call(f"attempt-{index}")],
                concurrent=False, external_lookup_counts={}, workspace_violation_counts={},
                hook=AgentHook(), context=AgentHookContext(iteration=0, messages=[]),
                tool_policy=policy, turn_context=context)
            assert events[0]["status"] == ("error" if index == 0 else "ok")
            assert counter.count == index
    finally:
        admission.retire(context)


async def test_live_revocation_after_preparation_refuses_dispatch(tmp_path):
    admission, context, policy = await issued_policy(tmp_path)
    registry = ToolRegistry()
    counter = CountExec()
    registry.register(counter)
    class RevokeHook(AgentHook):
        async def before_execute_tool(self, context, tool_call, tool, params):
            await admission.revoke("cli:source")
    results, _ = await execute_tool_calls(registry, [risky_call("revoked")], concurrent=False,
        external_lookup_counts={}, workspace_violation_counts={}, hook=RevokeHook(),
        context=AgentHookContext(iteration=0, messages=[]), tool_policy=policy,
        turn_context=context)
    assert results[0].stop_reason == "approval_required"
    assert counter.count == 0


async def test_effect_authority_cannot_transfer_to_another_run_context(tmp_path):
    admission, context, policy = await issued_policy(tmp_path)
    registry = ToolRegistry()
    counter = CountExec()
    registry.register(counter)
    results, _ = await execute_tool_calls(registry, [risky_call("cloned-context")], concurrent=False,
        external_lookup_counts={}, workspace_violation_counts={}, hook=AgentHook(),
        context=AgentHookContext(iteration=0, messages=[]), tool_policy=policy,
        turn_context=replace(context))
    assert results[0].stop_reason == "approval_required"
    assert counter.count == 0
    assert policy.effect_authority.permits(policy.effect_authority.effect_id)
    admission.retire(context)


@pytest.mark.parametrize("replacement", ["reset", "delete"])
async def test_legacy_source_replacement_invalidates_pending_consent(engine_config, replacement):
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[risky_call("blocked")]),
        LLMResponse(content=None, tool_calls=[risky_call("must-not-dispatch")]),
        LLMResponse(content="done"),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    counter = CountExec()
    loop.tools.replace(counter)
    try:
        await loop.execute_turn(TurnRequest(content="risky", session_key="cli:source",
            route=DeliveryTarget(channel="cli", chat_id="original")))
        pending = (await loop.approval_admission.snapshot_decision("cli:original", None)).pending
        if replacement == "reset":
            loop.sessions.reset_session("cli:source")
        else:
            loop.sessions.delete_session("cli:source")
        result = await approve(loop, pending)
        assert result.stop_reason == "approval_stale"
        assert counter.count == 0
        assert len(provider.calls) == 1
    finally:
        await loop.aclose()
