"""A stale execution fence recovers itself at acquisition without replaying tools."""

import pytest

from nanobot.session.execution_owner import ExecutionOwnershipError, acquire_execution
from nanobot.session.manager import JsonlSessionStore
from nanobot.session.source import new_source_event
from tests.session.test_execution_owner import seed


@pytest.fixture
def backend(tmp_path):
    return JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")


def fence(backend, key, *, lease: bool = False):
    snapshot = backend.source.load_source_snapshot(key)
    snapshot.header["metadata"]["execution_recovery_required"] = True
    if lease:
        snapshot.header["metadata"]["execution_lease"] = {
            "schema_version": 1,
            "token": "abandoned",
        }
    backend.source._write(snapshot.header, snapshot.events)


def add_unresolved_call(backend, key, *, tool="mcp_comfy_generate_image"):
    snapshot = backend.source.load_source_snapshot(key)
    call = {
        "id": "call-comfy",
        "type": "function",
        "function": {"name": tool, "arguments": "{}"},
    }
    event = new_source_event(
        {"role": "assistant", "content": "", "tool_calls": [call]},
        run_id="run-comfy",
    )
    event["_execution_ids"] = ["exec-comfy"]
    backend.source.append_source_events(key, snapshot.state, [event])


async def test_settled_fence_recovers_at_acquisition(backend):
    original = seed(backend)
    fence(backend, original.state.session_key)

    async with acquire_execution(backend, original.state.session_key, run_id="next", reconcile_stale=True) as owner:
        assert owner.recovery_required is False
        owner.validate_dispatch()

    snapshot = backend.source.load_source_snapshot(original.state.session_key)
    assert "execution_lease" not in snapshot.header["metadata"]
    assert "execution_recovery_required" not in snapshot.header["metadata"]


async def test_interrupted_comfy_call_is_recorded_and_does_not_block(backend):
    original = seed(backend)
    add_unresolved_call(backend, original.state.session_key)
    fence(backend, original.state.session_key, lease=True)

    async with acquire_execution(backend, original.state.session_key, run_id="next", reconcile_stale=True) as owner:
        assert owner.recovery_required is False
        owner.validate_dispatch()

    snapshot = backend.source.load_source_snapshot(original.state.session_key)
    assert "execution_recovery_required" not in snapshot.header["metadata"]
    assert "execution_lease" not in snapshot.header["metadata"]
    results = [event for event in snapshot.events if event.get("role") == "tool"]
    assert len(results) == 1
    assert results[0]["_execution_id"] == "exec-comfy"
    assert results[0]["tool_call_id"] == "call-comfy"
    assert "comfy_status(include_images=true)" in results[0]["content"]


async def test_structural_unresolved_call_keeps_the_fence(backend):
    original = seed(backend)
    add_unresolved_call(backend, original.state.session_key)
    snapshot = backend.source.load_source_snapshot(original.state.session_key)
    user = new_source_event(
        {"role": "user", "content": "next"}, run_id="run-next",
    )
    backend.source.append_source_events(
        original.state.session_key, snapshot.state, [user],
    )
    fence(backend, original.state.session_key, lease=True)

    async with acquire_execution(
        backend, original.state.session_key, run_id="next", reconcile_stale=True,
    ) as owner:
        assert owner.recovery_required is True
        with pytest.raises(ExecutionOwnershipError):
            owner.validate_dispatch()


async def test_default_acquisition_keeps_the_fence(backend):
    original = seed(backend)
    fence(backend, original.state.session_key)

    async with acquire_execution(backend, original.state.session_key, run_id="cron") as owner:
        assert owner.recovery_required is True
        with pytest.raises(ExecutionOwnershipError):
            owner.validate_dispatch()


async def test_checkpoint_keeps_the_fence(backend):
    original = seed(backend)
    fence(backend, original.state.session_key)
    backend.get_runtime_checkpoint_path(original.state.session_key).write_text(
        "{}", encoding="utf-8",
    )

    async with acquire_execution(backend, original.state.session_key, run_id="next", reconcile_stale=True) as owner:
        assert owner.recovery_required is True
        with pytest.raises(ExecutionOwnershipError):
            owner.validate_dispatch()

    snapshot = backend.source.load_source_snapshot(original.state.session_key)
    assert snapshot.header["metadata"]["execution_recovery_required"] is True
