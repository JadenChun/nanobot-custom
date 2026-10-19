"""Canonical follow-ups retain input without permitting unbound source writes."""

from dataclasses import replace

import pytest

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import SessionManager
from nanobot.session.pending_followups import (
    copy_pending_followup_proof,
    mint_pending_followup_id,
    parse_pending_followup_records,
    pending_followup_id,
    record_pending_followup,
)
from nanobot.session.persistence_metadata import SOURCE_PENDING_FOLLOWUPS_METADATA_KEY as JOURNAL
from nanobot.session.recovery import RecoveryCoordinator
from nanobot.session.source import SourceValidationError
from nanobot.session.turn_writer import SourceTurnWriter

KEY = "websocket:followups"


@pytest.fixture
def sessions(tmp_path):
    manager = SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    session = manager.get_or_create(KEY)
    manager.save(session)
    backend = manager._jsonl_store
    backend.source.upgrade_source_session(
        KEY, expected_revision=session.revision,
        expected_generation=session.generation, expected_incarnation=session.incarnation,
    )
    manager.invalidate(KEY)
    return manager


def message():
    msg = InboundMessage(
        channel="websocket", sender_id="webui", chat_id="followups", content="original input",
        media=["note.txt"], metadata={"webui": True, "remote": "private address",
                                     "_websocket_turn_owner": "private token",
                                     "_recovery_followup_id": "client-forged-id"},
    )
    mint_pending_followup_id(msg)
    return msg


async def test_admission_and_atomic_consumption_are_idempotent_and_bound(sessions):
    backend = sessions.source_backend(KEY)
    msg = message()
    record = record_pending_followup(msg)
    assert record.metadata == {"webui": True}
    assert record.identifier != msg.metadata["_recovery_followup_id"]
    cloned = replace(msg)
    assert pending_followup_id(cloned) is None
    copy_pending_followup_proof(msg, cloned)
    assert pending_followup_id(cloned) == record.identifier
    async with acquire_execution(backend, KEY, run_id="followup") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="followup")
        assert backend.source.admit_pending_followup(KEY, owner, record)
        assert backend.source.admit_pending_followup(KEY, owner, record)
        assert len(owner.read_snapshot().header["metadata"][JOURNAL]) == 1
        binding = writer.consume_pending_followup(record, {"role": "user", "content": "projection"})
        assert JOURNAL not in owner.read_snapshot().header["metadata"]
        assert len(owner.read_snapshot().events) == 1
        assert owner.read_snapshot().events[0]["content"] == "original input"
        assert writer.validate_injection_binding(binding) is binding
        assert not backend.source.admit_pending_followup(KEY, owner, record)
        assert writer.consume_pending_followup(record, binding.message).event_id == binding.event_id
        with pytest.raises(SourceValidationError):
            writer.validate_injection_binding(replace(binding))
        other = SourceTurnWriter(backend.source, owner, run_id="other")
        with pytest.raises(SourceValidationError):
            other.validate_injection_binding(binding)
        binding.message["content"] = "altered projection"
        with pytest.raises(SourceValidationError):
            writer.validate_injection_binding(binding)


async def test_failed_consumption_retains_journal_and_does_not_append(sessions, monkeypatch):
    backend = sessions.source_backend(KEY)
    async with acquire_execution(backend, KEY, run_id="failure") as owner:
        record = record_pending_followup(message())
        backend.source.admit_pending_followup(KEY, owner, record)
        writer = SourceTurnWriter(backend.source, owner, run_id="failure")
        before = owner.read_snapshot()
        with monkeypatch.context() as patch:
            def fail(*args, **kwargs):
                raise OSError("disk failure")
            patch.setattr(type(backend.source), "_write", fail)
            with pytest.raises(OSError, match="disk failure"):
                writer.consume_pending_followup(record, {"role": "user", "content": record.content})
        after = owner.read_snapshot()
        assert after.header == before.header and after.events == before.events
        assert writer.failed


async def test_generic_metadata_cannot_replace_or_remove_pending_input(sessions):
    backend = sessions.source_backend(KEY)
    async with acquire_execution(backend, KEY, run_id="reserved") as owner:
        record = record_pending_followup(message())
        backend.source.admit_pending_followup(KEY, owner, record)
        expected = owner.read_snapshot().header["metadata"][JOURNAL]
        backend.source.update_metadata(KEY, {JOURNAL: []}, removals=(JOURNAL,))
        assert owner.read_snapshot().header["metadata"][JOURNAL] == expected
        assert backend.source.discard_pending_followups(KEY, owner) == (record.identifier,)
        assert JOURNAL not in owner.read_snapshot().header["metadata"]


async def test_recovery_requeues_original_input_only_after_owner_releases(sessions):
    backend = sessions.source_backend(KEY)
    bus = MessageBus()
    coordinator = RecoveryCoordinator(sessions, bus)
    msg = message()
    async with acquire_execution(backend, KEY, run_id="recovery") as owner:
        backend.source.admit_pending_followup(KEY, owner, record_pending_followup(msg))
        await coordinator.scan()
        assert bus.inbound.empty()
    await coordinator.scan()
    restored = bus.inbound.get_nowait()
    assert restored.content == msg.content and restored.media == msg.media
    assert pending_followup_id(restored) == pending_followup_id(msg)
    assert restored.require_existing_session and restored.metadata == {"webui": True}


async def test_recovery_never_replays_uncertain_tool_effects(sessions):
    backend = sessions.source_backend(KEY)
    async with acquire_execution(backend, KEY, run_id="uncertain") as owner:
        backend.source.admit_pending_followup(KEY, owner, record_pending_followup(message()))
        writer = SourceTurnWriter(backend.source, owner, run_id="uncertain")
        writer.begin_tool_batch({"role": "assistant", "content": "", "tool_calls": [
            {"id": "effect", "type": "function", "function": {"name": "exec", "arguments": "{}"}},
        ]})
    bus = MessageBus()
    await RecoveryCoordinator(sessions, bus).scan()
    assert bus.inbound.empty()
    assert JOURNAL in backend.source.load_source_snapshot(KEY).header["metadata"]


@pytest.mark.parametrize("raw", [None, {}, [None], [{"id": "bad"}]])
def test_corrupt_journal_is_reported(raw):
    with pytest.raises(ValueError, match="source follow-up journal"):
        parse_pending_followup_records(raw)
