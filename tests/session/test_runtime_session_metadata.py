"""Runtime session metadata contracts for absent, valid, stale, and malformed facts."""

from typing import Literal

import pytest

from nanobot.session.manager import SessionManager
from nanobot.session.runtime_authority import (
    RUNTIME_SESSION_METADATA_KEY,
    RUNTIME_SESSION_METADATA_VERSION,
    RuntimeSessionMetadata,
    RuntimeSessionMetadataError,
    read_runtime_session,
    write_runtime_session,
)


def manager_for(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return SessionManager(workspace, sessions_root=tmp_path / "sessions")


def test_absent_runtime_metadata_uses_main_native_defaults(tmp_path):
    manager = manager_for(tmp_path)

    state = read_runtime_session(manager, "websocket:absent")
    assert state.metadata == RuntimeSessionMetadata()
    assert state.metadata.profile == "default"
    assert state.metadata.execution_kind == "native"
    assert state.metadata.delegation_depth == 0
    assert state.metadata.parent_key is None
    assert state.metadata.parent_incarnation is None
    assert state.metadata.runtime_generation == 0
    assert state.metadata.recovery_disposition == "none"
    assert state.present is False
    assert state.executable is True

    session = manager.get_or_create("websocket:plain")
    session.add_message("user", "history")
    manager.save(session)
    state = read_runtime_session(manager, "websocket:plain")
    assert state.metadata == RuntimeSessionMetadata()
    assert state.present is False
    assert state.executable is True


def test_runtime_metadata_round_trips_and_rejects_stale_parent(tmp_path):
    manager = manager_for(tmp_path)
    metadata = RuntimeSessionMetadata(
        profile="worker",
        execution_kind="external",
        delegation_depth=2,
        backend_binding="backend-a",
        runtime_generation=5,
    )
    assert write_runtime_session(manager, "worker:round-trip", metadata, create_if_absent=True) is True

    state = read_runtime_session(manager, "worker:round-trip")
    assert state.metadata == metadata
    assert state.present is True
    assert state.executable is True

    parent_facts = RuntimeSessionMetadata(profile="parent", runtime_generation=1)
    write_runtime_session(manager, "parent:main", parent_facts, create_if_absent=True)
    parent = manager.read_session_snapshot("parent:main")
    child_facts = RuntimeSessionMetadata(
        profile="worker",
        execution_kind="external",
        delegation_depth=1,
        parent_key="parent:main",
        parent_incarnation=parent.incarnation,
    )
    write_runtime_session(manager, "worker:child", child_facts, create_if_absent=True)
    child = manager.get_or_create("worker:child")
    child.add_message("user", "child history")
    manager.save(child)
    assert read_runtime_session(manager, "worker:child").executable is True

    with pytest.raises(RuntimeSessionMetadataError):
        write_runtime_session(
            manager,
            "worker:orphan",
            RuntimeSessionMetadata(
                parent_key="parent:missing",
                parent_incarnation=parent.incarnation,
            ),
            create_if_absent=True,
        )

    manager.delete_session("parent:main")
    write_runtime_session(
        manager,
        "parent:main",
        RuntimeSessionMetadata(profile="parent", runtime_generation=2),
        create_if_absent=True,
    )
    recreated = manager.read_session_snapshot("parent:main")
    assert recreated.incarnation != parent.incarnation

    with pytest.raises(RuntimeSessionMetadataError):
        write_runtime_session(manager, "worker:stale", child_facts, create_if_absent=True)

    stale = read_runtime_session(manager, "worker:child")
    assert stale.present is True
    assert stale.executable is False
    assert [row["content"] for row in manager.read_session_snapshot("worker:child").messages] == [
        "child history"
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"version": RUNTIME_SESSION_METADATA_VERSION, "profile": ""},
        {"version": RUNTIME_SESSION_METADATA_VERSION + 1, "profile": "future"},
        "not-a-mapping",
    ],
)
def test_malformed_runtime_metadata_keeps_history_readable(tmp_path, payload):
    manager = manager_for(tmp_path)
    key = "websocket:malformed"
    session = manager.get_or_create(key)
    session.add_message("user", "retained history")
    manager.save(session)
    manager.upsert_session_metadata(key, {RUNTIME_SESSION_METADATA_KEY: payload})

    state = read_runtime_session(manager, key)
    assert state.present is True
    assert state.executable is False
    assert state.issue is not None
    assert [row["content"] for row in manager.read_session_snapshot(key).messages] == [
        "retained history"
    ]


@pytest.mark.parametrize(
    "disposition",
    ["none", "interrupted", "completed", "uncertain"],
)
def test_recovery_disposition_round_trips(
    tmp_path, disposition: Literal["none", "interrupted", "completed", "uncertain"]
):
    manager = manager_for(tmp_path)
    key = f"websocket:recovery-{disposition}"
    metadata = RuntimeSessionMetadata(recovery_disposition=disposition)

    assert write_runtime_session(manager, key, metadata, create_if_absent=True) is True
    state = read_runtime_session(manager, key)
    assert state.metadata.recovery_disposition == disposition
    assert state.present is True
    assert state.executable is True
