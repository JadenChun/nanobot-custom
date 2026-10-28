"""Process-local workspace permission references are source and task bound."""

from __future__ import annotations

import pickle
import re

import pytest

from nanobot.security.desktop_permissions import (
    DesktopPermissionOwner,
    WorkspacePermissionConflictError,
    WorkspacePermissionUnavailableError,
)
from nanobot.session.manager import SessionManager
from nanobot.webui.workspaces import WebUIWorkspaceController
from nanobot.workspace_access_recovery import parse_workspace_access_once_payload


def _source(generation: int = 0) -> dict[str, object]:
    return {
        "store_identity": "store-a",
        "incarnation": "incarnation-a",
        "generation": generation,
    }


def test_session_manager_composes_one_permission_owner(tmp_path) -> None:
    home = tmp_path / "workspace"
    home.mkdir()
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    controller = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=home,
        default_restrict_to_workspace=True,
    )

    assert controller.permission_owner is sessions.desktop_permissions
    assert re.fullmatch(r"[0-9a-f]{32}", controller.permission_owner.runtime_id)


def test_once_payload_requires_exact_capabilities_and_revision_preconditions() -> None:
    payload = {
        "chat_id": "chat-a",
        "access_request_id": "0" * 32,
        "expected_revision": 1,
        "path": "/projects/alpha",
        "file_access": "read_only",
        "commands": False,
        "expected_grants_revision": 0,
        "expected_runtime_id": "1" * 32,
        "expected_permission_revision": 2,
    }
    assert parse_workspace_access_once_payload(payload) == payload
    for changed in (
        {**payload, "commands": 0},
        {**payload, "expected_permission_revision": True},
        {**payload, "unexpected": "authority"},
    ):
        with pytest.raises(ValueError, match="invalid workspace access once payload"):
            parse_workspace_access_once_payload(changed)


def test_projection_revision_tracks_source_lifetime_without_chat_switch_churn() -> None:
    owner = DesktopPermissionOwner()
    first_key = "websocket:chat-a"
    second_key = "websocket:chat-b"
    first_source = _source()
    first = owner.projection(session_key=first_key, source_binding=first_source)
    assert owner.projection(session_key=first_key, source_binding=first_source)["revision"] == (
        first["revision"]
    )

    # First observation of an unchanged second chat and switching back do not
    # invalidate an existing confirmation.
    second_source = _source()
    assert owner.projection(session_key=second_key, source_binding=second_source)["revision"] == (
        first["revision"]
    )
    assert owner.projection(session_key=first_key, source_binding=first_source)["revision"] == (
        first["revision"]
    )

    for changed_source in (
        {**first_source, "generation": 1},
        {**first_source, "incarnation": "incarnation-b"},
        {**first_source, "store_identity": "store-b"},
    ):
        prior_revision = owner.revision
        changed = owner.projection(session_key=first_key, source_binding=changed_source)
        assert changed["revision"] == prior_revision + 1
        assert owner.projection(
            session_key=first_key,
            source_binding=changed_source,
        )["revision"] == changed["revision"]
        first_source = changed_source


def test_once_reference_is_private_to_original_task_and_source(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    owner = DesktopPermissionOwner()
    source = _source()

    reference = owner.activate_once(
        session_key="websocket:chat-a",
        source_binding=source,
        task_turn_id="turn-original",
        task_input_event_id="event-original",
        path=root,
        file_access="read_only",
        commands=False,
    )

    assert re.fullmatch(r"[0-9a-f]{32}", owner.runtime_id)
    assert owner.authorizes_once(
        reference,
        session_key="websocket:chat-a",
        source_binding=source,
        task_turn_id="turn-original",
        task_input_event_id="event-original",
    )
    assert not owner.authorizes_once(
        reference,
        session_key="websocket:chat-a",
        source_binding=source,
        task_turn_id="turn-next",
        task_input_event_id="event-next",
    )
    assert not owner.authorizes_once(
        reference,
        session_key="websocket:chat-b",
        source_binding=source,
        task_turn_id="turn-original",
        task_input_event_id="event-original",
    )
    assert not owner.authorizes_once(
        reference,
        session_key="websocket:chat-a",
        source_binding=_source(generation=1),
        task_turn_id="turn-original",
        task_input_event_id="event-original",
    )
    with pytest.raises(TypeError):
        pickle.dumps(reference)


def test_once_roots_accumulate_only_after_exact_owner_update(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    owner = DesktopPermissionOwner()
    kwargs = {
        "session_key": "websocket:chat-a",
        "source_binding": _source(),
        "task_turn_id": "turn-original",
        "task_input_event_id": "event-original",
    }

    first_reference = owner.activate_once(
        **kwargs,
        path=first,
        file_access="read_only",
        commands=False,
    )
    second_reference = owner.activate_once(
        **kwargs,
        path=second,
        file_access="read_write",
        commands=True,
    )

    assert owner.once_roots(second_reference) == (
        (str(first.resolve()), "read_only", False),
        (str(second.resolve()), "read_write", True),
    )
    assert not owner.authorizes_once(first_reference, **kwargs)
    assert owner.authorizes_once(second_reference, **kwargs)


def test_failed_once_source_commit_leaves_no_private_authority(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    owner = DesktopPermissionOwner()
    kwargs = {
        "session_key": "websocket:chat-a",
        "source_binding": _source(),
        "task_turn_id": "turn-original",
        "task_input_event_id": "event-original",
    }

    def fail_commit(_epoch: int) -> None:
        raise RuntimeError("source compare-and-swap failed")

    with pytest.raises(RuntimeError, match="compare-and-swap"):
        owner.commit_once(
            **kwargs,
            path=root,
            file_access="read_only",
            commands=False,
            persist=fail_commit,
        )

    assert owner.task_reference(**kwargs) is None
    assert owner.revision == 0


def test_full_disable_fences_future_work_and_waits_for_owned_process_proof() -> None:
    owner = DesktopPermissionOwner()
    source = _source()
    kwargs = {
        "session_key": "websocket:chat-a",
        "source_binding": source,
    }
    reference = owner.enable_full(
        **kwargs,
        persistent=True,
        existing=True,
        local=True,
        idle=True,
        configured_sandbox=False,
        supervisor_available=True,
        stores_healthy=True,
    )

    with pytest.raises(WorkspacePermissionConflictError, match="native_processes_unknown"):
        owner.disable_full(
            **kwargs,
            active_turn=False,
            active_processes=0,
            unknown_processes=1,
        )
    assert owner.full_reference(**kwargs) is None
    assert not owner.authorizes_full(reference, **kwargs)
    assert owner.projection(**kwargs)["mode"] == "full"
    assert owner.projection(**kwargs)["state"] == "retiring"
    with pytest.raises(WorkspacePermissionConflictError, match="native_processes_unknown"):
        owner.disable_full(**kwargs, active_turn=False, active_processes=0, unknown_processes=1)
    assert owner.disable_full(
        **kwargs,
        active_turn=False,
        active_processes=0,
        unknown_processes=0,
    )
    assert owner.projection(**kwargs)["mode"] == "restricted"


def test_full_retirement_activity_observation_changes_only_exact_existing_retirement() -> None:
    owner = DesktopPermissionOwner()
    source = _source()
    kwargs = {
        "session_key": "websocket:chat-retirement-activity",
        "source_binding": source,
    }
    reference = owner.enable_full(
        **kwargs,
        persistent=True,
        existing=True,
        local=True,
        idle=True,
        configured_sandbox=False,
        supervisor_available=True,
        stores_healthy=True,
    )
    live_counts = [1]
    assert owner.register_native_stdin(
        reference,
        object(),
        lambda: True,
        lambda: (live_counts[0], 0),
    )

    with pytest.raises(WorkspacePermissionConflictError, match="native_processes_active"):
        owner.disable_full(
            **kwargs,
            active_turn=True,
            active_processes=0,
            unknown_processes=0,
            process_counts=lambda: (live_counts[0], 0),
        )
    initial = owner.projection(**kwargs)
    assert initial["state"] == "retiring"
    assert initial["retirement"] == {
        "reason": "mode_changed",
        "active_turn": True,
        "active_processes": 1,
        "unknown_processes": 0,
    }
    assert not owner.authorizes_full(reference, **kwargs)

    observe = getattr(owner, "observe_full_retirement_activity")
    assert observe(**kwargs, active_turn=False) is True
    observed = owner.projection(**kwargs)
    assert observed["mode"] == "full"
    assert observed["state"] == "retiring"
    assert observed["retirement"] == {
        "reason": "mode_changed",
        "active_turn": False,
        "active_processes": 1,
        "unknown_processes": 0,
    }
    revision = owner.revision
    assert observe(**kwargs, active_turn=False) is True
    assert owner.revision == revision

    new_source = _source(generation=1)
    assert observe(
        session_key=kwargs["session_key"],
        source_binding=new_source,
        active_turn=False,
    ) is False
    prior_revision = owner.revision
    assert owner.projection(
        session_key=kwargs["session_key"],
        source_binding=new_source,
    )["state"] == "active"
    assert owner.revision == prior_revision + 1


def test_full_requires_explicit_available_local_mode_action_and_is_volatile(tmp_path) -> None:
    owner = DesktopPermissionOwner()
    source = _source()
    kwargs = {
        "session_key": "websocket:chat-a",
        "source_binding": source,
    }

    with pytest.raises(WorkspacePermissionUnavailableError) as unavailable:
        owner.enable_full(
            **kwargs,
            persistent=True,
            existing=True,
            local=False,
            idle=True,
            configured_sandbox=False,
            supervisor_available=True,
            stores_healthy=True,
        )
    assert unavailable.value.reason == "session_required"
    assert owner.full_reference(**kwargs) is None

    reference = owner.enable_full(
        **kwargs,
        persistent=True,
        existing=True,
        local=True,
        idle=True,
        configured_sandbox=False,
        supervisor_available=True,
        stores_healthy=True,
    )
    assert owner.authorizes_full(reference, **kwargs)

    owner.restart()
    assert owner.full_reference(**kwargs) is None
    assert not owner.authorizes_full(reference, **kwargs)


def test_restart_residuals_project_retiring_without_recreating_authority() -> None:
    owner = DesktopPermissionOwner()
    source = _source()
    task = {
        "session_key": "websocket:chat-a",
        "source_binding": source,
        "task_turn_id": "turn-a",
        "task_input_event_id": "event-a",
    }
    owner.observe_full_residual_counts(
        session_key=task["session_key"],
        source_binding=source,
        active_processes=1,
        unknown_processes=0,
    )
    full_projection = owner.projection(
        session_key=task["session_key"], source_binding=source,
    )
    assert full_projection["mode"] == "full"
    assert full_projection["state"] == "retiring"
    assert full_projection["retirement"] == {
        "reason": "restart",
        "active_turn": False,
        "active_processes": 1,
        "unknown_processes": 0,
    }
    assert owner.full_reference(
        session_key=task["session_key"], source_binding=source,
    ) is None
    with pytest.raises(WorkspacePermissionUnavailableError) as unavailable:
        owner.enable_full(
            session_key=task["session_key"],
            source_binding=source,
            persistent=True,
            existing=True,
            local=True,
            idle=True,
            configured_sandbox=False,
            supervisor_available=True,
            stores_healthy=True,
        )
    assert unavailable.value.reason == "permission_retiring"

    owner.observe_full_residual_counts(
        session_key=task["session_key"],
        source_binding=source,
        active_processes=0,
        unknown_processes=0,
    )
    task_projection = owner.projection(**task)
    assert task_projection["mode"] == "restricted"
    assert task_projection["task_access"]["state"] == "expired"

    owner.observe_task_residual_counts(
        **task,
        active_processes=0,
        unknown_processes=1,
    )
    task_projection = owner.projection(**task)
    assert task_projection["task_access"]["state"] == "retiring"
    assert task_projection["retirement"]["reason"] == "restart"
    assert owner.task_reference(**task) is None

    owner.observe_task_residual_counts(
        **task,
        active_processes=0,
        unknown_processes=0,
    )
    assert owner.projection(**task)["task_access"]["state"] == "expired"


@pytest.mark.parametrize(
    ("counts", "expected"),
    [((1, 0), "native_processes_active"), ((0, 1), "native_processes_unknown")],
)
def test_once_reauthorization_waits_for_prior_retirement_receipt(
    tmp_path,
    counts: tuple[int, int],
    expected: str,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    owner = DesktopPermissionOwner()
    kwargs = {
        "session_key": "websocket:chat-a",
        "source_binding": _source(),
        "task_turn_id": "turn-a",
        "task_input_event_id": "event-a",
    }
    reference = owner.activate_once(
        **kwargs,
        path=root,
        file_access="read_only",
        commands=False,
    )
    observed = [counts]
    owner.retire_once(
        **kwargs,
        reason="restart",
        active_turn=False,
        active_processes=0,
        unknown_processes=0,
        process_counts=lambda: observed[0],
    )
    assert not owner.is_current_task_reference(reference)
    persisted: list[int] = []
    with pytest.raises(WorkspacePermissionConflictError, match=expected):
        owner.commit_once(
            **kwargs,
            path=root,
            file_access="read_only",
            commands=False,
            persist=persisted.append,
            process_counts=lambda: observed[0],
        )
    assert persisted == []
    retirement = owner.projection(**kwargs)
    assert retirement["task_access"]["state"] == "retiring"
    assert retirement["retirement"]["active_processes"] == counts[0]
    assert retirement["retirement"]["unknown_processes"] == counts[1]

    observed[0] = (0, 0)
    replacement = owner.commit_once(
        **kwargs,
        path=root,
        file_access="read_only",
        commands=False,
        persist=persisted.append,
        process_counts=lambda: observed[0],
    )
    assert persisted == [replacement.epoch]
    assert owner.is_current_task_reference(replacement)
    assert owner.projection(**kwargs)["task_access"]["state"] == "active"
