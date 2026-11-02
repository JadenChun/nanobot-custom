from __future__ import annotations

import pytest

from nanobot.remote.authority import RemoteAuthorityCeiling


def projection(**overrides):
    value = {
        "version": 1,
        "runtime_id": "a" * 32,
        "revision": 4,
        "mode": "restricted",
        "state": "active",
        "can_enable_full": True,
        "full_access_unavailable_reason": None,
        "task_access": {
            "task_turn_id": "turn-1",
            "state": "active",
            "folders": [],
        },
        "retirement": None,
    }
    value.update(overrides)
    return value


def test_restricted_task_allows_only_bounded_actions() -> None:
    ceiling = RemoteAuthorityCeiling()
    allowed = ceiling.qualify(projection(), action="chat.send", task_turn_id="turn-1")
    assert allowed.allowed is True
    assert allowed.reason == "restricted_task"
    assert allowed.task_turn_id == "turn-1"

    for action in (
        "settings.write",
        "system.exec",
        "workspace.add_folder",
        "folder.pick",
        "automation.create",
        "skill.install",
        "package.install",
        "temporary.create",
        "delete.session",
        "native.full",
        "shell.run",
        "unknown.action",
    ):
        decision = ceiling.qualify(projection(), action=action)
        assert decision.allowed is False
        assert decision.reason in {"action_not_remote", "action_not_allowed"}


def test_full_unmanaged_retiring_and_stale_tasks_are_denied() -> None:
    ceiling = RemoteAuthorityCeiling()
    assert ceiling.qualify(None, action="chat.send").reason == "unmanaged_installation"
    assert ceiling.qualify(projection(mode="full"), action="chat.send").reason == "local_full_not_remote"
    assert ceiling.qualify(projection(state="retiring"), action="chat.send").reason == "permission_retiring"
    assert ceiling.qualify(projection(task_access=None), action="chat.send").reason == "task_not_qualified"
    assert ceiling.qualify(
        projection(task_access={"task_turn_id": "turn-2", "state": "active", "folders": []}),
        action="chat.send",
        task_turn_id="turn-1",
    ).reason == "task_revision_stale"
    assert ceiling.qualify(projection(), action="chat.send").allowed is True
    with pytest.raises(PermissionError):
        ceiling.require(projection(mode="full"), action="chat.send")


def test_reads_are_authorized_when_restricted_but_send_requires_task() -> None:
    ceiling = RemoteAuthorityCeiling()
    no_task = projection(task_access=None)
    assert ceiling.qualify(no_task, action="chat.read").allowed is True
    assert ceiling.qualify(no_task, action="media.read").allowed is True
    assert ceiling.qualify(no_task, action="chat.send").allowed is False
    caps = ceiling.capabilities(no_task)
    assert caps["chat_read"] is True
    assert caps["media_read"] is True
    assert caps["chat_send"] is False
    assert caps["reason"] == "task_not_qualified"


def test_malicious_projection_fields_do_not_escalate() -> None:
    ceiling = RemoteAuthorityCeiling()
    value = projection()
    value["full_reference"] = {"token": "local-full"}
    value["task_access"]["folders"] = [{"path": "/", "file_access": "read_write", "commands": True}]
    decision = ceiling.qualify(value, action="chat.send", task_turn_id="turn-1")
    assert decision.allowed is True
    assert decision.reason == "restricted_task"
    assert ceiling.qualify(value, action="workspace.add_folder").allowed is False
