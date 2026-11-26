"""Linux parity for executor admission (A3-2M, WP1).

These tables freeze what the managed executor decides for every combination of
bubblewrap and whole-tree supervisor availability. They were written against
the code *before* the capability seam and must keep passing unchanged after it,
so the macOS work cannot alter a Linux decision or reason code (decision D-M2).

The gates are patched at the module names the production code and the older
suites already patch, so this file also guards that those patch points keep
working.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kohaku.agent.tools.shell import ExecTool
from kohaku.context_repo import ContextRepoManager, ResourceAccessPolicy
from kohaku.context_repo.catalog import RepositoryCatalog
from kohaku.security.folder_grants import FolderGrantStore
from kohaku.session.manager import SessionManager
from kohaku.session.workspace_focus import WorkspaceFocusOwner

CHAT = "websocket:chat-parity"
BINDING = {"store_identity": "store-par", "incarnation": "inc-p", "generation": 1}


def _force_gates(monkeypatch, *, bwrap: bool, supervisor: bool, linux: bool = True) -> None:
    for module in ("sandbox", "shell"):
        monkeypatch.setattr(f"kohaku.agent.tools.{module}.managed_bwrap_available", lambda: bwrap)
    for module in ("native_process", "shell"):
        monkeypatch.setattr(
            f"kohaku.agent.tools.{module}.native_supervisor_available", lambda: supervisor,
        )
    monkeypatch.setattr("kohaku.session.workspace_focus._linux_host", lambda: linux)


def _build(tmp_path):
    agent_home = tmp_path / "agent-home"
    agent_home.mkdir(parents=True)
    instance = tmp_path / "instance"
    instance.mkdir()
    manager = ContextRepoManager.from_config(context_repos=[])
    grants = FolderGrantStore(
        agent_home=agent_home, instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    ).view
    policy = ResourceAccessPolicy(
        workspace=tmp_path, context_manager=manager, folder_grants=grants,
    )
    sessions = SessionManager(agent_home, sessions_root=tmp_path / "sessions")
    owner = WorkspaceFocusOwner(
        sessions=sessions, agent_home=agent_home, grants=grants, policy=policy,
        catalog=RepositoryCatalog(instance, runtime_root=instance),
    )
    return owner, sessions, agent_home


def _commands(access) -> tuple[bool, str, str | None]:
    commands = access["commands"]
    return commands["available"], commands["backend"], commands["reason"]


@pytest.mark.parametrize(
    ("supervisor", "expected"),
    [
        (True, (True, "native", None)),
        (False, (False, "none", "supervisor_unavailable")),
    ],
)
def test_full_access_commands_follow_the_supervisor_only(
    tmp_path, monkeypatch, supervisor, expected,
) -> None:
    # Full access is unconfined, so bubblewrap availability must not matter.
    for bwrap in (True, False):
        _force_gates(monkeypatch, bwrap=bwrap, supervisor=supervisor)
        owner, sessions, _home = _build(tmp_path / f"full-{supervisor}-{bwrap}")
        sessions.desktop_permissions.enable_full(
            session_key=CHAT, source_binding=dict(BINDING), persistent=True, existing=True,
            local=True, idle=True, configured_sandbox=False, supervisor_available=True,
            stores_healthy=True,
        )
        reference = sessions.desktop_permissions.full_reference(
            session_key=CHAT, source_binding=dict(BINDING), request_turn_id="turn-1",
        )
        project = tmp_path / f"project-{supervisor}-{bwrap}"
        project.mkdir()

        access = owner.effective_access(project, permission_reference=reference)

        assert access["source"] == "full"
        assert _commands(access) == expected


@pytest.mark.parametrize(
    ("bwrap", "supervisor", "expected"),
    [
        (True, True, (True, "bwrap", None)),
        (True, False, (False, "none", "supervisor_unavailable")),
        (False, True, (False, "none", "sandbox_unavailable")),
        (False, False, (False, "none", "supervisor_unavailable")),
    ],
)
def test_once_commands_need_bwrap_and_supervisor(
    tmp_path, monkeypatch, bwrap, supervisor, expected,
) -> None:
    _force_gates(monkeypatch, bwrap=bwrap, supervisor=supervisor)
    owner, sessions, _home = _build(tmp_path)
    root = tmp_path / "task-root"
    (root / "sub").mkdir(parents=True)
    reference = sessions.desktop_permissions.activate_once(
        session_key=CHAT, source_binding=dict(BINDING), task_turn_id="turn-once",
        task_input_event_id="input-once", path=root, file_access="read_write", commands=True,
    )

    access = owner.effective_access(root / "sub", permission_reference=reference)

    assert access["source"] == "task"
    assert _commands(access) == expected


@pytest.mark.parametrize(
    ("bwrap", "linux", "expected"),
    [
        (True, True, (True, "bwrap", None)),
        (True, False, (True, "bwrap", None)),
        (False, True, (False, "none", "sandbox_unavailable")),
        (False, False, (False, "none", "unsupported_platform")),
    ],
)
def test_sandbox_commands_follow_bwrap_and_report_platform(
    tmp_path, monkeypatch, bwrap, linux, expected,
) -> None:
    _force_gates(monkeypatch, bwrap=bwrap, supervisor=True, linux=linux)
    owner, _sessions, home = _build(tmp_path)

    access = owner.effective_access(home)

    assert access["policy"] == "managed"
    assert _commands(access) == expected


_GRANTS_MANAGED = SimpleNamespace(is_managed=True)
_GRANTS_NATIVE = SimpleNamespace(is_managed=False)
_ADMISSION = "Error: native command admission is unavailable"
_PROOF = "Error: whole-tree native process proof is unavailable"


@pytest.mark.parametrize(
    ("supervisor", "grants", "managed", "owned", "expected"),
    [
        # No grant policy: legacy launch, nothing to prove.
        (True, None, False, False, None),
        (False, None, False, False, None),
        # An owned (Full) launch always needs admission and the supervisor.
        (True, None, False, True, _ADMISSION),
        (True, _GRANTS_NATIVE, False, True, None),
        (False, _GRANTS_NATIVE, False, True, _PROOF),
        (False, _GRANTS_MANAGED, True, True, _PROOF),
        # Un-owned native launch under a non-managed grant view needs the proof.
        (True, _GRANTS_NATIVE, False, False, None),
        (False, _GRANTS_NATIVE, False, False, _PROOF),
        # Managed launches and managed grant views never need the supervisor here.
        (False, _GRANTS_NATIVE, True, False, None),
        (False, _GRANTS_MANAGED, False, False, None),
        (False, _GRANTS_MANAGED, True, False, None),
    ],
)
def test_native_command_refusal_table(
    monkeypatch, supervisor, grants, managed, owned, expected,
) -> None:
    _force_gates(monkeypatch, bwrap=True, supervisor=supervisor)

    result = ExecTool._native_command_refusal(grants, managed=managed, owned=owned)

    assert (None if result is None else str(result)) == expected


@pytest.mark.parametrize(
    ("supervisor", "expected"), [(True, "native-supervisor"), (False, "process-tree")],
)
def test_process_ownership_identity(monkeypatch, supervisor, expected) -> None:
    _force_gates(monkeypatch, bwrap=True, supervisor=supervisor)

    assert ExecTool._process_ownership_identity() == expected

@pytest.mark.parametrize(
    ("platform", "bwrap", "full_expected", "once_expected"),
    [
        (
            "linux", True,
            (True, "native", None), (True, "bwrap", None),
        ),
        (
            "linux", False,
            (True, "native", None), (False, "none", "sandbox_unavailable"),
        ),
        (
            "darwin", False,
            (False, "none", "supervisor_unavailable"),
            (False, "none", "supervisor_unavailable"),
        ),
    ],
)
def test_projection_through_the_real_capability_seam(
    tmp_path, monkeypatch, platform, bwrap, full_expected, once_expected,
) -> None:
    """Force only the platform; the real wrappers must decide as before."""
    from kohaku.agent.tools import executor_capabilities

    monkeypatch.setattr(executor_capabilities, "current_platform", lambda: platform)
    monkeypatch.setattr("kohaku.agent.tools.sandbox.managed_bwrap_available", lambda: bwrap)
    monkeypatch.setattr("kohaku.session.workspace_focus._linux_host", lambda: platform == "linux")
    owner, sessions, _home = _build(tmp_path)
    sessions.desktop_permissions.enable_full(
        session_key=CHAT, source_binding=dict(BINDING), persistent=True, existing=True,
        local=True, idle=True, configured_sandbox=False, supervisor_available=True,
        stores_healthy=True,
    )
    full_reference = sessions.desktop_permissions.full_reference(
        session_key=CHAT, source_binding=dict(BINDING), request_turn_id="turn-1",
    )
    project = tmp_path / "project"
    project.mkdir()

    full = owner.effective_access(project, permission_reference=full_reference)

    assert (full["source"], _commands(full)) == ("full", full_expected)

    root = tmp_path / "task-root"
    (root / "sub").mkdir(parents=True)
    once_reference = sessions.desktop_permissions.activate_once(
        session_key=CHAT, source_binding=dict(BINDING), task_turn_id="turn-once",
        task_input_event_id="input-once", path=root, file_access="read_write", commands=True,
    )

    once = owner.effective_access(root / "sub", permission_reference=once_reference)

    assert (once["source"], _commands(once)) == ("task", once_expected)
