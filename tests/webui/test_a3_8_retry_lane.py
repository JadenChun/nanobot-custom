"""Regression tests for A3-8 AC8.2 and AC8.3 retry lanes."""

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.exec_session import ExecCompletion
from nanobot.bus.queue import MessageBus
from nanobot.process_completion import ProcessCompletionRuntime
from nanobot.security.folder_grants import FolderGrantStore
from nanobot.security.workspace_access import build_workspace_scope
from nanobot.session.manager import SessionManager
from nanobot.webui.media_maintenance import garbage_collect_generated_media
from nanobot.webui.temporary_chats import WebUITemporaryChats
from nanobot.webui.workspaces import WebUIWorkspaceController


class Assistant:
    def __init__(self, workspace):
        self.sessions = SessionManager(workspace)
        session = self.sessions.get_or_create("telegram:42")
        session.add_message("user", "Run project A's check")
        self.sessions.save(session, fsync=True)
        self.messages = []

    async def submit_local_trigger_turn(self, msg):
        self.messages.append(msg)
        return None


def _origin(project):
    return RequestContext(
        channel="telegram",
        chat_id="42",
        session_key="telegram:42",
        original_user_text="Run project A's check",
        turn_id="source-turn-a",
        metadata={},
        workspace=project,
        workspace_scope=build_workspace_scope(project, "restricted"),
    )


async def _wait_for_state(runtime, exec_id, state, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        entry = runtime.get(exec_id)
        if entry is not None and entry.state == state:
            return entry
        if asyncio.get_running_loop().time() >= deadline:
            actual = getattr(entry, "state", None)
            raise AssertionError(
                f"{exec_id!r} did not reach state {state!r} within {timeout}s; "
                f"got {actual!r}"
            )
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_process_completion_lane_keeps_ordered_durable_transitions(tmp_path):
    project = tmp_path / "project-a"
    project.mkdir()
    assistant = Assistant(tmp_path / "agent")
    runtime = ProcessCompletionRuntime(tmp_path / "runtime", assistant)
    runtime.start()
    origin = _origin(project)

    try:
        runtime.started("one", "first", str(project), origin)
        runtime.started("two", "second", str(project), origin)
        runtime.finished("one", ExecCompletion("one-out", 0, False))

        await _wait_for_state(runtime, "one", "done")
        assert runtime.get("two").state == "waiting"

        runtime.finished("two", ExecCompletion("two-out", 0, False))
        await _wait_for_state(runtime, "two", "done")

        assert len(assistant.messages) == 2
        rendered = "\n".join(f"{message!r}\n{message}" for message in assistant.messages)
        assert "one-out" in rendered
        assert "two-out" in rendered
    finally:
        await runtime.aclose()


@pytest.mark.asyncio
async def test_process_completion_restart_marks_unfinished_execution_unknown(tmp_path):
    project = tmp_path / "project-restart"
    project.mkdir()
    assistant = Assistant(tmp_path / "agent-restart")
    root = tmp_path / "runtime-restart"
    origin = _origin(project)

    runtime = ProcessCompletionRuntime(root, assistant)
    runtime.start()
    try:
        runtime.started("interrupted", "interrupted", str(project), origin)
    finally:
        await runtime.aclose()

    reopened = ProcessCompletionRuntime(root, assistant)
    reopened.start()
    try:
        assert reopened.get("interrupted").state == "unknown"
    finally:
        await reopened.aclose()


def test_partial_generated_media_deletion_retains_failed_ownership(tmp_path, monkeypatch):
    root = tmp_path / "generated"
    root.mkdir()
    for name in ("a.png", "a.json", "b.png", "b.json"):
        (root / name).write_text("generated\n", encoding="utf-8")

    now = datetime.now().astimezone()
    failed_path = root / "b.png"
    original_unlink = Path.unlink

    def selective_unlink(self, *args, **kwargs):
        if Path(self) == failed_path:
            raise OSError("busy")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", selective_unlink)

    result = garbage_collect_generated_media(
        artifact_root=root,
        referenced=set(),
        retention_days=0,
        orphan_grace_days=0,
        now=now,
    )

    assert result["partial"] is True
    assert result["retry_required"] is True
    assert str(failed_path) in {item["path"] for item in result["failed"]}
    assert not (root / "a.png").exists()
    assert (root / "b.png").exists()


@pytest.mark.asyncio
async def test_temporary_chat_retains_failed_cleanup_until_retry(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    instance = tmp_path / "instance"
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    controller = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=home,
        default_restrict_to_workspace=True,
        grant_store=FolderGrantStore(
            agent_home=home,
            instance_root=instance,
            path=instance / "webui" / "folder-grants.json",
        ),
        native_supervisor_available=True,
    )
    chats = WebUITemporaryChats(
        bus=MessageBus(),
        session_manager=sessions,
        workspaces=controller,
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )
    controller.set_temporary_owners(
        binding=lambda chat_id: chats.binding(None, chat_id),
        set_mode=lambda chat_id, mode: chats.set_access_mode(chat_id, mode),
    )

    owner = object()
    chat_id = chats.create(owner, trusted_webui=True)
    media = tmp_path / "attachment.txt"
    media.write_text("attachment\n", encoding="utf-8")
    chats.register_media(owner, chat_id, [str(media)])

    original_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if Path(self) == media:
            raise OSError("busy")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    try:
        await chats.discard(owner, chat_id)
        assert str(media) in chats.pending_cleanup(chat_id)
        assert media.exists()
    finally:
        monkeypatch.setattr(Path, "unlink", original_unlink)

    remaining = chats.retry_cleanup(chat_id)
    assert remaining == ()
    assert chats.pending_cleanup(chat_id) == ()
    assert not media.exists()
