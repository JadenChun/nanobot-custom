"""A process wakes the same assistant once; uncertain work never replays."""

import asyncio
import re
from dataclasses import replace

import pytest

from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.exec_session import ExecCompletion, ExecSessionManager
from nanobot.agent.tools.shell import ExecTool
from nanobot.bus.events import OutboundMessage
from nanobot.process_completion import ProcessCompletionRuntime
from nanobot.security.workspace_access import build_workspace_scope
from nanobot.session.manager import SessionManager
from tests.tools.test_exec_session_tools import _python_command


class Assistant:
    def __init__(self, workspace):
        self.sessions = SessionManager(workspace)
        session = self.sessions.get_or_create("telegram:42")
        session.add_message("user", "Run project A's check")
        self.sessions.save(session, fsync=True)
        self.messages = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def submit_local_trigger_turn(self, msg):
        self.messages.append(msg)
        self.entered.set()
        await self.release.wait()
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="continued")


def setup(tmp_path):
    project = tmp_path / "project-a"
    project.mkdir()
    assistant = Assistant(tmp_path / "agent")
    runtime = ProcessCompletionRuntime(tmp_path / "runtime", assistant)
    runtime.start()
    manager = ExecSessionManager()
    manager.install_completion_sink(runtime)
    tool = ExecTool(working_dir=str(project), timeout=3, session_manager=manager)
    origin = RequestContext(channel="telegram", chat_id="42", session_key="telegram:42",
                            original_user_text="Run project A's check", turn_id="source-turn-a",
                            metadata={"message_thread_id": 7}, workspace=project,
                            workspace_scope=build_workspace_scope(project, "restricted"))
    return project, assistant, runtime, manager, tool, origin


async def settled(runtime, identity, state):
    async with asyncio.timeout(5):
        while runtime.get(identity).state != state:
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 7])
async def test_process_exit_continues_original_owner_route_and_project_once(tmp_path, exit_code):
    project, assistant, runtime, manager, tool, origin = setup(tmp_path)
    script = "from pathlib import Path; import time,sys; "
    script += "\nwhile not Path('release').exists(): time.sleep(.01)\n"
    script += f"print('finished A'); sys.exit({exit_code})"
    try:
        with request_context(origin):
            initial = await tool.execute(command=_python_command(script), notify_on_exit=True, yield_time_ms=0)
        identity = re.search(r"session_id: ([a-f0-9]+)", initial).group(1)
        assert not assistant.messages
        (project / "release").touch()
        await settled(runtime, identity, "done")
        record = runtime.get(identity)
        assert record.exit_code == exit_code and "finished A" in record.output
        message = assistant.messages[0]
        assert message.session_key == origin.session_key
        assert message.channel == "telegram" and message.chat_id == "42"
        assert message.metadata["message_thread_id"] == 7
        assert message.workspace_scope.project_path == project
        assert "source-turn-a" in message.content and str(project) in message.content
        runtime.finished(identity, ExecCompletion("duplicate", exit_code))
        await asyncio.sleep(0)
        assert len(assistant.messages) == 1
        assert runtime.result(identity, "another-owner") is None
    finally:
        await manager.close_all()
        await runtime.aclose()


@pytest.mark.asyncio
async def test_completion_waits_for_assistant_and_interruption_is_not_replayed(tmp_path):
    _project, assistant, runtime, manager, _tool, origin = setup(tmp_path)
    assistant.release.clear()
    runtime.started("manual-job", "a command", str(origin.workspace), origin)
    runtime.finished("manual-job", ExecCompletion("result", 0))
    await asyncio.wait_for(assistant.entered.wait(), 2)
    assert runtime.get("manual-job").state == "dispatching"
    await runtime.aclose()
    assert runtime.get("manual-job").state == "unknown"
    reopened = ProcessCompletionRuntime(tmp_path / "runtime", assistant)
    reopened.start()
    await asyncio.sleep(0)
    assert len(assistant.messages) == 1
    assert reopened.get("manual-job").state == "unknown"
    await reopened.aclose()
    await manager.close_all()


@pytest.mark.asyncio
async def test_restart_preserves_unsubmitted_completion_and_marks_process_survival_unknown(tmp_path):
    _project, assistant, runtime, manager, _tool, origin = setup(tmp_path)
    runtime.started("finished", "a command", str(origin.workspace), origin)
    runtime.finished("finished", ExecCompletion("finished before restart", 0))
    runtime.started("interrupted", "another command", str(origin.workspace), origin)
    await runtime.aclose()  # ready task has not entered the assistant
    reopened = ProcessCompletionRuntime(tmp_path / "runtime", assistant)
    reopened.start()
    await settled(reopened, "finished", "done")
    assert reopened.get("interrupted").state == "unknown"
    assert reopened.result("interrupted", origin.session_key).state == "unknown"
    assert len(assistant.messages) == 1
    await reopened.aclose()
    await manager.close_all()


@pytest.mark.asyncio
async def test_cancelled_process_does_not_wake_another_turn(tmp_path):
    _project, assistant, runtime, manager, tool, origin = setup(tmp_path)
    try:
        with request_context(origin):
            initial = await tool.execute(command=_python_command("import time; time.sleep(30)"),
                                         notify_on_exit=True, yield_time_ms=0)
        identity = re.search(r"session_id: ([a-f0-9]+)", initial).group(1)
        assert await manager.terminate_by_owner(origin.session_key) == 1
        await settled(runtime, identity, "cancelled")
        assert not assistant.messages
    finally:
        await manager.close_all()
        await runtime.aclose()


@pytest.mark.asyncio
async def test_inline_completion_has_no_second_model_turn_and_temporary_work_is_rejected(tmp_path):
    _project, assistant, runtime, manager, tool, origin = setup(tmp_path)
    try:
        with request_context(origin):
            initial = await tool.execute(command="echo done", notify_on_exit=True, yield_time_ms=1000)
        assert "Exit code: 0" in initial and not assistant.messages
        with request_context(replace(origin, persistent_session=False)):
            refused = await tool.execute(command="echo should-not-run", notify_on_exit=True)
        assert refused.is_error and "temporary" in refused
    finally:
        await manager.close_all()
        await runtime.aclose()


@pytest.mark.asyncio
async def test_no_process_is_started_without_completion_owner(tmp_path):
    tool = ExecTool(working_dir=str(tmp_path), session_manager=ExecSessionManager())
    refused = await tool.execute(command="echo should-not-run", notify_on_exit=True)
    assert refused.is_error and "enabled gateway" in refused
