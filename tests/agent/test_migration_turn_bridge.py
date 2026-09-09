import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.tools.file_state import FileStateStore
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import HistoryMode, RunPolicyContext, TurnContext, TurnRequest
from nanobot.context_repo import ContextRepoManager


async def test_missing_policy_context_fails_before_runtime_access():
    loop = object.__new__(AgentLoop)
    with pytest.raises(ValueError, match='canonical run context'):
        await loop._run_agent_loop(None, runtime=None)
    with pytest.raises(ValueError, match='policy dependencies'):
        await loop._run_agent_loop(None, runtime=None, run_context=TurnContext())


@pytest.mark.parametrize("failure", [None, "cancel", "runner", "entry", "exit", "hook"])
@pytest.mark.parametrize("fresh", [False, True])
async def test_loop_forwards_same_context_and_mandatory_policy(tmp_path, monkeypatch, failure, fresh):
    import nanobot.agent.loop as module
    loop = object.__new__(AgentLoop)
    loop._sync_subagent_runtime_limits = lambda: None
    loop.workspace_scopes = SimpleNamespace(for_turn=lambda **kwargs: SimpleNamespace(project_path=tmp_path))
    loop.context = SimpleNamespace(build_transcript=lambda *args, **kwargs: None)
    loop.tools = ToolRegistry()
    from nanobot.agent.tools.message import MessageTool
    loop.tools.register(MessageTool())
    selected_tools = ToolRegistry()
    loop._file_state_store = FileStateStore()
    loop.tool_hint_max_length = 100
    loop._hook_factories = []
    loop._extra_hooks = []
    loop.max_iterations = 3
    loop.max_tool_result_chars = 1000
    loop.provider_retry_mode = 'standard'
    loop.sessions = None
    loop.runner = SimpleNamespace(run=AsyncMock(return_value=AgentRunResult(final_content='done', messages=[])))
    monkeypatch.setattr(module, 'build_agent_turn_hook', lambda spec: None)
    if not fresh:
        monkeypatch.setattr(module, 'runner_wall_llm_timeout_s', lambda *args, **kwargs: None)
    from nanobot.session.manager import Session
    active_session = Session(key='cron:fixture') if fresh else None
    if fresh:
        active_session.provider_state = object()
        loop._set_runtime_checkpoint = lambda *args: pytest.fail('shared checkpoint written')
    context = TurnContext(policy=RunPolicyContext(workspace=tmp_path, context_manager=ContextRepoManager(),
                                                  approval_granted=failure is None))
    if fresh:
        context.request = TurnRequest(history_mode=HistoryMode.FRESH)
    from nanobot.agent.delegation import current_turn_context
    async def run(spec):
        assert current_turn_context() is context
        if failure == "cancel":
            raise asyncio.CancelledError()
        if failure == "runner":
            raise RuntimeError("runner")
        return AgentRunResult(final_content='done', messages=[], provider_state=object())
    loop.runner.run.side_effect = run
    class Scope:
        def __enter__(self):
            if failure == 'entry':
                raise RuntimeError('entry')
        def __exit__(self, *args):
            assert current_turn_context() is context
            if failure == 'exit':
                raise RuntimeError('exit')
    if failure == 'hook':
        def hook(spec):
            raise RuntimeError('hook')
        monkeypatch.setattr(module, 'build_agent_turn_hook', hook)
    from nanobot.agent.delegation import reset_current_turn_context, set_current_turn_context
    from nanobot.agent.tools.context import current_request_context
    from nanobot.agent.tools.file_state import FileStates, current_file_states
    from nanobot.security.workspace_access import current_workspace_scope
    outer = TurnContext()
    token = set_current_turn_context(outer)
    sentinel = FileStates()
    previous_request, previous_scope = current_request_context(), current_workspace_scope()
    previous_files = current_file_states(sentinel)
    try:
        if failure:
            with pytest.raises(asyncio.CancelledError if failure == 'cancel' else RuntimeError):
                await loop._run_agent_loop(None, runtime=None, run_context=context, turn_scopes=[Scope()], session=active_session, tools=selected_tools)
        else:
            await loop._run_agent_loop(None, runtime=None, run_context=context, turn_scopes=[Scope()], session=active_session, tools=selected_tools)
        assert current_turn_context() is outer
        assert current_request_context() is previous_request
        assert current_workspace_scope() is previous_scope
        assert current_file_states(sentinel) is previous_files
    finally:
        reset_current_turn_context(token)
    if failure in ('entry', 'hook'):
        loop.runner.run.assert_not_awaited()
        return
    spec = loop.runner.run.await_args.args[0]
    assert spec.tools is selected_tools and not spec.tools.tool_names
    assert spec.turn_context is context
    assert spec.tool_policy.context_manager is context.policy.context_manager
    assert spec.tool_policy.workspace == tmp_path
    assert spec.tool_policy.approval_granted is (failure is None)
    assert spec.fail_on_tool_error is True
    if fresh:
        assert spec.continuation_callback is None
        assert spec.provider_state is None
        assert spec.consolidate_history is None
        assert spec.consolidate_provider_compaction is None
        await spec.checkpoint_callback({"provider_state": object()})
        if failure is None:
            assert active_session.provider_state is None
        assert await spec.injection_callback() == []
