from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.loop import AgentLoop, TurnKind
from nanobot.agent.loop import TurnContext as StageContext
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import HistoryMode, TurnContext, TurnRequest, TurnSource
from nanobot.bus.events import InboundMessage
from nanobot.config.schema import ToolsConfig
from nanobot.session.manager import Session


def stage():
    session = Session(key='cron:job')
    session.add_message('user', 'prior secret')
    session.add_message('assistant', 'prior answer')
    session.provider_state = object()
    request = TurnRequest(content='/new', history_mode=HistoryMode.FRESH, source=TurnSource.CRON)
    ctx = StageContext(msg=InboundMessage(channel='telegram', chat_id='42', sender_id='cron', content='/new'),
                       session_key=session.key, turn_id='run', kind=TurnKind.USER,
                       runtime=SimpleNamespace(), session=session,
                       delivery=SimpleNamespace(runtime_admitted=AsyncMock()))
    ctx.run_context = TurnContext(request=request)
    return ctx


async def test_fresh_build_excludes_prior_state_but_appends_detail_history():
    loop = object.__new__(AgentLoop)
    ctx = stage()
    session = ctx.session
    ctx.pending_summary = object()
    loop.auto_compact = SimpleNamespace(prepare_session=lambda *args: pytest.fail('old compaction used'))
    loop._request_context_for_turn = lambda ctx: None
    loop._resolve_runtime_context_for_turn = AsyncMock(return_value=[])
    def persist(msg, target, **kwargs):
        target.add_message('user', msg.content)
        return True
    loop._persist_user_message_early = persist
    await loop._compact_session(ctx)
    await loop._build_turn(ctx)
    assert ctx.history == [] and ctx.transcript_input.history == []
    assert ctx.transcript_input.session_summary is None
    assert ctx.transcript_input.current_message == '/new'
    assert ctx.provider_state is None and session.provider_state is None
    assert [m['content'] for m in session.messages] == ['prior secret', 'prior answer', '/new']
    # The save boundary uses the fresh projected input, not retained detail length.
    from nanobot.session.turn_continuation import prepare_save_boundary
    prepare_save_boundary(ctx)
    loop._save_turn(session, [{'role': 'system', 'content': 'system'},
                              {'role': 'user', 'content': '/new'},
                              {'role': 'assistant', 'content': 'new answer'}],
                    ctx.save_skip, input_persisted_early=True)
    assert [m['content'] for m in session.messages] == ['prior secret', 'prior answer', '/new', 'new answer']
    second = stage()
    second.session = session
    await loop._compact_session(second)
    await loop._build_turn(second)
    assert second.transcript_input.history == []
    assert second.transcript_input.session_summary is None
    assert len(session.messages) == 5
    assert session.messages[3]['content'] == 'new answer'



async def test_scheduled_text_does_not_execute_interactive_command():
    loop = object.__new__(AgentLoop)
    loop.commands = SimpleNamespace(dispatch=AsyncMock())
    assert not await loop._dispatch_command(stage())
    loop.commands.dispatch.assert_not_awaited()


async def test_fresh_restore_does_not_consume_old_recovery(monkeypatch):
    import nanobot.agent.loop as module
    loop = object.__new__(AgentLoop)
    loop._fork_dependencies = SimpleNamespace(
        resource_policy=SimpleNamespace(folder_grants=None),
    )
    ctx = stage()
    ctx.delivery.started = AsyncMock()
    loop.tools = ToolRegistry()
    loop.tools_config = ToolsConfig()
    loop._remember_session_route = lambda *args, **kwargs: None
    loop.workspace_scopes = SimpleNamespace(persist_message_scope=lambda *args: None)
    monkeypatch.setattr(module, 'restore_runtime_checkpoint', lambda *args: pytest.fail('checkpoint restored'))
    monkeypatch.setattr(module, 'restore_pending_interruption', lambda *args: pytest.fail('interruption restored'))
    await loop._restore_turn(ctx)
    assert ctx.session.messages[0]['content'] == 'prior secret'


async def test_fresh_runner_summary_cannot_replace_retained_history_boundary(monkeypatch):
    from nanobot.agent.runner import AgentRunResult
    from nanobot.session import turn_continuation
    loop = object.__new__(AgentLoop)
    ctx = stage()
    ctx.delivery.running = AsyncMock()
    ctx.delivery.record_usage = lambda usage: None
    ctx.delivery.route = SimpleNamespace(channel='telegram', chat_id='42')
    ctx.transcript_input = object()
    result = AgentRunResult(final_content='answer', messages=[], summary_checkpoint=object())
    loop._run_agent_loop = AsyncMock(return_value=result)
    monkeypatch.setattr(turn_continuation, 'maybe_continue_turn', AsyncMock(side_effect=AssertionError('continued')))
    await loop._run_turn(ctx)
    assert ctx.summary_checkpoint is None
    assert ctx.run_result is result
    assert loop._run_agent_loop.await_args.kwargs['pending_queue'] is None


@pytest.mark.parametrize('sent,suppressed', [('final', True), ('progress', False), ('', False)])
async def test_canonical_final_suppression_requires_matching_text(sent, suppressed):
    from nanobot.agent.runner import AgentRunResult
    from nanobot.agent.tools.message import MessageTool
    loop = object.__new__(AgentLoop)
    ctx = stage()
    ctx.delivery.running = AsyncMock()
    ctx.delivery.record_usage = lambda usage: None
    ctx.delivery.route = SimpleNamespace(channel='telegram', chat_id='42')
    ctx.transcript_input = object()
    tool = MessageTool(send_callback=AsyncMock())
    async def run(*args, **kwargs):
        await tool.execute(content=sent, channel='telegram', chat_id='42')
        return AgentRunResult(final_content='final', messages=[])
    loop._run_agent_loop = run
    await loop._run_turn(ctx)
    assert ctx.suppress_response is suppressed
