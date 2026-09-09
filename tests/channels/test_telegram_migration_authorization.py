from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.channels.telegram.runtime import TelegramChannel, TelegramConfig


@pytest.mark.parametrize('handler', ['_on_start', '_on_help', '_forward_command', '_on_message',
                                      '_process_forward_command', '_process_message_update'])
@pytest.mark.parametrize('senders,chats', [([], ['-10']), (['42'], ['-20'])])
async def test_denied_ingress_has_no_side_effects(handler, senders, chats):
    channel = TelegramChannel(TelegramConfig(allow_from=senders, allow_chats=chats), MessageBus())
    channel._running = True
    channel._handle_message = AsyncMock()
    channel._send_pairing_code_if_private = AsyncMock()
    channel._enqueue_ordered_update = Mock()
    channel._remember_thread_context = Mock()
    message = SimpleNamespace(chat_id=-10, chat=SimpleNamespace(type='group'), reply_text=AsyncMock())
    update = SimpleNamespace(message=message, effective_user=SimpleNamespace(id=42, username=None))
    await getattr(channel, handler)(update, None)
    channel._handle_message.assert_not_awaited()
    channel._send_pairing_code_if_private.assert_not_awaited()
    channel._enqueue_ordered_update.assert_not_called()
    channel._remember_thread_context.assert_not_called()
    message.reply_text.assert_not_awaited()


@pytest.mark.parametrize('sender,allowed', [('42', True), ('42|name', True), ('99|name', True),
                                          ('99|other', False), ('invalid|name', False)])
def test_explicit_sender_matching(sender, allowed):
    channel = TelegramChannel(TelegramConfig(allow_from=['42', 'name']), MessageBus())
    assert channel.is_allowed(sender) is allowed


def test_legacy_fields_roundtrip():
    data = {'allowChats': ['-10'], 'fastCommands': {'/report': ['python', 'report.py']},
            'fastCommandWorkdir': '/workspace', 'fastCommandTimeout': 7}
    config = TelegramConfig.model_validate(data)
    saved = config.model_dump(by_alias=True)
    assert all(saved[key] == value for key, value in data.items())


@pytest.mark.parametrize('chat_type,allowed', [('private', True), ('group', False), ('supergroup', False)])
def test_private_bypasses_only_chat_constraint(chat_type, allowed):
    channel = TelegramChannel(TelegramConfig(allow_chats=['-20']), MessageBus())
    assert channel.is_chat_allowed(-10, chat_type) is allowed
    assert not channel.is_allowed('42')


async def test_denied_callback_does_not_acknowledge_or_dispatch():
    channel = TelegramChannel(TelegramConfig(allow_from=['42'], allow_chats=['-20']), MessageBus())
    channel._handle_message = AsyncMock()
    query = SimpleNamespace(message=SimpleNamespace(chat=SimpleNamespace(id=-10, type='group')),
                            answer=AsyncMock(), data='button')
    await channel._on_callback_query(SimpleNamespace(callback_query=query,
        effective_user=SimpleNamespace(id=42, username=None)), None)
    query.answer.assert_not_awaited()
    channel._handle_message.assert_not_awaited()


async def test_buffered_album_rechecks_revoked_authorization(monkeypatch):
    channel = TelegramChannel(TelegramConfig(allow_from=['42'], allow_chats=['-10']), MessageBus())
    channel._handle_message = AsyncMock()
    channel._media_group_buffers['album'] = {'sender_id': '42', 'chat_id': '-10',
                                            'chat_type': 'group', 'contents': ['text'],
                                            'media': [], 'metadata': {}}
    channel.config.allow_chats = ['-20']
    monkeypatch.setattr('nanobot.channels.telegram.runtime.asyncio.sleep', AsyncMock())
    await channel._flush_media_group('album')
    channel._handle_message.assert_not_awaited()
    assert 'album' not in channel._media_group_buffers


@pytest.mark.parametrize('newer', [False, True])
async def test_revoked_album_cancels_only_its_owned_typing_task(monkeypatch, newer):
    import asyncio
    channel = TelegramChannel(TelegramConfig(allow_from=['42'], allow_chats=['-20']), MessageBus())
    blocker = asyncio.Event()
    owned = asyncio.create_task(blocker.wait())
    replacement = asyncio.create_task(blocker.wait()) if newer else owned
    channel._typing_tasks['-10'] = replacement
    channel._media_group_buffers['album'] = {'sender_id': '42', 'chat_id': '-10',
        'chat_type': 'group', 'typing_task': owned}
    channel._handle_message = AsyncMock()
    monkeypatch.setattr('nanobot.channels.telegram.runtime.asyncio.sleep', AsyncMock())
    try:
        await channel._flush_media_group('album')
        assert owned.cancelled()
        assert channel._typing_tasks.get('-10') is (replacement if newer else None)
        if newer:
            assert not replacement.done()
        channel._handle_message.assert_not_awaited()
    finally:
        replacement.cancel()
        await asyncio.gather(replacement, return_exceptions=True)


async def test_fast_command_uses_only_configured_argv_and_preserves_reply_thread(monkeypatch):
    channel = TelegramChannel(TelegramConfig(allow_from=['42'], allow_chats=['-10'],
        fast_commands={'/report': ['runner', '--fixed']}, fast_command_workdir='/configured',
        reply_to_message=True), MessageBus())
    process = SimpleNamespace(communicate=AsyncMock(return_value=(b'report output', b'')), returncode=0)
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr('nanobot.channels.telegram.runtime.asyncio.create_subprocess_exec', spawn)
    channel._send_text = AsyncMock()
    channel._handle_message = AsyncMock()
    channel._remember_thread_context = Mock()
    message = SimpleNamespace(text='/REPORT@mybot ; untrusted arguments', chat_id=-10,
        chat=SimpleNamespace(type='group'), message_id=9, message_thread_id=7)
    await channel._process_forward_command(SimpleNamespace(message=message,
        effective_user=SimpleNamespace(id=42, username=None)), None)
    assert spawn.call_args.args == ('runner', '--fixed')
    assert spawn.call_args.kwargs['cwd'] == '/configured'
    args = channel._send_text.call_args.args
    assert args[0:2] == (-10, 'report output')
    assert args[2].message_id == 9 and args[3] == {'message_thread_id': 7}
    channel._handle_message.assert_not_awaited()


@pytest.mark.parametrize('failure', ['timeout', 'cancel'])
async def test_fast_command_failure_reaps_owned_process(monkeypatch, failure):
    import asyncio
    import os
    channel = TelegramChannel(TelegramConfig(fast_commands={'/report': ['runner']}), MessageBus())
    error = asyncio.TimeoutError() if failure == 'timeout' else asyncio.CancelledError()
    process = SimpleNamespace(communicate=AsyncMock(side_effect=error), returncode=None,
                              pid=12345, wait=AsyncMock(), kill=Mock())
    monkeypatch.setattr('nanobot.channels.telegram.runtime.asyncio.create_subprocess_exec',
                        AsyncMock(return_value=process))
    kill = Mock()
    monkeypatch.setattr('nanobot.agent.tools.shell.os.killpg', kill)
    with pytest.raises(type(error)):
        await channel._try_fast_command(SimpleNamespace(text='/report'))
    process.wait.assert_awaited_once()
    if os.name != 'nt':
        kill.assert_called_once()


async def test_repeated_cancellation_does_not_abandon_fast_command_cleanup(monkeypatch):
    import asyncio
    channel = TelegramChannel(TelegramConfig(fast_commands={'/report': ['runner']}), MessageBus())
    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def communicate():
        started.set()
        await asyncio.Event().wait()
    process = SimpleNamespace(communicate=communicate)
    async def cleanup(child):
        assert child is process
        cleaning.set()
        await release.wait()
    monkeypatch.setattr('nanobot.channels.telegram.runtime.asyncio.create_subprocess_exec',
                        AsyncMock(return_value=process))
    clean = AsyncMock(side_effect=cleanup)
    monkeypatch.setattr('nanobot.agent.tools.shell.ExecTool._kill_process_tree', clean)
    task = asyncio.create_task(channel._try_fast_command(SimpleNamespace(text='/report')))
    await started.wait()
    task.cancel()
    await cleaning.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    clean.assert_awaited_once()


async def test_quota_menu_routing_and_reset_display(monkeypatch):
    from nanobot.command.builtin import build_help_text, cmd_quota, register_builtin_commands
    from nanobot.command.router import CommandRouter
    from nanobot.providers import codex_auth
    assert any(command.command == 'quota' for command in TelegramChannel.BOT_COMMANDS)
    assert TelegramChannel.TELEGRAM_BUS_SLASH_COMMAND_RE.match('/quota@mybot')
    assert '/quota' in build_help_text()
    router = CommandRouter()
    register_builtin_commands(router)
    assert router._priority['/quota'] is cmd_quota
    monkeypatch.setattr(codex_auth, 'get_codex_usage', lambda: {'rate_limit': {
        'primary_window': {'used_percent': 25, 'reset_at': 1700000000}}})
    result = await cmd_quota(SimpleNamespace(msg=SimpleNamespace(
        channel='telegram', chat_id='42', metadata={'message_thread_id': 7})))
    assert '75% remaining' in result.content and 'MYT' in result.content
    assert result.metadata['message_thread_id'] == 7


async def test_quota_error_does_not_expose_auth_response(monkeypatch):
    from nanobot.command.builtin import cmd_quota
    from nanobot.providers import codex_auth
    def fail():
        raise RuntimeError('secret-token-in-response')
    monkeypatch.setattr(codex_auth, 'get_codex_usage', fail)
    result = await cmd_quota(SimpleNamespace(msg=SimpleNamespace(
        channel='telegram', chat_id='42', metadata={})))
    assert 'secret-token' not in result.content
