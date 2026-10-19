"""Joined acquired-runtime acceptance, no live effects."""

import asyncio
import builtins
import json
import pickle
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.cron.runtime import RuntimeOwnedWorkError as OldError
from nanobot.runtime_composition import _close_reverse
from tests.migration.p5_stage4_fixtures import (
    Effects,
    Provider,
    config_at,
    define_graph,
    joined,
    offline_denial,  # noqa: F401
    seed_modes,
)


class ApiRequest:
    content_type = "application/json"

    def __init__(self, handle):
        self.app = {"agent_loop": handle.agent, "model_name": "synthetic-model", "request_timeout": 5}

    async def json(self):
        return {"session_id": "joined", "messages": [{"role": "user", "content": "api request"}]}


@pytest.mark.parametrize("mode", ["legacy", "canonical"])
async def test_four_surfaces_two_modes_share_one_acquired_real_runner(tmp_path, monkeypatch, mode):
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.runner import AgentRunner
    from nanobot.api.server import handle_chat_completions
    from nanobot.config.loader import load_config, set_config_path
    from nanobot.nanobot import Nanobot
    from tests.channels.test_telegram_headless_composition import _telegram_update

    config = config_at(tmp_path)
    # Actual entrypoints bind the global instance path as well as Config's source.
    # API media parsing uses that public context even for a text-only request.
    config_path = config.runtime_data_dir / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config.model_dump_json(by_alias=True))
    monkeypatch.setattr("nanobot.config.loader._current_config_path", None)
    set_config_path(config_path)
    config = load_config(config_path)
    keys = ("cli:direct", "api:joined", "sdk:joined", "telegram:123")
    await seed_modes(config, {**dict.fromkeys(keys, mode), "cli:other-mode":
                             "canonical" if mode == "legacy" else "legacy"})
    effects = Effects()
    provider = Provider(effects)
    composition = define_graph(config, provider, effects)
    handle = await composition.acquire()
    assert isinstance(handle.agent, AgentLoop) and isinstance(handle.agent.runner, AgentRunner)
    assert handle.agent.bus is handle.bus and handle.agent.sessions is handle.sessions
    assert handle.agent.tools is handle.tools
    execute = AsyncMock(wraps=handle.agent.execute_turn)
    run = AsyncMock(wraps=handle.agent.runner.run)
    monkeypatch.setattr(handle.agent, "execute_turn", execute)
    monkeypatch.setattr(handle.agent.runner, "run", run)
    try:
        await joined(handle.agent.process_direct("direct request", session_key=keys[0]))
        response = await joined(handle_chat_completions(ApiRequest(handle)))
        assert response.status == 200 and json.loads(response.body)["choices"][0]["message"]["content"] == "fake answer"
        bot = Nanobot(handle.agent)
        result = await joined(bot.run("sdk request", session_key=keys[2]))
        assert result.content == "fake answer" and bot._loop is handle.agent
        update = _telegram_update()
        update.message.reply_to_message = None
        update.message.message_thread_id = None
        update.message.text = "telegram request"
        await joined(handle.channels.telegram._on_message(update, None))
        inbound = await joined(handle.bus.consume_inbound())
        assert inbound.session_key == keys[3] and inbound.chat_id == "123"
        await joined(handle.agent._dispatch(inbound))
        assert execute.await_count == run.await_count == effects.provider == 4
        assert [call.args[0].session_key for call in execute.await_args_list] == list(keys)
        assert [handle.sessions.resolve_session_mode(key).mode for key in keys] == [mode] * 4
        assert effects.tools == 0 and effects.sends == []
    finally:
        await joined(composition.aclose())
    assert effects.lifecycle[-2:] == ["channels-close", "mcp-close"]


class ObservedLock(asyncio.Lock):
    def __init__(self):
        super().__init__()
        self.queued = asyncio.Event()

    async def acquire(self):
        if self.locked():
            self.queued.set()
        return await super().acquire()


@pytest.mark.parametrize("modes", [("legacy", "legacy"), ("canonical", "canonical"),
                                   ("legacy", "canonical"), ("canonical", "legacy")])
async def test_real_lock_serialization_and_cross_key_progress_with_barriers(tmp_path, modes):
    from nanobot.agent.turn import TurnRequest

    config = config_at(tmp_path)
    await seed_modes(config, dict(zip(("cli:a", "cli:b"), modes)))
    effects = Effects()
    provider = Provider(effects)
    provider.entered["first-blocked"] = asyncio.Event()
    provider.release["first-blocked"] = asyncio.Event()
    composition = define_graph(config, provider, effects)
    handle = await composition.acquire()
    lock = ObservedLock()
    handle.agent._session_locks["cli:a"] = lock
    tasks = []
    try:
        tasks.append(asyncio.create_task(handle.agent.execute_turn(TurnRequest(content="first-blocked", session_key="cli:a"))))
        await joined(provider.entered["first-blocked"].wait())
        tasks.append(asyncio.create_task(handle.agent.execute_turn(TurnRequest(content="same-key-second", session_key="cli:a"))))
        await joined(lock.queued.wait())
        other = await joined(handle.agent.execute_turn(TurnRequest(content="different-key", session_key="cli:b")))
        assert other.content == "fake answer" and effects.provider == 2
        assert not tasks[0].done() and not tasks[1].done()
        provider.release["first-blocked"].set()
        results = await joined(asyncio.gather(*tasks))
        assert all(result.content == "fake answer" for result in results)
        assert effects.provider == 3
        assert "fake answer" in str(provider.messages[-1])
    finally:
        provider.release["first-blocked"].set()
        await joined(asyncio.gather(*tasks, return_exceptions=True))
        await joined(composition.aclose())


async def test_failure_cleanup_does_not_import_optional_cron(monkeypatch):
    real_import = builtins.__import__

    def no_optional(name, *args, **kwargs):
        if name in {"nanobot.cron.runtime", "croniter"}:
            raise AssertionError("optional cleanup import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_optional)

    async def fail():
        raise ValueError("fixed fake failure")

    errors = await _close_reverse([fail])
    assert len(errors) == 1 and isinstance(errors[0], ValueError)


async def test_default_foreground_and_clean_close_do_not_import_optional_cron(tmp_path, monkeypatch):
    config = config_at(tmp_path)
    effects = Effects()
    provider = Provider(effects)
    composition = define_graph(config, provider, effects)
    real_import = builtins.__import__

    def no_optional(name, *args, **kwargs):
        if name in {"nanobot.cron.runtime", "croniter"}:
            raise AssertionError("optional runtime imported in default graph")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_optional)
    before = set(asyncio.all_tasks())
    handle = await composition.acquire()
    assert handle.cron_runtime is None and handle.cron.occurrence_store is None
    assert effects.provider == 0 and effects.sends == []
    assert handle.agent.auto_compact is None
    await joined(handle.agent.process_direct("ordinary", session_key="cli:new"))
    await joined(composition.aclose())
    assert effects.provider == 1 and effects.sends == []
    assert not [task for task in asyncio.all_tasks() - before if not task.done()]
    assert not any(path.name in {"cron-runtime", "callbacks", "heartbeat", "dream"}
                   for path in tmp_path.rglob("*"))


def test_shared_error_alias_and_old_pickle_lookup_are_preserved():
    from nanobot.runtime_errors import RuntimeOwnedWorkError

    assert OldError is RuntimeOwnedWorkError
    assert RuntimeOwnedWorkError.__module__ == "nanobot.runtime_errors"
    assert type(pickle.loads(pickle.dumps(OldError("fixed")))) is RuntimeOwnedWorkError
    assert pickle.loads(b"cnanobot.cron.runtime\nRuntimeOwnedWorkError\n.") is RuntimeOwnedWorkError


async def test_default_owned_failure_classification_and_explicit_retry_without_optional_imports(tmp_path, monkeypatch):
    from nanobot.runtime_composition import RuntimeState
    from nanobot.runtime_errors import RuntimeOwnedWorkError

    config = config_at(tmp_path)
    effects = Effects()
    value = define_graph(config, Provider(effects), effects)
    handle = await value.acquire()
    stop = handle.channels.stop_all
    blocked = True

    async def retain():
        if blocked:
            raise RuntimeOwnedWorkError("fixed fake resource fence")
        await stop()

    # Replace the registered channel closer only; graph identity/engine stay real.
    value._closers[-1] = retain
    real_import = builtins.__import__

    def no_optional(name, *args, **kwargs):
        if name in {"nanobot.cron.runtime", "croniter"}:
            raise AssertionError("optional runtime imported during failure retry")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_optional)
    with pytest.raises(ExceptionGroup) as error:
        await joined(value.aclose())
    assert isinstance(error.value.exceptions[0], RuntimeOwnedWorkError)
    assert value.state is RuntimeState.FAILED and effects.lifecycle == ["channels-acquire"]
    blocked = False
    await joined(value.aclose())
    assert value.state is RuntimeState.CLOSED and effects.lifecycle[-2:] == ["channels-close", "mcp-close"]


@pytest.mark.parametrize("unsafe", ["witness", "journal", "sidecar", "ttl", "idle", "recovery", "quarantine"])
async def test_assembled_startup_unsafe_state_denies_before_dependency_factories(tmp_path, unsafe):
    from nanobot.session.manager import JsonlSessionStore

    config = config_at(tmp_path)
    effects = Effects()
    if unsafe in {"witness", "journal", "sidecar"}:
        await seed_modes(config, {"cli:unsafe": "canonical"})
        root = config.runtime_data_dir / "sessions"
        namespace = next(path for path in root.iterdir() if path.is_dir())
        if unsafe == "witness":
            (config.workspace_path / ".nanobot" / "session-mode-binding.json").write_bytes(b"{}")
        elif unsafe == "journal":
            next((namespace / ".session-mode" / "journals").rglob("0001-prepared.json")).write_bytes(b"{}")
        else:
            (namespace / (JsonlSessionStore.storage_key("cli:unsafe") + ".checkpoint.json")).write_bytes(b"{}")
    elif unsafe == "ttl":
        config.agents.defaults.session_ttl_minutes = 1
    elif unsafe == "idle":
        config.agents.defaults.idle_compaction.enabled = True
    else:
        memory = config.workspace_path / "memory"
        memory.mkdir(mode=0o700)
        if unsafe == "recovery":
            (memory / ".consolidation.pending.json").write_bytes(b"pending fake receipt")
        else:
            (memory / ".recovery" / "receipt-quarantine").mkdir(parents=True, mode=0o700)
            (memory / ".recovery").chmod(0o700)
    before = set(asyncio.all_tasks())
    value = None
    with pytest.raises((ValueError, RuntimeError)):
        value = define_graph(config, Provider(effects), effects)

        def forbidden():
            pytest.fail("unsafe state reached foreground dependency factory")

        value._factories = replace(value._factories, bus=forbidden)
        await joined(value.acquire())
    if value is not None:
        await joined(value.aclose())
        # Lifetime ownership was rolled back; no leaked lease on failed admission.
        from nanobot.session.runtime_authority import OfflineSessionOperator

        lease = await OfflineSessionOperator(config.workspace_path,
            copied_sessions_root=config.runtime_data_dir / "sessions").acquire()
        await lease.close()
    assert effects == Effects()
    assert not [task for task in asyncio.all_tasks() - before if not task.done()]


@pytest.mark.parametrize("unsafe", ["telegram", "plugin-conflict"])
async def test_channel_admission_conflicts_unwind_real_assembled_resources(tmp_path, monkeypatch, unsafe):
    from nanobot.channels.manager import ChannelManager
    from nanobot.runtime_composition import ChannelBuildOptions
    from nanobot.session.runtime_authority import OfflineSessionOperator

    config = config_at(tmp_path)
    if unsafe == "telegram":
        config.channels.telegram["allowFrom"] = []
    else:
        config.channels.telegram["enabled"] = False
        config.channels.broken = {"enabled": True}
        monkeypatch.setattr("nanobot.channels.registry._channel_package_names", lambda: ["broken"])
        monkeypatch.setattr("nanobot.channels.registry.load_channel_package",
            lambda _: (_ for _ in ()).throw(TypeError("malformed built-in manifest")))
        monkeypatch.setattr("nanobot.channels.registry.metadata.entry_points", lambda **_: [
            SimpleNamespace(name="broken", value="fake:Channel",
                            load=lambda: pytest.fail("conflicting plugin loaded"))])
    effects = Effects()
    value = define_graph(config, Provider(effects), effects)
    value._channel_options = ChannelBuildOptions(factory=ChannelManager)
    before = set(asyncio.all_tasks())
    with pytest.raises(SystemExit if unsafe == "telegram" else ValueError):
        # SystemExit must stay in the owning task, not escape a wait_for child.
        await value.acquire()
    await joined(value.aclose())
    assert effects.provider == effects.tools == effects.callbacks == 0 and effects.sends == []
    assert effects.lifecycle == ["mcp-close"]
    lease = await OfflineSessionOperator(config.workspace_path,
        copied_sessions_root=config.runtime_data_dir / "sessions").acquire()
    await lease.close()
    assert not [task for task in asyncio.all_tasks() - before if not task.done()]
