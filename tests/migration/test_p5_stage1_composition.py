from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config
from nanobot.runtime_composition import (
    ChannelBuildOptions,
    RuntimeComposition,
    RuntimeFactories,
    RuntimeLifecycleError,
    RuntimeLoopError,
    RuntimePreflightError,
    RuntimeState,
    inspect_session_modes,
    runtime_preflight,
)
from nanobot.session.runtime_authority import (
    SessionAuthorityLease,
    SessionAuthorityUnavailableError,
    Stage1UnavailableSessionAuthority,
)


def _config(tmp_path: Path, **defaults: object) -> Config:
    config = Config.model_validate({
        "agents": {"defaults": {"workspace": str(tmp_path / "workspace"), **defaults}},
        "modelPresets": {"safe": {"model": "test/model", "provider": "auto"}},
    })
    config.bind_source_path(tmp_path / "runtime" / "config.json")
    return config


def _tree(root: Path) -> tuple[tuple[str, str, int, int, bytes], ...]:
    rows: list[tuple[str, str, int, int, bytes]] = []
    if not root.exists():
        return ()
    for path in sorted(root.rglob("*")):
        stat = path.lstat()
        rows.append((str(path.relative_to(root)), "d" if path.is_dir() else "f", stat.st_ino, stat.st_mtime_ns, path.read_bytes() if path.is_file() else b""))
    return tuple(rows)


class Authority:
    def __init__(self, manager: object, events: list[str], close: bool = False) -> None:
        self.manager = manager
        self.events = events
        self.close_owned = close
        self.calls = 0

    async def acquire(self) -> SessionAuthorityLease:
        self.calls += 1
        self.events.append("authority")

        async def close() -> None:
            self.events.append("authority-close")

        return SessionAuthorityLease(self.manager, close if self.close_owned else None)  # type: ignore[arg-type]


def _factories(events: list[str], counts: dict[str, int] | None = None) -> RuntimeFactories:
    counts = counts if counts is not None else {}

    def hit(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1
        events.append(name)

    class MCP:
        async def aclose(self) -> None:
            events.append("mcp-close")

    def bus():
        hit("bus")
        return MessageBus()

    def tools():
        hit("tools")
        return object()

    def mcp(_config, _tools):
        hit("mcp")
        return MCP()

    def cron(_preflight):
        hit("cron")
        return object()

    def agent(_config, bus_value, sessions, _tools, _mcp, _cron, delivery):
        hit("agent")

        async def close() -> None:
            events.append("agent-close")

        return SimpleNamespace(
            bus=bus_value,
            sessions=sessions,
            turn_delivery_factory=delivery,
            aclose=close,
        )

    return RuntimeFactories(bus, tools, mcp, cron, agent)  # type: ignore[arg-type]


@pytest.mark.parametrize("case", ["absent", "legacy", "canonical", "mixed"])
async def test_default_definition_and_acquisition_are_read_only_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
) -> None:
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    workspace = tmp_path / "workspace"
    root = tmp_path / "runtime" / "sessions"
    identity = "a" * 32
    namespace = root / identity
    if case != "absent":
        (workspace / ".nanobot").mkdir(parents=True)
        namespace.mkdir(parents=True)
        (workspace / ".nanobot" / "workspace-id").write_text(identity + "\n")
        (namespace / ".workspace").write_text(str(workspace.resolve()) + "\n")
    if case in {"legacy", "mixed"}:
        (namespace / "legacy.jsonl").write_text(
            json.dumps({"_type": "metadata", "key": "cli:legacy", "metadata": {}}) + "\n"
        )
    if case in {"canonical", "mixed"}:
        (namespace / "canonical.jsonl").write_text(json.dumps({
            "_type": "metadata", "key": "cli:canonical",
            "metadata": {"source_schema_version": 1}, "revision": 0,
            "generation": 0, "incarnation": "test", "last_archived": 0,
            "last_consolidated": 0, "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }) + "\n")
    config = _config(tmp_path)
    before = _tree(tmp_path)
    calls: list[str] = []
    composition = RuntimeComposition.define(
        config,
        session_authority=Stage1UnavailableSessionAuthority(),
        factories=_factories(calls),
        startup_receipt_admitted=True,
    )
    assert composition.state is RuntimeState.NEW
    assert _tree(tmp_path) == before
    with pytest.raises(SessionAuthorityUnavailableError) as first_failure:
        await composition.acquire()
    assert composition.state is RuntimeState.FAILED
    assert calls == []
    assert _tree(tmp_path) == before
    with pytest.raises(SessionAuthorityUnavailableError) as replayed:
        await composition.acquire()
    assert replayed.value is first_failure.value


@pytest.mark.parametrize("raise_on_bool", [False, True])
async def test_explicit_authority_is_not_selected_by_truthiness(
    tmp_path: Path,
    raise_on_bool: bool,
) -> None:
    events: list[str] = []
    manager = object()

    class FalseyAuthority(Authority):
        def __bool__(self) -> bool:
            events.append("bool")
            if raise_on_bool:
                raise AssertionError("definition must not evaluate authority truthiness")
            return False

    authority = FalseyAuthority(manager, events)
    before = _tree(tmp_path)
    composition = RuntimeComposition.define(
        _config(tmp_path),
        session_authority=authority,
        factories=_factories(events),
        startup_receipt_admitted=True,
    )
    assert composition._authority is authority
    assert events == []
    assert _tree(tmp_path) == before
    handle = await composition.acquire()
    assert handle.sessions is manager
    assert authority.calls == 1
    assert "bool" not in events
    await composition.aclose()


async def test_shared_handle_concurrent_acquire_and_reverse_exact_close(tmp_path: Path) -> None:
    events: list[str] = []
    counts: dict[str, int] = {}
    manager = object()
    authority = Authority(manager, events, close=True)
    composition = RuntimeComposition.define(
        _config(tmp_path), session_authority=authority, factories=_factories(events, counts),
        startup_receipt_admitted=True,
    )
    handles = await asyncio.gather(*(composition.acquire() for _ in range(100)))
    handle = handles[0]
    assert all(candidate is handle for candidate in handles)
    assert handle.sessions is manager
    assert handle.agent.bus is handle.bus
    assert handle.agent.sessions is handle.sessions
    assert handle.agent.turn_delivery_factory is handle.delivery
    assert authority.calls == 1
    assert counts == {"bus": 1, "tools": 1, "mcp": 1, "cron": 1, "agent": 1}
    await composition.aclose()
    await composition.aclose()
    assert composition.state is RuntimeState.CLOSED
    assert events[-3:] == ["agent-close", "mcp-close", "authority-close"]
    with pytest.raises(RuntimeLifecycleError):
        await composition.acquire()


async def test_foreign_loop_cannot_acquire_or_close(tmp_path: Path) -> None:
    composition = RuntimeComposition.define(
        _config(tmp_path), session_authority=Authority(object(), []), factories=_factories([]),
        startup_receipt_admitted=True,
    )
    await composition.acquire()
    errors: list[BaseException] = []

    def foreign() -> None:
        for operation in (composition.acquire, composition.aclose):
            try:
                asyncio.run(operation())
            except BaseException as exc:
                errors.append(exc)

    thread = threading.Thread(target=foreign)
    thread.start()
    thread.join()
    assert len(errors) == 2
    assert all(isinstance(error, RuntimeLoopError) for error in errors)
    await composition.aclose()


async def test_acquisition_failure_and_cleanup_failure_are_ordered(tmp_path: Path) -> None:
    events: list[str] = []
    factories = _factories(events)

    class MCP:
        async def aclose(self) -> None:
            events.append("mcp-close")
            raise LookupError("cleanup")

    def broken_agent(*_args):
        raise ValueError("acquisition")

    factories = RuntimeFactories(
        factories.bus, factories.tools, lambda _config, _tools: MCP(),
        factories.cron, broken_agent,
    )
    composition = RuntimeComposition.define(
        _config(tmp_path), session_authority=Authority(object(), events), factories=factories,
        startup_receipt_admitted=True,
    )
    with pytest.raises(ExceptionGroup) as caught:
        await composition.acquire()
    assert [str(error) for error in caught.value.exceptions] == ["acquisition", "cleanup"]
    assert composition.state is RuntimeState.FAILED
    with pytest.raises(ExceptionGroup) as replayed:
        await composition.acquire()
    assert replayed.value is caught.value


async def test_composition_close_aggregates_all_reverse_cleanup_failures_once(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class OwnedAuthority(Authority):
        async def acquire(self):
            self.calls += 1

            async def close():
                events.append("authority")
                raise LookupError("authority")

            return SessionAuthorityLease(self.manager, close)  # type: ignore[arg-type]

    class MCP:
        async def aclose(self):
            events.append("mcp")
            raise LookupError("mcp")

    class Channels:
        def __init__(self, *_args, **_kwargs):
            pass

        async def acquire(self):
            return self

        async def stop_all(self):
            events.append("channels")
            raise LookupError("channels")

    factories = _factories([])

    def agent(_config, bus, sessions, _tools, _mcp, _cron, delivery):
        async def close():
            events.append("agent")
            raise LookupError("agent")

        return SimpleNamespace(
            bus=bus,
            sessions=sessions,
            turn_delivery_factory=delivery,
            aclose=close,
        )

    composition = RuntimeComposition.define(
        _config(tmp_path),
        session_authority=OwnedAuthority(object(), events),
        factories=RuntimeFactories(
            factories.bus,
            factories.tools,
            lambda _config, _tools: MCP(),
            factories.cron,
            agent,
        ),
        channel_options=ChannelBuildOptions(factory=Channels),
        startup_receipt_admitted=True,
    )
    await composition.acquire()
    with pytest.raises(ExceptionGroup) as caught:
        await composition.aclose()
    assert events == ["channels", "agent", "mcp", "authority"]
    assert [str(error) for error in caught.value.exceptions] == [
        "channels",
        "agent",
        "mcp",
        "authority",
    ]
    await composition.aclose()
    assert events == ["channels", "agent", "mcp", "authority"]


@pytest.mark.parametrize("failure_owner", ["authority", "bus"])
async def test_terminal_failure_replays_identical_exception_without_retry(
    tmp_path: Path,
    failure_owner: str,
) -> None:
    failure = LookupError(failure_owner)
    calls = {"authority": 0, "bus": 0}

    class FailingAuthority:
        async def acquire(self):
            calls["authority"] += 1
            if failure_owner == "authority":
                raise failure
            return SessionAuthorityLease(object())  # type: ignore[arg-type]

    factories = _factories([])

    def bus():
        calls["bus"] += 1
        if failure_owner == "bus":
            raise failure
        return MessageBus()

    composition = RuntimeComposition.define(
        _config(tmp_path),
        session_authority=FailingAuthority(),  # type: ignore[arg-type]
        factories=RuntimeFactories(
            bus,
            factories.tools,
            factories.mcp,
            factories.cron,
            factories.agent,
        ),
        startup_receipt_admitted=True,
    )
    for _ in range(2):
        with pytest.raises(LookupError) as caught:
            await composition.acquire()
        assert caught.value is failure
    assert calls == {
        "authority": 1,
        "bus": 0 if failure_owner == "authority" else 1,
    }


async def test_failed_and_closed_compositions_keep_foreign_loop_fence(tmp_path: Path) -> None:
    failure = LookupError("failed")

    class FailingAuthority:
        async def acquire(self):
            raise failure

    failed = RuntimeComposition.define(
        _config(tmp_path),
        session_authority=FailingAuthority(),  # type: ignore[arg-type]
        factories=_factories([]),
        startup_receipt_admitted=True,
    )
    with pytest.raises(LookupError):
        await failed.acquire()
    closed = RuntimeComposition.define(
        _config(tmp_path),
        session_authority=Authority(object(), []),
        factories=_factories([]),
        startup_receipt_admitted=True,
    )
    await closed.acquire()
    await closed.aclose()
    errors: list[BaseException] = []

    def foreign() -> None:
        for composition in (failed, closed):
            for operation in (composition.acquire, composition.aclose):
                try:
                    asyncio.run(operation())
                except BaseException as exc:
                    errors.append(exc)

    thread = threading.Thread(target=foreign)
    thread.start()
    thread.join()
    assert len(errors) == 4
    assert all(isinstance(error, RuntimeLoopError) for error in errors)


async def test_builder_cancellation_rolls_back_on_owner_loop(tmp_path: Path) -> None:
    events: list[tuple[str, int]] = []
    entered = asyncio.Event()

    class Channels:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def acquire(self) -> None:
            entered.set()
            await asyncio.Event().wait()

        async def stop_all(self) -> None:
            events.append(("channels", id(asyncio.get_running_loop())))

    factories = _factories([])
    original_mcp = factories.mcp
    original_agent = factories.agent

    def mcp(config, tools):
        value = original_mcp(config, tools)
        close = value.aclose

        async def tracked_close():
            events.append(("mcp", id(asyncio.get_running_loop())))
            await close()

        value.aclose = tracked_close
        return value

    def agent(*args):
        value = original_agent(*args)
        close = value.aclose

        async def tracked_close():
            events.append(("agent", id(asyncio.get_running_loop())))
            await close()

        value.aclose = tracked_close
        return value

    composition = RuntimeComposition.define(
        _config(tmp_path), session_authority=Authority(object(), []),
        factories=RuntimeFactories(factories.bus, factories.tools, mcp, factories.cron, agent),
        channel_options=ChannelBuildOptions(factory=Channels),
        startup_receipt_admitted=True,
    )
    task = asyncio.create_task(composition.acquire())
    await entered.wait()
    owner = id(asyncio.get_running_loop())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == [("agent", owner), ("mcp", owner)]
    assert composition.state is RuntimeState.FAILED


def test_public_source_validator_rejects_incomplete_canonical_v1(tmp_path: Path) -> None:
    from nanobot.session.source import SourceValidationError
    from nanobot.session.source_store import validate_source_session_file

    location = tmp_path / "sessions"
    location.mkdir()
    path = location / "incomplete.jsonl"
    path.write_text(json.dumps({
        "_type": "metadata", "key": "cli:test",
        "metadata": {"source_schema_version": 1},
    }) + "\n")
    with pytest.raises(SourceValidationError, match="Missing source state"):
        validate_source_session_file(path, expected_key="cli:test")
    with pytest.raises(RuntimePreflightError, match="malformed"):
        inspect_session_modes(tmp_path, tmp_path)


def test_preflight_rejects_background_idle_construction(tmp_path: Path) -> None:
    with pytest.raises(RuntimePreflightError):
        runtime_preflight(_config(tmp_path, idleCompactAfterMinutes=5))


async def test_default_definition_and_acquire_never_construct_or_migrate_jsonl_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from nanobot.session.manager import JsonlSessionStore, SessionManager

    sentinel = AssertionError("persistent store construction is unreachable")
    monkeypatch.setattr(
        SessionManager,
        "__init__",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sentinel),
    )
    monkeypatch.setattr(
        JsonlSessionStore,
        "__init__",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sentinel),
    )
    monkeypatch.setattr(
        JsonlSessionStore,
        "_migrate_from_workspace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(sentinel),
    )
    composition = RuntimeComposition.define(
        _config(tmp_path),
        session_authority=Stage1UnavailableSessionAuthority(),
        startup_receipt_admitted=True,
    )
    with pytest.raises(SessionAuthorityUnavailableError):
        await composition.acquire()


async def test_channel_start_failure_stops_only_started_channels_in_reverse() -> None:
    from nanobot.channels.manager import ChannelManager

    events: list[str] = []

    class Channel:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        async def start(self) -> None:
            events.append(f"start:{self.name}")
            if self.fail:
                raise RuntimeError(f"{self.name} failed")
            await asyncio.Event().wait()

        async def stop(self) -> None:
            events.append(f"stop:{self.name}")

        def start_error_message(self, exc: Exception) -> str:
            return str(exc)

    manager = ChannelManager(Config(), MessageBus())
    manager.channels = {
        "first": Channel("first"),  # type: ignore[assignment]
        "second": Channel("second", fail=True),  # type: ignore[assignment]
        "never": Channel("never"),  # type: ignore[assignment]
    }
    with pytest.raises(RuntimeError, match="second failed"):
        await manager.start_all()
    assert events == ["start:first", "start:second", "stop:second", "stop:first"]
    await manager.stop_all()
    assert events == [
        "start:first", "start:second", "stop:second", "stop:first", "stop:never",
    ]
    await manager.stop_all()
    assert events == [
        "start:first", "start:second", "stop:second", "stop:first", "stop:never",
    ]


async def test_channel_acquisition_cancellation_restores_every_provisional_field() -> None:
    from nanobot.channels.manager import ChannelManager

    manager = ChannelManager(Config(), MessageBus())
    entered = asyncio.Event()

    async def provisional():
        manager.channels["partial"] = object()  # type: ignore[assignment]
        manager._channel_owners["partial"] = "owner"
        manager._channel_runtime_specs["partial"] = ("owner", "default")
        manager._channel_errors["partial"] = "partial"
        entered.set()
        await asyncio.Event().wait()

    manager._acquire_channels = provisional  # type: ignore[method-assign]
    task = asyncio.create_task(manager.acquire())
    await entered.wait()
    with pytest.raises(RuntimeError, match="already in progress"):
        await manager.acquire()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.channels == {}
    assert manager._channel_owners == {}
    assert manager._channel_runtime_specs == {}
    assert manager._channel_errors == {}
    assert manager._acquiring is False
    assert manager._acquired is False

    async def success():
        manager._acquired = True
        return manager

    manager._acquire_channels = success  # type: ignore[method-assign]
    assert await manager.acquire() is manager


async def test_channel_shutdown_attempts_reverse_cleanup_once_and_aggregates() -> None:
    from nanobot.channels.manager import ChannelManager

    events: list[str] = []

    class Channel:
        def __init__(self, name: str) -> None:
            self.name = name

        async def stop(self) -> None:
            events.append(self.name)
            raise LookupError(self.name)

    manager = ChannelManager(Config(), MessageBus())
    manager.channels = {
        "first": Channel("first"),  # type: ignore[assignment]
        "second": Channel("second"),  # type: ignore[assignment]
    }
    with pytest.raises(ExceptionGroup) as caught:
        await manager.stop_all()
    assert events == ["second", "first"]
    assert [str(error) for error in caught.value.exceptions] == ["second", "first"]
    await manager.stop_all()
    assert events == ["second", "first"]


async def test_restart_notice_task_is_cancelled_before_start_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.channels.manager import ChannelManager
    from nanobot.utils.restart import RestartNotice

    fail = asyncio.Event()
    sent: list[str] = []

    class Channel:
        def __init__(self, name: str, *, failing: bool = False) -> None:
            self.name = name
            self.failing = failing
            self.is_running = False

        async def start(self) -> None:
            if self.failing:
                await fail.wait()
                raise RuntimeError("delayed failure")
            await asyncio.Event().wait()

        async def stop(self) -> None:
            pass

        async def send(self, _message) -> None:
            sent.append(self.name)

        def start_error_message(self, exc: Exception) -> str:
            return str(exc)

    notice = RestartNotice(
        channel="first",
        chat_id="chat",
        started_at_raw="2026-01-01T00:00:00+00:00",
    )
    monkeypatch.setattr(
        "nanobot.channels.manager.consume_restart_notice_from_env",
        lambda: notice,
    )
    manager = ChannelManager(Config(), MessageBus())
    manager.channels = {
        "first": Channel("first"),  # type: ignore[assignment]
        "second": Channel("second", failing=True),  # type: ignore[assignment]
    }
    task = asyncio.create_task(manager.start_all())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    fail.set()
    with pytest.raises(RuntimeError, match="delayed failure"):
        await task
    assert manager._restart_notice_task is None
    await asyncio.sleep(0)
    assert sent == []
