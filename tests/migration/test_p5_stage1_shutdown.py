"""Cancellation and instance ownership at the Stage 1 shutdown boundary."""

import asyncio
from types import SimpleNamespace

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.channels.manager import ChannelManager
from nanobot.config.schema import Config


def _leaf_exceptions(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for child in exc.exceptions for leaf in _leaf_exceptions(child)]
    return [exc]


class Channel:
    def __init__(self):
        self.stop_entered = asyncio.Event()
        self.stop_release = asyncio.Event()
        self.start_entered = asyncio.Event()
        self.start_cleanup = asyncio.Event()
        self.start_release = asyncio.Event()
        self.stop_calls = 0

    async def start(self):
        self.start_entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.start_cleanup.set()
            await self.start_release.wait()

    async def stop(self):
        self.stop_calls += 1
        self.stop_entered.set()
        await self.stop_release.wait()


@pytest.mark.parametrize("surface", ["manager", "channel"])
async def test_repeated_shutdown_cancellation_joins_stop_and_start_cleanup(surface):
    manager = ChannelManager(Config(), MessageBus())
    channel = Channel()
    manager.channels["probe"] = channel
    start = manager._start_channel_task("probe", channel)
    await channel.start_entered.wait()
    shutdown = asyncio.create_task(
        manager.stop_all() if surface == "manager" else manager._stop_channel_runtime("probe")
    )
    try:
        await channel.stop_entered.wait()
        shutdown.cancel()
        await asyncio.sleep(0)
        assert not shutdown.done(), "cancellation detached channel.stop cleanup"
        channel.stop_release.set()
        await channel.start_cleanup.wait()
        shutdown.cancel()
        await asyncio.sleep(0)
        assert not shutdown.done(), "second cancellation detached channel.start cleanup"
        channel.start_release.set()
        with pytest.raises(asyncio.CancelledError):
            await shutdown
        assert start.done()
        assert manager._channel_tasks == {}
        await manager.stop_all()
        assert channel.stop_calls == 1
    finally:
        channel.stop_release.set()
        channel.start_release.set()
        start.cancel()
        await asyncio.gather(start, shutdown, return_exceptions=True)


async def test_channel_stop_self_cancellation_still_joins_start_task():
    manager = ChannelManager(Config(), MessageBus())
    channel = Channel()

    async def cancelled_stop():
        channel.stop_calls += 1
        raise asyncio.CancelledError()

    channel.stop = cancelled_stop
    channel.start_release.set()
    manager.channels["probe"] = channel
    start = manager._start_channel_task("probe", channel)
    await channel.start_entered.wait()
    await manager.stop_all()
    assert start.done()
    assert manager._channel_tasks == {}
    await manager.stop_all()
    assert channel.stop_calls == 1


async def test_concurrent_shutdown_coalesces_one_stop_per_instance():
    manager = ChannelManager(Config(), MessageBus())
    channel = Channel()
    manager.channels["probe"] = channel
    shutdowns = [asyncio.create_task(manager.stop_all()) for _ in range(20)]
    try:
        await channel.stop_entered.wait()
        channel.stop_release.set()
        await asyncio.gather(*shutdowns)
        assert channel.stop_calls == 1
    finally:
        channel.stop_release.set()
        await asyncio.gather(*shutdowns, return_exceptions=True)


async def test_replacement_at_same_runtime_name_has_independent_cleanup():
    manager = ChannelManager(Config(), MessageBus())
    first = Channel()
    first.stop_release.set()
    manager.channels["probe"] = first
    await manager.stop_all()
    second = Channel()
    second.stop_release.set()
    manager.channels["probe"] = second
    await manager.stop_all()
    await manager.stop_all()
    assert first.stop_calls == second.stop_calls == 1


async def test_cancelled_shutdown_attempts_remaining_channels_and_surfaces_errors():
    manager = ChannelManager(Config(), MessageBus())
    blocked = Channel()
    events = []

    class Failing:
        async def stop(self):
            events.append("failing")
            raise LookupError("stop failure")

    manager.channels = {"failing": Failing(), "blocked": blocked}
    shutdown = asyncio.create_task(manager.stop_all())
    try:
        await blocked.stop_entered.wait()
        shutdown.cancel()
        await asyncio.sleep(0)
        blocked.stop_release.set()
        with pytest.raises(BaseExceptionGroup) as caught:
            await shutdown
        assert events == ["failing"]
        assert any(isinstance(error, asyncio.CancelledError) for error in caught.value.exceptions)
        assert any(isinstance(error, LookupError) for error in caught.value.exceptions)
        await manager.stop_all()
        assert blocked.stop_calls == 1
        assert events == ["failing"]
    finally:
        blocked.stop_release.set()
        await asyncio.gather(shutdown, return_exceptions=True)


@pytest.mark.parametrize("repeat_cancel", [False, True])
async def test_start_rollback_stops_before_joining_cancellation_resistant_start(
    repeat_cancel: bool,
):
    manager = ChannelManager(Config(), MessageBus())

    class StopReleasesStart(Channel):
        def __init__(self):
            super().__init__()
            self.stop_completed = asyncio.Event()
            self.finalizer_entered = asyncio.Event()

        async def start(self):
            self.start_entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.start_cleanup.set()
                await self.stop_completed.wait()
                self.finalizer_entered.set()
                await self.start_release.wait()

        async def stop(self):
            self.stop_calls += 1
            self.stop_entered.set()
            await self.stop_release.wait()
            self.stop_completed.set()

    channel = StopReleasesStart()
    manager.channels["probe"] = channel
    startup = asyncio.create_task(manager.start_all())
    try:
        await channel.start_entered.wait()
        startup.cancel()
        async with asyncio.timeout(1):
            await channel.stop_entered.wait()
        if repeat_cancel:
            startup.cancel()
            await asyncio.sleep(0)
            assert not startup.done(), "repeated cancellation detached startup rollback"
        channel.stop_release.set()
        async with asyncio.timeout(1):
            await channel.finalizer_entered.wait()
        if repeat_cancel:
            startup.cancel()
            await asyncio.sleep(0)
            assert not startup.done(), "cancellation detached start finalizer"
        channel.start_release.set()
        result = (await asyncio.gather(startup, return_exceptions=True))[0]
        leaves = _leaf_exceptions(result)
        assert any(isinstance(error, asyncio.CancelledError) for error in leaves)
        assert manager._channel_tasks == {}
        assert manager._dispatch_task is None
        assert channel.stop_calls == 1
    finally:
        channel.stop_release.set()
        channel.start_release.set()
        startup.cancel()
        await asyncio.gather(startup, return_exceptions=True)


async def test_start_rollback_preserves_cancellation_and_stop_failure():
    manager = ChannelManager(Config(), MessageBus())

    class FailingStop(Channel):
        async def stop(self):
            self.stop_calls += 1
            self.stop_entered.set()
            self.start_release.set()
            raise LookupError("rollback stop")

    channel = FailingStop()
    manager.channels["probe"] = channel
    startup = asyncio.create_task(manager.start_all())
    try:
        await channel.start_entered.wait()
        startup.cancel()
        result = (await asyncio.gather(startup, return_exceptions=True))[0]
        leaves = _leaf_exceptions(result)
        assert any(isinstance(error, asyncio.CancelledError) for error in leaves)
        assert any(isinstance(error, LookupError) for error in leaves)
        assert channel.stop_calls == 1
        assert manager._channel_tasks == {}
    finally:
        channel.start_release.set()
        startup.cancel()
        await asyncio.gather(startup, return_exceptions=True)


async def test_start_failure_rollback_preserves_primary_cancellation_and_cleanup_error():
    manager = ChannelManager(Config(), MessageBus())

    class FailingStartAndStop(Channel):
        async def start(self):
            self.start_entered.set()
            raise RuntimeError("startup")

        def start_error_message(self, exc):
            return str(exc)

        async def stop(self):
            self.stop_calls += 1
            self.stop_entered.set()
            await self.stop_release.wait()
            raise LookupError("rollback stop")

    channel = FailingStartAndStop()
    manager.channels["probe"] = channel
    startup = asyncio.create_task(manager.start_all())
    try:
        async with asyncio.timeout(1):
            await channel.stop_entered.wait()
        startup.cancel()
        await asyncio.sleep(0)
        assert not startup.done(), "cancellation detached failed-start rollback"
        channel.stop_release.set()
        result = (await asyncio.gather(startup, return_exceptions=True))[0]
        leaves = _leaf_exceptions(result)
        assert any(
            isinstance(error, RuntimeError) and str(error) == "startup" for error in leaves
        ), repr(leaves)
        assert any(isinstance(error, asyncio.CancelledError) for error in leaves)
        assert any(isinstance(error, LookupError) and str(error) == "rollback stop" for error in leaves)
        assert channel.stop_calls == 1
    finally:
        channel.stop_release.set()
        startup.cancel()
        await asyncio.gather(startup, return_exceptions=True)


@pytest.mark.parametrize("cancel_startup", [False, True])
async def test_stop_all_cancellation_of_start_all_waits_for_cleanup(
    cancel_startup: bool,
):
    manager = ChannelManager(Config(), MessageBus())
    channel = Channel()
    manager.channels["probe"] = channel
    startup = asyncio.create_task(manager.start_all())
    shutdown = None
    try:
        await channel.start_entered.wait()
        shutdown = asyncio.create_task(manager.stop_all())
        await channel.stop_entered.wait()
        if cancel_startup:
            startup.cancel()
            await asyncio.sleep(0)
            assert not startup.done()
        channel.stop_release.set()
        await channel.start_cleanup.wait()
        assert not startup.done()
        channel.start_release.set()
        await shutdown
        if cancel_startup:
            with pytest.raises(asyncio.CancelledError):
                await startup
        else:
            await startup
        assert channel.stop_calls == 1
        assert manager._channel_tasks == {}
        assert (manager._dispatch_task is None) is cancel_startup
    finally:
        channel.stop_release.set()
        channel.start_release.set()
        startup.cancel()
        pending = [startup]
        if shutdown is not None:
            pending.append(shutdown)
        await asyncio.gather(*pending, return_exceptions=True)


async def test_gateway_order_does_not_double_stop_started_instance():
    manager = ChannelManager(Config(), MessageBus())

    class ImmediateStop(Channel):
        async def stop(self):
            self.stop_calls += 1
            self.stop_entered.set()
            self.start_release.set()

    channel = ImmediateStop()
    manager.channels["probe"] = channel
    startup = asyncio.create_task(manager.start_all())
    await channel.start_entered.wait()
    startup.cancel()
    await asyncio.gather(startup, return_exceptions=True)
    await manager.stop_all()
    assert channel.stop_calls == 1
    assert manager._channel_tasks == {}


async def test_acquisition_rollback_is_reverse_owned_and_preserves_all_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    config = Config.model_validate({
        "channels": {
            "websocket": {"enabled": False},
            "probe": {"enabled": True},
        }
    })
    manager = ChannelManager(config, MessageBus())
    events: list[str] = []

    class Acquired:
        display_name = "Probe"

        def __init__(self, name: str, *, fail: bool = False):
            self.name = name
            self.fail = fail
            self.stop_entered = asyncio.Event()
            self.stop_release = asyncio.Event()
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1
            events.append(self.name)
            self.stop_entered.set()
            await self.stop_release.wait()
            if self.fail:
                raise LookupError(f"cleanup:{self.name}")

    first = Acquired("first", fail=True)
    second = Acquired("second")
    built = iter([first, second])
    plugin = SimpleNamespace(default_enabled=False)
    specs = [
        SimpleNamespace(instance_id="first", config={}),
        SimpleNamespace(instance_id="second", config={}),
    ]
    monkeypatch.setattr("nanobot.channels.registry.discover_plugins", lambda: {"probe": plugin})
    monkeypatch.setattr("nanobot.channels.registry.packaged_channel_names", lambda: set())
    monkeypatch.setattr(
        "nanobot.channels.registry.discover_legacy_plugins",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "nanobot.optional_features.check_enabled_channel_dependencies",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr("nanobot.channels.manager.channel_setup_spec", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("nanobot.channels.manager.channel_instance_specs", lambda *_args: specs)
    monkeypatch.setattr(
        "nanobot.channels.manager.channel_runtime_name",
        lambda _plugin, instance_id: f"probe.{instance_id}",
    )
    monkeypatch.setattr(plugin, "load_channel_class", lambda: object, raising=False)
    monkeypatch.setattr(manager, "_build_channel", lambda *_args, **_kwargs: next(built))
    monkeypatch.setattr(
        manager,
        "_validate_allow_from",
        lambda: (_ for _ in ()).throw(ValueError("construction")),
    )

    acquisition = asyncio.create_task(manager.acquire())
    try:
        async with asyncio.timeout(1):
            await second.stop_entered.wait()
        acquisition.cancel()
        await asyncio.sleep(0)
        acquisition.cancel()
        await asyncio.sleep(0)
        assert not acquisition.done(), "cancellation detached acquisition rollback"
        second.stop_release.set()
        async with asyncio.timeout(1):
            await first.stop_entered.wait()
        first.stop_release.set()
        result = (await asyncio.gather(acquisition, return_exceptions=True))[0]
        leaves = _leaf_exceptions(result)
        assert events == ["second", "first"]
        assert any(isinstance(error, ValueError) and str(error) == "construction" for error in leaves)
        assert any(isinstance(error, LookupError) and str(error) == "cleanup:first" for error in leaves)
        assert any(isinstance(error, asyncio.CancelledError) for error in leaves)
        assert first.stop_calls == second.stop_calls == 1
        assert manager.channels == {}
        assert manager._acquiring is False
        assert manager._acquired is False
    finally:
        first.stop_release.set()
        second.stop_release.set()
        acquisition.cancel()
        await asyncio.gather(acquisition, return_exceptions=True)
