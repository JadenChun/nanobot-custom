"""Native gather output is heterogeneous; the notice owner filters it unchanged."""

import asyncio
from contextlib import suppress

import pytest

from nanobot.channels.manager import ChannelManager


class _ChildFailure(BaseException):
    pass


@pytest.mark.parametrize("present", [False, True])
async def test_absent_and_none_notice_tasks_remain_noops(present, monkeypatch) -> None:
    manager = object.__new__(ChannelManager)
    if present:
        manager._restart_notice_task = None

    def unavailable(*args, **kwargs):
        pytest.fail("absent task reached gather")

    monkeypatch.setattr(asyncio, "gather", unavailable)
    assert await manager._cancel_restart_notice() == []
    assert manager._restart_notice_task is None


@pytest.mark.parametrize("kind", ["future", "task"])
@pytest.mark.parametrize("outcome", ["none", "value", "error", "base_error", "cancelled"])
async def test_completed_notice_keeps_value_filtering_and_exception_identity(kind, outcome) -> None:
    manager = object.__new__(ChannelManager)
    marker = object()
    error = _ChildFailure("child failure") if outcome == "base_error" else RuntimeError("child failure")
    if kind == "future":
        task = asyncio.get_running_loop().create_future()
        if outcome == "cancelled":
            task.cancel()
        elif outcome in {"error", "base_error"}:
            task.set_exception(error)
        else:
            task.set_result(None if outcome == "none" else marker)
    else:
        async def child():
            if outcome == "cancelled":
                raise asyncio.CancelledError()
            if outcome in {"error", "base_error"}:
                raise error
            return None if outcome == "none" else marker

        task = asyncio.create_task(child())
        with suppress(asyncio.CancelledError, RuntimeError, _ChildFailure):
            await task
    manager._restart_notice_task = task
    result = await manager._cancel_restart_notice()
    assert manager._restart_notice_task is None and task.done()
    if outcome in {"error", "base_error"}:
        assert len(result) == 1 and result[0] is error
    else:
        assert result == []
    if outcome == "value":
        assert task.result() is marker
    elif outcome == "cancelled":
        assert task.cancelled()


async def test_running_notice_is_reset_then_cancelled_and_joined() -> None:
    manager = object.__new__(ChannelManager)
    entered, cleaning, release = (asyncio.Event() for _ in range(3))
    phases = []

    async def child():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            assert manager._restart_notice_task is None
            phases.append("cleaning")
            cleaning.set()
            await release.wait()
            phases.append("finished")

    task = asyncio.create_task(child())
    manager._restart_notice_task = task
    await asyncio.wait_for(entered.wait(), 1.0)
    owner = asyncio.create_task(manager._cancel_restart_notice())
    try:
        await asyncio.wait_for(cleaning.wait(), 1.0)
        assert manager._restart_notice_task is None
        assert not owner.done() and not task.done() and task.cancelling() == 1
        release.set()
        assert await owner == []
        assert task.done() and task.cancelled() and phases == ["cleaning", "finished"]
    finally:
        release.set()
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(owner), 1.0)


async def test_native_gather_fault_keeps_task_identity_after_the_reset(monkeypatch) -> None:
    manager = object.__new__(ChannelManager)
    task = asyncio.get_running_loop().create_future()
    task.set_result(object())
    manager._restart_notice_task = task
    error = RuntimeError("gather failure")
    calls = []

    def gather(received, *, return_exceptions):
        assert manager._restart_notice_task is None
        calls.append((received, return_exceptions))
        raise error

    monkeypatch.setattr(asyncio, "gather", gather)
    with pytest.raises(RuntimeError) as raised:
        await manager._cancel_restart_notice()
    assert raised.value is error and calls == [(task, True)]
    assert task.done() and not task.cancelled() and manager._restart_notice_task is None
