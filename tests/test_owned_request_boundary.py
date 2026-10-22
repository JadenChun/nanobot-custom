"""Exercise the API's owned task boundary without opening an HTTP listener."""

import asyncio
import sys
from collections.abc import Coroutine, Generator
from contextlib import suppress
from types import TracebackType, coroutine

import pytest

import nanobot.api.server as api


class _CoroutineWrapper(Coroutine[object, object, object]):
    """A native-supported ABC coroutine that delegates to the actual child."""

    def __init__(self, child: Coroutine[object, object, object]) -> None:
        self.child = child

    def send(self, value: object) -> object:
        return self.child.send(value)

    def throw(
        self,
        typ: type[BaseException] | BaseException,
        value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> object:
        if value is None and traceback is None:
            return self.child.throw(typ)
        return self.child.throw(typ, value, traceback)

    def close(self) -> None:
        self.child.close()

    def __await__(self) -> Generator[object, None, object]:
        return self.child.__await__()


def _input(
    child: Coroutine[object, object, object], wrapped: bool,
) -> Coroutine[object, object, object]:
    return _CoroutineWrapper(child) if wrapped else child


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("result", [None, {"value": "unchanged"}, object()])
async def test_coroutine_result_identity_is_preserved(wrapped, result) -> None:
    async def child() -> object:
        await asyncio.sleep(0)
        return result

    assert await api._run_owned_with_timeout(_input(child(), wrapped), 1.0) is result


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
async def test_child_failure_identity_is_preserved(wrapped, error_type) -> None:
    failure = error_type("child failure")

    async def child() -> object:
        await asyncio.sleep(0)
        raise failure

    with pytest.raises(error_type) as raised:
        await api._run_owned_with_timeout(_input(child(), wrapped), 1.0)
    assert raised.value is failure


class _OnlyAwaitable:
    def __await__(self) -> Generator[object, None, object]:
        yield None
        return object()


@pytest.mark.parametrize("kind", ["future", "task", "only_awaitable"])
async def test_non_coroutine_awaitables_keep_native_rejection(kind) -> None:
    if kind == "future":
        value = asyncio.get_running_loop().create_future()
        value.set_result(object())
    elif kind == "task":
        async def child() -> object:
            return object()

        value = asyncio.create_task(child())
        await value
    else:
        value = _OnlyAwaitable()

    # Deliberate invalid runtime inputs exercise the existing native contract;
    # there is no cast claiming that an arbitrary awaitable is coroutine-like.
    with pytest.raises(TypeError) as native:
        asyncio.create_task(value)
    with pytest.raises(TypeError) as owned:
        await api._run_owned_with_timeout(value, 1.0)
    assert str(owned.value) == str(native.value)
    if isinstance(value, asyncio.Future):
        assert value.done() and not value.cancelled()


async def test_generator_coroutine_matches_the_actual_native_python_contract() -> None:
    marker = object()

    @coroutine
    def child() -> Generator[object, None, object]:
        yield None
        return marker

    first, second = child(), child()
    try:
        if sys.version_info < (3, 12):
            assert await asyncio.create_task(first) is marker
            assert await api._run_owned_with_timeout(second, 1.0) is marker
        else:
            with pytest.raises(TypeError) as native:
                asyncio.create_task(first)
            with pytest.raises(TypeError) as owned:
                await api._run_owned_with_timeout(second, 1.0)
            # Repr includes each generator's address; compare the native type
            # refusal, not two different objects' rendered identities.
            assert "a coroutine was expected" in str(native.value)
            assert "a coroutine was expected" in str(owned.value)
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("wrapped", [False, True])
async def test_native_task_creation_failure_keeps_exact_input_and_exception(
    wrapped, monkeypatch,
) -> None:
    failure = RuntimeError("task factory failure")
    seen = []

    async def child() -> object:
        pytest.fail("child must not execute after task creation fails")

    value = _input(child(), wrapped)

    def reject(awaitable):
        seen.append(awaitable)
        raise failure

    monkeypatch.setattr(api.asyncio, "create_task", reject)
    try:
        with pytest.raises(RuntimeError) as raised:
            await api._run_owned_with_timeout(value, 1.0)
        assert raised.value is failure
        assert len(seen) == 1 and seen[0] is value
    finally:
        value.close()


@pytest.mark.parametrize("wrapped", [False, True])
async def test_child_self_cancellation_is_joined(wrapped) -> None:
    children = []

    async def child() -> object:
        current = asyncio.current_task()
        assert current is not None
        children.append(current)
        current.cancel()
        await asyncio.sleep(0)
        pytest.fail("self-cancelled child must not resume")

    with pytest.raises(asyncio.CancelledError):
        await api._run_owned_with_timeout(_input(child(), wrapped), 1.0)
    assert len(children) == 1 and children[0].done() and children[0].cancelled()


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("cause", ["timeout", "owner_cancel"])
@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("cleanup_error", [False, True])
async def test_owner_waits_for_cleanup_without_forwarding_repeated_cancellation(
    wrapped, cause, repeated, cleanup_error,
) -> None:
    started, cleaning, release, finished = (asyncio.Event() for _ in range(4))
    children = []
    phases = []
    failure = RuntimeError("cleanup failure")

    async def child() -> object:
        current = asyncio.current_task()
        assert current is not None
        children.append(current)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            phases.append("cleanup_started")
            cleaning.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                phases.append("cleanup_interrupted")
                raise
            phases.append("cleanup_finished")
            finished.set()
            if cleanup_error:
                raise failure

    owner = asyncio.create_task(api._run_owned_with_timeout(
        _input(child(), wrapped), 0.01 if cause == "timeout" else 10.0,
    ))
    try:
        await asyncio.wait_for(started.wait(), 1.0)
        if cause == "owner_cancel":
            owner.cancel()
        await asyncio.wait_for(cleaning.wait(), 1.0)
        assert len(children) == 1 and children[0].cancelling() == 1
        assert not owner.done() and not finished.is_set()
        if repeated:
            for _ in range(2):
                owner.cancel()
                await asyncio.sleep(0)
                assert not owner.done() and not children[0].done()
                assert children[0].cancelling() == 1
        release.set()
        expected = asyncio.CancelledError if cause == "owner_cancel" or repeated else TimeoutError
        with pytest.raises(expected):
            await owner
        assert finished.is_set() and children[0].done()
        assert phases == ["cleanup_started", "cleanup_finished"]
        if cleanup_error:
            assert children[0].exception() is failure
        else:
            assert children[0].cancelled()
    finally:
        release.set()
        if not owner.done():
            owner.cancel()
        with suppress(asyncio.CancelledError, TimeoutError, RuntimeError):
            await asyncio.wait_for(asyncio.shield(owner), 1.0)


@pytest.mark.parametrize("failed", [False, True])
async def test_join_of_completed_child_does_not_change_its_result(failed) -> None:
    marker, failure = object(), RuntimeError("completed child failure")

    async def child() -> object:
        if failed:
            raise failure
        return marker

    task = asyncio.create_task(child())
    if failed:
        with pytest.raises(RuntimeError) as raised:
            await task
        assert raised.value is failure
    else:
        assert await task is marker
    assert await api._cancel_and_join_owned(task) is None
    assert task.done() and task.cancelling() == 0
    if failed:
        assert task.exception() is failure
    else:
        assert task.result() is marker
