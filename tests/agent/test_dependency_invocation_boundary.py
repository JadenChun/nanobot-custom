"""Shared dependency work remains live; async execution retains task ownership."""

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from nanobot.agent import idle_summary as idle
from tests.agent.test_idle_summary import _authority


@pytest.fixture(autouse=True)
def invocation_mode():
    assert "_defer_invocation" in inspect.signature(idle._verify_dependencies).parameters, (
        "live deferred dependency invocation mode is missing"
    )


@pytest.fixture
def bound(tmp_path, invocation_mode):
    _, policy, _, binding, capture = _authority(tmp_path, [{"role": "user", "content": "bound"}])
    references = ("artifact:dependency-invocation",)
    verification = idle.IdleDependencyVerification(
        capture, binding, references, policy.fingerprint, binding.artifact_namespace,
    )
    return capture, binding, references, verification


def arguments(bound):
    capture, binding, references, _ = bound
    return dict(capture=capture, binding=binding, references=references)


def test_empty_dependencies_do_not_observe_the_verifier():
    class Bomb:
        @property
        def verify(self):
            pytest.fail("empty dependencies observed the verifier")

    assert idle._verify_dependencies(
        Bomb(), capture=object(), binding=object(), references=(), _defer_invocation=True,
    ) is None


def test_sync_default_still_invokes_and_validates_once(bound, monkeypatch):
    events = []
    original = idle._validate_dependency_result

    class Verifier:
        def verify(self, **kwargs):
            events.append("invoke")
            assert all(kwargs[k] is v for k, v in arguments(bound).items())
            return bound[-1]

    def validate(result, **kwargs):
        events.append("validate")
        return original(result, **kwargs)

    monkeypatch.setattr(idle, "_validate_dependency_result", validate)
    assert idle._verify_dependencies(Verifier(), **arguments(bound)) is None
    assert events == ["invoke", "validate"]


def test_deferred_work_does_not_cache_a_bound_method_or_result(bound):
    calls, lookups = [], []

    class Verifier:
        def __init__(self):
            self.current = lambda **_kwargs: pytest.fail("obsolete method was cached")

        @property
        def verify(self):
            lookups.append("select")
            return self.current

    verifier = Verifier()
    pending = idle._verify_dependencies(verifier, **arguments(bound), _defer_invocation=True)
    assert callable(pending) and calls == [] and lookups == ["select"]

    def live(**kwargs):
        calls.append(kwargs)
        return bound[-1]

    verifier.current = live
    assert pending() is bound[-1]
    assert lookups == ["select", "select"] and len(calls) == 1
    assert all(calls[0][k] is v for k, v in arguments(bound).items())


@pytest.mark.asyncio
async def test_deferred_work_returns_the_actual_unresolved_awaitable(bound):
    events = []

    class Verifier:
        def verify(self, **_kwargs):
            events.append("invoke")

            async def complete():
                events.append("await")
                return bound[-1]

            return complete()

    pending = idle._verify_dependencies(Verifier(), **arguments(bound), _defer_invocation=True)
    assert events == []
    result = pending()
    assert inspect.isawaitable(result) and events == ["invoke"]
    assert await result is bound[-1] and events == ["invoke", "await"]


def test_sync_default_rejects_unbound_result_after_invocation(bound):
    class Verifier:
        def verify(self, **_kwargs):
            return True

    with pytest.raises(idle.IdleSummaryDependencyError, match="returned no bound result"):
        idle._verify_dependencies(Verifier(), **arguments(bound))


@pytest.mark.parametrize("error", [ValueError("live"), RuntimeError("live"), OSError("live")])
def test_sync_default_keeps_exception_normalization(bound, error):
    class Verifier:
        def verify(self, **_kwargs):
            raise error

    with pytest.raises(idle.IdleSummaryDependencyError, match="dependency verification failed") as caught:
        idle._verify_dependencies(Verifier(), **arguments(bound))
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("error", [ValueError("live"), RuntimeError("live"), OSError("live")])
def test_deferred_work_preserves_raw_exception_identity(bound, error):
    class Verifier:
        def verify(self, **_kwargs):
            raise error

    pending = idle._verify_dependencies(Verifier(), **arguments(bound), _defer_invocation=True)
    with pytest.raises(type(error)) as caught:
        pending()
    assert caught.value is error


@pytest.mark.parametrize("verifier", [None, object(), SimpleNamespace(verify=17)])
@pytest.mark.parametrize("defer", [False, True])
def test_both_modes_reject_missing_verifier_before_scheduling(bound, verifier, defer):
    with pytest.raises(idle.IdleSummaryDependencyError, match="requires an injected verifier"):
        idle._verify_dependencies(verifier, **arguments(bound), _defer_invocation=defer)


@pytest.mark.asyncio
@pytest.mark.parametrize("awaitable", [False, True])
async def test_async_driver_executes_shared_live_work_in_owned_task(bound, monkeypatch, awaitable):
    events = []
    owner = asyncio.current_task()
    original_plan = idle._verify_dependencies
    original_validate = idle._validate_dependency_result

    def shared(*args, **kwargs):
        assert asyncio.current_task() is owner and kwargs["_defer_invocation"] is True
        events.append("prepare")
        work = original_plan(*args, **kwargs)

        def run():
            assert asyncio.current_task() is not owner
            events.append("invoke")
            return work()

        return run

    class Verifier:
        @property
        def verify(self):
            events.append("select")

            def verify(**kwargs):
                assert asyncio.current_task() is not owner
                assert all(kwargs[k] is v for k, v in arguments(bound).items())
                events.append("verify")
                if not awaitable:
                    return bound[-1]

                async def complete():
                    assert asyncio.current_task() is not owner
                    events.append("await")
                    return bound[-1]

                return complete()

            return verify

    def clock():
        events.append("clock")
        return 0.0

    def validate(result, **kwargs):
        assert asyncio.current_task() is owner
        events.append("validate")
        return original_validate(result, **kwargs)

    monkeypatch.setattr(idle, "_verify_dependencies", shared)
    monkeypatch.setattr(idle, "_validate_dependency_result", validate)
    await idle._verify_dependencies_async(
        Verifier(), **arguments(bound), deadline=100.0, clock=clock,
    )
    expected = ["prepare", "select", "clock", "invoke", "select", "verify"]
    if awaitable:
        expected.append("await")
    assert events == [*expected, "validate"]


@pytest.mark.asyncio
@pytest.mark.parametrize("deadline", [0.0, -1.0, float("nan"), float("inf")])
async def test_invalid_async_deadline_does_not_invoke_or_create_task(bound, monkeypatch, deadline):
    events = []
    original_create = asyncio.create_task

    class Verifier:
        @property
        def verify(self):
            events.append("select")
            return lambda **_kwargs: pytest.fail("expired dependency work was invoked")

    def clock():
        events.append("clock")
        return 0.0

    def create(*args, **kwargs):
        events.append("task")
        return original_create(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", create)
    with pytest.raises(idle.IdleSummaryTimeoutError, match="elapsed before dependency verification"):
        await idle._verify_dependencies_async(
            Verifier(), **arguments(bound), deadline=deadline, clock=clock,
        )
    assert events == ["select", "clock"]


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["capture", "references", "policy_fingerprint", "artifact_namespace"])
async def test_async_driver_validates_actual_result_not_the_deferred_plan(bound, tmp_path, changed):
    capture, binding, references, verification = bound
    values = dict(capture=capture, binding=binding, references=references,
        policy_fingerprint=verification.policy_fingerprint, artifact_namespace=verification.artifact_namespace)
    if changed == "capture":
        values[changed] = _authority(tmp_path / "other", [{"role": "user", "content": "other"}])[-1]
    elif changed == "references":
        values[changed] = ("artifact:other",)
    elif changed == "policy_fingerprint":
        values[changed] = "other-policy"
    else:
        values[changed] = object()
    wrong = idle.IdleDependencyVerification(**values)
    calls = []

    class Verifier:
        def verify(self, **_kwargs):
            calls.append("invoke")
            return wrong

    verifier = Verifier()
    work = idle._verify_dependencies(verifier, **arguments(bound), _defer_invocation=True)
    assert calls == [] and work() is wrong and calls == ["invoke"]
    with pytest.raises(idle.IdleSummaryDependencyError, match="changed state"):
        await idle._verify_dependencies_async(
            verifier, **arguments(bound), deadline=asyncio.get_running_loop().time() + 100,
            clock=asyncio.get_running_loop().time,
        )
    assert calls == ["invoke", "invoke"]


@pytest.mark.asyncio
async def test_async_cancellation_waits_for_owned_callback_settlement(bound):
    started, settling, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events = []

    class Verifier:
        async def verify(self, **_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                settling.set()
                await release.wait()
                events.append("joined")

    task = asyncio.create_task(idle._verify_dependencies_async(
        Verifier(), **arguments(bound), deadline=asyncio.get_running_loop().time() + 100,
        clock=asyncio.get_running_loop().time,
    ))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        await asyncio.wait_for(settling.wait(), timeout=2)
        assert not task.done() and events == []
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == ["joined"] and task.done()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_async_timeout_waits_for_owned_callback_settlement(bound, monkeypatch):
    started, settling, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events = []
    original_wait = asyncio.wait_for

    class Verifier:
        async def verify(self, **_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                settling.set()
                await release.wait()
                events.append("joined")

    async def expire_owned(future, timeout):
        if timeout == 100.0:
            await original_wait(started.wait(), timeout=2)
            future.cancel()
            raise asyncio.TimeoutError
        return await original_wait(future, timeout=timeout)

    monkeypatch.setattr(asyncio, "wait_for", expire_owned)
    task = asyncio.create_task(idle._verify_dependencies_async(
        Verifier(), **arguments(bound), deadline=100.0, clock=lambda: 0.0,
    ))
    try:
        await original_wait(settling.wait(), timeout=2)
        assert not task.done() and events == []
        release.set()
        with pytest.raises(idle.IdleSummaryTimeoutError, match="dependency verification timed out"):
            await task
        assert events == ["joined"] and task.done()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
