"""Tests for CompositeHook fan-out, error isolation, and integration."""

from __future__ import annotations

import pytest

from nanobot.agent.hook import AgentHook, AgentHookContext, CompositeHook


def _ctx() -> AgentHookContext:
    return AgentHookContext(iteration=0, messages=[])


# ---------------------------------------------------------------------------
# Fan-out: every hook is called in order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_composite_fans_out_before_iteration():
    calls: list[str] = []

    class H(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            calls.append(f"A:{context.iteration}")

    class H2(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            calls.append(f"B:{context.iteration}")

    hook = CompositeHook([H(), H2()])
    ctx = _ctx()
    await hook.before_iteration(ctx)
    assert calls == ["A:0", "B:0"]


@pytest.mark.asyncio
async def test_composite_fans_out_all_async_methods():
    """Verify all async methods fan out to every hook."""
    events: list[str] = []

    class RecordingHook(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            events.append("before_iteration")

        async def on_stream(self, context: AgentHookContext, delta: str) -> None:
            events.append(f"on_stream:{delta}")

        async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
            events.append(f"on_stream_end:{resuming}")

        async def before_execute_tools(self, context: AgentHookContext) -> None:
            events.append("before_execute_tools")

        async def after_iteration(self, context: AgentHookContext) -> None:
            events.append("after_iteration")

    hook = CompositeHook([RecordingHook(), RecordingHook()])
    ctx = _ctx()

    await hook.before_iteration(ctx)
    await hook.on_stream(ctx, "hi")
    await hook.on_stream_end(ctx, resuming=True)
    await hook.before_execute_tools(ctx)
    await hook.after_iteration(ctx)

    assert events == [
        "before_iteration", "before_iteration",
        "on_stream:hi", "on_stream:hi",
        "on_stream_end:True", "on_stream_end:True",
        "before_execute_tools", "before_execute_tools",
        "after_iteration", "after_iteration",
    ]


# ---------------------------------------------------------------------------
# Error isolation: one hook raises, others still run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_composite_error_isolation_before_iteration():
    calls: list[str] = []

    class Bad(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            raise RuntimeError("boom")

    class Good(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            calls.append("good")

    hook = CompositeHook([Bad(), Good()])
    await hook.before_iteration(_ctx())
    assert calls == ["good"]


@pytest.mark.asyncio
async def test_composite_error_isolation_on_stream():
    calls: list[str] = []

    class Bad(AgentHook):
        async def on_stream(self, context: AgentHookContext, delta: str) -> None:
            raise RuntimeError("stream-boom")

    class Good(AgentHook):
        async def on_stream(self, context: AgentHookContext, delta: str) -> None:
            calls.append(delta)

    hook = CompositeHook([Bad(), Good()])
    await hook.on_stream(_ctx(), "delta")
    assert calls == ["delta"]


@pytest.mark.asyncio
async def test_composite_error_isolation_all_async():
    """Error isolation for on_stream_end, before_execute_tools, after_iteration."""
    calls: list[str] = []

    class Bad(AgentHook):
        async def on_stream_end(self, context, *, resuming):
            raise RuntimeError("err")
        async def before_execute_tools(self, context):
            raise RuntimeError("err")
        async def after_iteration(self, context):
            raise RuntimeError("err")

    class Good(AgentHook):
        async def on_stream_end(self, context, *, resuming):
            calls.append("on_stream_end")
        async def before_execute_tools(self, context):
            calls.append("before_execute_tools")
        async def after_iteration(self, context):
            calls.append("after_iteration")

    hook = CompositeHook([Bad(), Good()])
    ctx = _ctx()
    await hook.on_stream_end(ctx, resuming=False)
    await hook.before_execute_tools(ctx)
    await hook.after_iteration(ctx)
    assert calls == ["on_stream_end", "before_execute_tools", "after_iteration"]


@pytest.mark.asyncio
async def test_composite_reraise_hook_propagates_without_isolation():
    calls: list[str] = []

    class Strict(AgentHook):
        def __init__(self) -> None:
            super().__init__(reraise=True)

        async def before_iteration(self, context: AgentHookContext) -> None:
            raise RuntimeError("strict hook failed")

    class Good(AgentHook):
        async def before_iteration(self, context: AgentHookContext) -> None:
            calls.append("good")

    with pytest.raises(RuntimeError, match="strict hook failed"):
        await CompositeHook([Strict(), Good()]).before_iteration(_ctx())
    assert calls == []


# ---------------------------------------------------------------------------
# finalize_content: pipeline semantics with error isolation
# ---------------------------------------------------------------------------


def test_composite_finalize_content_pipeline():
    class Upper(AgentHook):
        def finalize_content(self, context, content):
            return content.upper() if content else content

    class Suffix(AgentHook):
        def finalize_content(self, context, content):
            return (content + "!") if content else content

    hook = CompositeHook([Upper(), Suffix()])
    result = hook.finalize_content(_ctx(), "hello")
    assert result == "HELLO!"


def test_composite_finalize_content_none_passthrough():
    hook = CompositeHook([AgentHook()])
    assert hook.finalize_content(_ctx(), None) is None


def test_composite_finalize_content_ordering():
    """First hook transforms first, result feeds second hook."""
    steps: list[str] = []

    class H1(AgentHook):
        def finalize_content(self, context, content):
            steps.append(f"H1:{content}")
            return content.upper()

    class H2(AgentHook):
        def finalize_content(self, context, content):
            steps.append(f"H2:{content}")
            return content + "!"

    hook = CompositeHook([H1(), H2()])
    result = hook.finalize_content(_ctx(), "hi")
    assert result == "HI!"
    assert steps == ["H1:hi", "H2:HI"]


def test_composite_finalize_content_isolates_failures_and_continues(monkeypatch):
    """A failed transform keeps the last content and does not block later hooks."""
    calls: list[str] = []
    logged: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "nanobot.agent.hook.logger.error",
        lambda *args, **kwargs: logged.append(args),
    )

    class Bad(AgentHook):
        def finalize_content(self, context, content):
            calls.append(f"bad:{content}")
            raise RuntimeError(f"failed while handling {content}")

    class Good(AgentHook):
        def finalize_content(self, context, content):
            calls.append(f"good:{content}")
            return f"{content}!"

    content = "sensitive output"
    result = CompositeHook([Bad(), Good()]).finalize_content(_ctx(), content)

    assert result == f"{content}!"
    assert calls == [f"bad:{content}", f"good:{content}"]
    assert all(content not in repr(args) for args in logged)


# ---------------------------------------------------------------------------
# wants_streaming: any-semantics
# ---------------------------------------------------------------------------


def test_composite_wants_streaming_any_true():
    class No(AgentHook):
        def wants_streaming(self):
            return False

    class Yes(AgentHook):
        def wants_streaming(self):
            return True

    hook = CompositeHook([No(), Yes(), No()])
    assert hook.wants_streaming() is True


def test_composite_wants_streaming_all_false():
    hook = CompositeHook([AgentHook(), AgentHook()])
    assert hook.wants_streaming() is False


def test_composite_wants_streaming_empty():
    hook = CompositeHook([])
    assert hook.wants_streaming() is False


def test_composite_wants_streaming_isolates_failures_and_continues(monkeypatch):
    """A failed query is false, but later hooks still decide the any-result."""
    calls: list[str] = []
    logged: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "nanobot.agent.hook.logger.error",
        lambda *args, **kwargs: logged.append(args),
    )

    class Bad(AgentHook):
        def wants_streaming(self):
            calls.append("bad")
            raise RuntimeError("streaming preference failed")

    class Good(AgentHook):
        def wants_streaming(self):
            calls.append("good")
            return True

    assert CompositeHook([Bad(), Good()]).wants_streaming() is True
    assert calls == ["bad", "good"]
    assert all("streaming preference failed" not in repr(args) for args in logged)


# ---------------------------------------------------------------------------
# Empty hooks list: behaves like no-op AgentHook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_composite_empty_hooks_no_ops():
    hook = CompositeHook([])
    ctx = _ctx()
    await hook.before_iteration(ctx)
    await hook.on_stream(ctx, "delta")
    await hook.on_stream_end(ctx, resuming=False)
    await hook.before_execute_tools(ctx)
    await hook.after_iteration(ctx)
    assert hook.finalize_content(ctx, "test") == "test"
