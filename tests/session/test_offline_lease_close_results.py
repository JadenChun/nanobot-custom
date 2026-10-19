"""Existing close callback contract, including synchronous completion.

Callback adapters borrow a real manager under a separate test-owned lease.
They qualify dispatch/error behavior, not release by a production adapter.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
from unittest.mock import Mock

import pytest

from nanobot.cli import commands
from nanobot.config.schema import Config
from nanobot.session import artifact_operator, runtime_authority
from nanobot.session.artifact_operator import ArtifactOperatorError
from nanobot.session.runtime_authority import OfflineSessionOperator, SessionAuthorityLease
from tests.session.test_artifact_operator import copied_artifact_fixture

_ENTRIES = ("cli", "inspect", "rematerialize")
_STYLES = (
    "missing", "sync_none", "async", "factory", "awaitable", "falsy_awaitable",
    "bad_return", "sync_error", "async_error", "cancel",
)


def _denied(*_args, **_kwargs):
    raise AssertionError("offline close unit external effect denied")


@pytest.fixture(autouse=True)
def _fence_external_effects(monkeypatch):
    for name in ("connect", "connect_ex", "bind", "listen", "sendto"):
        monkeypatch.setattr(socket.socket, name, _denied)
    monkeypatch.setattr(socket, "getaddrinfo", _denied)
    monkeypatch.setattr(socket, "create_connection", _denied)
    monkeypatch.setattr(subprocess, "Popen", _denied)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _denied)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", _denied)


class _CloseAwaitable:
    def __init__(self, finish, *, falsy=False):
        self.finish = finish
        self.falsy = falsy

    def __await__(self):
        return self.finish().__await__()

    def __bool__(self):
        if self.falsy:
            return False
        raise AssertionError("close result truthiness must not be inspected")


def _callback(style, events, failure):
    owner_loop = asyncio.get_running_loop()

    async def finish():
        assert asyncio.get_running_loop() is owner_loop
        events.append("await")
        await asyncio.sleep(0)
        if style in ("async_error", "cancel"):
            raise failure

    def synchronous():
        assert asyncio.get_running_loop() is owner_loop
        events.append("sync")
        if style == "sync_error":
            raise failure
        if style == "bad_return":
            return 0  # Intentionally invalid, falsy, non-None and non-awaitable.
        if style in ("factory", "async_error", "cancel"):
            return finish()
        if style in ("awaitable", "falsy_awaitable"):
            return _CloseAwaitable(finish, falsy=style == "falsy_awaitable")
        return None

    if style == "missing":
        return None
    return Mock(side_effect=finish if style == "async" else synchronous)


async def _exercise(tmp_path, monkeypatch, entry, close, *, primary=None):
    async with copied_artifact_fixture(tmp_path) as fixture:
        operator = fixture["operator"]
        authority = operator._sessions
        workspace, sessions_root = authority.workspace, authority.sessions_root
        real_lease = await authority.acquire()
        source_before = fixture["source_path"].read_bytes()
        mode_before = fixture["record_path"].read_bytes()
        borrowed = SessionAuthorityLease(real_lease.manager, close)
        marker = object()

        async def acquire():
            return borrowed

        class CallbackUnitAuthority:
            async def acquire(self):
                return await acquire()

        try:
            with monkeypatch.context() as patches:
                if entry == "cli":
                    config = Config()
                    config.agents.defaults.workspace = str(fixture["live_workspace"])

                    def define(received_workspace, **options):
                        assert received_workspace == workspace
                        assert options == {
                            "copied_sessions_root": sessions_root,
                            "live_workspace": fixture["live_workspace"],
                            "live_sessions_root": fixture["live_sessions_root"],
                        }
                        return CallbackUnitAuthority()

                    def operation(manager):
                        assert manager is real_lease.manager
                        if primary is not None:
                            raise primary
                        return marker

                    patches.setattr(runtime_authority, "OfflineSessionOperator", define)
                    result = await commands._with_offline_manager(
                        config, fixture["live_sessions_root"], workspace, sessions_root, operation,
                    )
                    assert result is marker
                else:
                    patches.setattr(operator, "_sessions", CallbackUnitAuthority())
                    if primary is not None:
                        def fail_snapshot(*_args, **_kwargs):
                            raise primary

                        patches.setattr(artifact_operator, "_canonical_snapshot", fail_snapshot)
                    if entry == "inspect":
                        result = await operator.inspect(fixture["key"])
                        assert result.status == "complete"
                    else:
                        result = await operator.rematerialize(
                            fixture["key"], incarnation=fixture["incarnation"],
                            generation=fixture["generation"],
                            reference=fixture["descriptor"].reference,
                        )
                        assert result.status == "already_present"
        finally:
            # Scripted callbacks never claim to release this separate real lease.
            assert real_lease.close is not None
            completion = real_lease.close()
            assert completion is not None
            await completion
            assert fixture["source_path"].read_bytes() == source_before
            assert fixture["record_path"].read_bytes() == mode_before
            # A new native authority must acquire after the actual owner released.
            fresh = await OfflineSessionOperator(
                workspace, copied_sessions_root=sessions_root,
            ).acquire()
            try:
                assert fresh.manager.has_managed_authority
            finally:
                assert fresh.close is not None
                completion = fresh.close()
                assert completion is not None
                await completion


@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize("style", _STYLES)
async def test_all_callers_obey_close_result_contract(tmp_path, monkeypatch, entry, style):
    events = []
    failure = asyncio.CancelledError("unit cancellation") if style == "cancel" else RuntimeError(
        "private close failure payload",
    )
    close = _callback(style, events, failure)
    exceptional = style in ("bad_return", "sync_error", "async_error", "cancel")
    if exceptional:
        expected = asyncio.CancelledError if style == "cancel" else (
            TypeError if style == "bad_return" and entry == "cli" else
            RuntimeError if entry == "cli" else ArtifactOperatorError
        )
        with pytest.raises(expected) as caught:
            await _exercise(tmp_path, monkeypatch, entry, close)
        if style == "cancel" or (entry == "cli" and style != "bad_return"):
            assert caught.value is failure
        elif entry != "cli":
            assert str(caught.value) == "Copied artifact operation is unavailable"
            assert "private close failure payload" not in str(caught.value)
            assert caught.value.__suppress_context__ is True
    else:
        await _exercise(tmp_path, monkeypatch, entry, close)
    if close is None:
        assert events == []
    else:
        close.assert_called_once_with()
        expected_events = ["await"] if style == "async" else ["sync"]
        if style in ("factory", "awaitable", "falsy_awaitable", "async_error", "cancel"):
            expected_events.append("await")
        assert events == expected_events


@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize("style", ("missing", "sync_none", "async"))
@pytest.mark.parametrize("kind", ("error", "cancel"))
async def test_operation_failure_not_masked_by_valid_close(
    tmp_path, monkeypatch, entry, style, kind,
):
    primary = RuntimeError("private operation failure") if kind == "error" else asyncio.CancelledError(
        "operation cancellation",
    )
    events = []
    close = _callback(style, events, RuntimeError("unused"))
    expected = asyncio.CancelledError if kind == "cancel" else (
        RuntimeError if entry == "cli" else ArtifactOperatorError
    )
    with pytest.raises(expected) as caught:
        await _exercise(tmp_path, monkeypatch, entry, close, primary=primary)
    if kind == "cancel" or entry == "cli":
        assert caught.value is primary
    else:
        assert caught.value.__context__ is primary
        assert caught.value.__suppress_context__ is True
        assert str(caught.value) == "Copied artifact operation is unavailable"
        assert "private operation failure" not in str(caught.value)
    if close is None:
        assert events == []
    else:
        close.assert_called_once_with()
        assert events == (["sync"] if style == "sync_none" else ["await"])
