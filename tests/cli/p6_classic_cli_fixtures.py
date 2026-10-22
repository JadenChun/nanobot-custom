"""Positive-only classic CLI infrastructure; no readiness or authority bypass."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.channels.manager import ChannelManager
from nanobot.config.schema import Config
from nanobot.cron.runtime import CronRuntimeInjection
from nanobot.cron.service import CronService
from nanobot.runtime_composition import (
    ChannelBuildOptions,
    RuntimeComposition,
    RuntimeFactories,
    RuntimeState,
)
from nanobot.session.runtime_authority import DurableSessionAuthority, SessionAuthority


def _denied(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("classic CLI fixture external effect denied")


class ClassicCliRuntime:
    def __init__(self, root: Path, patches: pytest.MonkeyPatch) -> None:
        assert root.is_absolute()
        self.root = root
        self.compositions: list[RuntimeComposition] = []
        original_define = RuntimeComposition.define.__func__

        def define(
            cls: type[RuntimeComposition], config: Config, *,
            sessions_root: Path | None = None,
            session_authority: SessionAuthority | None = None,
            factories: RuntimeFactories | None = None,
            channel_options: ChannelBuildOptions | None = None,
            startup_receipt_admitted: bool = False,
            cron_injection: CronRuntimeInjection | None = None,
        ) -> RuntimeComposition:
            workspace = config.workspace_path
            assert workspace.is_absolute() and workspace.is_relative_to(root)
            parts = workspace.relative_to(root).parts
            assert ".." not in parts
            cursor = root
            assert not cursor.is_symlink()
            for part in parts:
                cursor /= part
                assert not cursor.is_symlink()
            assert not config.tools.mcp_servers
            assert channel_options is None and cron_injection is None
            # The unit test owns this pre-existing workspace. Production define
            # remains read-only and the actual writer lease is still acquired.
            workspace.mkdir(parents=True, exist_ok=True)
            value = original_define(
                cls, config, sessions_root=sessions_root, session_authority=session_authority,
                factories=factories, channel_options=channel_options,
                startup_receipt_admitted=startup_receipt_admitted, cron_injection=cron_injection,
            )
            self.compositions.append(value)
            return value

        patches.setattr(RuntimeComposition, "define", classmethod(define))
        for name in ("connect", "connect_ex", "bind", "listen", "sendto"):
            patches.setattr(socket.socket, name, _denied)
        for name in ("getaddrinfo", "create_connection"):
            patches.setattr(socket, name, _denied)
        patches.setattr(subprocess, "Popen", _denied)
        for name in ("system", "fork", "posix_spawn", "posix_spawnp", "execv", "execve"):
            if hasattr(os, name):
                patches.setattr(os, name, _denied)
        patches.setattr(asyncio, "create_subprocess_exec", _denied)
        patches.setattr(asyncio, "create_subprocess_shell", _denied)
        patches.setattr(ChannelManager, "acquire", _denied)
        patches.setattr(CronService, "start", _denied)
        patches.setattr(AgentLoop, "run", _denied)

    def verify_closed(self) -> None:
        for graph in self.compositions:
            assert graph.state in (RuntimeState.CLOSED, RuntimeState.FAILED)
            assert graph._owner_loop is not None and graph._owner_loop.is_closed()

            async def verify_release() -> None:
                lease = await DurableSessionAuthority(
                    graph.preflight.workspace, sessions_root=graph.preflight.sessions_root,
                ).acquire()
                try:
                    assert lease.manager.has_managed_authority
                finally:
                    assert lease.close is not None
                    result = lease.close()
                    assert result is not None
                    await result

            asyncio.run(verify_release())


@pytest.fixture
def classic_cli_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[ClassicCliRuntime]:
    value = ClassicCliRuntime(tmp_path, monkeypatch)
    try:
        yield value
    finally:
        value.verify_closed()
