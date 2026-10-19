from __future__ import annotations

import asyncio
import gc
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from tests.agent.p6_admitted_fixtures import AgentInputs
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async
from tests.agent.p6_admitted_fixtures import admitted_sync as admitted_sync


def _make_loop(inputs: AgentInputs) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(
        bus=inputs.bus,
        provider=provider,
        workspace=inputs.config.workspace_path,
        model="test-model",
        session_manager=inputs.sessions,
        tool_registry=inputs.tools,
        cron_service=inputs.cron,
        turn_delivery_factory=inputs.delivery,
        tools_config=inputs.config.tools,
        restrict_to_workspace=True,
        enable_legacy_auto_compact=False,
    )


def test_idle_agent_session_locks_are_released(tmp_path: Path, admitted_sync):
    loop = admitted_sync(tmp_path, _make_loop)

    for index in range(1000):
        lock = loop._get_session_lock(f"api:temporary-{index}")

    del lock
    gc.collect()

    assert len(loop._session_locks) == 0


@pytest.mark.asyncio
async def test_waiter_keeps_agent_session_lock_alive(tmp_path: Path, admitted_async):
    loop = await admitted_async(tmp_path, _make_loop)
    owner_lock = loop._get_session_lock("api:shared")
    await owner_lock.acquire()
    waiter_started = asyncio.Event()
    waiter_entered = asyncio.Event()

    async def wait_for_lock() -> None:
        lock = loop._get_session_lock("api:shared")
        waiter_started.set()
        async with lock:
            waiter_entered.set()

    waiter = asyncio.create_task(wait_for_lock())
    await waiter_started.wait()

    assert loop._get_session_lock("api:shared") is owner_lock
    assert not waiter_entered.is_set()

    owner_lock.release()
    await waiter
    del owner_lock
    gc.collect()

    assert "api:shared" not in loop._session_locks
