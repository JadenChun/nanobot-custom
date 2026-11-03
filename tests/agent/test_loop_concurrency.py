from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.turn import TurnRequest
from tests.agent.p6_admitted_fixtures import AgentInputs
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async
from tests.agent.p6_admitted_fixtures import admitted_sync as admitted_sync
from tests.agent.test_migration_canonical_admission import host


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = SimpleNamespace(
        max_tokens=4096,
        temperature=0.1,
        reasoning_effort=None,
    )
    return provider


def _make_loop(inputs: AgentInputs) -> AgentLoop:
    return AgentLoop(
        bus=inputs.bus,
        provider=_provider(),
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


@pytest.mark.parametrize("value", [None, "2", "not-a-number"])
def test_retired_concurrency_environment_does_not_block_loop_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    admitted_sync,
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv("NANOBOT_MAX_CONCURRENT_REQUESTS", raising=False)
    else:
        monkeypatch.setenv("NANOBOT_MAX_CONCURRENT_REQUESTS", value)

    loop = admitted_sync(tmp_path, _make_loop)

    assert loop.model == "test-model"


@pytest.mark.parametrize("value", ["2", "not-a-number"])
async def test_retired_concurrency_environment_does_not_serialize_unrelated_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str,
) -> None:
    monkeypatch.setenv("NANOBOT_MAX_CONCURRENT_REQUESTS", value)
    loop = host(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    calls: list[str] = []

    async def stage(msg, **kwargs):
        calls.append(msg.content)
        if msg.content == "first":
            entered.set()
            await release.wait()
        return SimpleNamespace(
            run_result=AgentRunResult(final_content=msg.content, messages=[]),
            outbound=None,
        )

    loop._process_turn.side_effect = stage
    first = asyncio.create_task(
        loop.execute_turn(TurnRequest(content="first", session_key="cli:first"))
    )
    await entered.wait()
    second = await loop.execute_turn(TurnRequest(content="second", session_key="cli:second"))
    assert second.status.value == "completed"
    assert calls == ["first", "second"]
    release.set()
    await first
