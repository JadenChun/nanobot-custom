from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from tests.agent.p6_admitted_fixtures import AgentInputs
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async
from tests.agent.p6_admitted_fixtures import admitted_sync as admitted_sync


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


def test_request_concurrency_is_unlimited_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    admitted_sync,
) -> None:
    monkeypatch.delenv("NANOBOT_MAX_CONCURRENT_REQUESTS", raising=False)

    loop = admitted_sync(tmp_path, _make_loop)

    assert loop._concurrency_gate is None


@pytest.mark.asyncio
async def test_positive_request_concurrency_keeps_explicit_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    admitted_async,
) -> None:
    monkeypatch.setenv("NANOBOT_MAX_CONCURRENT_REQUESTS", "2")
    loop = await admitted_async(tmp_path, _make_loop)
    gate = loop._concurrency_gate

    assert gate is not None
    for _ in range(2):
        await gate.acquire()
    try:
        assert gate.locked()
    finally:
        for _ in range(2):
            gate.release()
