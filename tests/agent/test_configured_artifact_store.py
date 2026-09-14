"""Configured artifact storage stays opt-in and invocation-local."""

from __future__ import annotations

import pytest

from agent.test_migration_engine import ScriptedProvider
from agent.test_migration_engine import engine_config as engine_config
from agent.test_migration_source_engine import upgrade
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import TurnRequest
from nanobot.providers.base import LLMResponse
from nanobot.session.artifact import ArtifactValidationError
from nanobot.session.manager import SessionPolicy


async def test_enabled_config_captures_default_root_and_registers_only_per_invocation(engine_config) -> None:
    engine_config.artifacts.enabled = True
    original_data_dir = engine_config.runtime_data_dir
    assert original_data_dir is not None
    provider = ScriptedProvider([LLMResponse(content="done")])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        upgrade(loop, "cli:artifact")
        factory = loop._artifact_reader_factory
        assert factory is not None
        assert factory.settings.root == original_data_dir / "context-artifacts"
        assert not loop.tools.has("read_artifact")

        engine_config.bind_source_path(original_data_dir.parent / "other-instance" / "config.json")
        assert factory.settings.root == original_data_dir / "context-artifacts"

        result = await loop.execute_turn(TurnRequest(content="hello", session_key="cli:artifact"))

        assert result.content == "done"
        assert "read_artifact" in {
            tool["function"]["name"] for tool in provider.calls[0][1]
        }
        assert (original_data_dir / "context-artifacts").is_dir()
        assert not loop.tools.has("read_artifact")
    finally:
        await loop.aclose()


async def test_disabled_config_never_constructs_private_artifact_storage(engine_config) -> None:
    data_dir = engine_config.runtime_data_dir
    assert data_dir is not None
    provider = ScriptedProvider([LLMResponse(content="done")])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        upgrade(loop, "cli:disabled-artifact")
        assert loop._artifact_reader_factory is None
        result = await loop.execute_turn(TurnRequest(content="hello", session_key="cli:disabled-artifact"))

        assert result.content == "done"
        assert not (data_dir / "context-artifacts").exists()
        assert "read_artifact" not in {
            tool["function"]["name"] for tool in provider.calls[0][1]
        }
    finally:
        await loop.aclose()


async def test_enabled_loops_capture_independent_config_data_roots(engine_config, tmp_path) -> None:
    engine_config.artifacts.enabled = True
    first_data_dir = engine_config.runtime_data_dir
    assert first_data_dir is not None
    second_config = engine_config.model_copy(deep=True)
    second_config.bind_source_path(tmp_path / "second-instance" / "config.json")
    second_config.artifacts.enabled = True
    second_data_dir = second_config.runtime_data_dir
    assert second_data_dir is not None

    first = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=ScriptedProvider([]),
    )
    second = AgentLoop.from_config(
        second_config,
        tool_registry=ToolRegistry(),
        provider=ScriptedProvider([]),
    )
    try:
        assert first._artifact_reader_factory is not None
        assert second._artifact_reader_factory is not None
        assert first._artifact_reader_factory.settings.root == first_data_dir / "context-artifacts"
        assert second._artifact_reader_factory.settings.root == second_data_dir / "context-artifacts"
        assert first._artifact_reader_factory.settings.root != second._artifact_reader_factory.settings.root
    finally:
        await first.aclose()
        await second.aclose()


async def test_session_capability_disable_removes_configured_artifact_reader(engine_config) -> None:
    engine_config.artifacts.enabled = True
    provider = ScriptedProvider([LLMResponse(content="done")])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        upgrade(loop, "cli:artifact-disabled")
        session = loop.sessions.get_or_create("cli:artifact-disabled")
        session.policy = SessionPolicy(disabled_tools=frozenset({"read_artifact"}))
        result = await loop.execute_turn(
            TurnRequest(content="hello", session_key="cli:artifact-disabled")
        )

        assert result.content == "done"
        assert "read_artifact" not in {
            tool["function"]["name"] for tool in provider.calls[0][1]
        }
    finally:
        await loop.aclose()


async def test_enabled_config_preserves_an_injected_artifact_reader_factory(engine_config) -> None:
    engine_config.artifacts.enabled = True

    def injected_factory(*_args):
        return None

    loop = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=ScriptedProvider([]),
        artifact_reader_factory=injected_factory,
    )
    try:
        assert loop._artifact_reader_factory is injected_factory
        assert loop.runner.artifact_reader_factory is injected_factory
    finally:
        await loop.aclose()


async def test_unsafe_enabled_artifact_root_rejects_loop_startup(engine_config) -> None:
    engine_config.artifacts.enabled = True
    engine_config.artifacts.root = str(engine_config.workspace_path)

    with pytest.raises(ArtifactValidationError, match="overlaps"):
        AgentLoop.from_config(
            engine_config,
            tool_registry=ToolRegistry(),
            provider=ScriptedProvider([]),
        )
