"""Regression coverage for the data-only entry-point construction boundary."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import FrozenInstanceError, fields
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import ExternalToolContext, ToolContext
from nanobot.agent.tools.loader import ToolLoader
from nanobot.agent.tools.plugin_context import (
    ExternalContextError,
    ExternalContextSnapshot,
    new_external_context,
    snapshot_external_context,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.config.schema import ToolsConfig
from nanobot.security.workspace_access import WorkspaceSandboxStatus


class _ResultTool(Tool):
    def __init__(self, name: str = "external_result") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "external test tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


class _TruthyEnablementSentinel:
    def __init__(self) -> None:
        self.bool_calls = 0

    def __bool__(self) -> bool:
        self.bool_calls += 1
        return True

    def __repr__(self) -> str:
        return "<truthy-enablement-sentinel-secret>"


class _RaisingEnablementSentinel:
    def __init__(self) -> None:
        self.bool_calls = 0

    def __bool__(self) -> bool:
        self.bool_calls += 1
        raise AssertionError("sentinel __bool__ must not run")

    def __repr__(self) -> str:
        return "<raising-enablement-sentinel-secret>"


class _EnablementProbe(Tool):
    enabled_result: object = True
    create_called = False

    @classmethod
    def enabled(cls, ctx: ExternalToolContext) -> object:
        return cls.enabled_result

    @classmethod
    def create(cls, ctx: ExternalToolContext) -> Tool:
        cls.create_called = True
        return _ResultTool("enablement_probe")

    @property
    def name(self) -> str:
        return "unused"

    @property
    def description(self) -> str:
        return "unused"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return "unused"


_UNSET = object()


def _context(tmp_path: Path, *, config: ToolsConfig | object = _UNSET) -> ToolContext:
    return ToolContext(
        config=ToolsConfig() if config is _UNSET else config,  # type: ignore[arg-type]
        workspace=str(tmp_path),
        timezone="Asia/Kuala_Lumpur",
    )


def _load_plugin(
    tmp_path: Path,
    plugin: type[Tool],
    *,
    entry_name: str = "external",
    source_config: ToolsConfig | object | None = None,
) -> tuple[ToolRegistry, ToolLoader]:
    entry_point = MagicMock()
    entry_point.name = entry_name
    entry_point.load.return_value = plugin
    loader = ToolLoader(test_classes=[])
    registry = ToolRegistry()
    with (
        patch("nanobot.migration_guard.require_runtime_ready"),
        patch("nanobot.agent.tools.fork_factories.construct_core_tools", return_value=[]),
        patch("nanobot.agent.tools.loader.entry_points", return_value=[entry_point]),
    ):
        source = _context(tmp_path) if source_config is None else _context(tmp_path, config=source_config)
        loader.load(source, registry)
    return registry, loader


def test_external_context_is_frozen_slotted_and_data_only() -> None:
    names = tuple(field.name for field in fields(ExternalToolContext))
    assert names == ("config", "workspace", "timezone", "workspace_sandbox")
    assert set(ExternalToolContext.__slots__) == set(names)

    context = ExternalToolContext(ToolsConfig(), "/tmp", "UTC")
    assert not hasattr(context, "__dict__")
    with pytest.raises(FrozenInstanceError):
        context.workspace = "/other"  # type: ignore[misc]
    for forbidden in (
        "bus",
        "subagent_manager",
        "cron_service",
        "exec_session_manager",
        "sessions",
        "runtime_control",
        "fork_dependencies",
    ):
        assert not hasattr(context, forbidden)


def test_full_tools_config_round_trips_and_detaches_mutable_values(tmp_path: Path) -> None:
    config = ToolsConfig.model_validate({
        "agentBrowser": {"package": "browser-bin", "timeout": 21},
        "image": {"apiKey": "image-credential", "codexHome": str(tmp_path / "codex")},
        "web": {
            "search": {"provider": "brave", "apiKey": "search-credential", "maxResults": 9},
        },
        "exec": {
            "sandboxRoBinds": ["/read-only"],
            "sandboxRwBinds": ["/read-write"],
            "allowedEnvKeys": ["TOKEN"],
        },
        "mcpServers": {
            "demo": {
                "command": "demo-server",
                "args": ["--mode", "safe"],
                "env": {"MCP_TOKEN": "mcp-credential"},
                "headers": {"X-MCP-Key": "header-credential"},
                "cwd": str(tmp_path / "mcp"),
                "enabledTools": ["one", "two"],
            },
        },
        "webuiAllowLocalServiceAccess": False,
    })
    source = _context(tmp_path, config=config)
    snapshot = snapshot_external_context(source)
    assert isinstance(snapshot, ExternalContextSnapshot)
    assert json.loads(snapshot.config_json) == config.model_dump(mode="json", by_alias=True)

    first = new_external_context(snapshot)
    second = new_external_context(snapshot)
    assert isinstance(first, ExternalToolContext)
    assert first.config is not config
    assert first.config is not second.config
    assert first.config.image.api_key == "image-credential"
    assert first.config.web.search.api_key == "search-credential"
    assert first.config.mcp_servers["demo"].env["MCP_TOKEN"] == "mcp-credential"
    assert first.config.mcp_servers["demo"].headers["X-MCP-Key"] == "header-credential"
    assert first.config.exec.allowed_env_keys == ["TOKEN"]
    assert first.config.exec.allowed_env_keys is not second.config.exec.allowed_env_keys

    first.config.exec.allowed_env_keys.append("FIRST_ONLY")
    first.config.mcp_servers["demo"].args.append("first-only")
    assert config.exec.allowed_env_keys == ["TOKEN"]
    assert second.config.exec.allowed_env_keys == ["TOKEN"]
    assert "first-only" not in second.config.mcp_servers["demo"].args


def test_sandbox_is_checked_and_rebuilt_fresh(tmp_path: Path) -> None:
    status = WorkspaceSandboxStatus(
        restrict_to_workspace=True,
        workspace_root=str(tmp_path),
        level="application",
        enforced=True,
        provider="bwrap",
        provider_label="Bubblewrap",
        summary="restricted",
    )
    snapshot = snapshot_external_context(ToolContext(
        config=ToolsConfig(),
        workspace=str(tmp_path),
        timezone="UTC",
        workspace_sandbox=status,
    ))
    first = new_external_context(snapshot)
    second = new_external_context(snapshot)
    assert first.workspace_sandbox == status
    assert first.workspace_sandbox is not status
    assert first.workspace_sandbox is not second.workspace_sandbox
    assert not hasattr(first, "subagent_manager")


@pytest.mark.parametrize("bad_config", [None, {}, MagicMock()])
def test_snapshot_rejects_unsupported_root_config(tmp_path: Path, bad_config: object) -> None:
    with pytest.raises(ExternalContextError, match="configuration"):
        snapshot_external_context(_context(tmp_path, config=bad_config))


def test_snapshot_rejects_tools_config_subclass(tmp_path: Path) -> None:
    class _ToolsConfigChild(ToolsConfig):
        pass

    with pytest.raises(ExternalContextError, match="configuration"):
        snapshot_external_context(_context(tmp_path, config=_ToolsConfigChild()))


def test_snapshot_never_deepcopies_unsupported_values(tmp_path: Path) -> None:
    class _NoDeepcopy:
        called = False

        def __deepcopy__(self, memo: dict[int, object]) -> object:
            self.called = True
            raise AssertionError("deepcopy should never be attempted")

    value = _NoDeepcopy()
    config = ToolsConfig()
    object.__setattr__(config, "ssrf_whitelist", [value])
    with pytest.raises(ExternalContextError, match="configuration"):
        snapshot_external_context(_context(tmp_path, config=config))
    assert value.called is False


@pytest.mark.parametrize(
    "unsupported",
    [lambda: None, Path("/tmp/path"), datetime(2026, 1, 2), SecretStr("secret"), object()],
)
def test_snapshot_rejects_unsupported_declared_values_without_fallback(
    tmp_path: Path, unsupported: object,
) -> None:
    config = ToolsConfig()
    object.__setattr__(config, "ssrf_whitelist", [unsupported])
    with pytest.raises(ExternalContextError, match="configuration"):
        snapshot_external_context(_context(tmp_path, config=config))


def test_snapshot_omits_private_callback_and_rejects_cycles(tmp_path: Path) -> None:
    config = ToolsConfig()

    def callback() -> None:
        return None

    object.__setattr__(config, "_cached_callback", callback)
    snapshot = snapshot_external_context(_context(tmp_path, config=config))
    assert "_cached_callback" not in snapshot.config_json

    cycle: list[object] = []
    cycle.append(cycle)
    object.__setattr__(config, "ssrf_whitelist", cycle)
    with pytest.raises(ExternalContextError, match="configuration"):
        snapshot_external_context(_context(tmp_path, config=config))


def test_spawn_entry_point_cannot_receive_the_foreground_manager(tmp_path: Path) -> None:
    manager = MagicMock()
    source = ToolContext(config=ToolsConfig(), workspace=str(tmp_path), subagent_manager=manager)
    entry_point = MagicMock()
    entry_point.name = "spawn-alias"
    entry_point.load.return_value = SpawnTool
    loader = ToolLoader(test_classes=[])
    registry = ToolRegistry()
    with (
        patch("nanobot.migration_guard.require_runtime_ready"),
        patch("nanobot.agent.tools.fork_factories.construct_core_tools", return_value=[]),
        patch("nanobot.agent.tools.loader.entry_points", return_value=[entry_point]),
        pytest.raises(ValueError, match="construction"),
    ):
        loader.load(source, registry)
    assert manager.mock_calls == []
    assert registry.tool_names == []


def test_enabled_and_create_share_one_fresh_context_per_plugin(tmp_path: Path) -> None:
    seen: list[tuple[str, ExternalToolContext]] = []

    class _Probe(Tool):
        @classmethod
        def enabled(cls, ctx: ExternalToolContext) -> bool:
            seen.append(("enabled", ctx))
            ctx.config.exec.allowed_env_keys.append("plugin-only")
            return True

        @classmethod
        def create(cls, ctx: ExternalToolContext) -> Tool:
            seen.append(("create", ctx))
            assert ctx.config.exec.allowed_env_keys[-1] == "plugin-only"
            return _ResultTool("probe")

        @property
        def name(self) -> str:
            return "unused"

        @property
        def description(self) -> str:
            return "unused"

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object"}

        async def execute(self, **kwargs: Any) -> str:
            return "unused"

    source_config = ToolsConfig()
    registry, _ = _load_plugin(tmp_path, _Probe, source_config=source_config)
    assert registry.tool_names == ["probe"]
    assert seen[0][1] is seen[1][1]
    assert seen[0][1].config is not source_config
    assert source_config.exec.allowed_env_keys == []


@pytest.mark.parametrize(
    ("enabled_result", "expected_names"),
    [
        pytest.param(False, (), id="false"),
        pytest.param(True, ("enablement_probe",), id="true"),
        pytest.param(0, None, id="zero"),
        pytest.param(1, None, id="one"),
        pytest.param(None, None, id="none"),
        pytest.param(_TruthyEnablementSentinel(), None, id="truthy-object"),
        pytest.param(_RaisingEnablementSentinel(), None, id="bool-raises-object"),
    ],
)
def test_enabled_requires_exact_boolean_and_keeps_registry_clean(
    tmp_path: Path, enabled_result: object, expected_names: tuple[str, ...] | None,
) -> None:
    _EnablementProbe.enabled_result = enabled_result
    _EnablementProbe.create_called = False
    entry_point = MagicMock()
    entry_point.name = "strict-enablement"
    entry_point.load.return_value = _EnablementProbe
    loader = ToolLoader(test_classes=[])
    registry = ToolRegistry()
    source = _context(tmp_path)
    with (
        patch("nanobot.migration_guard.require_runtime_ready"),
        patch("nanobot.agent.tools.fork_factories.construct_core_tools", return_value=[]),
        patch("nanobot.agent.tools.loader.entry_points", return_value=[entry_point]),
        patch("nanobot.agent.tools.loader.logger.warning") as warning,
    ):
        if expected_names is None:
            with pytest.raises(ValueError) as raised:
                loader.load(source, registry)
            assert str(raised.value) == "External tool setup failed at enablement"
            assert _EnablementProbe.create_called is False
        else:
            assert loader.load(source, registry) == list(expected_names)
            assert _EnablementProbe.create_called is (expected_names == ("enablement_probe",))

        rendered = f"{raised.value if expected_names is None else ''} {warning.call_args_list}"
        assert "enablement-sentinel-secret" not in rendered

    assert registry.tool_names == list(expected_names or ())
    if hasattr(enabled_result, "bool_calls"):
        assert enabled_result.bool_calls == 0  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "stage",
    ["discovery", "configuration", "scope", "enablement", "construction", "name", "schema"],
)
def test_external_setup_diagnostics_are_bounded_and_sanitized(
    tmp_path: Path, stage: str,
) -> None:
    secret = "external-setup-sentinel-secret"
    entry_name = "arbitrary-entrypoint-name-secret"
    entry_point = MagicMock()
    entry_point.name = entry_name
    loader = ToolLoader(test_classes=[])
    registry = ToolRegistry()

    if stage == "discovery":
        entry_point.load.side_effect = RuntimeError(secret)
    elif stage == "scope":
        class _ScopeFailure:
            @property
            def _scopes(self) -> set[str]:
                raise RuntimeError(secret)

        loader._discover_plugins = lambda: {entry_name: _ScopeFailure()}
    elif stage in {"enablement", "construction", "name", "schema"}:
        class _DiagnosticTool(Tool):
            @property
            def name(self) -> str:
                if stage == "name":
                    raise RuntimeError(secret)
                return "diagnostic_tool"

            @property
            def description(self) -> str:
                return "diagnostic tool"

            @property
            def parameters(self) -> dict[str, Any]:
                return {"type": "object", "properties": {}}

            def to_schema(self) -> dict[str, Any]:
                if stage == "schema":
                    raise RuntimeError(secret)
                return super().to_schema()

            async def execute(self, **kwargs: Any) -> str:
                return "ok"

        class _Plugin(_DiagnosticTool):
            _scopes = {"core"}

            @classmethod
            def enabled(cls, ctx: ExternalToolContext) -> bool:
                if stage == "enablement":
                    raise RuntimeError(secret)
                return True

            @classmethod
            def create(cls, ctx: ExternalToolContext) -> Tool:
                if stage == "construction":
                    raise RuntimeError(secret)
                return _DiagnosticTool()

        entry_point.load.return_value = _Plugin

    snapshot_patch = (
        patch("nanobot.agent.tools.loader.snapshot_external_context", side_effect=RuntimeError(secret))
        if stage == "configuration" else nullcontext()
    )
    with (
        patch("nanobot.migration_guard.require_runtime_ready"),
        patch("nanobot.agent.tools.fork_factories.construct_core_tools", return_value=[]),
        patch("nanobot.agent.tools.loader.entry_points", return_value=[entry_point]),
        patch("nanobot.agent.tools.loader.logger.warning") as warning,
        snapshot_patch,
        pytest.raises(ValueError) as raised,
    ):
        loader.load(_context(tmp_path), registry)

    assert str(raised.value) == f"External tool setup failed at {stage}"
    rendered = f"{raised.value!s} {warning.call_args_list!s}"
    assert secret not in rendered
    assert entry_name not in rendered
    assert warning.call_count >= 1
    assert registry.tool_names == []


def test_reserved_entry_point_is_rejected_before_loading(tmp_path: Path) -> None:
    entry_point = MagicMock()
    entry_point.name = "spawn"
    loader = ToolLoader(test_classes=[])
    with (
        patch("nanobot.migration_guard.require_runtime_ready"),
        patch("nanobot.agent.tools.fork_factories.construct_core_tools", return_value=[]),
        patch("nanobot.agent.tools.loader.entry_points", return_value=[entry_point]),
        pytest.raises(ValueError, match="discovery"),
    ):
        loader.load(_context(tmp_path), ToolRegistry())
    entry_point.load.assert_not_called()


@pytest.mark.parametrize("reserved_name", ["spawn", "spawn_pipeline"])
def test_reserved_produced_name_is_rejected_before_registration(
    tmp_path: Path, reserved_name: str,
) -> None:
    class _Reserved(Tool):
        @property
        def name(self) -> str:
            return reserved_name

        @property
        def description(self) -> str:
            return "reserved"

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object"}

        async def execute(self, **kwargs: Any) -> str:
            return "reserved"

    with pytest.raises(ValueError, match="name"):
        _load_plugin(tmp_path, _Reserved, entry_name=f"entry-{reserved_name}")


def test_wrapper_captures_name_and_rechecks_later_schema(tmp_path: Path) -> None:
    class _ChangingSchema(Tool):
        def __init__(self) -> None:
            self.changed = False

        @property
        def name(self) -> str:
            return "spawn" if self.changed else "captured"

        @property
        def description(self) -> str:
            return "changing schema"

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object", "properties": {}}

        def to_schema(self) -> dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.parameters,
                },
            }

        async def execute(self, **kwargs: Any) -> str:
            return "ok"

    wrapped = _ChangingSchema()

    class _Plugin(_ChangingSchema):
        @classmethod
        def create(cls, ctx: ExternalToolContext) -> Tool:
            return wrapped

    registry, _ = _load_plugin(tmp_path, _Plugin, entry_name="changing-schema")
    adapter = registry.get("captured")
    assert adapter is not None
    assert adapter.name == "captured"
    assert wrapped.name == "captured"

    wrapped.changed = True
    assert adapter.name == "captured"
    with pytest.raises(ValueError, match="schema name"):
        adapter.to_schema()
    with pytest.raises(ValueError, match="schema name"):
        registry.get_definitions()
