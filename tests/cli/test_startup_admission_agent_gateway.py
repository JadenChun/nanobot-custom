"""CLI startup must admit receipt evidence before local setup and child launch."""

import importlib
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from nanobot.cli import gateway_runtime
from nanobot.config.schema import Config
from tests.agent.test_memory_recovery_quarantine_storage import _real_layout, _seal_real_receipt
from tests.cli.test_gateway_commands import _test_app


def _pending(workspace: Path) -> None:
    memory = workspace / "memory"
    memory.mkdir(parents=True)
    (memory / ".consolidation.pending.json").write_text("private receipt")


@pytest.mark.parametrize("classic", [False, True])
def test_agent_denies_before_native_tui_or_classic_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, classic: bool,
) -> None:
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "workspace")
    _pending(config.workspace_path)
    provider, launch = Mock(), Mock()
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr("nanobot.cli.agent._load_runtime_config", lambda *_: config)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", provider)
    tui_module = ModuleType("nanobot.cli.tui_launcher")
    tui_module.launch_tui = launch  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nanobot.cli.tui_launcher", tui_module)
    agent_module = importlib.import_module("nanobot.cli.agent")
    with pytest.raises(RuntimeError, match="memory recovery required") as failure:
        agent_module.agent(
            message="hello" if classic else None,
            session_id=None, workspace=None, config=None,
            markdown=True, logs=False, log_file=None, classic=classic, theme="auto",
        )
    assert "private receipt" not in str(failure.value)
    provider.assert_not_called()
    launch.assert_not_called()


@pytest.mark.parametrize("mode", ["--foreground", "--background"])
def test_gateway_denies_before_bundle_and_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "workspace")
    _pending(config.workspace_path)
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    app, runtime, _service, run_calls, prepare_calls = _test_app(tmp_path, config)
    result = CliRunner().invoke(app, ["gateway", mode])
    assert result.exit_code != 0
    assert "memory recovery required" in str(result.exception)
    assert prepare_calls == run_calls == []
    assert runtime.started_options is None


def test_direct_gateway_denies_before_bundle_or_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "workspace")
    _pending(config.workspace_path)
    bundle, templates = Mock(), Mock()
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr(gateway_runtime, "_prepare_webui_bundle_for_gateway", bundle)
    monkeypatch.setattr(gateway_runtime, "sync_workspace_templates", templates)
    with pytest.raises(RuntimeError, match="memory recovery required"):
        gateway_runtime._run_gateway(config)
    bundle.assert_not_called()
    templates.assert_not_called()


def test_sealed_gateway_uses_exact_configured_root_after_mutable_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    _seal_real_receipt(workspace, sessions, receipt, binding)
    (workspace / "memory" / "MEMORY.md").write_bytes(b"legitimate later update\n")
    config = Config()
    config.agents.defaults.workspace = str(workspace)
    config.bind_source_path(sessions.parent / "config.json")
    assert config.runtime_data_dir / "sessions" == sessions
    app, runtime, _service, _calls, bundle_calls = _test_app(tmp_path, config)
    result = CliRunner().invoke(app, ["gateway", "--background"])
    assert result.exit_code == 0, str(result.exception)
    assert len(bundle_calls) == 1
    assert runtime.started_options is not None

    config.bind_source_path(tmp_path / "other" / "config.json")
    app, runtime, _service, _calls, bundle_calls = _test_app(tmp_path, config)
    result = CliRunner().invoke(app, ["gateway", "--background"])
    assert result.exit_code != 0
    assert "memory recovery required" in str(result.exception)
    assert bundle_calls == []
    assert runtime.started_options is None


def test_corrupt_historical_seal_denies_gateway_before_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    _seal_real_receipt(workspace, sessions, receipt, binding)
    manifests = workspace / "memory" / ".recovery" / "transactions" / binding.transaction_identity
    next(manifests.glob("*.json")).write_bytes(b"corrupt private evidence\n")
    config = Config()
    config.agents.defaults.workspace = str(workspace)
    config.bind_source_path(sessions.parent / "config.json")
    app, runtime, _service, run_calls, bundle_calls = _test_app(tmp_path, config)
    result = CliRunner().invoke(app, ["gateway", "--background"])
    assert result.exit_code != 0
    assert str(result.exception) == "memory recovery required"
    assert "corrupt private evidence" not in result.output
    assert bundle_calls == run_calls == []
    assert runtime.started_options is None


@pytest.mark.parametrize("evidence", ["pending", "corrupt_seal"])
@pytest.mark.parametrize("command", ["restart", "install-service"])
def test_gateway_management_writer_denies_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evidence: str, command: str,
) -> None:
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    config = Config()
    if evidence == "pending":
        config.agents.defaults.workspace = str(tmp_path / "workspace")
        _pending(config.workspace_path)
    else:
        workspace, sessions, receipt, binding = _real_layout(tmp_path)
        _seal_real_receipt(workspace, sessions, receipt, binding)
        manifests = workspace / "memory" / ".recovery" / "transactions" / binding.transaction_identity
        next(manifests.glob("*.json")).write_bytes(b"private corrupt seal evidence\n")
        config.agents.defaults.workspace = str(workspace)
        config.bind_source_path(sessions.parent / "config.json")
    app, runtime, installer, run_calls, bundle_calls = _test_app(tmp_path, config)
    result = CliRunner().invoke(app, ["gateway", command])
    assert result.exit_code != 0
    assert str(result.exception) == "memory recovery required"
    assert "private" not in result.output
    assert bundle_calls == run_calls == []
    assert runtime.restarted_options is None
    assert installer.installed_options is None


def test_clean_install_service_still_runs_and_dry_run_remains_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "workspace")
    app, _runtime, installer, _run_calls, _bundle_calls = _test_app(tmp_path, config)
    result = CliRunner().invoke(app, ["gateway", "install-service"])
    assert result.exit_code == 0, str(result.exception)
    assert installer.installed_options is not None
    assert installer.install_dry_run is False

    _pending(config.workspace_path)
    app, _runtime, installer, _run_calls, _bundle_calls = _test_app(tmp_path, config)
    result = CliRunner().invoke(app, ["gateway", "install-service", "--dry-run"])
    assert result.exit_code == 0, str(result.exception)
    assert installer.installed_options is not None
    assert installer.install_dry_run is True
