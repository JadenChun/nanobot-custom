"""CLI generator lifetimes, using native gates/aiohttp without a listener."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import contextmanager
from pathlib import Path

import pytest
import typer
from aiohttp import web

from nanobot import migration_guard
from nanobot.agent import memory_recovery_startup_admission as admission
from nanobot.agent import memory_recovery_storage as storage
from nanobot.cli import commands
from nanobot.config.schema import Config


@pytest.fixture
def admitted_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "instance" / "config.json"
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "workspace")
    config.bind_source_path(config_path)
    gates = []
    events = []
    original_mutation = admission.admitted_startup_mutation

    @contextmanager
    def tracked_mutation(workspace, *, sessions_root):
        assert workspace == config.workspace_path
        assert sessions_root == config_path.parent / "sessions"
        with original_mutation(workspace, sessions_root=sessions_root) as gate:
            assert gate is not None
            gates.append(gate)
            events.append("enter")
            try:
                yield gate
            finally:
                events.append("exit")

    # Positive-only, function-scoped readiness; real receipt admission is retained.
    monkeypatch.setattr(migration_guard, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(commands, "_load_runtime_config", lambda *_: config)
    monkeypatch.setattr(admission, "admitted_startup_mutation", tracked_mutation)
    return config, config_path, gates, events


def _assert_gate_closed(gate) -> None:
    with pytest.raises(storage.MemoryStorageError) as caught:
        gate.require_current()
    assert caught.value.code == "closed"


def test_artifact_context_holds_real_gate_until_exit(admitted_cli) -> None:
    config, config_path, gates, events = admitted_cli
    assert inspect.isgeneratorfunction(commands._admitted_browser_artifacts_dir.__wrapped__)
    with commands._admitted_browser_artifacts_dir(str(config_path), None) as directory:
        assert directory == config.workspace_path / "artifacts" / "browser-tests"
        assert directory.is_dir()
        assert events == ["enter"]
        assert len(gates) == 1
        gates[0].require_current()
    assert events == ["enter", "exit"]
    _assert_gate_closed(gates[0])
    with storage.open_existing_workspace_memory_lock(config.workspace_path) as reopened:
        reopened.require_current()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyError, asyncio.CancelledError])
def test_artifact_body_error_keeps_identity_and_releases_gate(admitted_cli, error_type):
    config, config_path, gates, events = admitted_cli
    error = error_type("synthetic artifact body failure")
    with pytest.raises(error_type) as caught:
        with commands._admitted_browser_artifacts_dir(str(config_path), None):
            gates[0].require_current()
            raise error
    assert caught.value is error
    assert events == ["enter", "exit"]
    _assert_gate_closed(gates[0])
    with storage.open_existing_workspace_memory_lock(config.workspace_path) as reopened:
        reopened.require_current()


def test_artifact_pending_receipt_denies_before_gate_or_directory(admitted_cli, capsys):
    config, config_path, gates, events = admitted_cli
    memory = config.workspace_path / "memory"
    memory.mkdir(parents=True)
    pending = memory / ".consolidation.pending.json"
    pending.write_bytes(b"synthetic private pending receipt")
    with pytest.raises(typer.Exit) as caught:
        with commands._admitted_browser_artifacts_dir(str(config_path), None):
            pytest.fail("pending receipt must prevent yield")
    assert caught.value.exit_code == 1
    assert caught.value.__cause__ is None
    assert gates == [] and events == []
    assert not (config.workspace_path / "artifacts").exists()
    assert pending.read_bytes() == b"synthetic private pending receipt"
    output = capsys.readouterr().out
    assert "memory recovery required" in output
    assert "synthetic private" not in output
    assert ".consolidation.pending.json" not in output


@pytest.mark.parametrize("symlink_parent", [False, True])
def test_artifact_symlink_refusal_precedes_gate(admitted_cli, tmp_path, symlink_parent):
    config, config_path, gates, events = admitted_cli
    external = tmp_path / "external"
    external.mkdir()
    saved = external / "keep.txt"
    saved.write_bytes(b"unchanged")
    config.workspace_path.mkdir()
    parent = config.workspace_path / "artifacts"
    if symlink_parent:
        parent.symlink_to(external, target_is_directory=True)
    else:
        parent.mkdir()
        (parent / "browser-tests").symlink_to(external, target_is_directory=True)
    with pytest.raises(typer.Exit) as caught:
        with commands._admitted_browser_artifacts_dir(str(config_path), None):
            pytest.fail("symlink must prevent yield")
    assert caught.value.exit_code == 1
    assert gates == [] and events == []
    assert saved.read_bytes() == b"unchanged"
    assert list(external.iterdir()) == [saved]


def test_artifact_production_guard_precedes_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(
        commands, "_load_runtime_config",
        lambda *_: pytest.fail("configuration must follow production readiness"),
    )
    workspace = tmp_path / "absent"
    with pytest.raises(migration_guard.MigrationIncompleteError):
        with commands._admitted_browser_artifacts_dir(None, str(workspace)):
            pytest.fail("unadmitted runtime must not yield")
    assert not workspace.exists()


def _capture_api_app(admitted_cli, monkeypatch):
    config, config_path, gates, events = admitted_cli
    app = web.Application()
    seen = {}

    def create_app(**kwargs):
        seen["create"] = kwargs
        assert kwargs["composition"].config is config
        return app

    def run_app(api_app, **kwargs):
        assert api_app is app
        seen["run"] = kwargs

    monkeypatch.setattr("nanobot.api.server.create_app", create_app)
    monkeypatch.setattr(web, "run_app", run_app)
    monkeypatch.setattr(commands, "_set_nanobot_logs", lambda *_: None)
    commands.serve(
        port=18888, host="127.0.0.1", timeout=12.5, verbose=False,
        workspace=None, config=str(config_path),
    )
    assert len(app.cleanup_ctx) == 1
    assert inspect.isasyncgenfunction(app.cleanup_ctx[0])
    assert gates == [] and events == []
    assert seen["run"]["host"] == "127.0.0.1"
    assert seen["run"]["port"] == 18888
    assert seen["create"]["request_timeout"] == 12.5
    assert seen["create"]["api_key"] == ""
    return app


async def test_api_templates_run_once_and_gate_closes_before_yield(admitted_cli, monkeypatch):
    config, _, gates, events = admitted_cli
    app = _capture_api_app(admitted_cli, monkeypatch)
    calls = []

    def sync(workspace):
        assert workspace == config.workspace_path
        assert events == ["enter"]
        gates[0].require_current()
        calls.append(workspace)

    monkeypatch.setattr(commands, "sync_workspace_templates", sync)
    runner = web.AppRunner(app)
    try:
        await runner.setup()
        assert calls == [config.workspace_path]
        assert events == ["enter", "exit"]
        _assert_gate_closed(gates[0])
        with storage.open_existing_workspace_memory_lock(config.workspace_path) as reopened:
            reopened.require_current()
    finally:
        await runner.cleanup()
    assert calls == [config.workspace_path]
    assert events == ["enter", "exit"]


@pytest.mark.parametrize("error_type", [RuntimeError, KeyError, asyncio.CancelledError])
async def test_api_template_error_preserves_identity_and_releases_gate(
    admitted_cli, monkeypatch, error_type,
):
    config, _, gates, events = admitted_cli
    app = _capture_api_app(admitted_cli, monkeypatch)
    error = error_type("synthetic template failure")
    calls = []

    def fail(workspace):
        gates[0].require_current()
        calls.append(workspace)
        raise error

    monkeypatch.setattr(commands, "sync_workspace_templates", fail)
    runner = web.AppRunner(app)
    try:
        with pytest.raises(error_type) as caught:
            await runner.setup()
        assert caught.value is error
    finally:
        await runner.cleanup()
    assert calls == [config.workspace_path]
    assert events == ["enter", "exit"]
    _assert_gate_closed(gates[0])
    with storage.open_existing_workspace_memory_lock(config.workspace_path) as reopened:
        reopened.require_current()


async def test_api_later_startup_failure_does_not_repeat_templates(admitted_cli, monkeypatch):
    config, _, gates, events = admitted_cli
    app = _capture_api_app(admitted_cli, monkeypatch)
    calls = []
    error = RuntimeError("synthetic later startup failure")
    monkeypatch.setattr(commands, "sync_workspace_templates", lambda path: calls.append(path))

    async def fail(_app):
        assert _app is app
        assert events == ["enter", "exit"]
        _assert_gate_closed(gates[0])
        raise error

    app.on_startup.append(fail)
    runner = web.AppRunner(app)
    try:
        with pytest.raises(RuntimeError) as caught:
            await runner.setup()
        assert caught.value is error
    finally:
        await runner.cleanup()
    assert calls == [config.workspace_path]
    assert events == ["enter", "exit"]
