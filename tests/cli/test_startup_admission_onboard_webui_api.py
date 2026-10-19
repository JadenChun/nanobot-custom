"""CLI setup must admit the final workspace before its first mutation."""

import asyncio
import importlib
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from nanobot.agent.memory_recovery_startup_admission import StartupReceiptAdmissionError
from nanobot.cli import commands
from nanobot.config.schema import Config
from tests.agent.test_memory_recovery_quarantine_storage import _real_layout, _seal_real_receipt
from tests.agent.test_memory_recovery_transaction_terminal import _transaction_chain


def _deny(workspace: Path, *, sessions_root: Path) -> None:
    raise StartupReceiptAdmissionError()


def _unexpected(*args: object, **kwargs: object) -> None:
    pytest.fail("mutation preceded startup receipt admission")


@pytest.mark.skipif(os.name != "posix", reason="real seal requires POSIX descriptor protocol")
@pytest.mark.parametrize("entrypoint", ["onboard", "webui", "serve"])
@pytest.mark.parametrize("damage", ["corrupt-semantic-blob", "wrong-sessions-root"])
def test_real_seal_cli_denial_precedes_any_write(
    tmp_path, monkeypatch, capsys, entrypoint, damage,
):
    """A structural v2 seal does not admit corrupt history or another instance root."""
    from nanobot import migration_guard
    from nanobot.config import loader
    from nanobot.gateway import GatewayInstance
    from nanobot.session import manager as session_module

    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    _seal_real_receipt(workspace, sessions, receipt, binding)
    assert not receipt.exists()
    config_path = tmp_path / "config.json"
    if damage == "corrupt-semantic-blob":
        _, records = _transaction_chain(workspace, receipt_bytes=(
            workspace / "memory" / ".recovery" / "receipt-quarantine"
            / binding.transaction_identity / "receipt.bin"
        ).read_bytes())
        digest = records[1].payload["source"]["new_digest"]
        blob = workspace / "memory" / ".recovery" / "blobs" / "sha256" / digest
        blob.write_bytes(b"corrupt semantic blob\n")
    else:
        config_path = tmp_path / "other-instance" / "config.json"
    config = Config()
    config.agents.defaults.workspace = str(workspace)
    config.bind_source_path(config_path)

    monkeypatch.setattr(migration_guard, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(loader, "save_config", _unexpected)
    monkeypatch.setattr(commands, "sync_workspace_templates", _unexpected)
    monkeypatch.setattr(commands, "_onboard_plugins", _unexpected)
    monkeypatch.setattr(GatewayInstance, "resolve", _unexpected)
    monkeypatch.setattr(session_module, "SessionManager", _unexpected)

    if entrypoint == "onboard":
        # No pre-existing config: final --workspace is known before save.
        def action():
            commands.onboard(
                workspace=str(workspace), config=str(config_path), wizard=False,
                non_interactive_refresh=False,
            )
    elif entrypoint == "webui":
        webui_module = importlib.import_module("nanobot.cli.webui")
        monkeypatch.setattr(webui_module.cli_terminal, "_ensure_interactive_tty_mode", lambda: None)
        monkeypatch.setattr(webui_module, "_load_webui_setup_config", lambda *_: config)
        monkeypatch.setattr(webui_module, "_ensure_local_webui_channel", lambda *a, **kw: (True, None))
        monkeypatch.setattr(webui_module, "_warn_webui_bind_scope", lambda *_: None)
        monkeypatch.setattr(webui_module, "_webui_browser_url", lambda *_: "http://localhost/")
        monkeypatch.setattr(webui_module, "sync_workspace_templates", _unexpected)
        monkeypatch.setattr(webui_module, "_prepare_webui_bundle_for_gateway", _unexpected)
        def action():
            webui_module.webui(
                port=None, gateway_port=None, workspace=str(workspace),
                config=str(config_path), background=False, dev=False, no_open=True, yes=True,
            )
    else:
        monkeypatch.setattr(commands, "_load_runtime_config", lambda *_: config)
        def action():
            commands.serve(
                port=None, host=None, timeout=None, verbose=False,
                workspace=None, config=str(config_path),
            )

    with pytest.raises(typer.Exit) as exc:
        action()
    assert exc.value.exit_code == 1
    output = capsys.readouterr().out
    assert "memory recovery required" in output
    assert binding.transaction_identity not in output
    assert ".consolidation.pending.json" not in output
    assert "corrupt semantic blob" not in output
    assert not config_path.exists()


def test_onboard_denial_precedes_config_save_and_plugins(tmp_path, monkeypatch):
    from nanobot import migration_guard
    from nanobot.agent import memory_recovery_startup_admission as admission
    from nanobot.config import loader

    monkeypatch.setattr(migration_guard, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(admission, "require_startup_receipt_admission", _deny)
    monkeypatch.setattr(loader, "save_config", _unexpected)
    monkeypatch.setattr(commands, "_onboard_plugins", _unexpected)
    monkeypatch.setattr(commands, "sync_workspace_templates", _unexpected)

    with pytest.raises(typer.Exit) as exc:
        commands.onboard(
            workspace=str(tmp_path / "selected"), config=str(tmp_path / "config.json"),
            wizard=False, non_interactive_refresh=False,
        )
    assert exc.value.exit_code != 0
    assert not (tmp_path / "selected").exists()


def test_onboard_real_pending_receipt_denies_without_revealing_name(tmp_path, monkeypatch, capsys):
    from nanobot import migration_guard
    from nanobot.config import loader

    workspace = tmp_path / "selected"
    memory = workspace / "memory"
    memory.mkdir(parents=True)
    (memory / ".consolidation.pending.json").write_text("secret", encoding="utf-8")
    monkeypatch.setattr(migration_guard, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(loader, "save_config", _unexpected)
    monkeypatch.setattr(commands, "_onboard_plugins", _unexpected)
    monkeypatch.setattr(commands, "sync_workspace_templates", _unexpected)
    with pytest.raises(typer.Exit) as exc:
        commands.onboard(
            workspace=str(workspace), config=str(tmp_path / "config.json"),
            wizard=False, non_interactive_refresh=False,
        )
    assert exc.value.exit_code == 1
    output = capsys.readouterr().out
    assert "memory recovery required" in output
    assert "secret" not in output
    assert ".consolidation.pending.json" not in output
    assert not (tmp_path / "config.json").exists()


def test_serve_denial_precedes_templates_and_session(tmp_path, monkeypatch):
    from nanobot import migration_guard
    from nanobot.agent import memory_recovery_startup_admission as admission

    monkeypatch.setattr(migration_guard, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(admission, "require_startup_receipt_admission", _deny)
    monkeypatch.setattr(commands, "_load_runtime_config", lambda *_: _configured(tmp_path))
    monkeypatch.setattr(commands, "sync_workspace_templates", _unexpected)

    with pytest.raises(typer.Exit) as exc:
        commands.serve(
            port=None, host=None, timeout=None, verbose=False,
            workspace=None, config=str(tmp_path / "config.json"),
        )
    assert exc.value.exit_code != 0
    assert not (tmp_path / "selected").exists()


def test_serve_explicit_root_and_authority_outside_setup_lock(tmp_path, monkeypatch):
    from aiohttp import web

    from nanobot import migration_guard
    from nanobot.agent import memory_recovery_startup_admission as admission
    from nanobot.runtime_composition import RuntimeComposition

    monkeypatch.setattr(migration_guard, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(commands, "_load_runtime_config", lambda *_: _configured(tmp_path))
    held = False
    calls = []

    def preflight(workspace, *, sessions_root):
        calls.append(("admit", workspace, sessions_root))

    @contextmanager
    def mutation(workspace, *, sessions_root):
        nonlocal held
        held = True
        calls.append(("enter", workspace, sessions_root))
        yield None
        held = False

    def sync(workspace):
        assert held
        calls.append(("templates", workspace))

    class Authority:
        async def acquire(self):
            assert not held
            calls.append(("authority",))
            raise StartupReceiptAdmissionError()

    original_define = RuntimeComposition.define

    def define(config, **kwargs):
        assert kwargs["sessions_root"] == tmp_path / "sessions"
        return original_define(config, session_authority=Authority(), **kwargs)

    def run_app(app, **_kwargs):
        async def setup():
            runner = web.AppRunner(app)
            try:
                await runner.setup()
            finally:
                await runner.cleanup()

        asyncio.run(setup())

    monkeypatch.setattr(admission, "require_startup_receipt_admission", preflight)
    monkeypatch.setattr(admission, "admitted_startup_mutation", mutation)
    monkeypatch.setattr(commands, "sync_workspace_templates", sync)
    monkeypatch.setattr(RuntimeComposition, "define", define)
    monkeypatch.setattr(web, "run_app", run_app)
    with pytest.raises(StartupReceiptAdmissionError):
        commands.serve(
            port=None, host=None, timeout=None, verbose=False,
            workspace=None, config=str(tmp_path / "config.json"),
        )
    assert calls == [
        ("admit", tmp_path / "selected", tmp_path / "sessions"),
        ("authority",),
    ]


def _configured(tmp_path: Path) -> Config:
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "selected")
    config.bind_source_path(tmp_path / "config.json")
    return config


def test_webui_denial_precedes_config_save_templates_and_bundle(tmp_path, monkeypatch):
    from nanobot import migration_guard
    from nanobot.agent import memory_recovery_startup_admission as admission
    from nanobot.config import loader

    webui_module = importlib.import_module("nanobot.cli.webui")
    monkeypatch.setattr(migration_guard, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(admission, "require_startup_receipt_admission", _deny)
    monkeypatch.setattr(webui_module.cli_terminal, "_ensure_interactive_tty_mode", lambda: None)
    monkeypatch.setattr(webui_module, "_load_webui_setup_config", lambda *_: _configured(tmp_path))
    monkeypatch.setattr(webui_module, "_ensure_local_webui_channel", lambda *a, **kw: (True, None))
    monkeypatch.setattr(webui_module, "_warn_webui_bind_scope", lambda *_: None)
    monkeypatch.setattr(webui_module, "_webui_browser_url", lambda *_: "http://localhost/")
    monkeypatch.setattr(loader, "save_config", _unexpected)
    monkeypatch.setattr(webui_module, "sync_workspace_templates", _unexpected)
    monkeypatch.setattr(webui_module, "_prepare_webui_bundle_for_gateway", _unexpected)

    with pytest.raises(typer.Exit) as exc:
        webui_module.webui(
            port=None, gateway_port=None, workspace=str(tmp_path / "selected"),
            config=str(tmp_path / "config.json"), background=False, dev=False,
            no_open=True, yes=True,
        )
    assert exc.value.exit_code != 0
    assert not (tmp_path / "selected").exists()


def test_onboard_declined_wizard_has_no_admission_or_mutation(tmp_path, monkeypatch):
    from nanobot import migration_guard
    from nanobot.agent import memory_recovery_startup_admission as admission
    from nanobot.cli import onboard as onboard_module
    from nanobot.config import loader

    monkeypatch.setattr(migration_guard, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(admission, "require_startup_receipt_admission", _unexpected)
    monkeypatch.setattr(loader, "save_config", _unexpected)
    monkeypatch.setattr(commands, "_onboard_plugins", _unexpected)
    monkeypatch.setattr(commands, "sync_workspace_templates", _unexpected)
    monkeypatch.setattr(onboard_module, "run_onboard", lambda **_: SimpleNamespace(should_save=False))

    commands.onboard(
        workspace=str(tmp_path / "selected"), config=str(tmp_path / "config.json"),
        wizard=True, non_interactive_refresh=False,
    )
    assert not (tmp_path / "selected").exists()
    assert not (tmp_path / "config.json").exists()


def test_onboard_final_wizard_workspace_and_explicit_root(tmp_path, monkeypatch):
    from nanobot import migration_guard
    from nanobot.agent import memory_recovery_startup_admission as admission
    from nanobot.cli import onboard as onboard_module
    from nanobot.config import loader

    monkeypatch.setattr(migration_guard, "require_runtime_ready", lambda: None)
    final = _configured(tmp_path)
    final.agents.defaults.workspace = str(tmp_path / "wizard-selected")
    monkeypatch.setattr(onboard_module, "run_onboard", lambda **_: SimpleNamespace(
        should_save=True, config=final,
    ))
    calls = []

    def preflight(workspace, *, sessions_root):
        calls.append(("admit", workspace, sessions_root))

    @contextmanager
    def mutation(workspace, *, sessions_root):
        calls.append(("enter", workspace, sessions_root))
        yield None
        calls.append(("exit", workspace, sessions_root))

    monkeypatch.setattr(admission, "require_startup_receipt_admission", preflight)
    monkeypatch.setattr(admission, "admitted_startup_mutation", mutation)
    monkeypatch.setattr(loader, "save_config", lambda *_: calls.append(("save",)))
    monkeypatch.setattr(commands, "_onboard_plugins", lambda *_: calls.append(("plugins",)))
    monkeypatch.setattr(commands, "sync_workspace_templates", lambda *_: calls.append(("templates",)))
    commands.onboard(
        workspace=None, config=str(tmp_path / "config.json"), wizard=True,
        non_interactive_refresh=False,
    )
    root = tmp_path / "sessions"
    assert calls == [
        ("admit", tmp_path / "wizard-selected", root),
        ("enter", tmp_path / "wizard-selected", root),
        ("save",), ("plugins",), ("templates",),
        ("exit", tmp_path / "wizard-selected", root),
    ]


@pytest.mark.parametrize("overwrite", [True, False])
def test_onboard_existing_config_choice_admitted_before_save(tmp_path, monkeypatch, overwrite):
    from nanobot import migration_guard
    from nanobot.agent import memory_recovery_startup_admission as admission
    from nanobot.config import loader

    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(migration_guard, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(typer, "confirm", lambda *_: overwrite)
    monkeypatch.setattr(loader, "load_config", lambda *_: _configured(tmp_path))
    calls = []
    monkeypatch.setattr(admission, "require_startup_receipt_admission", lambda *a, **kw: (
        calls.append("admit")
    ))

    @contextmanager
    def mutation(*args, **kwargs):
        calls.append("enter")
        yield None
        calls.append("exit")

    monkeypatch.setattr(admission, "admitted_startup_mutation", mutation)
    monkeypatch.setattr(loader, "save_config", lambda *_: calls.append("save"))
    monkeypatch.setattr(commands, "_onboard_plugins", lambda *_: calls.append("plugins"))
    monkeypatch.setattr(commands, "sync_workspace_templates", lambda *_: calls.append("templates"))
    commands.onboard(
        workspace=str(tmp_path / "selected"), config=str(config_path), wizard=False,
        non_interactive_refresh=False,
    )
    assert calls == ["admit", "enter", "save", "plugins", "templates", "exit"]


def test_webui_rechecks_after_releasing_setup_lock_before_bundle(tmp_path, monkeypatch):
    from nanobot import migration_guard
    from nanobot.agent import memory_recovery_startup_admission as admission
    from nanobot.config import loader

    webui_module = importlib.import_module("nanobot.cli.webui")
    monkeypatch.setattr(migration_guard, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(webui_module.cli_terminal, "_ensure_interactive_tty_mode", lambda: None)
    monkeypatch.setattr(webui_module, "_load_webui_setup_config", lambda *_: _configured(tmp_path))
    monkeypatch.setattr(webui_module, "_load_runtime_config", lambda *_: _configured(tmp_path))
    monkeypatch.setattr(webui_module, "_ensure_local_webui_channel", lambda *a, **kw: (True, None))
    monkeypatch.setattr(webui_module, "_warn_webui_bind_scope", lambda *_: None)
    monkeypatch.setattr(webui_module, "_webui_browser_url", lambda *_: "http://localhost/")
    monkeypatch.setattr(webui_module, "_webui_build_mode_for_interactive", lambda **_: "warn")
    monkeypatch.setattr(webui_module, "sync_workspace_templates", lambda *_: None)
    monkeypatch.setattr(loader, "save_config", lambda *_: None)
    held = False
    checks = []

    def preflight(workspace, *, sessions_root):
        checks.append((workspace, sessions_root, held))

    @contextmanager
    def mutation(workspace, *, sessions_root):
        nonlocal held
        held = True
        yield None
        held = False

    def bundle(*args, **kwargs):
        assert not held
        assert len(checks) == 2
        assert all(root == tmp_path / "sessions" for _, root, _ in checks)
        raise typer.Exit(3)

    monkeypatch.setattr(admission, "require_startup_receipt_admission", preflight)
    monkeypatch.setattr(admission, "admitted_startup_mutation", mutation)
    monkeypatch.setattr(webui_module, "_prepare_webui_bundle_for_gateway", bundle)
    with pytest.raises(typer.Exit) as exc:
        webui_module.webui(
            port=None, gateway_port=None, workspace=str(tmp_path / "selected"),
            config=str(tmp_path / "config.json"), background=False, dev=False,
            no_open=True, yes=True,
        )
    assert exc.value.exit_code == 3


def test_webui_rechecks_after_bundle_before_gateway_lifecycle(tmp_path, monkeypatch):
    from nanobot import migration_guard
    from nanobot.agent import memory_recovery_startup_admission as admission
    from nanobot.config import loader
    from nanobot.gateway import GatewayInstance

    webui_module = importlib.import_module("nanobot.cli.webui")
    monkeypatch.setattr(migration_guard, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(webui_module.cli_terminal, "_ensure_interactive_tty_mode", lambda: None)
    monkeypatch.setattr(webui_module, "_load_webui_setup_config", lambda *_: _configured(tmp_path))
    monkeypatch.setattr(webui_module, "_load_runtime_config", lambda *_: _configured(tmp_path))
    monkeypatch.setattr(webui_module, "_ensure_local_webui_channel", lambda *a, **kw: (True, None))
    monkeypatch.setattr(webui_module, "_warn_webui_bind_scope", lambda *_: None)
    monkeypatch.setattr(webui_module, "_webui_browser_url", lambda *_: "http://localhost/")
    monkeypatch.setattr(webui_module, "_webui_build_mode_for_interactive", lambda **_: "warn")
    monkeypatch.setattr(webui_module, "sync_workspace_templates", lambda *_: None)
    monkeypatch.setattr(loader, "save_config", lambda *_: None)
    monkeypatch.setattr(webui_module, "_prepare_webui_bundle_for_gateway", lambda *a, **kw: None)
    monkeypatch.setattr(GatewayInstance, "resolve", _unexpected)
    checks = []

    def preflight(workspace, *, sessions_root):
        checks.append((workspace, sessions_root))
        if len(checks) == 3:
            raise StartupReceiptAdmissionError()

    @contextmanager
    def mutation(*args, **kwargs):
        yield None

    monkeypatch.setattr(admission, "require_startup_receipt_admission", preflight)
    monkeypatch.setattr(admission, "admitted_startup_mutation", mutation)
    with pytest.raises(typer.Exit) as exc:
        webui_module.webui(
            port=None, gateway_port=None, workspace=str(tmp_path / "selected"),
            config=str(tmp_path / "config.json"), background=False, dev=False,
            no_open=True, yes=True,
        )
    assert exc.value.exit_code == 1
    assert checks == [(tmp_path / "selected", tmp_path / "sessions")] * 3
