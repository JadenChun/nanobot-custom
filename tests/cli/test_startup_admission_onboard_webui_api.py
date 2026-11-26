"""Preserve onboarding choices after the retirement of receipt admission."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from kohaku.cli import commands
from kohaku.config.schema import Config
from kohaku.utils import workspace_lock


def _unexpected(*args: object, **kwargs: object) -> None:
    pytest.fail("declined onboarding performed a write")


def _configured(tmp_path: Path) -> Config:
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "selected")
    config.bind_source_path(tmp_path / "config.json")
    return config


def test_onboard_declined_wizard_does_not_write(tmp_path, monkeypatch):
    from kohaku import migration_guard
    from kohaku.cli import onboard as onboard_module
    from kohaku.config import loader

    monkeypatch.setattr(migration_guard, "require_runtime_ready", lambda: None)
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


def test_onboard_uses_final_wizard_workspace(tmp_path, monkeypatch):
    from kohaku import migration_guard
    from kohaku.cli import onboard as onboard_module
    from kohaku.config import loader

    monkeypatch.setattr(migration_guard, "require_runtime_ready", lambda: None)
    final = _configured(tmp_path)
    final.agents.defaults.workspace = str(tmp_path / "wizard-selected")
    monkeypatch.setattr(onboard_module, "run_onboard", lambda **_: SimpleNamespace(
        should_save=True, config=final,
    ))
    calls = []

    @contextmanager
    def mutation(workspace):
        calls.append(("enter", workspace))
        yield None
        calls.append(("exit", workspace))

    monkeypatch.setattr(workspace_lock, "open_workspace_memory_lock", mutation)
    monkeypatch.setattr(loader, "save_config", lambda *_: calls.append(("save",)))
    monkeypatch.setattr(commands, "_onboard_plugins", lambda *_: calls.append(("plugins",)))
    monkeypatch.setattr(commands, "sync_workspace_templates", lambda *_: calls.append(("templates",)))
    commands.onboard(
        workspace=None, config=str(tmp_path / "config.json"), wizard=True,
        non_interactive_refresh=False,
    )
    assert calls == [
        ("save",), ("plugins",), ("enter", tmp_path / "wizard-selected"),
        ("templates",), ("exit", tmp_path / "wizard-selected"),
    ]


@pytest.mark.parametrize("overwrite", [True, False])
def test_onboard_existing_config_choice_saves_and_syncs(tmp_path, monkeypatch, overwrite):
    from kohaku import migration_guard
    from kohaku.config import loader

    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(migration_guard, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(typer, "confirm", lambda *_: overwrite)
    monkeypatch.setattr(loader, "load_config", lambda *_: _configured(tmp_path))
    calls = []
    @contextmanager
    def mutation(*args, **kwargs):
        calls.append("enter")
        yield None
        calls.append("exit")

    monkeypatch.setattr(workspace_lock, "open_workspace_memory_lock", mutation)
    monkeypatch.setattr(loader, "save_config", lambda *_: calls.append("save"))
    monkeypatch.setattr(commands, "_onboard_plugins", lambda *_: calls.append("plugins"))
    monkeypatch.setattr(commands, "sync_workspace_templates", lambda *_: calls.append("templates"))
    commands.onboard(
        workspace=str(tmp_path / "selected"), config=str(config_path), wizard=False,
        non_interactive_refresh=False,
    )
    assert calls == ["save", "plugins", "enter", "templates", "exit"]
