from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path

from kohaku.agent.tools.cli_apps import CliAppsTool
from kohaku.agent.tools.context import RequestContext
from kohaku.apps.cli.service import CliAppManager, CliAppsRuntimeConfig


def _write_cache(path: Path, registry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"_cached_at": time.time(), "data": registry}),
        encoding="utf-8",
    )


def test_run_cli_app_uses_installed_registry_app(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = tmp_path / "data"
    registry = {
        "meta": {"updated": "2026-04-16"},
        "clis": [
            {
                "name": "gimp",
                "display_name": "GIMP",
                "version": "1.0.0",
                "description": "Image editing",
                "category": "image",
                "install_cmd": "pip install cli-anything-gimp",
                "entry_point": "cli-anything-gimp",
            }
        ],
    }
    _write_cache(data_dir / "harness_registry_cache.json", registry)
    _write_cache(data_dir / "public_registry_cache.json", {"meta": {}, "clis": []})
    _write_cache(data_dir / "extensions_registry_cache.json", {"meta": {}, "clis": []})
    CliAppManager(workspace=workspace, data_dir=data_dir)._save_installed(
        {"gimp": {"entry_point": "cli-anything-gimp"}}
    )
    resolved = str(tmp_path / "bin" / "cli-anything-gimp")
    monkeypatch.setattr(
        "kohaku.apps.cli.service.shutil.which",
        lambda entry: resolved if entry == "cli-anything-gimp" else None,
    )

    def fake_run_owned(
        self,
        argv: list[str],
        *,
        timeout: int,
        cwd: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="tool:" + " ".join(argv[1:]),
            stderr="",
        )

    monkeypatch.setattr(CliAppManager, "_run_owned_argv", fake_run_owned)
    monkeypatch.setattr("kohaku.apps.cli.service.get_runtime_subdir", lambda _name: data_dir)

    tool = CliAppsTool(
        workspace=workspace,
        restrict_to_workspace=True,
        runtime=CliAppsRuntimeConfig(run_timeout=5),
    )
    assert tool.name == "run_cli_app"

    result = asyncio.run(
        tool.execute(
            name="gimp",
            args=["project", "list"],
            json=True,
            working_dir=str(workspace),
        )
    )

    assert "CLI app 'gimp' exited 0" in result
    assert "tool:--json project list" in result


def test_run_cli_app_rejects_uninstalled_app(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = tmp_path / "data"
    registry = {
        "meta": {"updated": "2026-04-16"},
        "clis": [
            {
                "name": "gimp",
                "display_name": "GIMP",
                "version": "1.0.0",
                "description": "Image editing",
                "category": "image",
                "install_cmd": "pip install cli-anything-gimp",
                "entry_point": "cli-anything-gimp",
            }
        ],
    }
    _write_cache(data_dir / "harness_registry_cache.json", registry)
    _write_cache(data_dir / "public_registry_cache.json", {"meta": {}, "clis": []})
    _write_cache(data_dir / "extensions_registry_cache.json", {"meta": {}, "clis": []})
    monkeypatch.setattr("kohaku.apps.cli.service.get_runtime_subdir", lambda _name: data_dir)
    tool = CliAppsTool(workspace=workspace, restrict_to_workspace=True)

    result = asyncio.run(tool.execute(name="gimp"))

    assert "not installed" in result


def test_run_cli_app_keeps_installed_inventory_in_runtime_context(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = tmp_path / "data"
    monkeypatch.setattr("kohaku.apps.cli.service.get_runtime_subdir", lambda _name: data_dir)
    tool = CliAppsTool(workspace=workspace)
    before = tool.description
    CliAppManager(workspace=workspace, data_dir=data_dir)._save_installed(
        {"drawio": {"entry_point": "cli-anything-drawio"}}
    )

    assert before == tool.description
    assert "ordinary system CLIs such as git, gh" in tool.description
    provider = tool.runtime_context_provider()
    blocks = asyncio.run(provider(RequestContext(
        channel="cli", chat_id="chat", original_user_text="hello", workspace=workspace,
    )))
    assert blocks and blocks[0].source == "cli_apps_index"
    assert "Installed CLI Apps: drawio" in blocks[0].content


def test_cli_app_tool_provides_attachment_context(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("kohaku.apps.cli.service.get_runtime_subdir", lambda _name: tmp_path)
    tool = CliAppsTool(workspace=tmp_path)
    provider = tool.runtime_context_provider()
    assert provider is not None

    empty = asyncio.run(provider(RequestContext(
        channel="websocket",
        chat_id="chat",
        original_user_text="hello",
        workspace=tmp_path,
    )))
    attached = asyncio.run(provider(RequestContext(
        channel="websocket",
        chat_id="chat",
        original_user_text="use @drawio",
        metadata={
            "cli_apps": [{
                "name": "drawio",
                "entry_point": "cli-anything-drawio",
            }],
        },
        workspace=tmp_path,
    )))

    assert empty is None
    assert attached and len(attached) == 1
    assert attached[0].source == "cli_apps"
    assert "CLI App Attachment: @drawio" in attached[0].content
