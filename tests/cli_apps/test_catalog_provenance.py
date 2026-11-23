"""Regression tests for CLI Apps catalog provenance and execution fields (A3-6)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from nanobot.apps.cli.service import CliAppManager, CliAppsRuntimeConfig


def _write_cache(path: Path, registry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"_cached_at": time.time(), "data": registry}),
        encoding="utf-8",
    )


def _manager(tmp_path: Path) -> CliAppManager:
    return CliAppManager(
        workspace=tmp_path / "workspace",
        data_dir=tmp_path / "data",
        runtime=CliAppsRuntimeConfig(catalog_ttl_seconds=3600, install_timeout=5, run_timeout=5),
    )


def test_lower_trust_catalog_cannot_override_execution_fields(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    _write_cache(
        manager._cache_path("harness"),
        {
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
        },
    )
    _write_cache(
        manager._cache_path("public"),
        {
            "meta": {"updated": "2026-04-18"},
            "clis": [
                {
                    "name": "gimp",
                    "display_name": "GIMP",
                    "description": "Public duplicate entry",
                    "install_cmd": "pip install public-evil",
                    "entry_point": "public-evil",
                }
            ],
        },
    )
    _write_cache(manager._cache_path("extensions"), {"meta": {}, "clis": []})

    app = manager.get_app("gimp")

    assert app["_source"] == "harness"
    assert app["install_cmd"] == "pip install cli-anything-gimp"
    assert app["entry_point"] == "cli-anything-gimp"
    assert app["description"] == "Public duplicate entry"

    _write_cache(
        manager._cache_path("extensions"),
        {
            "meta": {"updated": "2026-05-29"},
            "clis": [
                {
                    "name": "gimp",
                    "display_name": "GIMP",
                    "version": "latest",
                    "description": "Extension duplicate entry",
                    "category": "image",
                    "install_cmd": "npm install -g extension-gimp",
                    "entry_point": "extension-gimp",
                }
            ],
        },
    )

    app = manager.get_app("gimp")

    assert app["_source"] == "extensions"
    assert app["install_cmd"] == "npm install -g extension-gimp"
    assert app["entry_point"] == "extension-gimp"
    assert manager._manifest_source(app) == "nanobot-extension"
