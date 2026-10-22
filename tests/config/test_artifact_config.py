"""Configuration contract for opt-in private context artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nanobot.config.loader import load_config, save_config
from nanobot.config.schema import Config


def test_artifacts_default_to_disabled_with_documented_caps() -> None:
    artifacts = Config().artifacts

    assert artifacts.enabled is False
    assert artifacts.root is None
    assert artifacts.max_blob_bytes == 64 * 1024 * 1024
    assert artifacts.max_namespace_bytes == 1024 * 1024 * 1024


def test_artifact_settings_round_trip_with_camel_case_names(tmp_path: Path) -> None:
    root = tmp_path / "private-artifacts"
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"artifacts": {
        "enabled": True,
        "root": str(root),
        "maxBlobBytes": 1024,
        "maxNamespaceBytes": 4096,
    }}), encoding="utf-8")

    config = load_config(source)
    target = tmp_path / "saved.json"
    save_config(config, target)
    reloaded = load_config(target)

    assert reloaded.artifacts == config.artifacts
    assert json.loads(target.read_text(encoding="utf-8"))["artifacts"] == {
        "enabled": True,
        "root": str(root),
        "maxBlobBytes": 1024,
        "maxNamespaceBytes": 4096,
    }


def test_legacy_config_omitting_artifacts_stays_disabled_and_lossless(tmp_path: Path) -> None:
    source = tmp_path / "legacy.json"
    payload = {"agents": {"defaults": {"workspace": str(tmp_path / "workspace")}}}
    source.write_text(json.dumps(payload), encoding="utf-8")

    config = load_config(source)

    assert config.artifacts.enabled is False
    assert config.artifacts.root is None
    assert json.loads(source.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize("artifacts", [
    {"enabled": 1},
    {"enabled": "false"},
    {"root": ""},
    {"root": "relative/artifacts"},
    {"root": "/"},
    {"maxBlobBytes": True},
    {"maxNamespaceBytes": 1, "maxBlobBytes": 2},
])
def test_artifact_settings_reject_non_strict_or_unsafe_values(artifacts: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"artifacts": artifacts})
