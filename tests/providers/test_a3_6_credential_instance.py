"""Regression tests for instance-scoped Codex credential paths (A3-6)."""

from __future__ import annotations

from nanobot.config import loader
from nanobot.providers import codex_auth


def test_codex_default_credential_path_is_instance_scoped(tmp_path, monkeypatch):
    monkeypatch.delenv("NANOBOT_CODEX_AUTH_PATH", raising=False)
    original = loader._current_config_path
    try:
        loader.set_config_path(tmp_path / "instance-a" / "config.json")
        path_a = codex_auth._nanobot_token_path()

        loader.set_config_path(tmp_path / "instance-b" / "config.json")
        path_b = codex_auth._nanobot_token_path()
    finally:
        loader.set_config_path(original)

    assert path_a != path_b
    assert path_a.parent.parent == tmp_path / "instance-a"
    assert path_b.parent.parent == tmp_path / "instance-b"


def test_codex_explicit_credential_path_override_wins(tmp_path, monkeypatch):
    original = loader._current_config_path
    try:
        loader.set_config_path(tmp_path / "instance-a" / "config.json")
        monkeypatch.setenv("NANOBOT_CODEX_AUTH_PATH", str(tmp_path / "explicit" / "codex.json"))

        assert codex_auth._nanobot_token_path().resolve() == (
            tmp_path / "explicit" / "codex.json"
        ).resolve()
    finally:
        loader.set_config_path(original)
