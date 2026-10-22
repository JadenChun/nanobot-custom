"""Test-local home expansion without accessing the operator's real home."""

from pathlib import Path

import pytest


def isolate_path_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    original_expanduser = Path.expanduser

    def expanduser(path: Path) -> Path:
        if path.parts and path.parts[0] == "~":
            return home.joinpath(*path.parts[1:])
        return original_expanduser(path)

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(Path, "expanduser", expanduser)
