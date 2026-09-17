"""Descriptor-backed source-root authority checks."""

from __future__ import annotations

import copy
import os
import pickle
from pathlib import Path

import pytest

import nanobot.session.source_root as source_root_module
from nanobot.session.manager import JsonlSessionStore
from nanobot.session.source_root import (
    SourceRootAuthority,
    SourceRootAuthorityError,
    SourceRootUnavailableError,
)


def test_source_root_authority_is_opaque_and_process_bound(tmp_path: Path) -> None:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    authority = backend.source.create_source_root_authority()
    try:
        assert type(authority) is SourceRootAuthority
        with pytest.raises(TypeError):
            copy.copy(authority)
        with pytest.raises(TypeError):
            copy.deepcopy(authority)
        with pytest.raises(TypeError):
            pickle.dumps(authority)
        authority.require_current(backend)
    finally:
        authority.close()
    assert authority.closed
    with pytest.raises(SourceRootAuthorityError):
        authority.require_current(backend)


def test_root_replacement_fails_closed_before_descriptor_transaction(tmp_path: Path) -> None:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    authority = backend.source.create_source_root_authority()
    old_root = backend.sessions_dir.with_name("old-source-root")
    try:
        os.rename(backend.sessions_dir, old_root)
        backend.sessions_dir.mkdir()
        with pytest.raises(SourceRootAuthorityError):
            authority.require_current(backend)
    finally:
        authority.close()


def test_source_root_fails_closed_when_posix_lock_primitives_are_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    monkeypatch.setattr(source_root_module, "fcntl", None)
    with pytest.raises(SourceRootUnavailableError):
        SourceRootAuthority.issue(backend)
