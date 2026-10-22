"""Cross-suite test infrastructure."""

from __future__ import annotations

import json
import os
import ssl
import sys
from collections.abc import Iterator
from pathlib import Path

import certifi
import pytest
from loguru import logger


@pytest.fixture(autouse=True)
def _isolate_active_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default runtime paths must never resolve into the operator's live home.

    Tests can override the public config path themselves. Register the original
    binding with monkeypatch so both the default and test-specific changes are
    restored, including an originally unbound loader.
    """
    from nanobot.config import loader

    monkeypatch.setattr(loader, "_current_config_path", loader._current_config_path)
    reference = tmp_path.parent / f"{tmp_path.name}-test-config" / "config.json"
    reference.parent.mkdir()
    reference.write_text(json.dumps({
        "agents": {"defaults": {"workspace": str(tmp_path / "workspace")}},
    }), encoding="utf-8")
    loader.set_config_path(reference)


@pytest.fixture(autouse=True)
def _isolate_nanobot_log_activation() -> Iterator[None]:
    """Keep CLI log settings from leaking into later tests in the same process."""
    logger.enable("nanobot")
    try:
        yield
    finally:
        logger.enable("nanobot")


@pytest.fixture(autouse=True)
def _isolate_sessions_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Redirect session storage away from the real active config data directory.

    Session storage lives under the active runtime data root (outside the workspace,
    per ADR-0001), so without redirection tests would write into the real home.
    """
    runtime_root = tmp_path.parent / f"{tmp_path.name}-runtime-root"
    legacy_root = tmp_path.parent / f"{tmp_path.name}-legacy-sessions-root"

    def sessions_root() -> Path:
        # Admission must observe the intended root without creating it.
        return runtime_root / "sessions"

    for target in (
        "nanobot.config.paths.get_default_sessions_root",
        "nanobot.agent.memory.get_default_sessions_root",
        "nanobot.session.manager.get_default_sessions_root",
    ):
        monkeypatch.setattr(target, sessions_root)
    monkeypatch.setattr(
        "nanobot.session.manager.get_legacy_sessions_dir",
        lambda: legacy_root,
    )
    yield


@pytest.fixture(autouse=True)
def _isolate_pairing_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep channel pairing tests out of the user's active pairing store."""
    pairing_path = tmp_path / "pairing.json"
    monkeypatch.setattr(
        "nanobot.pairing.store._store_path",
        lambda: pairing_path,
    )


@pytest.fixture(scope="session", autouse=True)
def _use_windows_system_ca_for_default_http_clients() -> Iterator[None]:
    """Avoid reparsing certifi's CA bundle for every offline HTTP client.

    Loading certifi takes roughly 0.7 seconds per client on Windows. The test
    suite constructs hundreds of clients while mocking their I/O. System roots
    preserve certificate verification for accidental local requests; explicit
    ``cafile``, ``capath``, and ``cadata`` arguments still use the real loader.
    """
    if sys.platform != "win32":
        yield
        return

    original = ssl.create_default_context
    certifi_path = os.path.normcase(os.path.abspath(certifi.where()))

    def create_default_context(
        purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH,
        *,
        cafile: str | None = None,
        capath: str | None = None,
        cadata: str | bytes | None = None,
    ) -> ssl.SSLContext:
        requested_path = os.path.normcase(os.path.abspath(cafile)) if cafile else None
        if requested_path == certifi_path and capath is None and cadata is None:
            return original(purpose)
        return original(
            purpose,
            cafile=cafile,
            capath=capath,
            cadata=cadata,
        )

    ssl.create_default_context = create_default_context
    try:
        yield
    finally:
        ssl.create_default_context = original
