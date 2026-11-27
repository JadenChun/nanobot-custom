"""Cross-suite test infrastructure."""

from __future__ import annotations

import json
import os
import ssl
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

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
    from kohaku.config import loader

    monkeypatch.setattr(loader, "_current_config_path", loader._current_config_path)
    reference = tmp_path.parent / f"{tmp_path.name}-test-config" / "config.json"
    reference.parent.mkdir()
    reference.write_text(json.dumps({
        "agents": {"defaults": {"workspace": str(tmp_path / "workspace")}},
    }), encoding="utf-8")
    loader.set_config_path(reference)


@pytest.fixture(autouse=True)
def _isolate_kohaku_log_activation() -> Iterator[None]:
    """Keep CLI log settings from leaking into later tests in the same process."""
    logger.enable("kohaku")
    try:
        yield
    finally:
        logger.enable("kohaku")


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
        "kohaku.config.paths.get_default_sessions_root",
        "kohaku.session.manager.get_default_sessions_root",
    ):
        monkeypatch.setattr(target, sessions_root)
    monkeypatch.setattr(
        "kohaku.session.manager.get_legacy_sessions_dir",
        lambda: legacy_root,
    )
    yield


@pytest.fixture(autouse=True)
def _isolate_pairing_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep channel pairing tests out of the user's active pairing store."""
    pairing_path = tmp_path / "pairing.json"
    monkeypatch.setattr(
        "kohaku.pairing.store._store_path",
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


_REAL_SEATBELT_SELF_TEST: Callable[[], Any] | None = None


@pytest.fixture(autouse=True)
def _hermetic_macos_self_test(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Keep the real macOS self-test out of tests unless one asks for it.

    The executor is gated by the host self-test alone. Without this, a test run on
    a Mac would launch real Seatbelt probes and change what every test sees. A test
    that wants a capable macOS host patches the self-test or uses native_supervisor_host.
    A test of the self-test itself opts out with the real_macos_self_test marker.
    """
    global _REAL_SEATBELT_SELF_TEST
    from kohaku.agent.tools import executor_capabilities
    from kohaku.security import darwin_proc

    if _REAL_SEATBELT_SELF_TEST is None:
        _REAL_SEATBELT_SELF_TEST = darwin_proc.seatbelt_self_test
    if request.node.get_closest_marker("real_macos_self_test") is None:
        monkeypatch.setattr(
            darwin_proc,
            "seatbelt_self_test",
            lambda: darwin_proc.SelfTestResult(False, ("not exercised in tests",)),
        )
    executor_capabilities._macos_self_test_passed.cache_clear()
    yield
    executor_capabilities._macos_self_test_passed.cache_clear()


@pytest.fixture
def native_supervisor_host(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run one test against the real macOS self-test and require that it passes."""
    from kohaku.agent.tools import executor_capabilities
    from kohaku.security import darwin_proc

    assert _REAL_SEATBELT_SELF_TEST is not None
    monkeypatch.setattr(darwin_proc, "seatbelt_self_test", _REAL_SEATBELT_SELF_TEST)
    executor_capabilities._macos_self_test_passed.cache_clear()
    assert executor_capabilities.tree_ownership_available()
    yield


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "darwin_host: signals real processes; runs only on an opted-in macOS host",
    )
    config.addinivalue_line(
        "markers",
        "real_macos_self_test: runs the real Seatbelt self-test instead of the test double",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep real-signal macOS tests off developer machines (A3-2M, WP0)."""
    from tests.darwin.gate import darwin_host_skip_reason

    reason = darwin_host_skip_reason(sys.platform, os.environ)
    if reason is None:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "darwin_host" in item.keywords:
            item.add_marker(skip)
