from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config
from nanobot.runtime_composition import RuntimeComposition, RuntimeFactories
from nanobot.session.runtime_authority import (
    DurableSessionAuthority,
    Stage1UnavailableSessionAuthority,
)


def _config(tmp_path: Path) -> Config:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config.model_validate(
        {"agents": {"defaults": {"workspace": str(workspace)}}}
    )
    config.bind_source_path(tmp_path / "runtime" / "config.json")
    return config


def _factories() -> RuntimeFactories:
    class MCP:
        async def aclose(self) -> None:
            return None

    def agent(_config, bus, sessions, _tools, _mcp, _cron, delivery):
        async def close() -> None:
            return None

        return SimpleNamespace(
            bus=bus,
            sessions=sessions,
            turn_delivery_factory=delivery,
            aclose=close,
        )

    return RuntimeFactories(
        MessageBus,
        lambda: object(),
        lambda _config, _tools: MCP(),
        lambda _preflight: object(),
        agent,
    )  # type: ignore[arg-type]


@pytest.mark.parametrize("sessions_root_exists", [False, True])
def test_default_durable_definition_is_byte_inode_mtime_stable(
    tmp_path: Path,
    sessions_root_exists: bool,
) -> None:
    config = _config(tmp_path)
    (config.workspace_path / "notes.txt").write_bytes(b"unchanged workspace evidence\n")
    root = config.runtime_data_dir / "sessions"  # type: ignore[operator]
    if sessions_root_exists:
        root.mkdir(parents=True)

    def snapshot():
        return tuple(
            (
                str(path.relative_to(tmp_path)),
                path.stat().st_ino,
                path.stat().st_size,
                path.stat().st_mtime_ns,
                path.read_bytes() if path.is_file() else None,
            )
            for path in [tmp_path, *sorted(tmp_path.rglob("*"))]
        )

    before = snapshot()
    composition = RuntimeComposition.define(
        config,
        factories=_factories(),
        startup_receipt_admitted=True,
    )
    assert isinstance(composition._authority, DurableSessionAuthority)
    assert snapshot() == before
    assert composition.preflight.sessions_root_exists is sessions_root_exists


@pytest.mark.parametrize("explicit_root", [False, True])
async def test_default_authority_acquires_only_captured_preflight_roots(
    tmp_path: Path,
    explicit_root: bool,
) -> None:
    config = _config(tmp_path)
    composition = RuntimeComposition.define(
        config,
        sessions_root=tmp_path / "selected-sessions" if explicit_root else None,
        factories=_factories(),
        startup_receipt_admitted=True,
    )
    report = composition.preflight
    assert isinstance(composition._authority, DurableSessionAuthority)
    assert composition._authority.workspace == report.workspace
    assert composition._authority.sessions_root == report.sessions_root
    config.agents.defaults.workspace = str(tmp_path / "changed-workspace")
    config.bind_source_path(tmp_path / "changed-runtime" / "config.json")
    handle = await composition.acquire()
    try:
        assert handle.sessions.workspace == report.workspace
        assert handle.sessions.sessions_root == report.sessions_root
        assert not config.workspace_path.exists()
        assert not config.runtime_data_dir.exists()  # type: ignore[union-attr]
        handle.sessions.admit_persistent_write("cli:captured")
        assert handle.sessions.resolve_session_mode("cli:captured").mode == "canonical"
    finally:
        await composition.aclose()


def test_explicit_falsey_authority_wins_unchanged(tmp_path: Path) -> None:
    class FalseyAuthority(Stage1UnavailableSessionAuthority):
        def __bool__(self) -> bool:
            raise AssertionError("authority truthiness must not be evaluated")

    authority = FalseyAuthority()
    composition = RuntimeComposition.define(
        _config(tmp_path),
        session_authority=authority,
        factories=_factories(),
        startup_receipt_admitted=True,
    )
    assert composition._authority is authority


async def test_explicit_durable_authority_routes_one_managed_manager(tmp_path: Path) -> None:
    config = _config(tmp_path)
    authority = DurableSessionAuthority(
        config.workspace_path,
        sessions_root=config.runtime_data_dir / "sessions",  # type: ignore[operator]
    )
    composition = RuntimeComposition.define(
        config,
        session_authority=authority,
        factories=_factories(),
        startup_receipt_admitted=True,
    )
    handle = await composition.acquire()
    assert handle.sessions.has_managed_authority
    assert handle.agent.sessions is handle.sessions
    handle.sessions.admit_persistent_write("cli:item")
    assert handle.sessions.resolve_session_mode("cli:item").mode == "canonical"
    await composition.aclose()


async def test_explicit_durable_lifetime_is_released_after_graph_shutdown(tmp_path: Path) -> None:
    config = _config(tmp_path)
    authority = DurableSessionAuthority(
        config.workspace_path,
        sessions_root=config.runtime_data_dir / "sessions",  # type: ignore[operator]
    )
    composition = RuntimeComposition.define(
        config,
        session_authority=authority,
        factories=_factories(),
        startup_receipt_admitted=True,
    )
    await composition.acquire()
    contender = DurableSessionAuthority(
        config.workspace_path,
        sessions_root=config.runtime_data_dir / "sessions",  # type: ignore[operator]
    )
    with pytest.raises(RuntimeError):
        await contender.acquire()

    await composition.aclose()
    acquired = await contender.acquire()
    assert acquired.close is not None
    await acquired.close()
