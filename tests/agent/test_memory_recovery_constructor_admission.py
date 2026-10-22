"""Constructor writes require the exact historical startup admission root."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanobot.agent import memory as memory_module
from nanobot.agent import memory_recovery_startup_admission as admission
from nanobot.agent.context import ContextBuilder
from nanobot.agent.loop import AgentLoop
from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_recovery_startup_admission import StartupReceiptAdmissionError
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.queue import MessageBus
from nanobot.config import paths as config_paths
from nanobot.config.schema import Config
from nanobot.session.manager import JsonlSessionStore, SessionManager
from tests.agent.test_memory_recovery_quarantine_storage import _real_layout, _seal_real_receipt

_PURE_DEFAULT_SESSIONS_ROOT = config_paths.get_default_sessions_root


def test_default_sessions_root_resolution_does_not_create_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "not-created" / "config.json"
    monkeypatch.setattr(config_paths, "get_config_path", lambda: config)
    assert _PURE_DEFAULT_SESSIONS_ROOT() == config.parent / "sessions"
    assert not config.parent.exists()


@pytest.mark.skipif(os.name != "posix", reason="real seal uses POSIX descriptors")
@pytest.mark.parametrize("constructor", ["memory", "jsonl", "manager", "injected", "context"])
def test_sealed_constructor_uses_exact_root_after_mutable_edit(
    tmp_path: Path, constructor: str,
) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    _seal_real_receipt(workspace, sessions, receipt, binding)
    (workspace / "memory" / "MEMORY.md").write_text("later user edit\n")

    def construct(root: Path) -> object:
        if constructor == "memory":
            return MemoryStore(workspace, sessions_root=root)
        if constructor == "jsonl":
            return JsonlSessionStore(workspace, sessions_root=root)
        if constructor == "manager":
            return SessionManager(workspace, sessions_root=root)
        if constructor == "injected":
            return SessionManager(workspace, sessions_root=root, store=MagicMock())
        return ContextBuilder(workspace, sessions_root=root)

    wrong = tmp_path / "wrong-sessions"
    with pytest.raises(StartupReceiptAdmissionError, match="memory recovery required"):
        construct(wrong)
    assert not wrong.exists()
    result = construct(sessions)
    if isinstance(result, (JsonlSessionStore, SessionManager)):
        assert result.sessions_root == sessions.resolve()


@pytest.mark.skipif(os.name != "posix", reason="real seal uses POSIX descriptors")
def test_evidence_race_denies_before_memory_constructor_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    _seal_real_receipt(workspace, sessions, receipt, binding)
    manifest = next((workspace / "memory" / ".recovery" / "transactions"
                     / binding.transaction_identity).glob("*.json"))
    original_preflight = memory_module.require_startup_receipt_admission

    def corrupt_after_preflight(path: Path, *, sessions_root: Path) -> None:
        original_preflight(path, sessions_root=sessions_root)
        manifest.write_bytes(b"corrupt evidence\n")

    monkeypatch.setattr(memory_module, "require_startup_receipt_admission",
                        corrupt_after_preflight)
    monkeypatch.setattr(memory_module, "GitStore",
                        lambda *args, **kwargs: pytest.fail("GitStore entered after evidence race"))
    with pytest.raises(StartupReceiptAdmissionError, match="memory recovery required"):
        MemoryStore(workspace, sessions_root=sessions)
    assert manifest.read_bytes() == b"corrupt evidence\n"


@pytest.mark.skipif(os.name != "posix", reason="real seal uses POSIX descriptors")
def test_session_evidence_race_denies_before_namespace_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    _seal_real_receipt(workspace, sessions, receipt, binding)
    manifest = next((workspace / "memory" / ".recovery" / "transactions"
                     / binding.transaction_identity).glob("*.json"))
    original_preflight = admission.require_startup_receipt_admission

    def corrupt_after_preflight(path: Path, *, sessions_root: Path) -> None:
        original_preflight(path, sessions_root=sessions_root)
        manifest.write_bytes(b"corrupt evidence\n")

    monkeypatch.setattr(admission, "require_startup_receipt_admission",
                        corrupt_after_preflight)
    with pytest.raises(StartupReceiptAdmissionError, match="memory recovery required"):
        JsonlSessionStore(workspace, sessions_root=sessions)
    assert not (sessions / ".workspace-migration.lock").exists()
    assert manifest.read_bytes() == b"corrupt evidence\n"


@pytest.mark.skipif(os.name != "posix", reason="real seal uses POSIX descriptors")
def test_loop_preflights_root_before_dependencies_and_configured_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    _seal_real_receipt(workspace, sessions, receipt, binding)
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    wrong = tmp_path / "wrong-sessions"
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    monkeypatch.setattr("nanobot.agent.loop.build_foreground_dependencies",
                        lambda **kwargs: pytest.fail("foreground setup entered before admission"))
    with pytest.raises(StartupReceiptAdmissionError, match="memory recovery required"):
        AgentLoop(MessageBus(), provider, workspace, sessions_root=wrong)

    config = Config()
    config.agents.defaults.workspace = str(workspace)
    config.bind_source_path(tmp_path / "wrong-sessions" / "config.json")
    with pytest.raises(StartupReceiptAdmissionError, match="memory recovery required"):
        AgentLoop.from_config(config, tool_registry=MagicMock(), provider=provider)
    assert not wrong.exists()
    manager = SessionManager(workspace, sessions_root=sessions)
    with pytest.raises(ValueError, match="sessions root must match explicit root"):
        AgentLoop(MessageBus(), provider, workspace, session_manager=manager,
                  sessions_root=wrong)
    with pytest.raises(ValueError, match="configured sessions root must match explicit root"):
        AgentLoop.from_config(config, tool_registry=MagicMock(), provider=provider,
                              sessions_root=sessions)


@pytest.mark.skipif(os.name != "posix", reason="real seal uses POSIX descriptors")
def test_direct_and_configured_loop_propagate_sealed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    _seal_real_receipt(workspace, sessions, receipt, binding)
    (workspace / "memory" / "MEMORY.md").write_text("later user edit\n")
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    direct = AgentLoop(MessageBus(), provider, workspace, model="test-model",
                       sessions_root=sessions)
    assert direct.sessions.sessions_root == sessions

    config = Config()
    config.agents.defaults.workspace = str(workspace)
    config.bind_source_path(tmp_path / "config.json")
    configured = AgentLoop.from_config(config, tool_registry=ToolRegistry(), provider=provider)
    assert configured.sessions.sessions_root == sessions
    assert configured.context.memory.workspace == workspace
