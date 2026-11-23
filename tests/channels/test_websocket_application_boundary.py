"""Executable architecture constraints for the WebSocket transport adapter."""

from __future__ import annotations

import ast
from pathlib import Path

from kohaku.channels.websocket import runtime

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_PATH = _REPOSITORY_ROOT / "kohaku" / "channels" / "websocket" / "runtime.py"
_SESSION_IDENTITY_PATH = _REPOSITORY_ROOT / "kohaku" / "webui" / "session_identity.py"
_FORBIDDEN_RUNTIME_IMPORTS = (
    "kohaku.bus.outbound_events",
    "kohaku.command",
    "kohaku.runtime_context",
    "kohaku.security.workspace_access",
    "kohaku.session.goal_state",
    "kohaku.webui.cli_apps_api",
    "kohaku.webui.forking",
    "kohaku.webui.mcp_presets_api",
    "kohaku.webui.sidebar_state",
    "kohaku.webui.transcription_ws",
)


def _channel_method(name: str) -> ast.AsyncFunctionDef:
    tree = ast.parse(_RUNTIME_PATH.read_text(encoding="utf-8"))
    channel = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WebSocketChannel"
    )
    return next(
        node
        for node in channel.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name
    )


def _statements_without_docstring(node: ast.AsyncFunctionDef) -> list[ast.stmt]:
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    return body


def _contains_persisted_prefix_literal(source: str) -> bool:
    """Ignore identifiers/annotations while catching hardcoded session keys."""
    return any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("websocket:")
        for node in ast.walk(ast.parse(source))
    )


def test_websocket_runtime_does_not_import_application_command_trees() -> None:
    tree = ast.parse(_RUNTIME_PATH.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    violations = sorted(
        module
        for module in imported
        if module.startswith(_FORBIDDEN_RUNTIME_IMPORTS)
    )
    assert violations == []


def test_business_entrypoints_are_thin_transport_delegations() -> None:
    for method_name in ("_dispatch_envelope", "webui_hydrate", "send"):
        statements = _statements_without_docstring(_channel_method(method_name))
        assert len(statements) == 1, method_name
        assert isinstance(statements[0], ast.Expr), method_name
        assert isinstance(statements[0].value, ast.Await), method_name

    statements = _statements_without_docstring(_channel_method("_dispatch_http"))
    assert len(statements) == 1
    assert isinstance(statements[0], ast.Return)
    assert isinstance(statements[0].value, ast.Await)


def test_persisted_webui_session_prefix_has_one_production_owner() -> None:
    owners = []
    for path in (_REPOSITORY_ROOT / "kohaku").rglob("*.py"):
        if "tests" in path.parts or path == _SESSION_IDENTITY_PATH:
            continue
        if _contains_persisted_prefix_literal(path.read_text(encoding="utf-8")):
            owners.append(path.relative_to(_REPOSITORY_ROOT).as_posix())
    assert owners == []
    assert 'WEBUI_SESSION_STORAGE_PREFIX = "websocket:"' in _SESSION_IDENTITY_PATH.read_text(
        encoding="utf-8"
    )


def test_prefix_owner_scanner_ignores_annotations_but_catches_runtime_literals() -> None:
    assert not _contains_persisted_prefix_literal("websocket: dict[str, str] = {}")
    assert _contains_persisted_prefix_literal('return f"websocket:{chat_id}"')
    assert _contains_persisted_prefix_literal('return "websocket:" + chat_id')


def test_runtime_exports_compatibility_protocol_helpers() -> None:
    assert runtime._is_valid_chat_id("unified:default")  # pyright: ignore[reportPrivateUsage]
    assert not runtime._is_valid_chat_id("../escape")  # pyright: ignore[reportPrivateUsage]
