from __future__ import annotations

import json
from pathlib import Path

from nanobot.remote.operations import (
    CONTRACT_PATH,
    load_registry,
    owner_operation,
)

ROOT = Path(__file__).resolve().parents[2]
MOBILE_CONTRACT = ROOT / "mobile/contracts/operations-v1.json"


def test_contract_copies_are_identical() -> None:
    assert CONTRACT_PATH.read_bytes() == MOBILE_CONTRACT.read_bytes()


def test_registry_matches_m36_inventory() -> None:
    inventory = json.loads((ROOT / "mobile/contracts/desktop-parity.json").read_text())
    expected = {
        row["native"]["operation"]
        for category in inventory["categories"]
        for row in category["rows"]
        if row.get("native")
    }
    registry = load_registry()
    assert set(registry.operations) == expected
    assert len(registry.operations) == 109
    assert sum(1 for spec in registry.operations.values() if spec.enabled) == 109


def test_enabled_mutations_match_websocket_source_routes() -> None:
    import ast

    tree = ast.parse((ROOT / "nanobot/webui/ws_http.py").read_text())
    source: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "WEBSOCKET_MUTATION_ROUTES":
            assert isinstance(node.value, ast.Dict)
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                    source[key.value] = value.value
    settings_source: set[str] = set()
    settings_tree = ast.parse((ROOT / "nanobot/webui/settings_routes.py").read_text())
    for node in ast.walk(settings_tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            settings_source.add(node.value)
    registry = load_registry()
    for action, spec in registry.mutations.items():
        if not spec.enabled:
            continue
        if action == "session.delete":
            assert spec.path == "/api/sessions/{key}/delete"
        elif action.startswith("settings.channel.connect."):
            assert spec.path.startswith("/api/settings/channels/{channel}/connect/")
        elif action.startswith("settings."):
            assert spec.path in settings_source, action
        else:
            assert source.get(action) == spec.path, action


def test_workspace_operations_are_enabled_and_local_only_actions_stay_excluded() -> None:
    registry = load_registry()
    for operation in (
        "workspace.grant.revoke",
        "workspace.repository.discover",
        "workspace.repository.forget",
        "workspace.repository.remember",
        "workspace.folder.edit",
        "workspace.access.add_folder",
        "workspace.access.allow_once",
        "workspace.access.cancel",
        "workspace.access.continue",
        "workspace.access.resolve",
    ):
        assert owner_operation(registry, operation).enabled
    for operation in (
        "workspace.grant.upsert",
        "workspace.access.prepare_full",
        "workspace.access.mode.set",
        "workspace.directories.roots",
        "workspace.directories.list",
        "workspace.directories.resolve",
    ):
        assert owner_operation(registry, operation).enabled
    for operation in (
        "devices.remote.enable",
        "devices.remote.pairing_start",
        "workspace.pick_folder",
    ):
        assert registry.by_id(operation) is None


def test_read_matching_is_allowlisted_and_rejects_traversal() -> None:
    registry = load_registry()
    assert registry.match_read("/api/sessions") is not None
    assert registry.match_read("/api/sessions/websocket%3Aabc/webui-thread?limit=2") is not None
    assert registry.match_read("/api/media/abc/payload") is not None
    assert registry.match_read("/api/settings") is not None
    assert registry.match_read("/api/unknown") is None
    assert registry.match_read("/mobile/v1/read") is None
    assert registry.match_read("https://evil.example/api/sessions") is None
    assert registry.match_read("/api/../secret") is None
    assert registry.match_read("//evil/api/sessions") is None


def test_every_enabled_operation_has_bounds_and_replay_policy() -> None:
    registry = load_registry()
    for spec in registry.operations.values():
        if not spec.enabled:
            continue
        assert spec.scope == "owner"
        assert spec.timeout_s is not None and spec.timeout_s > 0
        if spec.kind in {"mutation", "attachment"}:
            assert spec.max_request_bytes is not None and spec.max_request_bytes > 0
            assert spec.replay in {"submit", "none"}
        if spec.kind == "read":
            assert spec.max_response_bytes is not None and spec.max_response_bytes > 0
