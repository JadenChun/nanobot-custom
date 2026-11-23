#!/usr/bin/env python3
"""Validate the frozen desktop-parity inventory against server source (M36).

Checks, without importing the gateway:
  * schema/enums/required fields/unique IDs per row;
  * every WebSocket mutation action and settings mutation path in source has a row;
  * every WS inbound command type and read route template has a row;
  * every source-enforced local-only path is marked current_local_only;
  * every target local-only row carries an explicit accepted exception;
  * every non-local row declares a bounded native operation and positive/negative checks.

Usage:
    python mobile/scripts/check-desktop-parity.py [--inventory PATH] [--self-test]

Exit 0 only when every check passes. ``--self-test`` mutates an in-memory copy to
prove omitted actions, silent exceptions and unbounded rows are rejected.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = REPO / "mobile/contracts/desktop-parity.json"
WS_HTTP = REPO / "nanobot/webui/ws_http.py"
SETTINGS_ROUTES = REPO / "nanobot/webui/settings_routes.py"
INBOUND = REPO / "nanobot/webui/inbound_commands.py"

REQUIRED_CATEGORIES = {
    "chat",
    "settings",
    "apps",
    "skills",
    "automations",
    "workspace",
    "devices_entry",
    "lifecycle",
    "media_files",
}
OWNER_VALUES = {"read", "write", "admin", "local_only", "denied"}
LEGACY_VALUES = {"chat.read", "chat.write", "media.read", "attachment.write", "none"}
STATUS_VALUES = {"present", "partial", "absent", "planned"}
ROW_KEYS = {
    "id", "category", "feature", "action", "desktop", "api", "ui", "auth",
    "native", "dependency", "checks", "status", "exception", "notes",
}

# Read routes that are matched by regex rather than exact string equality.
READ_TEMPLATES = {
    "/api/sessions/{key}/webui-thread": r"^/api/sessions/([^/]+)/webui-thread$",
    "/api/sessions/{key}/delegated-children": r"^/api/sessions/([^/]+)/delegated-children$",
    "/api/sessions/{key}/context": r"^/api/sessions/([^/]+)/context$",
    "/api/sessions/{key}/file-preview": r"^/api/sessions/([^/]+)/file-preview$",
    "/api/sessions/{key}/automations": r"^/api/sessions/([^/]+)/automations$",
    "/api/media/{sig}/{payload}": r"^/api/media/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)$",
    "/api/webui/skills/{name}": r"^/api/webui/skills/([^/]+)$",
}

# Settings module dicts that map path -> action.
SETTINGS_DICTS = {
    "_MODEL_ROUTES",
    "_CAPABILITY_ROUTES",
    "_SYSTEM_ROUTES",
    "_REMOTE_ROUTES",
    "_MCP_PRESET_ACTIONS_BY_PATH",
}


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def dict_constants(tree: ast.Module, name: str) -> dict[str, str]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                if isinstance(node.value, ast.Dict):
                    out = {}
                    for k, v in zip(node.value.keys, node.value.values):
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            out[k.value] = v.value if isinstance(v, ast.Constant) else ""
                    return out
    raise SystemExit(f"source dict {name} not found")


def set_constants(tree: ast.Module, name: str) -> set[str]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                value = node.value
                if isinstance(value, ast.Call) and value.args:
                    value = value.args[0]
                if isinstance(value, ast.Set):
                    out = set()
                    for elt in value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            out.add(elt.value)
                        elif isinstance(elt, ast.Starred):
                            # expanded later by caller
                            out.add("*")
                    return out
    raise SystemExit(f"source set {name} not found")


def ws_command_types(tree: ast.Module) -> set[str]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "dispatch":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Compare):
                    left = sub.left
                    if isinstance(left, ast.Name) and left.id == "command_type":
                        for comp in sub.comparators:
                            if isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                                out.add(comp.value)
    return out


DISPATCH_FUNCTIONS = ("_dispatch_misc_routes", "_dispatch_directory_routes")


def dispatch_get_routes(tree: ast.Module) -> set[str]:
    """Collect exact GET paths from synchronous and asynchronous dispatch."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in DISPATCH_FUNCTIONS:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                if sub.value.startswith("/api/"):
                    out.add(sub.value)
    return out


def async_recovery_routes(tree: ast.Module) -> set[str]:
    """Collect the recovery route alternatives from the async dispatcher."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_dispatch_recovery_route":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    if sub.value.startswith("/api/webui/recovery/"):
                        out.add(sub.value)
    return out


def inventory_index(data: dict) -> tuple[dict[str, dict], list[str]]:
    errors: list[str] = []
    rows: dict[str, dict] = {}
    for category in data.get("categories", []):
        cid = category.get("id")
        for row in category.get("rows", []):
            rid = row.get("id")
            if not isinstance(rid, str) or not rid:
                fail(errors, "row without a stable id")
                continue
            if rid in rows:
                fail(errors, f"duplicate row id: {rid}")
            if row.get("category") != cid:
                fail(errors, f"{rid}: category mismatch ({row.get('category')} != {cid})")
            rows[rid] = row
    return rows, errors


def validate_schema(data: dict, rows: dict[str, dict]) -> list[str]:
    errors: list[str] = []
    if data.get("contract") != "nanobot-desktop-parity":
        fail(errors, "unexpected contract name")
    if data.get("version") != 1:
        fail(errors, "unsupported inventory version")
    categories = {c.get("id") for c in data.get("categories", [])}
    missing = REQUIRED_CATEGORIES - categories
    if missing:
        fail(errors, f"missing required categories: {sorted(missing)}")
    for cid in categories:
        if cid not in REQUIRED_CATEGORIES:
            fail(errors, f"unclassified category: {cid}")
    for category in data.get("categories", []):
        if not category.get("required"):
            fail(errors, f"category {category.get('id')} not marked required")
        if category.get("row_count") != len(category.get("rows", [])):
            fail(errors, f"category {category.get('id')} row_count mismatch")
        if not category.get("rows"):
            fail(errors, f"category {category.get('id')} has no rows")

    for rid, row in rows.items():
        extra = set(row) - ROW_KEYS
        if extra:
            fail(errors, f"{rid}: unexpected fields {sorted(extra)}")
        for key in ("feature", "action"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                fail(errors, f"{rid}: missing {key}")
        api = row.get("api")
        if not isinstance(api, dict) or api.get("kind") not in {"http", "ws", "local"}:
            fail(errors, f"{rid}: invalid api block")
        auth = row.get("auth")
        if not isinstance(auth, dict):
            fail(errors, f"{rid}: missing auth block")
        else:
            if auth.get("owner") not in OWNER_VALUES:
                fail(errors, f"{rid}: invalid owner auth {auth.get('owner')!r}")
            if auth.get("legacy") not in LEGACY_VALUES:
                fail(errors, f"{rid}: invalid legacy scope {auth.get('legacy')!r}")
            if not isinstance(auth.get("local_only"), bool):
                fail(errors, f"{rid}: local_only must be boolean")
            if not isinstance(auth.get("current_local_only"), bool):
                fail(errors, f"{rid}: current_local_only must be boolean")
        status = row.get("status")
        if not isinstance(status, dict):
            fail(errors, f"{rid}: missing status block")
        else:
            if status.get("desktop_implementation") not in STATUS_VALUES:
                fail(errors, f"{rid}: invalid desktop status")
            if status.get("mobile_implementation") not in {"present", "partial", "absent"}:
                fail(errors, f"{rid}: invalid mobile status")
        checks = row.get("checks")
        if not isinstance(checks, dict) or not checks.get("positive") or not checks.get("negative"):
            fail(errors, f"{rid}: positive and negative checks are required")
        if not isinstance(row.get("dependency"), list) or not row.get("dependency"):
            fail(errors, f"{rid}: at least one dependency task is required")
        if auth and auth.get("local_only") and not row.get("exception"):
            fail(errors, f"{rid}: local-only row must record an explicit exception")
        native = row.get("native")
        if auth and auth.get("local_only"):
            if native is not None:
                fail(errors, f"{rid}: local-only row must not declare a native operation")
        elif api and api.get("kind") == "local":
            if native is not None:
                fail(errors, f"{rid}: client-local row must not declare a native operation")
        else:
            if not isinstance(native, dict) or not native.get("operation"):
                fail(errors, f"{rid}: missing bounded native operation")
            elif not re.fullmatch(r"[a-z][a-z0-9_.]*", str(native["operation"])):
                fail(errors, f"{rid}: invalid native operation name")
            else:
                bounds = native.get("bounds")
                if not isinstance(bounds, dict) or not bounds:
                    fail(errors, f"{rid}: native operation has no bounds")
    return errors


def validate_source(rows: dict[str, dict], data: dict) -> list[str]:
    errors: list[str] = []
    ws_tree = parse_module(WS_HTTP)
    settings_tree = parse_module(SETTINGS_ROUTES)
    inbound_tree = parse_module(INBOUND)

    ws_routes = dict_constants(ws_tree, "WEBSOCKET_MUTATION_ROUTES")
    settings_dicts: dict[str, str] = {}
    for name in SETTINGS_DICTS:
        settings_dicts.update(dict_constants(settings_tree, name))
    settings_mutations = set_constants(settings_tree, "_SETTINGS_MUTATION_PATHS")
    local_only = set_constants(ws_tree, "_LOCAL_ONLY_WEBUI_MUTATION_PATHS")
    if "*" in settings_mutations:
        settings_mutations.discard("*")
        settings_mutations |= set(dict_constants(settings_tree, "_MCP_PRESET_ACTIONS_BY_PATH"))

    # Index inventory mutations by action and by path.
    action_rows: dict[str, dict] = {}
    path_rows: dict[str, dict] = {}
    ws_types: set[str] = set()
    for rid, row in rows.items():
        api = row["api"]
        if api.get("mutation_action"):
            action_rows[api["mutation_action"]] = row
        if api.get("path"):
            path_rows[api["path"]] = row
        if api.get("kind") == "ws" and api.get("ws_type"):
            ws_types.add(api["ws_type"])

    for action, path in ws_routes.items():
        row = action_rows.get(action)
        if row is None:
            fail(errors, f"unclassified WebSocket mutation action: {action}")
        elif row["api"].get("path") != path:
            fail(errors, f"{action}: inventory path {row['api'].get('path')} != source {path}")

    for path in sorted(settings_mutations):
        if path in {row["api"].get("path") for row in rows.values()}:
            continue
        fail(errors, f"unclassified settings mutation path: {path}")

    for path in sorted(settings_dicts):
        if path in {row["api"].get("path") for row in rows.values()}:
            continue
        fail(errors, f"unclassified settings route: {path}")

    for command in sorted(ws_command_types(inbound_tree)):
        if command not in ws_types:
            fail(errors, f"unclassified WebSocket command type: {command}")

    settings_text = SETTINGS_ROUTES.read_text(encoding="utf-8")
    if 'prefix = "/api/settings/channels/"' not in settings_text:
        fail(errors, "source channel-connect route prefix not found (source drift)")
    for suffix in ("start", "poll", "cancel"):
        template = f"/api/settings/channels/{{channel}}/connect/{suffix}"
        if not any(row["api"].get("path") == template for row in rows.values()):
            fail(errors, f"unclassified channel connect route: {template}")

    source_text = WS_HTTP.read_text(encoding="utf-8")
    for template, pattern in READ_TEMPLATES.items():
        if pattern not in source_text:
            fail(errors, f"source regex for {template} not found (source drift)")
        if not any(row["api"].get("path") == template for row in rows.values()):
            fail(errors, f"unclassified read route template: {template}")

    for path in sorted(dispatch_get_routes(ws_tree)):
        if path in {row["api"].get("path") for row in rows.values()}:
            continue
        # The dispatch functions include the prefix check and a skills regex.
        if path == "/api/":
            continue
        fail(errors, f"unclassified dispatch GET route: {path}")

    # The recovery dispatcher is async and matches a regex with two alternatives.
    recovery_paths = async_recovery_routes(ws_tree)
    if recovery_paths != {"/api/webui/recovery/(continue|dismiss)"}:
        fail(errors, "async recovery dispatcher route drift")
    for suffix in ("continue", "dismiss"):
        template = f"/api/webui/recovery/{suffix}"
        if not any(row["api"].get("path") == template for row in rows.values()):
            fail(errors, f"unclassified async recovery route: {template}")

    for path in sorted(local_only):
        row = path_rows.get(path)
        if row is None:
            fail(errors, f"local-only source path has no inventory row: {path}")
        elif not row["auth"].get("current_local_only"):
            fail(errors, f"{row['id']}: source enforces local-only but current_local_only is false")

    # Every exception must point at real rows.
    for exception in data.get("exceptions", []):
        for rid in exception.get("rows", []):
            if rid not in rows:
                fail(errors, f"exception {exception.get('id')} references unknown row {rid}")
    return errors


def run(data: dict) -> tuple[dict[str, dict], list[str]]:
    rows, errors = inventory_index(data)
    return rows, errors + validate_schema(data, rows) + validate_source(rows, data)


def self_test(data: dict) -> list[str]:
    failures: list[str] = []

    def expect_failure(name: str, mutate) -> None:
        candidate = copy.deepcopy(data)
        mutate(candidate)
        _rows, errors = run(candidate)
        if not errors:
            failures.append(f"self-test {name}: expected rejection, got PASS")

    def drop_action(candidate: dict) -> None:
        for category in candidate["categories"]:
            if category["id"] == "settings":
                category["rows"] = [
                    r for r in category["rows"] if r["api"].get("mutation_action") != "settings.provider.create"
                ]
                category["row_count"] = len(category["rows"])

    def silent_exception(candidate: dict) -> None:
        for category in candidate["categories"]:
            for row in category["rows"]:
                if row["id"] == "devices.remote.enable":
                    row["exception"] = None

    def unbounded(candidate: dict) -> None:
        for category in candidate["categories"]:
            for row in category["rows"]:
                if row["id"] == "chat.message.send":
                    row["native"]["bounds"] = {}

    def drop_category(candidate: dict) -> None:
        candidate["categories"] = [
            c for c in candidate["categories"] if c["id"] != "skills"
        ]

    def fake_local(candidate: dict) -> None:
        for category in candidate["categories"]:
            for row in category["rows"]:
                if row["id"] == "workspace.grant.upsert":
                    row["auth"]["current_local_only"] = False

    def drop_async_route(candidate: dict) -> None:
        for category in candidate["categories"]:
            if category["id"] == "chat":
                category["rows"] = [
                    r
                    for r in category["rows"]
                    if r["api"].get("path") != "/api/webui/recovery/continue"
                ]
                category["row_count"] = len(category["rows"])

    expect_failure("omitted-action", drop_action)
    expect_failure("silent-exception", silent_exception)
    expect_failure("unbounded-native", unbounded)
    expect_failure("omitted-category", drop_category)
    expect_failure("unmarked-local-path", fake_local)
    expect_failure("missing-async-route", drop_async_route)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    data = load_json(args.inventory)
    rows, errors = run(data)
    if args.self_test:
        errors.extend(self_test(data))

    total_rows = len(rows)
    if errors:
        print(f"FAIL desktop-parity: {len(errors)} error(s), {total_rows} rows")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"PASS desktop-parity: {total_rows} rows, {len(data['categories'])} categories")
    print(f"  contract={data['contract']} version={data['version']} status={data['status']}")
    for category in data["categories"]:
        print(f"  {category['id']}: {category['row_count']} rows")
    if args.self_test:
        print(
            "  self-test: omitted action/category/async route, silent exception, "
            "unbounded native and unmarked local path all rejected"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
