"""Bounded mobile operation registry (M31).

The registry is the single allowlist for owner-authenticated standard feature
reads, mutations, WS frames and attachments. It is generated from the M36
parity inventory and shipped as versioned package data. Unknown operations,
disabled planned operations, raw URLs and undeclared methods are rejected.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

CONTRACT_PATH = Path(__file__).with_name("contracts") / "operations-v1.json"
CONTRACT_VERSION = 1
MAX_PATH_BYTES = 4096
MAX_TEMPLATE_SEGMENTS = 8


class OperationRegistryError(RuntimeError):
    """The registry contract is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class OperationSpec:
    id: str
    inventory_row: str
    category: str
    kind: str
    method: str | None
    path: str | None
    action: str | None
    ws_type: str | None
    scope: str
    enabled: bool
    planned_task: str | None
    replay: str
    max_request_bytes: int | None
    max_response_bytes: int | None
    timeout_s: int | None
    range_requests: bool


@dataclass(frozen=True)
class OperationRegistry:
    version: int
    operations: dict[str, OperationSpec]
    reads: dict[str, OperationSpec]
    mutations: dict[str, OperationSpec]
    ws_frames: dict[str, tuple[OperationSpec, ...]]
    attachments: tuple[OperationSpec, ...]

    def by_id(self, operation: str) -> OperationSpec | None:
        return self.operations.get(operation)

    def match_read(self, raw_path: str) -> OperationSpec | None:
        if not raw_path or len(raw_path.encode()) > MAX_PATH_BYTES:
            return None
        if raw_path.startswith("//") or ".." in raw_path.split("/"):
            return None
        if "?" in raw_path:
            raw_path = raw_path.split("?", 1)[0]
        if not raw_path.startswith("/api/"):
            return None
        for spec in self.reads.values():
            pattern = _path_pattern(spec.path or "")
            if pattern.fullmatch(raw_path):
                return spec
        return None


def _path_pattern(template: str) -> re.Pattern[str]:
    parts: list[str] = []
    for segment in template.strip("/").split("/"):
        if segment.startswith("{") and segment.endswith("}"):
            parts.append("[^/]+")
        else:
            parts.append(re.escape(segment))
    return re.compile("^/" + "/".join(parts) + "$")


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _spec(raw: dict[str, Any]) -> OperationSpec:
    bounds = raw.get("bounds")
    bounds = cast(dict[str, Any], bounds) if isinstance(bounds, dict) else {}
    kind = raw.get("kind")
    if kind not in {"read", "mutation", "ws_frame", "attachment"}:
        raise OperationRegistryError(f"unknown operation kind: {kind!r}")
    spec = OperationSpec(
        id=str(raw["id"]),
        inventory_row=str(raw["inventory_row"]),
        category=str(raw["category"]),
        kind=kind,
        method=raw.get("method") if isinstance(raw.get("method"), str) else None,
        path=raw.get("path") if isinstance(raw.get("path"), str) else None,
        action=raw.get("action") if isinstance(raw.get("action"), str) else None,
        ws_type=raw.get("ws_type") if isinstance(raw.get("ws_type"), str) else None,
        scope=str(raw.get("scope", "owner")),
        enabled=bool(raw.get("enabled")),
        planned_task=raw.get("planned_task") if isinstance(raw.get("planned_task"), str) else None,
        replay=str(raw.get("replay", "none")),
        max_request_bytes=_optional_int(bounds.get("max_request_bytes")),
        max_response_bytes=_optional_int(bounds.get("max_response_bytes")),
        timeout_s=_optional_int(bounds.get("timeout_s")),
        range_requests=bool(bounds.get("range_requests", False)),
    )
    if spec.kind == "read" and (spec.method != "GET" or not spec.path):
        raise OperationRegistryError(f"{spec.id}: read requires GET and a path")
    if spec.kind == "mutation" and (not spec.action or not spec.path):
        raise OperationRegistryError(f"{spec.id}: mutation requires action and path")
    if spec.kind == "ws_frame" and not spec.ws_type:
        raise OperationRegistryError(f"{spec.id}: ws_frame requires a ws_type")
    if spec.kind == "attachment" and not spec.path:
        raise OperationRegistryError(f"{spec.id}: attachment requires a path")
    if spec.enabled and spec.planned_task:
        raise OperationRegistryError(f"{spec.id}: enabled operation cannot be planned")
    if spec.path and spec.path.count("/") > MAX_TEMPLATE_SEGMENTS + 1:
        raise OperationRegistryError(f"{spec.id}: path template is too deep")
    return spec


@lru_cache(maxsize=1)
def load_registry(path: str | None = None) -> OperationRegistry:
    contract_path = Path(path) if path else CONTRACT_PATH
    try:
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperationRegistryError("operation contract cannot be read") from exc
    if not isinstance(raw, dict):
        raise OperationRegistryError("operation contract root must be an object")
    contract = cast(dict[str, Any], raw)
    if contract.get("contract") != "nanobot-mobile-operations":
        raise OperationRegistryError("unexpected operation contract name")
    if contract.get("version") != CONTRACT_VERSION:
        raise OperationRegistryError("unsupported operation contract version")
    raw_operations = contract.get("operations")
    if not isinstance(raw_operations, list):
        raise OperationRegistryError("operations must be a list")
    operations: dict[str, OperationSpec] = {}
    reads: dict[str, OperationSpec] = {}
    mutations: dict[str, OperationSpec] = {}
    ws_frames: dict[str, list[OperationSpec]] = {}
    attachments: list[OperationSpec] = []
    for item in cast(list[Any], raw_operations):
        if not isinstance(item, dict):
            raise OperationRegistryError("operation entries must be objects")
        spec = _spec(cast(dict[str, Any], item))
        if spec.id in operations:
            raise OperationRegistryError(f"duplicate operation id: {spec.id}")
        operations[spec.id] = spec
        if spec.kind == "read":
            reads.setdefault(cast(str, spec.path), spec)
        elif spec.kind == "mutation":
            if spec.action in mutations:
                raise OperationRegistryError(f"duplicate mutation action: {spec.action}")
            mutations[cast(str, spec.action)] = spec
        elif spec.kind == "ws_frame":
            ws_frames.setdefault(cast(str, spec.ws_type), []).append(spec)
        else:
            attachments.append(spec)
    return OperationRegistry(
        version=CONTRACT_VERSION,
        operations=operations,
        reads=reads,
        mutations=mutations,
        ws_frames={key: tuple(value) for key, value in ws_frames.items()},
        attachments=tuple(attachments),
    )


def owner_operation(registry: OperationRegistry, operation: str) -> OperationSpec:
    spec = registry.by_id(operation)
    if spec is None:
        raise OperationRegistryError("operation is not registered")
    if not spec.enabled:
        raise OperationRegistryError(f"operation is pending {spec.planned_task or 'a later task'}")
    if spec.scope != "owner":
        raise OperationRegistryError("operation is not owner scoped")
    return spec
