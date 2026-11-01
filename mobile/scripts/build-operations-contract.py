"""Build the shared bounded operation registry from the M36 parity inventory.

Writes both the authoritative server contract and the mobile copy, then a test
asserts they stay identical. Planned M37 operations are declared but disabled.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "mobile/contracts/desktop-parity.json"
SERVER_OUT = ROOT / "nanobot/remote/contracts/operations-v1.json"
MOBILE_OUT = ROOT / "mobile/contracts/operations-v1.json"

PLANNED_PREFIXES = (
    "workspace.grant.",
    "workspace.access.",
    "workspace.folder.",
    "workspace.repository.",
)


def kind_for(api: dict) -> str:
    if api.get("kind") == "ws":
        return "mutation" if api.get("mutation_action") else "ws_frame"
    if api.get("path") == "/mobile/v1/attachment":
        return "attachment"
    if api.get("method") == "GET":
        return "read"
    return "mutation"


def main() -> int:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    operations = []
    for category in inventory["categories"]:
        for row in category["rows"]:
            native = row.get("native")
            if not native:
                continue
            api = row["api"]
            kind = kind_for(api)
            planned = (
                row["id"].startswith(PLANNED_PREFIXES)
                and row["id"] not in {
                    "workspace.grant.upsert",
                    "workspace.access.prepare_full",
                    "workspace.access.mode.set",
                }
            )
            bounds = native.get("bounds", {})
            operations.append(
                {
                    "id": native["operation"],
                    "inventory_row": row["id"],
                    "category": category["id"],
                    "kind": kind,
                    "method": api.get("method"),
                    "path": api.get("path"),
                    "action": api.get("mutation_action"),
                    "ws_type": api.get("ws_type"),
                    "scope": "owner",
                    "enabled": not planned,
                    "planned_task": "M37" if planned else None,
                    "replay": native.get("replay", "none"),
                    "bounds": {
                        "max_request_bytes": bounds.get("max_request_bytes"),
                        "max_response_bytes": bounds.get("max_response_bytes"),
                        "timeout_s": bounds.get("timeout_s"),
                        "range_requests": bool(bounds.get("range_requests", False)),
                    },
                }
            )
    operations.sort(key=lambda item: item["id"])
    contract = {
        "contract": "nanobot-mobile-operations",
        "version": 1,
        "inventory_contract": inventory["contract"],
        "inventory_version": inventory["version"],
        "owner_scope": "owner",
        "local_only_exception_rows": [
            row["id"]
            for category in inventory["categories"]
            for row in category["rows"]
            if row["auth"].get("local_only")
        ],
        "operations": operations,
    }
    SERVER_OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(contract, indent=2, ensure_ascii=False) + "\n"
    SERVER_OUT.write_text(text, encoding="utf-8")
    MOBILE_OUT.write_text(text, encoding="utf-8")
    enabled = sum(1 for item in operations if item["enabled"])
    planned = len(operations) - enabled
    print(f"wrote {len(operations)} operations ({enabled} enabled, {planned} planned)")
    print(f"  {SERVER_OUT.relative_to(ROOT)}")
    print(f"  {MOBILE_OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
