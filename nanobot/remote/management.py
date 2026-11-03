"""Headless local management for remote-access settings (M18)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nanobot.remote.client import OwnerRequestError, OwnerUnavailableError, RemoteAccessClient


class RemoteAccessManager:
    def __init__(self, socket_path: Path | str, admin_token: str) -> None:
        self.client = RemoteAccessClient(socket_path, admin_token=admin_token)

    def status(self) -> dict[str, Any]:
        return self.client.status()

    def set_enabled(self, enabled: bool) -> bool:
        return bool(self.client.set_enabled(enabled))

    def devices(self) -> list[dict[str, Any]]:
        return self.client.list_devices()

    def revoke(
        self,
        installation_id: str,
        *,
        actor_installation_id: str | None = None,
    ) -> dict[str, Any]:
        return self.client.revoke(
            installation_id,
            actor_installation_id=actor_installation_id,
        )

    def create_pairing(
        self,
        *,
        origin: str,
        label: str,
        chat_consent: bool = True,
        media_consent: bool = True,
        full_control_consent: bool = True,
        shared_history_consent: bool = False,
    ) -> dict[str, Any]:
        return self.client.create_pairing(
            origin=origin,
            label=label,
            chat_consent=chat_consent,
            media_consent=media_consent,
            full_control_consent=full_control_consent,
            shared_history_consent=shared_history_consent,
        )

    def pending_pairings(self) -> dict[str, Any]:
        return self.client.pending_pairings()

    def approve_pairing(
        self,
        *,
        pairing_id: str,
        comparison_code: str | None = None,
        pairing_code: str | None = None,
        scopes: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.client.approve_pairing(
            pairing_id=pairing_id,
            comparison_code=comparison_code,
            pairing_code=pairing_code,
            scopes=scopes,
        )

    def deny_pairing(self, pairing_id: str) -> dict[str, Any]:
        return self.client.deny_pairing(pairing_id)

    def renew_pairing(self, pairing_id: str) -> dict[str, Any]:
        return self.client.renew_pairing(pairing_id)

    def cancel_pairing(self, pairing_id: str) -> bool:
        return self.client.cancel_pairing(pairing_id)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--admin-token-file", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    enabled = sub.add_parser("enable")
    enabled.add_argument("--enabled", choices=["true", "false"], default="true")
    sub.add_parser("devices")
    revoke = sub.add_parser("revoke")
    revoke.add_argument("installation_id")
    create = sub.add_parser("pair-create")
    create.add_argument("--origin", required=True)
    create.add_argument("--label", default="mobile")
    sub.add_parser("pair-pending")
    approve = sub.add_parser("pair-approve")
    approve.add_argument("--pairing-id", required=True)
    approve.add_argument("--code")
    approve.add_argument("--scope", action="append", default=[])
    deny = sub.add_parser("pair-deny")
    deny.add_argument("--pairing-id", required=True)
    cancel = sub.add_parser("pair-cancel")
    cancel.add_argument("--pairing-id", required=True)
    return parser.parse_args()


def _print(value: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, indent=2, sort_keys=True))
    elif isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        print(value)


def main() -> int:
    args = _parse_args()
    try:
        manager = RemoteAccessManager(
            args.socket,
            args.admin_token_file.read_text(encoding="ascii").strip(),
        )
        if args.command == "status":
            _print(manager.status(), args.json)
        elif args.command == "enable":
            _print(manager.set_enabled(args.enabled == "true"), args.json)
        elif args.command == "devices":
            _print(manager.devices(), args.json)
        elif args.command == "revoke":
            _print(manager.revoke(args.installation_id), args.json)
        elif args.command == "pair-create":
            _print(
                manager.create_pairing(
                    origin=args.origin,
                    label=args.label,
                ),
                args.json,
            )
        elif args.command == "pair-pending":
            _print(manager.pending_pairings(), args.json)
        elif args.command == "pair-approve":
            _print(
                manager.approve_pairing(
                    pairing_id=args.pairing_id,
                    comparison_code=args.code,
                    pairing_code=args.code,
                    scopes=args.scope or None,
                ),
                args.json,
            )
        elif args.command == "pair-deny":
            _print(manager.deny_pairing(args.pairing_id), args.json)
        elif args.command == "pair-cancel":
            _print(manager.cancel_pairing(args.pairing_id), args.json)
        return 0
    except (OwnerUnavailableError, OwnerRequestError, OSError) as exc:
        print(f"remote management failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
