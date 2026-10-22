"""Explicit offline cron-delivery reconciliation commands."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import typer

from nanobot.cron.delivery_store import (
    DeliveryOutboxCorruptError,
    DeliveryOutboxPersistenceError,
    DeliverySendBusyError,
)

cron_delivery_app = typer.Typer(
    help="Inspect and explicitly reconcile dormant cron delivery attempts offline.",
    no_args_is_help=True,
)
evidence_app = typer.Typer(help="Import immutable reconciliation evidence.", no_args_is_help=True)
cron_delivery_app.add_typer(evidence_app, name="evidence")

T = TypeVar("T")
_MAX_RECEIPT_FILE = 256 * 1024
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


def _emit(value: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True))
        return
    for key, item in value.items():
        typer.echo(f"{key}: {item}")


def _invoke(operation: Callable[[], T]) -> T:
    from nanobot.cron.delivery_reconciliation import (
        ReconciliationConflictError,
        ReconciliationUnsafeError,
    )

    try:
        return operation()
    except DeliverySendBusyError:
        typer.echo("Error: destination send fence is busy.", err=True)
        raise typer.Exit(3) from None
    except ReconciliationConflictError:
        typer.echo("Error: reconciliation state is stale or conflicts.", err=True)
        raise typer.Exit(4) from None
    except (ReconciliationUnsafeError, DeliveryOutboxCorruptError):
        typer.echo("Error: reconciliation authority or artifact is malformed or unsafe.", err=True)
        raise typer.Exit(5) from None
    except (DeliveryOutboxPersistenceError, OSError, ValueError):
        typer.echo("Error: reconciliation operation failed.", err=True)
        raise typer.Exit(1) from None


def _roots(
    outbox_root: Path,
    namespace: str,
    artifacts_root: Path,
) -> tuple[Path, str, Path]:
    # Typer converts paths but must not resolve them: lexical '..' and binding
    # ambiguity are rejected by the owning offline library.
    return outbox_root, namespace, artifacts_root


def _read_private_receipt_file(path: Path) -> bytes:
    from nanobot.cron.delivery_reconciliation import ReconciliationUnsafeError

    fd: int | None = None
    try:
        fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK)
        before = os.fstat(fd)
        entry = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino) != (entry.st_dev, entry.st_ino)
            or before.st_size > _MAX_RECEIPT_FILE
        ):
            raise ReconciliationUnsafeError("receipt evidence file is unsafe")
        signature = (
            before.st_mode,
            before.st_uid,
            before.st_gid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        chunks: list[bytes] = []
        remaining = _MAX_RECEIPT_FILE + 1
        while remaining:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(fd)
        rebound = os.stat(path, follow_symlinks=False)
        after_signature = (
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            len(data) > _MAX_RECEIPT_FILE
            or after_signature != signature
            or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or (rebound.st_dev, rebound.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ReconciliationUnsafeError("receipt evidence file changed during read")
        return data
    except ReconciliationUnsafeError:
        raise
    except OSError as exc:
        raise ReconciliationUnsafeError("receipt evidence file is unavailable or unsafe") from exc
    finally:
        if fd is not None:
            os.close(fd)


@cron_delivery_app.command("inspect")
def inspect_command(
    outbox_root: Path = typer.Option(..., "--outbox-root"),
    namespace: str = typer.Option(..., "--namespace"),
    artifacts_root: Path = typer.Option(..., "--artifacts-root"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Inspect existing E2 state without creating, syncing, or repairing it."""
    from nanobot.cron.delivery_reconciliation import inspect_reconciliation

    roots = _roots(outbox_root, namespace, artifacts_root)
    result = _invoke(lambda: inspect_reconciliation(roots[0], namespace=roots[1], artifacts_root=roots[2]))
    payload = {
        "clean": result.clean,
        "entries": [
            {
                "action_id": item.action_id,
                "attempt": item.attempt,
                "channel": item.channel,
                "destination_id": item.destination_id,
                "outbox_id": item.outbox_id,
                "owner": item.owner,
                "permitted_outcomes": list(item.permitted_outcomes),
                "record_revision": item.record_revision,
                "request_digest": item.request_digest,
                "status": item.status,
                "target_redacted": True,
            }
            for item in result.entries
        ],
        "schema": "nanobot.cron_delivery_reconciliation_inspection.v1",
    }
    _emit(payload, json_output=json_output)
    if not result.clean:
        raise typer.Exit(2)


@evidence_app.command("import")
def evidence_import_command(
    outbox_root: Path = typer.Option(..., "--outbox-root"),
    namespace: str = typer.Option(..., "--namespace"),
    artifacts_root: Path = typer.Option(..., "--artifacts-root"),
    outbox_id: str = typer.Option(..., "--outbox-id"),
    destination_id: str = typer.Option(..., "--destination-id"),
    outcome: str = typer.Option(..., "--outcome"),
    receipt_file: Path | None = typer.Option(None, "--receipt-file"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Import exact predecessor-bound evidence as a private immutable artifact."""
    from nanobot.cron.delivery_reconciliation import (
        ReconciliationUnsafeError,
        import_evidence,
    )

    def run():
        receipt = None
        if receipt_file is not None:
            raw = _read_private_receipt_file(receipt_file)
            def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
                value: dict[str, Any] = {}
                for key, item in items:
                    if key in value:
                        raise ValueError("duplicate")
                    value[key] = item
                return value

            def nonfinite(_value: str) -> None:
                raise ValueError("nonfinite")

            try:
                receipt = json.loads(
                    raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
                raise ReconciliationUnsafeError("receipt evidence is malformed") from exc
            if not isinstance(receipt, dict):
                raise ReconciliationUnsafeError("receipt evidence is malformed")
        if outcome not in {"confirmed_not_sent", "confirmed_delivered", "abandon"}:
            raise ReconciliationUnsafeError("outcome is unsupported")
        # JSON decoding constructs string keys; non-None receipts were checked above.
        return import_evidence(
            outbox_root,
            namespace=namespace,
            artifacts_root=artifacts_root,
            outbox_id=outbox_id,
            destination_id=destination_id,
            outcome=outcome,  # type: ignore[arg-type]
            receipt=cast(dict[str, Any] | None, receipt),
        )

    artifact = _invoke(run)
    _emit(
        {
            "artifact_ref": artifact.reference,
            "kind": artifact.kind,
            "schema": "nanobot.cron_delivery_reconciliation_import.v1",
        },
        json_output=json_output,
    )


@cron_delivery_app.command("plan")
def plan_command(
    outbox_root: Path = typer.Option(..., "--outbox-root"),
    namespace: str = typer.Option(..., "--namespace"),
    artifacts_root: Path = typer.Option(..., "--artifacts-root"),
    evidence_ref: str = typer.Option(..., "--evidence-ref"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Create an immutable plan from fresh exact predecessor evidence."""
    from nanobot.cron.delivery_reconciliation import create_plan

    artifact = _invoke(
        lambda: create_plan(
            outbox_root, namespace=namespace, artifacts_root=artifacts_root,
            evidence_ref=evidence_ref,
        )
    )
    _emit(
        {
            "artifact_ref": artifact.reference,
            "plan_id": artifact.semantic_id,
            "schema": "nanobot.cron_delivery_reconciliation_plan_created.v1",
        },
        json_output=json_output,
    )


@cron_delivery_app.command("plan-show")
def plan_show_command(
    outbox_root: Path = typer.Option(..., "--outbox-root"),
    namespace: str = typer.Option(..., "--namespace"),
    artifacts_root: Path = typer.Option(..., "--artifacts-root"),
    plan_id: str = typer.Option(..., "--plan-id"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show one immutable plan without message content or raw target."""
    from nanobot.cron.delivery_reconciliation import show_plan

    value = _invoke(
        lambda: show_plan(
            outbox_root, namespace=namespace, artifacts_root=artifacts_root, plan_id=plan_id
        )
    )
    value = {"schema": "nanobot.cron_delivery_reconciliation_plan_show.v1", **value}
    _emit(value, json_output=json_output)


@cron_delivery_app.command("apply")
def apply_command(
    outbox_root: Path = typer.Option(..., "--outbox-root"),
    namespace: str = typer.Option(..., "--namespace"),
    artifacts_root: Path = typer.Option(..., "--artifacts-root"),
    plan_id: str = typer.Option(..., "--plan-id"),
    confirm_plan: str = typer.Option(..., "--confirm-plan"),
    confirmation: str = typer.Option(..., "--confirmation"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Apply one exact plan after both irreversible confirmations."""
    from nanobot.cron.delivery_reconciliation import apply_plan

    value = _invoke(
        lambda: apply_plan(
            outbox_root,
            namespace=namespace,
            artifacts_root=artifacts_root,
            plan_id=plan_id,
            confirm_plan=confirm_plan,
            confirmation=confirmation,
        )
    )
    value = {"schema": "nanobot.cron_delivery_reconciliation_applied.v1", **value}
    _emit(value, json_output=json_output)


__all__ = ["cron_delivery_app"]
