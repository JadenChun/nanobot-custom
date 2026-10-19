"""Offline explicit-root cron inspection; imports no runtime/config credentials."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import typer

from nanobot.cron.runtime_inspection import (
    inspect_occurrences,
    inspect_scheduler,
    reconcile_occurrence,
)

cron_runtime_app = typer.Typer(no_args_is_help=True, help="Inspect copied dormant cron evidence offline.")


def _emit(operation: Callable[[], dict[str, object]]) -> None:
    try:
        result = operation()
    except Exception:
        typer.echo("Error: cron evidence unavailable or conflicting.", err=True)
        raise typer.Exit(5) from None
    typer.echo(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


@cron_runtime_app.command("scheduler-inspect")
def scheduler_inspect(root: Path = typer.Option(...), namespace: str = typer.Option(...),
                      runtime_identity: str = typer.Option(...)) -> None:
    _emit(lambda: inspect_scheduler(root, namespace=namespace, runtime_identity=runtime_identity))


@cron_runtime_app.command("occurrence-inspect")
def occurrence_inspect(root: Path = typer.Option(...), namespace: str = typer.Option(...)) -> None:
    _emit(lambda: inspect_occurrences(root, namespace=namespace))


@cron_runtime_app.command("occurrence-reconcile")
def occurrence_reconcile(root: Path = typer.Option(...), namespace: str = typer.Option(...),
                         occurrence_id: str = typer.Option(...), expected_owner: str = typer.Option(...),
                         expected_generation: int = typer.Option(...), resolution: str = typer.Option(...),
                         external_evidence: str = typer.Option(...), successor_owner: str = typer.Option(...),
                         scheduler_root: Path = typer.Option(...), runtime_identity: str = typer.Option(...)) -> None:
    _emit(lambda: reconcile_occurrence(root, namespace=namespace, occurrence_id=occurrence_id,
        expected_owner=expected_owner, expected_generation=expected_generation, resolution=resolution,
        external_evidence=external_evidence, successor_owner=successor_owner,
        scheduler_root=scheduler_root, runtime_identity=runtime_identity))
