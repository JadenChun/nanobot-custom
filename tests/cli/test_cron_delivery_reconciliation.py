"""CLI boundary tests for explicit offline delivery reconciliation."""

from __future__ import annotations

import json
import os
from pathlib import Path

from typer.testing import CliRunner

import nanobot.cli.cron_delivery as cron_cli
from nanobot.cli.cron_delivery import cron_delivery_app

runner = CliRunner()


def test_help_and_absent_inspection_create_nothing(tmp_path: Path) -> None:
    help_result = runner.invoke(cron_delivery_app, ["--help"])
    assert help_result.exit_code == 0
    assert list(tmp_path.iterdir()) == []

    outbox = tmp_path / "absent" / "outbox"
    artifacts = tmp_path / "absent" / "artifacts"
    result = runner.invoke(
        cron_delivery_app,
        [
            "inspect", "--outbox-root", str(outbox), "--namespace", "cron:test",
            "--artifacts-root", str(artifacts), "--json",
        ],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "clean": True,
        "entries": [],
        "schema": "nanobot.cron_delivery_reconciliation_inspection.v1",
    }
    assert not (tmp_path / "absent").exists()


def test_registered_group_help_creates_nothing(tmp_path: Path, monkeypatch) -> None:
    import nanobot.cli.commands as commands

    monkeypatch.setattr(commands, "set_cli_process_identity", lambda _args: None)
    before = set(tmp_path.rglob("*"))
    result = runner.invoke(commands.app, ["cron-delivery", "--help"])
    assert result.exit_code == 0
    assert "plan-show" in result.output
    assert set(tmp_path.rglob("*")) == before


def test_every_phase_requires_explicit_authority_selectors() -> None:
    invocations = [
        ["inspect"],
        ["evidence", "import"],
        ["plan"],
        ["plan-show"],
        ["apply"],
    ]
    for arguments in invocations:
        result = runner.invoke(cron_delivery_app, arguments)
        assert result.exit_code == 2
        assert "--outbox-root" in result.output
        help_result = runner.invoke(cron_delivery_app, [*arguments, "--help"])
        assert "--namespace" in help_result.output
        assert "--artifacts-root" in help_result.output


def test_unsafe_selectors_are_content_free_exit_five(tmp_path: Path) -> None:
    result = runner.invoke(
        cron_delivery_app,
        [
            "inspect", "--outbox-root", str(tmp_path / "a" / ".." / "outbox"),
            "--namespace", "cron:test", "--artifacts-root", str(tmp_path / "artifacts"),
        ],
    )
    assert result.exit_code == 5
    assert result.stdout == ""
    assert result.stderr == "Error: reconciliation authority or artifact is malformed or unsafe.\n"


def _receipt_import_args(tmp_path: Path, receipt: Path) -> list[str]:
    return [
        "evidence", "import",
        "--outbox-root", str(tmp_path / "outbox"),
        "--namespace", "cron:test",
        "--artifacts-root", str(tmp_path / "artifacts"),
        "--outbox-id", "a" * 64,
        "--destination-id", "b" * 64,
        "--outcome", "confirmed_delivered",
        "--receipt-file", str(receipt),
    ]


def test_receipt_file_rejects_symlink_fifo_and_oversize_without_leak(tmp_path: Path) -> None:
    private = tmp_path / "private.json"
    private.write_text('{"secret":"never-print"}', encoding="utf-8")
    private.chmod(0o600)
    symlink = tmp_path / "receipt-link.json"
    symlink.symlink_to(private)
    fifo = tmp_path / "receipt-fifo"
    os.mkfifo(fifo, 0o600)
    oversized = tmp_path / "receipt-large.json"
    oversized.write_bytes(b"x" * (256 * 1024 + 1))
    oversized.chmod(0o600)

    for candidate in (symlink, fifo, oversized):
        result = runner.invoke(cron_delivery_app, _receipt_import_args(tmp_path, candidate))
        assert result.exit_code == 5
        assert "never-print" not in result.output
        assert str(candidate) not in result.output
        assert result.stderr == (
            "Error: reconciliation authority or artifact is malformed or unsafe.\n"
        )


def test_receipt_file_binding_mutation_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"schema":"secret-value"}', encoding="utf-8")
    receipt.chmod(0o600)
    original_read = cron_cli.os.read
    changed = False

    def mutate_after_read(fd: int, size: int) -> bytes:
        nonlocal changed
        data = original_read(fd, size)
        if data and not changed:
            changed = True
            moved = tmp_path / "old-receipt"
            receipt.rename(moved)
            receipt.write_bytes(data)
            receipt.chmod(0o600)
        return data

    monkeypatch.setattr(cron_cli.os, "read", mutate_after_read)
    result = runner.invoke(cron_delivery_app, _receipt_import_args(tmp_path, receipt))
    assert changed
    assert result.exit_code == 5
    assert "secret-value" not in result.output
    assert result.stderr == (
        "Error: reconciliation authority or artifact is malformed or unsafe.\n"
    )
