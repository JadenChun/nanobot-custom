import json

import pytest

from nanobot._migration_preflight import _Copy, _UnsafeError
from nanobot.cli.migration_preflight import main


def test_shared_ancestor_sibling_change_does_not_change_copy_boundary(tmp_path):
    root = tmp_path / "copy"
    root.mkdir()
    copied = _Copy(root)
    try:
        (tmp_path / "unrelated-sibling").mkdir()
        copied.recheck()
    finally:
        copied.close()


@pytest.mark.parametrize("change", ["copy_contents", "ancestor_binding", "ancestor_permissions"])
def test_copy_or_ancestor_authority_change_still_fails_closed(tmp_path, change):
    parent = tmp_path / "parent"
    parent.mkdir(mode=0o700)
    root = parent / "copy"
    root.mkdir()
    copied = _Copy(root)
    try:
        if change == "copy_contents":
            (root / "changed").mkdir()
        elif change == "ancestor_binding":
            parent.rename(tmp_path / "moved")
            parent.mkdir(mode=0o700)
        else:
            parent.chmod(0o750)
        with pytest.raises(_UnsafeError):
            copied.recheck()
    finally:
        copied.close()


def test_missing_args_are_constant_json(capsys):
    assert main([]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["result"] == "NOT_READY"
    assert report["activation"] == "NOT_AUTHORIZED"


def test_bad_args_do_not_leak(capsys):
    assert main(["--private-token=secret-value"]) == 2
    output = capsys.readouterr()
    assert "secret" not in output.out + output.err
    assert json.loads(output.out)["result"] == "NOT_READY"


def test_clean_explicit_cli_is_deterministic_not_ready(tmp_path, capsys):
    root = tmp_path.resolve() / "copy"
    root.mkdir()
    workspace = root / "workspace"
    workspace.mkdir()
    runtime = root / "runtime"
    runtime.mkdir()
    config = runtime / "config.json"
    config.write_text(json.dumps({"agents": {"defaults": {"workspace": str(workspace)}},
                                  "gateway": {"heartbeat": {"enabled": False}}}))
    args = ["--copy-root", str(root), "--config", str(config), "--workspace", str(workspace),
            "--sessions-root", str(runtime / "sessions"), "--runtime-data-root", str(runtime),
            "--stopped-writer-copy"]
    assert main(args) == 1
    first = capsys.readouterr().out
    assert main(args) == 1
    assert first == capsys.readouterr().out
    assert json.loads(first)["result"] == "NOT_READY"
    assert str(root) not in first
