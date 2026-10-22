"""Copied-root inventory is diagnostic, never runtime authority."""
import json
import os
from dataclasses import replace

import pytest

from nanobot._migration_preflight import CHECKS, Check, inspect_copy, reduce_checks


@pytest.fixture
def copied(tmp_path):
    root = tmp_path / "copy"
    root.mkdir()
    workspace = root / "workspace"
    workspace.mkdir()
    runtime = root / "instance"
    runtime.mkdir()
    config = runtime / "config.json"
    data = {"agents": {"defaults": {"workspace": str(workspace)}},
            "gateway": {"heartbeat": {"enabled": False}}}
    config.write_text(json.dumps(data))
    return dict(copy_root=str(root), config=str(config), workspace=str(workspace),
                sessions_root=str(runtime / "sessions"), runtime_data_root=str(runtime),
                stopped_writer_copy=True)


def checks(report):
    return {item["check"]: item for item in report["checks"]}


def test_clean_copy_is_not_activation_authority(copied):
    report = inspect_copy(**copied)
    assert report["result"] == "NOT_READY"
    assert report["activation"] == "NOT_AUTHORIZED"
    assert checks(report)["sessions"]["status"] == "PASS"
    assert checks(report)["p6_release"]["status"] == "UNQUALIFIED"
    assert checks(report)["d9_authorization"]["status"] == "UNQUALIFIED"
    assert str(copied["copy_root"]) not in json.dumps(report)


def test_reducer_requires_exact_complete_evidence():
    complete = tuple(Check(name, "PASS", "CLEAN", 0) for name in CHECKS)
    assert reduce_checks(complete)["result"] == "READY"
    for invalid in (complete[:-1], complete + complete[:1],
                    (replace(complete[0], count=True),) + complete[1:],
                    (replace(complete[0], status="MAYBE"),) + complete[1:]):
        assert reduce_checks(invalid)["result"] == "NOT_READY"
    for invalid in (None, {}, (None,) * len(CHECKS),
                    (replace(complete[0], status=[]),) + complete[1:]):
        assert reduce_checks(invalid)["result"] == "NOT_READY"


@pytest.mark.parametrize("field,value", [("workspace", "/"), ("workspace", "~/x"),
    ("sessions_root", "/outside/secret"), ("copy_root", "relative"),
    ("stopped_writer_copy", False), ("stopped_writer_copy", 1)])
def test_explicit_boundary_fails_content_free(copied, field, value):
    copied[field] = value
    report = inspect_copy(**copied)
    assert report["result"] == "NOT_READY"
    assert checks(report)["copy_boundary"]["status"] == "FAIL"
    assert "secret" not in json.dumps(report)


@pytest.mark.parametrize("kind", ["symlink", "fifo", "hardlink"])
def test_unsafe_config_never_read(copied, tmp_path, kind):
    from pathlib import Path
    config = Path(copied["config"])
    config.unlink()
    target = tmp_path / "secret"
    target.write_text("private-value")
    if kind == "symlink":
        config.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(config)
    else:
        os.link(target, config)
    assert checks(inspect_copy(**copied))["copy_boundary"]["status"] == "FAIL"


@pytest.mark.parametrize("subdir,gate", [("sessions", "sessions"),
    ("cron-runtime/old-identity", "cron_runtime"), ("artifacts", "artifact_state")])
def test_retained_runtime_is_unqualified_without_parsing(copied, subdir, gate):
    from pathlib import Path
    directory = Path(copied["runtime_data_root"]) / subdir
    directory.mkdir(parents=True)
    (directory / "record").write_text('{"sessions_root":"/host/private"}')
    assert checks(inspect_copy(**copied))[gate]["status"] == "UNQUALIFIED"


def test_memory_residue_and_workspace_cron_not_ignored(copied):
    from pathlib import Path
    workspace = Path(copied["workspace"])
    for name in ("memory", "cron"):
        (workspace / name).mkdir()
        (workspace / name / "private").write_text("sensitive")
    report = checks(inspect_copy(**copied))
    assert report["recovery"]["status"] == "UNQUALIFIED"
    assert report["cron_definitions"]["status"] == "UNQUALIFIED"


@pytest.mark.parametrize("value", [True, "false", 0, None])
def test_heartbeat_requires_explicit_exact_false(copied, value):
    from pathlib import Path
    config = Path(copied["config"])
    data = json.loads(config.read_text())
    data["gateway"]["heartbeat"]["enabled"] = value
    config.write_text(json.dumps(data))
    assert checks(inspect_copy(**copied))["background_dormancy"]["status"] == "FAIL"


def test_omitted_heartbeat_not_silently_rewritten(copied):
    from pathlib import Path
    config = Path(copied["config"])
    data = json.loads(config.read_text())
    del data["gateway"]
    config.write_text(json.dumps(data))
    before = config.read_bytes()
    assert checks(inspect_copy(**copied))["background_dormancy"]["status"] == "FAIL"
    assert config.read_bytes() == before


def test_env_and_mutation_sentinels(copied, monkeypatch):
    monkeypatch.setenv("NANOBOT_AGENTS__DEFAULTS__WORKSPACE", "/host/private")
    monkeypatch.setenv("NANOBOT_GATEWAY__HEARTBEAT__ENABLED", "true")
    def forbidden(*args, **kwargs):
        raise AssertionError("mutation")
    for name in ("mkdir", "chmod", "fsync", "fork"):
        monkeypatch.setattr(os, name, forbidden)
    assert checks(inspect_copy(**copied))["config_structure"]["status"] == "PASS"


@pytest.mark.parametrize("payload", [
    '{"agents":{"defaults":{"workspace":"PLACEHOLDER","sessionTtlMinutes":0,"idleCompactAfterMinutes":0}}}',
    '{"agents":{"defaults":{"workspace":"PLACEHOLDER","workspace":"PLACEHOLDER"}}}',
    '{"artifacts":{"maxBlobBytes":1,"max_blob_bytes":1}}',
])
def test_duplicate_and_compatibility_aliases_fail(copied, payload):
    from pathlib import Path
    Path(copied["config"]).write_text(payload.replace("PLACEHOLDER", copied["workspace"]))
    report = checks(inspect_copy(**copied))
    assert report["copy_boundary"]["status"] == "FAIL" or report["config_structure"]["status"] == "FAIL"


@pytest.mark.parametrize("ttl_fields", [
    {"sessionTtlMinutes": 0, "idleCompactAfterMinutes": 1},
    {"sessionTtlMinutes": 1, "idleCompactAfterMinutes": 0},
    {"idleCompactAfterMinutes": 1, "sessionTtlMinutes": 0},
    {"idleCompactAfterMinutes": 0, "sessionTtlMinutes": 1},
    {"sessionTtlMinutes": 0, "idleCompactAfterMinutes": 0},
    {"sessionTtlMinutes": False},
    {"sessionTtlMinutes": True},
    {"idleCompactAfterMinutes": False},
    {"idleCompactAfterMinutes": True},
    {"sessionTtlMinutes": -1},
    {"sessionTtlMinutes": "0"},
])
def test_ambiguous_or_invalid_ttl_fails_structure_and_dormancy(copied, ttl_fields):
    from pathlib import Path

    config = Path(copied["config"])
    data = json.loads(config.read_text())
    data["agents"]["defaults"].update(ttl_fields)
    config.write_text(json.dumps(data))
    report = checks(inspect_copy(**copied))
    assert report["config_structure"]["status"] == "FAIL"
    assert report["background_dormancy"]["status"] == "FAIL"


def test_inventory_bound_and_symlink_directory(copied, monkeypatch, tmp_path):
    from pathlib import Path

    import nanobot._migration_preflight as preflight
    directory = Path(copied["sessions_root"])
    directory.mkdir()
    (directory / "nested").symlink_to(tmp_path, target_is_directory=True)
    assert checks(inspect_copy(**copied))["copy_boundary"]["status"] == "FAIL"
    (directory / "nested").unlink()
    (directory / "first").write_text("private")
    monkeypatch.setattr(preflight, "MAX_ENTRIES", 0)
    assert inspect_copy(**copied)["result"] == "NOT_READY"


def test_config_bound_and_snapshot_change(copied, monkeypatch):
    from pathlib import Path

    import nanobot._migration_preflight as preflight
    original = preflight._Copy.recheck
    def changed(self):
        Path(copied["config"]).write_text("{}")
        original(self)
    monkeypatch.setattr(preflight._Copy, "recheck", changed)
    assert checks(inspect_copy(**copied))["copy_boundary"]["status"] == "FAIL"
    monkeypatch.setattr(preflight._Copy, "recheck", original)
    Path(copied["config"]).write_text("x" * (preflight.MAX_BYTES + 1))
    assert checks(inspect_copy(**copied))["copy_boundary"]["status"] == "FAIL"


@pytest.mark.parametrize("mode,expected", [("missing", "FAIL"), ("duplicate", "FAIL"),
    ("reserved", "FAIL"), ("legacy", "PASS"), ("malformed", "FAIL")])
def test_channel_metadata_without_entrypoint_load(copied, monkeypatch, mode, expected):
    from pathlib import Path
    from types import SimpleNamespace

    import nanobot._migration_preflight as preflight
    config = Path(copied["config"])
    data = json.loads(config.read_text())
    name = "telegram" if mode == "reserved" else "external"
    data["channels"] = {name: {"enabled": True}}
    config.write_text(json.dumps(data))
    def forbidden():
        raise AssertionError("entry_point.load must never run")
    point = SimpleNamespace(name=name, value="invalid" if mode == "malformed" else "thirdparty:Channel", load=forbidden)
    selected = [] if mode == "missing" else [point, point] if mode == "duplicate" else [point]
    monkeypatch.setattr(preflight.metadata, "entry_points", lambda **kw: selected)
    report = checks(inspect_copy(**copied))
    assert report["channel_plugins"]["status"] == expected
    if mode == "legacy":
        assert report["dependencies"]["status"] == "UNQUALIFIED"


def test_descriptor_missing_dependency_never_loads_runtime(copied, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    import nanobot._migration_preflight as preflight
    import nanobot.channels.plugin as plugin
    import nanobot.channels.registry as registry
    config = Path(copied["config"])
    data = json.loads(config.read_text())
    data["channels"] = {"telegram": {"enabled": True}}
    config.write_text(json.dumps(data))
    monkeypatch.setattr(preflight.metadata, "entry_points", lambda **kw: [])
    monkeypatch.setattr(registry, "packaged_channel_names", lambda: {"telegram"})
    monkeypatch.setattr(plugin, "load_channel_package", lambda name: SimpleNamespace(dependencies=("missing-package>=1",)))
    def missing(name):
        raise preflight.metadata.PackageNotFoundError
    monkeypatch.setattr(preflight.metadata, "version", missing)
    assert checks(inspect_copy(**copied))["dependencies"]["status"] == "FAIL"


def test_clean_content_inode_mtime_unchanged(copied):
    from pathlib import Path
    config = Path(copied["config"])
    before = config.read_bytes(), config.stat().st_ino, config.stat().st_mtime_ns
    inspect_copy(**copied)
    assert before == (config.read_bytes(), config.stat().st_ino, config.stat().st_mtime_ns)


@pytest.mark.parametrize("field,value", [("enabled", "false"), ("maxBlobBytes", True),
    ("maxNamespaceBytes", 1), ("root", "/host/private")])
def test_artifact_policy_strict_and_contained(copied, field, value):
    from pathlib import Path
    config = Path(copied["config"])
    data = json.loads(config.read_text())
    data["artifacts"] = {field: value}
    config.write_text(json.dumps(data))
    report = checks(inspect_copy(**copied))
    assert report["copy_boundary"]["status"] == "FAIL" or report["artifact_policy"]["status"] == "FAIL"


@pytest.mark.parametrize("subdir", [".nanobot", "sessions"])
def test_workspace_authority_evidence_unqualified(copied, subdir):
    from pathlib import Path
    directory = Path(copied["workspace"]) / subdir
    directory.mkdir()
    (directory / "session-mode-binding.json").write_text("private")
    assert checks(inspect_copy(**copied))["sessions"]["status"] == "UNQUALIFIED"


def test_actual_default_artifact_state_unqualified(copied):
    from pathlib import Path
    directory = Path(copied["runtime_data_root"]) / "context-artifacts"
    directory.mkdir()
    (directory / "blob").write_text("private")
    assert checks(inspect_copy(**copied))["artifact_state"]["status"] == "UNQUALIFIED"


@pytest.mark.parametrize("kind", ["absent", "overlapping", "nested"])
def test_workspace_runtime_exist_and_are_disjoint(copied, kind):
    from pathlib import Path
    workspace = Path(copied["workspace"])
    workspace.rmdir()
    if kind == "overlapping":
        copied["workspace"] = copied["runtime_data_root"]
    elif kind == "nested":
        copied["workspace"] = str(Path(copied["runtime_data_root"]) / "nested")
    assert checks(inspect_copy(**copied))["copy_boundary"]["status"] == "FAIL"


def test_json_decoder_keeps_string_keys_through_nested_containers(copied):
    from pathlib import Path

    from nanobot._migration_preflight import _Copy

    config = Path(copied["config"])
    payload = {"NestedConfig": {"0": [{"TagName": "ordinary"}], "Enabled": False}}
    config.write_text(json.dumps(payload))
    before = config.read_bytes()
    reader = _Copy(Path(copied["copy_root"]))
    try:
        decoded = reader.config(config)
        assert decoded == {"nested_config": {"0": [{"tag_name": "ordinary"}],
                                             "enabled": False}}
        reader.recheck()
    finally:
        reader.close()
    assert config.read_bytes() == before


@pytest.mark.parametrize("value", [None, 42, b"not-a-path", {"root": "not-a-path"}])
def test_preflight_path_rejects_non_string_without_coercion(copied, value):
    copied["workspace"] = value
    report = inspect_copy(**copied)
    assert checks(report)["copy_boundary"]["status"] == "FAIL"
    assert report["activation"] == "NOT_AUTHORIZED"
    assert "not-a-path" not in json.dumps(report)
