from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from nanobot.cli.commands import app
from nanobot.context_repo import ContextRepoManager
from nanobot.session.manager import SessionManager
from nanobot.session.runtime_authority import OfflineSessionOperator

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_process_title(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)


def _config(tmp_path: Path, *, artifacts: dict[str, object] | None = None) -> tuple[Path, Path, Path]:
    live_workspace = tmp_path / "live-workspace"
    live_workspace.mkdir()
    config_path = tmp_path / "live-instance" / "config.json"
    config_path.parent.mkdir()
    payload: dict[str, object] = {
        "agents": {"defaults": {"workspace": str(live_workspace)}},
        "providers": {"openrouter": {"apiKey": "${P5_STAGE2_UNSET_SECRET}"}},
    }
    if artifacts is not None:
        payload["artifacts"] = artifacts
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path, live_workspace, config_path.parent / "sessions"


def _legacy_copy(tmp_path: Path) -> tuple[Path, Path, str, Path, bytes]:
    copied_workspace = tmp_path / "copied-workspace"
    copied_workspace.mkdir()
    copied_sessions_root = tmp_path / "copied-sessions"
    manager = SessionManager(copied_workspace, sessions_root=copied_sessions_root)
    key = "cli:operator-fixture"
    session = manager.get_or_create(key)
    session.add_message("user", "copied fixture")
    manager.save(session)
    source = manager._get_session_path(key)
    return copied_workspace, copied_sessions_root, key, source, source.read_bytes()


def _copy_options(config: Path, workspace: Path, sessions: Path) -> list[str]:
    return [
        "--config",
        str(config),
        "--copied-workspace",
        str(workspace),
        "--copied-sessions-root",
        str(sessions),
        "--offline-copy",
    ]


async def _artifact_copy(tmp_path: Path) -> dict[str, object]:
    from nanobot.agent.tools.execution import KnownToolResult
    from nanobot.agent.tools.filesystem import ReadFileTool
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.session.artifact import parse_artifact_descriptor
    from nanobot.session.artifact_factory import ConfiguredArtifactReaderFactory
    from nanobot.session.artifact_store import PrivateArtifactStore
    from nanobot.session.execution_owner import acquire_execution
    from nanobot.session.manager import JsonlSessionStore, Session
    from nanobot.session.source_store import SourceSessionStore
    from nanobot.session.turn_writer import SourceTurnWriter

    live_workspace = tmp_path / "live-artifact-workspace"
    live_workspace.mkdir()
    live_sessions_root = tmp_path / "live-artifact-sessions"
    backend = JsonlSessionStore(live_workspace, sessions_root=live_sessions_root)
    session = Session("test:read")
    backend.save(session)
    backend.source.upgrade_source_session(
        session.key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    content_text = "retained canonical 中文 🌟"
    source_file = live_workspace / "file"
    source_file.write_text(content_text, encoding="utf-8")
    read_file = ReadFileTool(workspace=live_workspace, allowed_dir=live_workspace)
    async with acquire_execution(backend, session.key, run_id="run") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="run")
        outcome = await ToolRegistry.execute_prepared(
            read_file,
            {"path": "file"},
            SimpleNamespace(lock_owner=None),
        )
        writer.begin_tool_batch(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "read",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            }
        )
        await writer.record_tool_result(
            KnownToolResult(
                0,
                "read",
                "read_file",
                outcome,
                outcome.content,
                {"status": "ok"},
                True,
                outcome.producer_receipt,
            )
        )
        snapshot = owner.read_snapshot()
        event = snapshot.events[-1]
        content = event["content"].encode()
        descriptor = parse_artifact_descriptor(
            {
                "schema_version": 1,
                "artifact_id": "b" * 32,
                "workspace_id": backend.sessions_dir.name,
                "session_key": session.key,
                "incarnation": snapshot.state.incarnation,
                "generation": snapshot.state.generation,
                "event_id": event["_event_id"],
                "execution_id": event["_execution_id"],
                "stream_id": writer.stream_id,
                "run_id": "run",
                "block_index": None,
                "mime_type": "text/plain; charset=utf-8",
                "byte_length": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "created_at": "2026-09-30T00:00:00+00:00",
                "source_access": outcome.producer_receipt.to_dict()["source_access"],
            }
        )
        live_artifact_root = tmp_path / "live-artifacts"
        store = PrivateArtifactStore(
            live_artifact_root,
            workspace_id=descriptor.workspace_id,
            session_key=session.key,
            incarnation=descriptor.incarnation,
            generation=descriptor.generation,
        )
        source = SourceSessionStore(backend, artifact_store=store)
        with store.transaction() as blobs:
            blobs.put(descriptor, content)
            source.commit_artifact_descriptor(
                session.key,
                snapshot.state,
                descriptor,
                owner=owner,
                blobs=blobs,
            )

    copied_workspace = tmp_path / "copied-artifact-workspace"
    copied_sessions_root = tmp_path / "copied-artifact-sessions"
    copied_artifact_root = tmp_path / "copied-artifacts"
    shutil.copytree(live_workspace, copied_workspace)
    shutil.copytree(live_sessions_root, copied_sessions_root)
    shutil.copytree(live_artifact_root, copied_artifact_root)
    bootstrap = await OfflineSessionOperator(
        copied_workspace,
        copied_sessions_root=copied_sessions_root,
        live_workspace=live_workspace,
        live_sessions_root=live_sessions_root,
    ).acquire()
    bootstrap.manager.bootstrap_existing_sessions(offline_copy=True)
    assert bootstrap.close is not None
    await bootstrap.close()
    factory = ConfiguredArtifactReaderFactory(
        root=live_artifact_root,
        max_blob_bytes=64 * 1024 * 1024,
        max_namespace_bytes=1024 * 1024 * 1024,
    )
    return {
        "copied_workspace": copied_workspace,
        "copied_sessions_root": copied_sessions_root,
        "artifact_root": copied_artifact_root,
        "live_workspace": live_workspace,
        "factory": factory,
        "key": session.key,
        "incarnation": descriptor.incarnation,
        "generation": descriptor.generation,
        "descriptor": descriptor,
        "content": content,
    }


def test_cli_bootstrap_convert_recover_and_export_real_copy(tmp_path: Path) -> None:
    config, _live_workspace, _live_sessions = _config(tmp_path)
    workspace, sessions, key, source, legacy_bytes = _legacy_copy(tmp_path)
    options = _copy_options(config, workspace, sessions)

    bootstrap = runner.invoke(app, ["sessions", "bootstrap", *options])
    assert bootstrap.exit_code == 0, bootstrap.output
    assert "older nanobot binaries" in bootstrap.output
    assert "do not honor the Stage 2 writer lease" in bootstrap.output
    assert source.read_bytes() == legacy_bytes

    convert = runner.invoke(app, ["sessions", "convert", key, *options])
    assert convert.exit_code == 0, convert.output
    header = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
    assert header["metadata"]["source_schema_version"] == 1

    recover = runner.invoke(app, ["sessions", "recover", key, *options])
    assert recover.exit_code == 0, recover.output

    destination = tmp_path / "export-bundle"
    export = runner.invoke(
        app,
        ["sessions", "export", key, *options, "--destination", str(destination)],
    )
    assert export.exit_code == 0, export.output
    assert (destination / "canonical-source.jsonl").read_bytes() == source.read_bytes()
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["classification"] == "canonical-lossless"


def test_cli_requires_explicit_offline_qualification(tmp_path: Path) -> None:
    config, _live_workspace, _live_sessions = _config(tmp_path)
    workspace = tmp_path / "copy"
    sessions = tmp_path / "sessions-copy"

    result = runner.invoke(
        app,
        [
            "sessions",
            "bootstrap",
            "--config",
            str(config),
            "--copied-workspace",
            str(workspace),
            "--copied-sessions-root",
            str(sessions),
        ],
    )

    assert result.exit_code == 1
    assert "--offline-copy is required" in result.output
    assert not workspace.exists() and not sessions.exists()


def test_cli_rejects_live_identity_with_content_free_error(tmp_path: Path) -> None:
    config, live_workspace, _live_sessions = _config(tmp_path)
    copied_sessions = tmp_path / "copied-sessions"
    secret_key = "cli:must-not-appear"

    result = runner.invoke(
        app,
        [
            "sessions",
            "convert",
            secret_key,
            "--config",
            str(config),
            "--copied-workspace",
            str(live_workspace),
            "--copied-sessions-root",
            str(copied_sessions),
            "--offline-copy",
        ],
    )

    assert result.exit_code == 1
    assert "Copied-state session operation is unavailable" in result.output
    assert secret_key not in result.output
    assert str(live_workspace) not in result.output
    assert not copied_sessions.exists()


def test_cli_artifact_disabled_is_aggregate_and_creates_no_storage(tmp_path: Path) -> None:
    config, _live_workspace, _live_sessions = _config(tmp_path, artifacts={"enabled": False})
    workspace = tmp_path / "absent-copy-workspace"
    sessions = tmp_path / "absent-copy-sessions"
    artifacts = (tmp_path / "absent-copy-artifacts").resolve()
    key = "cli:hidden"

    result = runner.invoke(
        app,
        [
            "sessions",
            "artifacts",
            "inspect",
            key,
            *_copy_options(config, workspace, sessions),
            "--copied-artifact-root",
            str(artifacts),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "status=disabled" in result.output
    assert key not in result.output
    assert not workspace.exists() and not sessions.exists() and not artifacts.exists()


def test_cli_artifact_failure_redacts_exact_inputs(tmp_path: Path) -> None:
    config, _live_workspace, _live_sessions = _config(tmp_path, artifacts={"enabled": False})
    workspace = tmp_path / "absent-copy-workspace"
    sessions = tmp_path / "absent-copy-sessions"
    artifacts = (tmp_path / "absent-copy-artifacts").resolve()
    key = "cli:private-key"
    reference = "artifact:" + "a" * 32

    result = runner.invoke(
        app,
        [
            "sessions",
            "artifacts",
            "rematerialize",
            key,
            *_copy_options(config, workspace, sessions),
            "--copied-artifact-root",
            str(artifacts),
            "--incarnation",
            "private-incarnation",
            "--generation",
            "0",
            "--reference",
            reference,
        ],
    )

    assert result.exit_code == 1
    assert "Copied-state session operation is unavailable" in result.output
    assert key not in result.output
    assert reference not in result.output
    assert "private-incarnation" not in result.output


@pytest.mark.parametrize("generation", ["P5_SECRET_SENTINEL", "-1"])
def test_cli_rematerialize_rejects_generation_before_config_or_root_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation: str,
) -> None:
    def unexpected_config(_path: Path) -> None:
        raise AssertionError("configuration load must remain unreachable")

    monkeypatch.setattr("nanobot.config.loader.load_config", unexpected_config)
    workspace = tmp_path / "absent-workspace"
    sessions = tmp_path / "absent-sessions"
    artifacts = (tmp_path / "absent-artifacts").resolve()
    result = runner.invoke(
        app,
        [
            "sessions",
            "artifacts",
            "rematerialize",
            "cli:hidden",
            "--copied-workspace",
            str(workspace),
            "--copied-sessions-root",
            str(sessions),
            "--copied-artifact-root",
            str(artifacts),
            "--incarnation",
            "hidden-incarnation",
            "--generation",
            generation,
            "--reference",
            "artifact:" + "a" * 32,
            "--offline-copy",
        ],
    )

    assert result.exit_code == 1
    assert "Copied-state session operation is unavailable" in result.output
    assert generation not in result.output
    assert not workspace.exists() and not sessions.exists() and not artifacts.exists()


def test_cli_enabled_artifact_inspect_and_rematerialize_real_copy(tmp_path: Path) -> None:
    fixture = asyncio.run(_artifact_copy(tmp_path))
    config_path = tmp_path / "artifact-instance" / "config.json"
    config_path.parent.mkdir()
    factory = fixture["factory"]
    config_path.write_text(
        json.dumps(
            {
                "agents": {"defaults": {"workspace": str(fixture["live_workspace"])}},
                "artifacts": {
                    "enabled": True,
                    "root": str(factory.settings.root),  # type: ignore[union-attr]
                    "maxBlobBytes": factory.settings.max_blob_bytes,  # type: ignore[union-attr]
                    "maxNamespaceBytes": factory.settings.max_namespace_bytes,  # type: ignore[union-attr]
                },
            }
        ),
        encoding="utf-8",
    )
    options = _copy_options(
        config_path,
        fixture["copied_workspace"],  # type: ignore[arg-type]
        fixture["copied_sessions_root"],  # type: ignore[arg-type]
    )
    artifact_root = fixture["artifact_root"]
    key = fixture["key"]

    inspected = runner.invoke(
        app,
        [
            "sessions",
            "artifacts",
            "inspect",
            key,
            *options,
            "--copied-artifact-root",
            str(artifact_root),
        ],
    )
    assert inspected.exit_code == 0, inspected.output
    assert "status=complete" in inspected.output
    assert "reachable_blobs=1" in inspected.output
    assert key not in inspected.output

    blob = next(artifact_root.rglob("*.blob"))  # type: ignore[union-attr]
    blob.unlink()
    rematerialized = runner.invoke(
        app,
        [
            "sessions",
            "artifacts",
            "rematerialize",
            key,
            *options,
            "--copied-artifact-root",
            str(artifact_root),
            "--incarnation",
            fixture["incarnation"],
            "--generation",
            str(fixture["generation"]),
            "--reference",
            fixture["descriptor"].reference,  # type: ignore[union-attr]
        ],
    )
    assert rematerialized.exit_code == 0, rematerialized.output
    assert "status=installed" in rematerialized.output
    assert blob.read_bytes() == fixture["content"]


def test_cli_malformed_config_failure_is_path_and_content_free(tmp_path: Path) -> None:
    sentinel = "PRIVATE-CONFIG-SENTINEL"
    config = tmp_path / sentinel / "config.json"
    config.parent.mkdir()
    config.write_text("{" + sentinel, encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "sessions",
            "bootstrap",
            "--config",
            str(config),
            "--copied-workspace",
            str(tmp_path / "copy-workspace"),
            "--copied-sessions-root",
            str(tmp_path / "copy-sessions"),
            "--offline-copy",
        ],
    )

    assert result.exit_code == 1
    assert "Copied-state session operation is unavailable" in result.output
    assert sentinel not in result.output
    assert str(config) not in result.output


def test_managed_restore_workspace_has_fixed_export_guidance(tmp_path: Path) -> None:
    workspace = tmp_path / "managed-workspace"
    workspace.mkdir()
    instance = tmp_path / "managed-instance"
    config = instance / "config.json"
    instance.mkdir()
    config.write_text(
        json.dumps({"agents": {"defaults": {"workspace": str(workspace)}}}),
        encoding="utf-8",
    )
    sessions = instance / "sessions"
    unmanaged = SessionManager(workspace, sessions_root=sessions)
    session = unmanaged.get_or_create("cli:managed-private")
    session.add_message("user", "private-content")
    unmanaged.save(session)

    async def bootstrap() -> None:
        lease = await OfflineSessionOperator(
            workspace,
            copied_sessions_root=sessions,
        ).acquire()
        try:
            lease.manager.bootstrap_existing_sessions(offline_copy=True)
        finally:
            assert lease.close is not None
            await lease.close()

    asyncio.run(bootstrap())
    result = runner.invoke(app, ["sessions", "restore-workspace", "--config", str(config)])

    assert result.exit_code == 1
    assert "Use sessions export against a verified" in result.output
    assert "offline copy" in result.output
    assert str(config) not in result.output
    assert str(workspace) not in result.output
    assert "cli:managed-private" not in result.output


@pytest.mark.parametrize("checker", ["workspace_id_if_present", "namespace_claimed"])
def test_restore_workspace_classification_failure_never_falls_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checker: str,
) -> None:
    sentinel = "P5_PRIVATE_CLASSIFIER_SENTINEL"
    workspace = tmp_path / "untrusted-workspace"
    workspace.mkdir()
    config = tmp_path / "instance" / "config.json"
    config.parent.mkdir()
    config.write_text(
        json.dumps({"agents": {"defaults": {"workspace": str(workspace)}}}),
        encoding="utf-8",
    )

    def fail_classification(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(sentinel)

    def forbidden_generic_loader(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("generic loader must remain unreachable")

    monkeypatch.setattr(f"nanobot.session.mode_storage.{checker}", fail_classification)
    monkeypatch.setattr("nanobot.cli.commands._load_runtime_config", forbidden_generic_loader)
    before = tuple(tmp_path.rglob("*"))
    result = runner.invoke(app, ["sessions", "restore-workspace", "--config", str(config)])

    assert result.exit_code == 1
    assert "Use sessions export against a verified" in result.output
    assert sentinel not in result.output
    assert str(workspace) not in result.output
    assert tuple(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("root_relation", ["disjoint", "within", "ancestor"])
def test_cli_artifact_roots_omit_absent_targets_and_retain_restricted_targets(
    tmp_path: Path, root_relation: str,
) -> None:
    context_root = tmp_path / "context"
    context_root.mkdir()
    configured_root = tmp_path / "target" / "restricted"
    (context_root / "nanobot.context.json").write_text(json.dumps({
        "targetRepos": {
            "unconfigured": {},
            "restricted": {"path": str(configured_root), "read": [], "write": []},
        },
    }), encoding="utf-8")
    manager = ContextRepoManager.from_config(context_repos=[str(context_root)])
    assert any(target.path is None for target in manager.target_repos())
    assert configured_root not in manager.read_roots()
    assert configured_root not in manager.write_roots()
    root = {
        "disjoint": tmp_path / "live-private-artifacts",
        "within": configured_root / "private-artifacts",
        "ancestor": configured_root.parent,
    }[root_relation]
    config_path, live_workspace, _live_sessions = _config(
        tmp_path, artifacts={"enabled": True, "root": str(root)},
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["agents"]["defaults"]["contextRepos"] = [str(context_root)]
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    workspace, sessions, key, source, _legacy_bytes = _legacy_copy(tmp_path)
    options = _copy_options(config_path, workspace, sessions)
    bootstrap = runner.invoke(app, ["sessions", "bootstrap", *options])
    assert bootstrap.exit_code == 0, bootstrap.output
    convert = runner.invoke(app, ["sessions", "convert", key, *options])
    assert convert.exit_code == 0, convert.output
    before = source.read_bytes()
    copied_artifacts = tmp_path / "copied-private-artifacts"

    result = runner.invoke(app, [
        "sessions", "artifacts", "inspect", key, *options,
        "--copied-artifact-root", str(copied_artifacts),
    ])

    if root_relation == "disjoint":
        assert result.exit_code == 0, result.output
        assert "status=complete" in result.output
        assert "reachable_blobs=0" in result.output
    else:
        assert result.exit_code == 1
        assert "Copied-state session operation is unavailable" in result.output
        assert not copied_artifacts.exists()
    assert key not in result.output
    assert str(root) not in result.output
    assert str(configured_root) not in result.output
    assert str(live_workspace) not in result.output
    assert source.read_bytes() == before
    assert not root.exists()
    assert not configured_root.exists()
