"""Production publication and destructive maintenance share storage custody."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nanobot.attachments.lifecycle import migrate_legacy_media, prune_unreferenced
from nanobot.attachments.store import AttachmentStore
from nanobot.config.loader import set_config_path
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.source import new_source_event
from nanobot.utils.artifacts import store_generated_image_artifact
from nanobot.webui.media_maintenance import garbage_collect_generated_media

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+yF9kAAAAASUVORK5CYII="


def backend_store(tmp_path: Path):
    set_config_path(tmp_path / "config.json")
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    media = tmp_path / "media"
    return backend, AttachmentStore(media / "attachments", media_root=media)


def publish(backend, key, messages):
    session = Session(key, messages=messages)
    backend.save(session)
    return backend.source.upgrade_source_session(key, expected_revision=session.revision,
        expected_generation=session.generation, expected_incarnation=session.incarnation)


def orphan(tmp_path: Path, *, generated=False):
    backend, store = backend_store(tmp_path)
    if generated:
        record = store_generated_image_artifact(PNG, prompt="one", model="fixture")
        path = Path(record["path"])
        messages = [{"role": "user", "content": json.dumps({"artifacts": [record]})}]
    else:
        record = store.commit(store.stage_bytes(b"orphan media", mime="text/plain", name="one.txt", source="telegram"))
        path = store.path_for(record)
        messages = [{"role": "user", "content": "one", "attachments": [record.to_dict()]}]
    snapshot = publish(backend, "telegram:orphan", messages)
    assert store.custody.records("pending") == []
    backend.source.delete_source_session(snapshot.state.session_key, snapshot.state)
    return backend, store, record, path


def tree(path: Path) -> dict[str, tuple[int, str]]:
    return {
        str(item.relative_to(path)): (
            stat.S_IFMT(item.lstat().st_mode) | stat.S_IMODE(item.lstat().st_mode),
            hashlib.sha256(item.read_bytes()).hexdigest() if item.is_file() else "",
        )
        for item in path.rglob("*")
    }


def committed(tmp_path: Path):
    set_config_path(tmp_path / "config.json")
    media = tmp_path / "media"
    store = AttachmentStore(media / "attachments", media_root=media)
    record = store.commit(store.stage_bytes(
        b"publication bytes", mime="text/plain", name="note.txt", source="telegram",
    ))
    path = store.path_for(record)
    old = time.time() - 90 * 86400
    os.utime(path, (old, old))
    return store, record, path


def test_committed_unacknowledged_upload_survives_grace_and_reopen(tmp_path: Path) -> None:
    store, _record, path = committed(tmp_path)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    reopened = AttachmentStore(store.root, media_root=store.media_root)
    assert prune_unreferenced(reopened, sessions, grace_seconds=0) == []
    assert path.is_file(), "a pending publication is not abandoned by time or restart"


@pytest.mark.parametrize("payload", ["{bad\n", "[]\n", '{"role":"user","role":"tool"}\n'])
def test_corrupt_reference_inventory_never_authorizes_prune(tmp_path: Path, payload: str) -> None:
    store, _record, path = committed(tmp_path)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "telegram.jsonl").write_text(payload)
    assert prune_unreferenced(store, sessions, grace_seconds=0) == []
    assert path.is_file()


def test_incomplete_evidence_cannot_reach_quota_branch(tmp_path: Path) -> None:
    root = tmp_path / "media" / "generated"
    root.mkdir(parents=True)
    path = root / "asset.bin"
    path.write_bytes(b"x" * 32)
    path.with_suffix(".json").write_text(json.dumps({"id": "asset", "path": str(path)}))
    result = garbage_collect_generated_media(
        artifact_root=root, referenced=set(), evidence_complete=False,
        orphan_grace_days=999, max_bytes=1,
    )
    assert path.is_file()
    assert result["deleted"] == 0
    assert result["quota_unmet_bytes"] == 31


@pytest.mark.parametrize("dry_run", [False, True])
def test_retired_migration_refuses_without_any_tree_mutation(tmp_path: Path, dry_run: bool) -> None:
    store, _record, path = committed(tmp_path)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "telegram.jsonl").write_text(json.dumps({
        "role": "user", "content": "image", "media": [str(path)],
    }) + "\n")
    before = tree(tmp_path)
    with pytest.raises(RuntimeError, match="retired"):
        migrate_legacy_media(sessions, store, dry_run=dry_run)
    assert tree(tmp_path) == before


def test_supported_complete_source_reclaims_orphan_and_sidecar(tmp_path: Path) -> None:
    backend, store, _record, path = orphan(tmp_path)
    with store.custody.maintenance() as (_gate, evidence):
        assert evidence.complete, evidence.reasons
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == [path]
    assert not path.exists()
    assert not path.with_suffix(".json").exists()


def test_supported_complete_source_satisfies_generated_quota(tmp_path: Path) -> None:
    _backend, store, _record, path = orphan(tmp_path, generated=True)
    result = garbage_collect_generated_media(artifact_root=store.media_root / "generated",
        max_bytes=1, orphan_grace_days=999)
    assert result["evidence_complete"], result["reasons"]
    assert result["deleted"] == 1
    assert result["quota_unmet_bytes"] == 0
    assert not path.exists() and not path.with_suffix(".json").exists()


def test_nullable_folder_target_survives_canonical_publication(tmp_path: Path) -> None:
    from nanobot.workspace_access_recovery import parse_folder_access_need

    backend, store = backend_store(tmp_path)
    need = parse_folder_access_need({
        "version": 1, "reason": "folder_selection_required", "operation": "select",
        "target": {"kind": "unknown", "path": None}, "purpose": "Select the requested folder",
        "allowed_roots": [], "roots_truncated": False,
    })
    content = json.dumps({"folder_access": need})
    snapshot = publish(backend, "telegram:nullable-owner", [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call", "type": "function", "function": {"name": "select", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call", "name": "select", "content": content},
    ])
    assert snapshot.events[-1]["content"] == content
    with store.custody.maintenance() as (_gate, evidence):
        assert evidence.complete, evidence.reasons


@pytest.mark.parametrize("branch", ["attachment", "retention", "orphan", "quota"])
@pytest.mark.parametrize("malformed", ["nonobject", "unknown-role", "unknown-kind", "content", "media-string", "media-null-path", "images", "unknown-field", "partial-page"])
def test_malformed_retained_thread_rows_fence_every_deletion_branch(tmp_path: Path, branch, malformed) -> None:
    backend, store, _record, path = orphan(tmp_path, generated=branch != "attachment")
    display = tmp_path / "webui"
    display.mkdir()
    store.custody.declare("webui", display)
    row = {"id": "retained", "role": "assistant", "content": "retained", "createdAt": 1}
    if malformed == "nonobject":
        message = str(path)
    elif malformed == "partial-page":
        message = row
    else:
        message = row
        row.update({
            "unknown-role": {"role": "future"}, "unknown-kind": {"kind": "future"},
            "content": {"content": [str(path)]}, "media-string": {"media": [str(path)]},
            "media-null-path": {"media": [{"kind": "image", "path": None}]},
            "images": {"images": [str(path)]}, "unknown-field": {"future_media": str(path)},
        }[malformed])
    payload = {"messages": [message]}
    if malformed == "partial-page":
        payload["page"] = {"has_more_before": True, "before_cursor": "older"}
    (display / "retained.json").write_text(json.dumps(payload))
    with store.custody.maintenance() as (_gate, evidence):
        assert not evidence.complete
        assert evidence.reasons
    if branch == "attachment":
        assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
    else:
        result = garbage_collect_generated_media(artifact_root=store.media_root / "generated",
            retention_days=1 if branch == "retention" else 0,
            orphan_grace_days=0 if branch == "orphan" else 999,
            max_bytes=1 if branch == "quota" else 0)
        assert result["evidence_complete"] is False and result["deleted"] == 0
        if branch == "quota":
            assert result["quota_unmet_bytes"] > 0 or result["quota_unknown"]
    assert path.is_file() and path.with_suffix(".json").is_file()


@pytest.mark.parametrize("generated", [False, True])
def test_real_retained_display_messages_protect_then_release_media(tmp_path: Path, generated) -> None:
    from nanobot.webui.media_api import sign_existing_media_path, signed_media_attachments
    from nanobot.webui.transcript import replay_transcript_to_ui_messages

    backend, store, _record, path = orphan(tmp_path, generated=generated)
    display = tmp_path / "webui"
    display.mkdir()
    store.custody.declare("webui", display)
    (display / "..native-process-admissions.json.lock").write_bytes(b"")
    messages = replay_transcript_to_ui_messages([
        {"event": "message", "text": "retained media", "media": [str(path)]},
    ], augment_assistant_media=lambda paths: signed_media_attachments(paths,
        sign_path=lambda selected: sign_existing_media_path(selected, secret=b"historical",
            media_dir=lambda _channel: store.media_root)))
    assert messages and messages[0]["media"]
    retained = display / "retained.json"
    retained.write_text(json.dumps({"schemaVersion": 3, "messages": messages}))
    pending = store.custody.records("pending")
    with store.custody.maintenance() as (_gate, evidence):
        assert evidence.complete, evidence.reasons
        assert path in evidence.paths
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
    retained.unlink()
    if generated:
        result = garbage_collect_generated_media(artifact_root=store.media_root / "generated", max_bytes=1)
        assert result["evidence_complete"] and result["deleted"] == 1
        assert result["quota_unmet_bytes"] == 0
    else:
        assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == [path]
    assert store.custody.records("pending") == pending == []
    assert not path.exists() and not path.with_suffix(".json").exists()


def test_authorized_workspace_move_rebinds_exact_media_and_memory_roots(tmp_path: Path) -> None:
    from nanobot.agent.memory import MemoryStore

    backend, store, _record, path = orphan(tmp_path)
    memory = MemoryStore(backend.workspace, sessions_root=backend.sessions_root)
    memory.append_history(f"Retained [file: {path}]")
    source = publish(backend, "telegram:moved-source", [{"role": "user", "content": "survives move"}])
    namespace = backend.sessions_dir
    moved = tmp_path / "workspace-moved"
    backend.workspace.rename(moved)
    reopened = JsonlSessionStore(moved, sessions_root=backend.sessions_root)
    assert reopened.sessions_dir == namespace
    restored = reopened.source.load_source_snapshot(source.state.session_key)
    assert restored.events == source.events
    assert restored.state.incarnation == source.state.incarnation
    registry, _raw = store.custody.declarations()
    declarations = registry["roots"]
    assert any(row["kind"] == "memory" and row["path"] == str(moved / "memory") for row in declarations)
    with store.custody.maintenance() as (_gate, evidence):
        assert evidence.complete, evidence.reasons
        assert path in evidence.paths
    assert prune_unreferenced(store, reopened.sessions_root, grace_seconds=0) == []


@pytest.mark.parametrize("replacement", ["workspace", "namespace", "memory", "workspace-symlink", "memory-symlink"])
def test_workspace_relocation_never_refreshes_replaced_root_identity(tmp_path: Path, replacement) -> None:
    from nanobot.agent.memory import MemoryStore
    from nanobot.attachments._media import MediaRetentionError

    backend, store, _record, path = orphan(tmp_path)
    memory = MemoryStore(backend.workspace, sessions_root=backend.sessions_root)
    memory.append_history("ordinary history")
    moved = tmp_path / "workspace-moved"
    backend.workspace.rename(moved)
    if replacement == "workspace":
        moved.rename(tmp_path / "saved-workspace")
        moved.mkdir()
        (moved / ".nanobot").mkdir()
        (moved / ".nanobot" / "workspace-id").write_text(backend.sessions_dir.name + "\n")
    elif replacement == "workspace-symlink":
        moved.rename(tmp_path / "saved-workspace")
        moved.symlink_to(tmp_path / "saved-workspace", target_is_directory=True)
    elif replacement == "namespace":
        backend.sessions_dir.rename(backend.sessions_root / "saved-namespace")
        backend.sessions_dir.mkdir()
        (backend.sessions_dir / ".workspace").write_text(str(moved) + "\n")
    else:
        (moved / "memory").rename(moved / "saved-memory")
        if replacement == "memory-symlink":
            (moved / "memory").symlink_to(moved / "saved-memory", target_is_directory=True)
        else:
            (moved / "memory").mkdir()
    registry_before = store.custody.registry.read_bytes()
    if replacement in {"workspace-symlink", "memory-symlink"}:
        with pytest.raises(OSError):
            store.custody.declare("sessions", backend.sessions_dir,
                sessions_root=backend.sessions_root, workspace=moved)
        assert store.custody.registry.read_bytes() == registry_before
    else:
        with pytest.raises(MediaRetentionError):
            JsonlSessionStore(moved, sessions_root=backend.sessions_root)
        assert store.custody.registry.read_bytes() == registry_before
    assert path.is_file() and path.with_suffix(".json").is_file()


@pytest.mark.parametrize("binding", ["unclaimed", "wrong-workspace-id", "old-path-replaced"])
def test_workspace_relocation_requires_current_owner_binding(tmp_path: Path, binding) -> None:
    from nanobot.attachments._media import MediaRetentionError

    backend, store, _record, path = orphan(tmp_path)
    moved = tmp_path / "workspace-moved"
    backend.workspace.rename(moved)
    if binding != "unclaimed":
        (backend.sessions_dir / ".workspace").write_text(str(moved) + "\n")
    if binding == "wrong-workspace-id":
        (moved / ".nanobot" / "workspace-id").write_text("0" * 32 + "\n")
    elif binding == "old-path-replaced":
        backend.workspace.mkdir()
    before = store.custody.registry.read_bytes()
    with pytest.raises(MediaRetentionError):
        store.custody.declare("sessions", backend.sessions_dir,
            sessions_root=backend.sessions_root, workspace=moved)
    assert store.custody.registry.read_bytes() == before
    assert path.is_file() and path.with_suffix(".json").is_file()


def test_identical_hashes_have_independent_exact_publication_transfers(tmp_path: Path) -> None:
    backend, store = backend_store(tmp_path)
    first = store.commit(store.stage_bytes(b"same", mime="text/plain", name="a.txt", source="telegram"))
    second = store.commit(store.stage_bytes(b"same", mime="text/plain", name="b.txt", source="telegram"))
    assert first.sha256 == second.sha256 and first.id != second.id
    assert len(store.custody.records("pending")) == 2
    snapshot = publish(backend, "telegram:first", [{"role": "user", "content": "first", "attachments": [first.to_dict()]}])
    pending = store.custody.records("pending")
    assert [row["publication_id"] for row in pending] == [second.id]
    backend.source.delete_source_session(snapshot.state.session_key, snapshot.state)
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
    snapshot = publish(backend, "child:second", [{"role": "user", "content": "second", "attachments": [second.to_dict()]}])
    assert store.custody.records("pending") == []
    backend.source.delete_source_session(snapshot.state.session_key, snapshot.state)
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == [store.path_for(second)]


@pytest.mark.parametrize("key", ["telegram:private", "child:canonical-history"])
def test_non_webui_canonical_source_protects_generated_media(tmp_path: Path, key: str) -> None:
    backend, store = backend_store(tmp_path)
    artifact = store_generated_image_artifact(PNG, prompt="raw tool JSON", model="fixture")
    publish(backend, key, [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call", "type": "function", "function": {"name": "image", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call", "name": "image", "content": json.dumps({"artifacts": [artifact]})},
    ])
    assert store.custody.records("pending") == []
    result = garbage_collect_generated_media(artifact_root=store.media_root / "generated", orphan_grace_days=0, max_bytes=1)
    assert result["evidence_complete"], result["reasons"]
    assert result["deleted"] == 0
    assert Path(artifact["path"]).is_file()
    assert result["quota_unmet_bytes"] > 0


def test_memory_history_breadcrumb_is_retained_evidence(tmp_path: Path) -> None:
    from nanobot.agent.memory import MemoryStore

    backend, store, _record, path = orphan(tmp_path)
    memory = MemoryStore(backend.workspace, sessions_root=backend.sessions_root)
    memory.append_history(f"Remember this [image: {path}]")
    with store.custody.maintenance() as (_gate, evidence):
        assert evidence.complete, evidence.reasons
        assert path in evidence.paths
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_failed_source_publication_retains_exact_pin(tmp_path: Path, monkeypatch, failure) -> None:
    backend, store = backend_store(tmp_path)
    record = store.commit(store.stage_bytes(b"failed publication", mime="text/plain", name="a.txt", source="telegram"))
    original = backend._write_text_atomic
    session = Session("telegram:failed", messages=[{"role": "user", "content": "x", "attachments": [record.to_dict()]}])
    backend.save(session)
    monkeypatch.setattr(backend, "_write_text_atomic", lambda *_args, **_kwargs: (_ for _ in ()).throw(failure("uncertain")))
    with pytest.raises(failure):
        backend.source.upgrade_source_session(session.key, expected_revision=session.revision,
            expected_generation=session.generation, expected_incarnation=session.incarnation)
    monkeypatch.setattr(backend, "_write_text_atomic", original)
    backend.delete(session.key)
    reopened = AttachmentStore(store.root, media_root=store.media_root)
    assert [row["publication_id"] for row in reopened.custody.records("pending")] == [record.id]
    assert prune_unreferenced(reopened, backend.sessions_root, grace_seconds=0) == []
    assert store.path_for(record).is_file()


@pytest.mark.parametrize("branch", ["attachment", "retention", "orphan", "quota"])
@pytest.mark.parametrize("corruption", ["malformed", "duplicate", "nonobject", "version", "missing", "traversal", "unreadable", "budget", "media-null", "native-lock-nonempty", "native-lock-symlink", "native-lock-replaced", "unknown-control"])
def test_qualified_evidence_failure_protects_each_destructive_branch(tmp_path: Path, monkeypatch, branch, corruption) -> None:
    import nanobot.attachments._inventory as inventory

    backend, store, _record, path = orphan(tmp_path, generated=branch != "attachment")
    snapshot = publish(backend, "telegram:evidence", [{"role": "user", "content": "ordinary"}])
    evidence_path = backend.get_session_path(snapshot.state.session_key)
    if corruption == "malformed":
        evidence_path.write_bytes(b"{broken\n")
    elif corruption == "duplicate":
        evidence_path.write_bytes(evidence_path.read_bytes().replace(b'"key":', b'"key":"duplicate","key":', 1))
    elif corruption == "nonobject":
        evidence_path.write_bytes(b"[]\n")
    elif corruption == "media-null":
        records = [json.loads(line) for line in evidence_path.read_text().splitlines()]
        records[-1]["attachments"] = [{"path": None}]
        evidence_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    elif corruption == "version":
        evidence_path.write_bytes(evidence_path.read_bytes().replace(b'"source_schema_version": 1', b'"source_schema_version": 99'))
    elif corruption == "missing":
        backend.sessions_dir.rename(backend.sessions_dir.with_name("missing-owner"))
    elif corruption == "traversal":
        evidence_path.unlink()
        evidence_path.symlink_to(tmp_path / "not-a-source")
    elif corruption == "unreadable":
        original = inventory.read_bytes
        def unavailable(selected, **kwargs):
            if selected == evidence_path:
                raise OSError("source unavailable")
            return original(selected, **kwargs)
        monkeypatch.setattr(inventory, "read_bytes", unavailable)
    elif corruption == "budget":
        monkeypatch.setattr(inventory, "MAX_FILES", 0)
    else:
        display = tmp_path / "webui"
        display.mkdir()
        store.custody.declare("webui", display)
        lock = display / "..native-process-admissions.json.lock"
        if corruption == "native-lock-nonempty":
            lock.write_bytes(b"unqualified lock contents")
        elif corruption == "native-lock-symlink":
            lock.symlink_to(evidence_path)
        elif corruption == "native-lock-replaced":
            lock.write_bytes(b"")
            original_stat = os.stat
            observed = 0
            def replace_during_read(selected, *args, **kwargs):
                nonlocal observed
                if selected == lock.name and kwargs.get("dir_fd") is not None:
                    observed += 1
                    if observed == 2:
                        replacement = tmp_path / "replacement-control-lock"
                        replacement.write_bytes(b"")
                        os.replace(replacement, lock)
                return original_stat(selected, *args, **kwargs)
            monkeypatch.setattr(os, "stat", replace_during_read)
        else:
            (display / "future-control.json").write_text('{}')
    if branch == "attachment":
        assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
    else:
        result = garbage_collect_generated_media(artifact_root=store.media_root / "generated",
            retention_days=1 if branch == "retention" else 0,
            orphan_grace_days=0 if branch == "orphan" else 999,
            max_bytes=1 if branch == "quota" else 0)
        assert result["evidence_complete"] is False
        assert result["deleted"] == 0
        assert result["reasons"]
        if branch == "quota":
            assert result["quota_unmet_bytes"] > 0 or result["quota_unknown"]
    assert path.is_file() and path.with_suffix(".json").is_file()


def test_live_fence_is_exact_process_scope_and_not_copyable(tmp_path: Path) -> None:
    from nanobot.attachments._media import MediaFence

    _backend, store, _record, path = orphan(tmp_path)
    with store.custody.maintenance() as (gate, evidence):
        for operation in (copy.copy, copy.deepcopy, pickle.dumps):
            with pytest.raises(TypeError):
                operation(gate)
        other = AttachmentStore(store.root, media_root=store.media_root)
        with pytest.raises(OSError):
            other.custody.delete(gate, evidence, path)
        fabricated = type(evidence)(evidence.hashes, evidence.paths, True, (), evidence.roots, evidence.declaration_generation)
        with pytest.raises(OSError):
            store.custody.delete(gate, fabricated, path)
        forged = MediaFence(store.custody, gate.lock)
        forged.evidence = evidence
        with pytest.raises(OSError):
            store.custody.delete(forged, evidence, path)
    with store.custody.fence() as publication_gate:
        publication_gate.evidence = evidence
        with pytest.raises(OSError):
            store.custody.delete(publication_gate, evidence, path)
    with pytest.raises(OSError):
        store.custody.delete(gate, evidence, path)
    assert path.is_file()


def test_separate_process_media_fence_refuses_maintenance_boundedly(tmp_path: Path) -> None:
    backend, store, _record, path = orphan(tmp_path)
    code = """import sys\nfrom pathlib import Path\nfrom nanobot.attachments.store import AttachmentStore\ns = AttachmentStore(Path(sys.argv[1]) / 'attachments', media_root=Path(sys.argv[1]))\nwith s.custody.fence():\n print('locked', flush=True)\n sys.stdin.readline()\n"""
    child = subprocess.Popen([sys.executable, "-B", "-c", code, str(store.media_root)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
        start = time.monotonic()
        assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
        assert time.monotonic() - start < 2
        assert path.is_file()
    finally:
        child.communicate("release\n", timeout=10)
    assert child.returncode == 0
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == [path]


@pytest.mark.parametrize("fault", ["blob", "sidecar"])
def test_partial_unlink_reopens_with_fresh_evidence_and_consumes_own_intent(tmp_path: Path, monkeypatch, fault) -> None:
    import nanobot.attachments._media as media

    backend, store, _record, path = orphan(tmp_path)
    original = media.os.unlink
    target = path.name if fault == "blob" else path.with_suffix(".json").name
    def fail_unlink(name, *args, **kwargs):
        if str(name) == target:
            raise OSError("unlink busy")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(media.os, "unlink", fail_unlink)
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
    assert path.with_suffix(".json").is_file()
    assert path.exists() is (fault == "blob")
    assert len(store.custody.records("deletions")) == 1
    monkeypatch.setattr(media.os, "unlink", original)
    reopened = AttachmentStore(store.root, media_root=store.media_root)
    assert prune_unreferenced(reopened, backend.sessions_root, grace_seconds=0) == [path]
    assert reopened.custody.records("deletions") == []
    assert not path.exists() and not path.with_suffix(".json").exists()


def test_preview_of_absent_roots_creates_no_files_or_locks(tmp_path: Path) -> None:
    set_config_path(tmp_path / "config.json")
    media = tmp_path / "media"
    before = tree(tmp_path)
    store = AttachmentStore(media / "attachments", media_root=media)
    assert prune_unreferenced(store, tmp_path / "sessions", dry_run=True) == []
    result = garbage_collect_generated_media(artifact_root=media / "generated", dry_run=True)
    assert result["evidence_complete"] is False
    assert tree(tmp_path) == before


async def test_header_only_nested_followup_publication_transfers_exact_pin(tmp_path: Path) -> None:
    from uuid import uuid4

    from nanobot.session.execution_owner import acquire_execution
    from nanobot.session.pending_followups import PendingFollowupRecord

    backend, store = backend_store(tmp_path)
    snapshot = publish(backend, "websocket:pending", [{"role": "user", "content": "start"}])
    record = store.commit(store.stage_bytes(b"nested followup", mime="text/plain", name="pending.txt", source="websocket"))
    followup = PendingFollowupRecord(uuid4().hex, "webui", "pending", "later", (),
        {"nested": {"attachments": [record.to_dict()]}})
    async with acquire_execution(backend, snapshot.state.session_key, run_id="followup") as owner:
        assert backend.source.admit_pending_followup(snapshot.state.session_key, owner, followup)
    assert store.custody.records("pending") == []
    with store.custody.maintenance() as (_gate, evidence):
        assert evidence.complete, evidence.reasons
        assert record.sha256 in evidence.hashes
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []


def test_segmented_display_and_unlisted_rotation_residue_protect_media(tmp_path: Path, monkeypatch) -> None:
    import nanobot.webui.transcript as transcript

    backend, store, record, path = orphan(tmp_path)
    monkeypatch.setattr(transcript, "_ACTIVE_TRANSCRIPT_ROTATE_BYTES", 1)
    monkeypatch.setattr(transcript, "_TARGET_ACTIVE_TRANSCRIPT_BYTES", 1)
    transcript.append_transcript_object("websocket:segments", {"event": "user", "chat_id": "segments", "text": "one", "attachments": [record.to_dict()]})
    transcript.append_transcript_object("websocket:segments", {"event": "turn_end", "chat_id": "segments"})
    transcript.append_transcript_object("websocket:segments", {"event": "user", "chat_id": "segments", "text": "two"})
    transcript.append_transcript_object("websocket:segments", {"event": "turn_end", "chat_id": "segments"})
    segments = transcript.webui_transcript_segments_dir("websocket:segments")
    assert list(segments.glob("*.jsonl"))
    with store.custody.maintenance() as (_gate, evidence):
        assert evidence.complete, evidence.reasons
        assert path in evidence.paths
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
    # Physical crash residue is inventoried even when absent from the manifest.
    (segments / "999999.jsonl").write_text(json.dumps({"event": "message", "text": f"[file: {path}]"}) + "\n")
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []


def test_pending_upload_expiry_and_registry_restart_never_release_storage_pin(tmp_path: Path, monkeypatch) -> None:
    import nanobot.attachments.uploads as uploads
    from nanobot.attachments.uploads import UploadRegistry

    backend, store = backend_store(tmp_path)
    record = store.commit(store.stage_bytes(b"upload", mime="text/plain", name="up.txt", source="websocket"))
    registry = UploadRegistry(ttl_seconds=1)
    registry.add(record, installation_id="installation", session_key="websocket:upload", source="websocket")
    monkeypatch.setattr(uploads.time, "time", lambda: record.created_at + 100 * 86400)
    registry.sweep()
    assert registry.get(record.id) is None
    assert UploadRegistry().get(record.id) is None
    reopened = AttachmentStore(store.root, media_root=store.media_root)
    assert len(reopened.custody.records("pending")) == 1
    assert prune_unreferenced(reopened, backend.sessions_root, grace_seconds=0) == []


async def test_retained_legacy_rollback_after_reset_protects_media(tmp_path: Path) -> None:
    from nanobot.session.execution_owner import acquire_execution
    from nanobot.session.runtime_authority import OfflineSessionOperator

    backend, store = backend_store(tmp_path)
    record = store.commit(store.stage_bytes(b"rollback", mime="text/plain", name="old.txt", source="telegram"))
    legacy = Session("telegram:retained", messages=[{"role": "user", "content": "old", "attachments": [record.to_dict()]}])
    backend.save(legacy)
    lease = await OfflineSessionOperator(backend.workspace, copied_sessions_root=backend.sessions_root).acquire()
    try:
        manager = lease.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        manager.convert_legacy_session(legacy.key, offline_copy=True)
        async with acquire_execution(manager._jsonl_store, legacy.key, run_id="reset") as owner:
            await owner.quiesce()
            manager._mode_resolver.reset(legacy.key, [], owner=owner)
        assert store.custody.records("pending") == []
        with store.custody.maintenance() as (_gate, evidence):
            assert evidence.complete, evidence.reasons
            assert record.sha256 in evidence.hashes
        assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
    finally:
        await lease.close()


def test_long_raw_tool_json_keeps_references_beyond_retrieval_preview(tmp_path: Path) -> None:
    backend, store = backend_store(tmp_path)
    artifact = store_generated_image_artifact(PNG, prompt="large", model="fixture")
    payload = json.dumps({"padding": "x" * 20_000, "artifacts": [artifact]})
    publish(backend, "telegram:full-tool-result", [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "large", "type": "function", "function": {"name": "image", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "large", "name": "image", "content": payload},
    ])
    assert store.custody.records("pending") == []
    result = garbage_collect_generated_media(artifact_root=store.media_root / "generated", max_bytes=1, orphan_grace_days=0)
    assert result["evidence_complete"], result["reasons"]
    assert result["deleted"] == 0 and Path(artifact["path"]).is_file()


def test_publication_before_delete_is_excluded_by_real_namespace_lock(tmp_path: Path, monkeypatch) -> None:
    backend, store = backend_store(tmp_path)
    record = store.commit(store.stage_bytes(b"publication first", mime="text/plain", name="a.txt", source="telegram"))
    original = backend._write_text_atomic
    entered, release = threading.Event(), threading.Event()
    def blocked_write(path, text, **kwargs):
        if '"source_schema_version": 1' in text:
            entered.set()
            assert release.wait(10)
        return original(path, text, **kwargs)
    monkeypatch.setattr(backend, "_write_text_atomic", blocked_write)
    with ThreadPoolExecutor(max_workers=1) as executor:
        writing = executor.submit(publish, backend, "telegram:race", [{"role": "user", "content": "one", "attachments": [record.to_dict()]}])
        assert entered.wait(10)
        assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
        release.set()
        writing.result(timeout=10)
    assert store.custody.records("pending") == []
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
    assert store.path_for(record).is_file()


def test_delete_before_publication_refuses_missing_asset_after_real_lock_barrier(tmp_path: Path, monkeypatch) -> None:
    from nanobot.attachments._media import MediaCustody

    backend, store, record, path = orphan(tmp_path)
    snapshot = publish(backend, "telegram:later", [{"role": "user", "content": "base"}])
    entered, release, attempting = threading.Event(), threading.Event(), threading.Event()
    original = MediaCustody.delete
    def blocked_delete(self, gate, evidence, selected):
        entered.set()
        assert release.wait(10)
        return original(self, gate, evidence, selected)
    monkeypatch.setattr(MediaCustody, "delete", blocked_delete)
    event = new_source_event({"role": "user", "content": "too late", "attachments": [record.to_dict()]}, run_id="late")
    def publish_after_delete():
        attempting.set()
        return backend.source.append_source_events(snapshot.state.session_key, snapshot.state, [event])
    with ThreadPoolExecutor(max_workers=2) as executor:
        deleting = executor.submit(prune_unreferenced, store, backend.sessions_root, grace_seconds=0)
        assert entered.wait(10)
        writing = executor.submit(publish_after_delete)
        assert attempting.wait(10)
        assert not writing.done()
        release.set()
        assert deleting.result(timeout=10) == [path]
        with pytest.raises(OSError):
            writing.result(timeout=10)
    assert not path.exists()
    assert len(backend.source.load_source_snapshot(snapshot.state.session_key).events) == 1


@pytest.mark.parametrize("replacement", ["namespace", "asset", "sidecar", "lock", "root"])
def test_replaced_inventory_generation_cannot_be_deleted(tmp_path: Path, monkeypatch, replacement) -> None:
    from nanobot.attachments._media import MediaCustody

    backend, store, _record, path = orphan(tmp_path)
    original = MediaCustody.delete
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / path.name
    sentinel.write_bytes(b"must survive")
    def replace_then_delete(self, gate, evidence, selected):
        if replacement == "namespace":
            (backend.sessions_root / ("f" * 32)).mkdir()
        elif replacement == "asset":
            selected.rename(selected.with_name("saved-original"))
            selected.symlink_to(sentinel)
        elif replacement == "sidecar":
            selected.with_suffix(".json").unlink()
            selected.with_suffix(".json").symlink_to(sentinel)
        elif replacement == "lock":
            lock = store.custody.state / "publication.lock"
            lock.rename(lock.with_name("saved.lock"))
            lock.write_bytes(b"")
        else:
            store.media_root.rename(tmp_path / "saved-media")
            store.media_root.symlink_to(outside, target_is_directory=True)
        return original(self, gate, evidence, selected)
    monkeypatch.setattr(MediaCustody, "delete", replace_then_delete)
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
    assert sentinel.read_bytes() == b"must survive"
    if replacement in {"namespace", "sidecar", "lock"}:
        assert path.is_file()


async def test_temporary_cleanup_preserves_shared_then_reclaims_exact_exclusive_custody(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from nanobot.bus.queue import MessageBus
    from nanobot.security.folder_grants import FolderGrantStore
    from nanobot.session.manager import SessionManager
    from nanobot.webui.temporary_chats import WebUITemporaryChats
    from nanobot.webui.workspaces import WebUIWorkspaceController

    backend, store = backend_store(tmp_path)
    manager = SessionManager(backend.workspace, sessions_root=backend.sessions_root)
    controller = WebUIWorkspaceController(session_manager=manager, default_workspace=backend.workspace,
        default_restrict_to_workspace=True, grant_store=FolderGrantStore(agent_home=backend.workspace,
            instance_root=tmp_path, path=tmp_path / "grants.json"), native_supervisor_available=True)
    chats = WebUITemporaryChats(bus=MessageBus(), session_manager=manager, workspaces=controller,
        logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None))
    owner = object()
    chat_id = chats.create(owner, trusted_webui=True)
    permanent = store.commit(store.stage_bytes(b"shared temporary", mime="text/plain", name="shared.txt", source="telegram"))
    snapshot = publish(backend, "telegram:shared", [{"role": "user", "content": "permanent", "attachments": [permanent.to_dict()]}])
    temporary = store.commit(store.stage_bytes(b"shared temporary", mime="text/plain", name="shared.txt", source="websocket"))
    path = store.path_for(temporary)
    chats.register_media(owner, chat_id, [str(path)], records=(temporary,))
    await chats.discard(owner, chat_id)
    assert path.is_file() and chats.pending_cleanup(chat_id) == (str(path),)
    backend.source.delete_source_session(snapshot.state.session_key, snapshot.state)
    assert chats.retry_cleanup(chat_id) == ()
    assert not path.exists() and not path.with_suffix(".json").exists()
    assert store.custody.records("pending") == []
    assert store.custody.records("deletions") == []


def test_normal_delegated_child_activity_allows_positive_reclamation(tmp_path: Path) -> None:
    from nanobot.webui.transcript import append_transcript_object

    backend, store, _record, path = orphan(tmp_path)
    append_transcript_object("websocket:parent", {"event": "delegated_child", "chat_id": "parent",
        "child_id": "child-1", "child_session_key": "child:task-1", "profile": "worker",
        "state": "complete", "runtime_generation": 1})
    with store.custody.maintenance() as (_gate, evidence):
        assert evidence.complete, evidence.reasons
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == [path]


@pytest.mark.parametrize("row", [{"event": "delegated_child", "child_id": "missing"},
    {"event": "unknown_future_activity", "text": "unsupported"}])
def test_unsupported_display_activity_refuses_reclamation(tmp_path: Path, row) -> None:
    from nanobot.attachments.lifecycle import prune_unreferenced_result
    from nanobot.webui.transcript import append_transcript_object

    backend, store, _record, path = orphan(tmp_path)
    append_transcript_object("websocket:future", row)
    result = prune_unreferenced_result(store, backend.sessions_root, grace_seconds=0)
    assert not result.evidence_complete and result.reasons
    assert path.exists() and path.with_suffix(".json").exists()


def test_custom_backend_cannot_reuse_prior_complete_declarations(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from nanobot.session.manager import SessionManager, SessionStore

    backend, store, _record, path = orphan(tmp_path)
    with store.custody.maintenance() as (_gate, evidence):
        assert evidence.complete, evidence.reasons
    SessionManager(backend.workspace, sessions_root=backend.sessions_root, store=MagicMock(spec=SessionStore))
    with store.custody.maintenance() as (_gate, evidence):
        assert not evidence.complete
        assert any("custom session backend" in reason for reason in evidence.reasons)
    reopened = AttachmentStore(store.root, media_root=store.media_root)
    assert prune_unreferenced(reopened, backend.sessions_root, grace_seconds=0) == []
    assert path.exists()


def test_evidence_budget_is_shared_by_display_and_memory(tmp_path: Path, monkeypatch) -> None:
    import nanobot.attachments._inventory as inventory
    from nanobot.agent.memory import MemoryStore
    from nanobot.webui.transcript import append_transcript_object

    backend, store, _record, path = orphan(tmp_path)
    append_transcript_object("websocket:budget", {"event": "message", "text": "x" * 12_000})
    MemoryStore(backend.workspace, sessions_root=backend.sessions_root).append_history("y" * 12_000)
    monkeypatch.setattr(inventory, "MAX_BYTES", 20_000)
    with store.custody.maintenance() as (_gate, evidence):
        assert not evidence.complete
        assert any("byte budget" in reason for reason in evidence.reasons)
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
    assert path.exists()


def test_unknown_custody_state_refuses_without_mutation(tmp_path: Path) -> None:
    backend, store, _record, path = orphan(tmp_path)
    (store.custody.state / "future.json").write_text('{"version":2}')
    before = tree(tmp_path)
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
    assert path.exists() and tree(tmp_path) == before


@pytest.mark.parametrize("changed", ["replacement", "new_reference"])
def test_reopened_deletion_intent_protects_new_generation_or_reference(tmp_path: Path, monkeypatch, changed) -> None:
    import nanobot.attachments._media as media

    backend, store, record, path = orphan(tmp_path)
    original = media.os.unlink
    def busy(name, *args, **kwargs):
        if str(name) == path.name:
            raise OSError("busy blob")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(media.os, "unlink", busy)
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
    assert len(store.custody.records("deletions")) == 1
    monkeypatch.setattr(media.os, "unlink", original)
    if changed == "replacement":
        replacement = path.with_name("replacement")
        replacement.write_bytes(b"replacement generation")
        replacement.replace(path)
    else:
        publish(backend, "telegram:new-reference", [{"role": "user", "content": "now retained", "attachments": [record.to_dict()]}])
    reopened = AttachmentStore(store.root, media_root=store.media_root)
    assert prune_unreferenced(reopened, backend.sessions_root, grace_seconds=0) == []
    assert path.exists() and path.with_suffix(".json").exists()
    if changed == "replacement":
        assert reopened.custody.records("deletions") == []
        assert path.read_bytes() == b"replacement generation"


def test_namespace_allocation_waits_for_maintenance_migration_fence(tmp_path: Path, monkeypatch) -> None:
    from nanobot.attachments._media import MediaCustody

    backend, store, _record, path = orphan(tmp_path)
    entered, release, allocating = threading.Event(), threading.Event(), threading.Event()
    original = MediaCustody.delete
    def pause(self, gate, evidence, selected):
        entered.set()
        assert release.wait(10)
        return original(self, gate, evidence, selected)
    def allocate():
        allocating.set()
        return JsonlSessionStore(tmp_path / "second-workspace", sessions_root=backend.sessions_root)
    monkeypatch.setattr(MediaCustody, "delete", pause)
    with ThreadPoolExecutor(max_workers=2) as executor:
        deleting = executor.submit(prune_unreferenced, store, backend.sessions_root, grace_seconds=0)
        assert entered.wait(10)
        creating = executor.submit(allocate)
        assert allocating.wait(10)
        assert not creating.done()
        release.set()
        assert deleting.result(timeout=10) == [path]
        second = creating.result(timeout=10)
    assert second.sessions_dir != backend.sessions_dir
    with store.custody.maintenance() as (_gate, evidence):
        assert evidence.complete, evidence.reasons


def test_cli_preview_decline_and_retired_command_leave_whole_tree_unchanged(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from nanobot.cli.commands import app

    backend, store, _record, path = orphan(tmp_path)
    config = tmp_path / "config.json"
    config.write_text('{}')
    runner = CliRunner()
    before = tree(tmp_path)
    args = ["attachments", "prune", "--config", str(config), "--grace-days", "0"]
    preview = runner.invoke(app, [*args, "--dry-run"])
    assert preview.exit_code == 0, preview.output
    assert "Would remove 1" in preview.output
    assert tree(tmp_path) == before
    declined = runner.invoke(app, args, input="n\n")
    assert declined.exit_code == 1 and "Cancelled" in declined.output
    assert tree(tmp_path) == before and path.exists()
    refused = runner.invoke(app, ["attachments", "migrate", "--config", str(config)])
    assert refused.exit_code != 0 and "No such command" in refused.output
    assert tree(tmp_path) == before
    assert backend.sessions_root.exists()


def test_historical_media_and_markdown_signing_remain_pure_and_reclaimable(tmp_path: Path) -> None:
    from nanobot.webui.media_gateway import WebUIMediaGateway

    backend, store, _record, path = orphan(tmp_path)
    _backend, _store, _artifact, image = orphan(tmp_path, generated=True)
    gateway = WebUIMediaGateway(workspace_path=backend.workspace, logger=None,
        media_dir=lambda _channel: store.media_root, secret=b"fixture-read")
    before = tree(tmp_path)
    for _ in range(3):
        assert gateway.sign_media_read(path)
        assert gateway.augment_transcript_media([str(path)])
        rewritten = gateway.rewrite_local_markdown_images(f"![old]({image})", workspace_path=store.media_root,
            publication=False)
        assert "/api/media/" in rewritten and "publication=" not in rewritten
    assert store.custody.records("pending") == []
    assert tree(tmp_path) == before
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == [path]
    result = garbage_collect_generated_media(artifact_root=store.media_root / "generated", orphan_grace_days=0)
    assert result["evidence_complete"] and result["deleted"] == 1


def test_initial_media_root_symlink_cannot_authorize_reclamation(tmp_path: Path) -> None:
    backend, store, _record, path = orphan(tmp_path)
    actual = tmp_path / "moved-media"
    store.media_root.rename(actual)
    store.media_root.symlink_to(actual, target_is_directory=True)
    reopened = AttachmentStore(store.media_root / "attachments", media_root=store.media_root)
    before = tree(actual)
    assert prune_unreferenced(reopened, backend.sessions_root, grace_seconds=0) == []
    assert tree(actual) == before and path.exists()


async def test_private_artifact_manifest_validates_full_canonical_media_result(tmp_path: Path) -> None:
    from nanobot.session.artifact import parse_artifact_descriptor
    from nanobot.session.artifact_store import PrivateArtifactStore
    from nanobot.session.execution_owner import acquire_execution
    from nanobot.session.source_store import SourceSessionStore
    from nanobot.session.turn_writer import SourceTurnWriter
    from tests.session.test_artifact import fixture
    from tests.session.test_source_turn_writer import assistant, result

    backend, store = backend_store(tmp_path)
    artifact = store_generated_image_artifact(PNG, prompt="private binding", model="fixture")
    publish(backend, "telegram:private-binding", [])
    content = json.dumps({"padding": "x" * 20_000, "artifacts": [artifact]})
    async with acquire_execution(backend, "telegram:private-binding", run_id="run") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="run")
        writer.begin_tool_batch(assistant())
        await writer.record_tool_result_event(result(content=content))
        await writer.record_tool_result_event(result(1, "second"))
        snapshot = owner.read_snapshot()
        event = snapshot.events[1]
        raw, _ = fixture()
        encoded = content.encode()
        raw.update(workspace_id=backend.sessions_dir.name, session_key=snapshot.state.session_key,
            incarnation=snapshot.state.incarnation, generation=snapshot.state.generation,
            event_id=event["_event_id"], execution_id=event["_execution_id"],
            stream_id=event["_stream_id"], byte_length=len(encoded), sha256=hashlib.sha256(encoded).hexdigest())
        descriptor = parse_artifact_descriptor(raw)
        private = PrivateArtifactStore(tmp_path / "private-artifacts", workspace_id=descriptor.workspace_id,
            session_key=descriptor.session_key, incarnation=descriptor.incarnation, generation=descriptor.generation)
        source = SourceSessionStore(backend, artifact_store=private)
        with private.transaction() as transaction:
            transaction.put(descriptor, encoded)
            source.commit_artifact_descriptor(descriptor.session_key, snapshot.state, descriptor,
                owner=owner, blobs=transaction)
    assert store.custody.records("pending") == []
    with store.custody.maintenance() as (_gate, evidence):
        assert evidence.complete, evidence.reasons
        assert Path(artifact["path"]) in evidence.paths
    source_path = backend.sessions_dir / f"{backend.storage_key(descriptor.session_key)}.jsonl"
    rows = [json.loads(line) for line in source_path.read_bytes().splitlines()]
    rows[0]["metadata"]["artifact_manifest"]["descriptors"][descriptor.artifact_id]["sha256"] = "0" * 64
    source_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = garbage_collect_generated_media(artifact_root=store.media_root / "generated", max_bytes=1, orphan_grace_days=0)
    assert not result["evidence_complete"] and result["reasons"]
    assert result["deleted"] == 0 and Path(artifact["path"]).exists()


async def test_export_retained_files_protect_and_missing_stage_refuses(tmp_path: Path) -> None:
    from nanobot.session.runtime_authority import OfflineSessionOperator
    from nanobot.session.session_export import export_canonical_session

    backend, store = backend_store(tmp_path)
    record = store.commit(store.stage_bytes(b"export media", mime="text/plain", name="export.txt", source="telegram"))
    session = Session("telegram:export-media", messages=[{"role": "user", "content": "export", "attachments": [record.to_dict()]}])
    backend.save(session)
    lease = await OfflineSessionOperator(backend.workspace, copied_sessions_root=backend.sessions_root).acquire()
    try:
        manager = lease.manager
        manager.bootstrap_existing_sessions(offline_copy=True)
        manager.convert_legacy_session(session.key, offline_copy=True)
        export_canonical_session(manager, session.key, tmp_path / "export-bundle", include_compatibility=True)
        with store.custody.maintenance() as (_gate, evidence):
            assert evidence.complete, evidence.reasons
            assert record.sha256 in evidence.hashes
        retained = next((manager.sessions_dir / ".session-mode" / "exports").glob("*/canonical-source.jsonl"))
        retained.unlink()
        with store.custody.maintenance() as (_gate, evidence):
            assert not evidence.complete and evidence.reasons
        assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
        assert store.path_for(record).exists()
    finally:
        await lease.close()


def test_inventory_covers_more_sources_than_library_page(tmp_path: Path, monkeypatch) -> None:
    backend, store = backend_store(tmp_path)
    record = store.commit(store.stage_bytes(b"beyond page", mime="text/plain", name="last.txt", source="telegram"))
    for index in range(205):
        message = {"role": "user", "content": "ordinary"}
        if index == 204:
            message["attachments"] = [record.to_dict()]
        publish(backend, f"telegram:page-{index:04}", [message])
    monkeypatch.setattr(JsonlSessionStore, "list_sessions", lambda *_args: (_ for _ in ()).throw(AssertionError("paged discovery is not evidence")))
    assert store.custody.records("pending") == []
    with store.custody.maintenance() as (_gate, evidence):
        assert evidence.complete, evidence.reasons
        assert record.sha256 in evidence.hashes
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []


async def test_interrupted_event_archive_remains_media_evidence_after_reset(tmp_path: Path) -> None:
    from tests.cron.test_interruption_recovery import _interrupted_gateway

    set_config_path(tmp_path / "config.json")
    gateway, lease, sessions, key, _claim, prefix, _tail, _calls, _clock = await _interrupted_gateway(
        tmp_path, leave_stale_lease=False)
    store = AttachmentStore(tmp_path / "media" / "attachments", media_root=tmp_path / "media")
    record = store.commit(store.stage_bytes(b"interrupted media", mime="text/plain", name="tail.txt", source="cron"))
    try:
        backend = sessions._jsonl_store
        with backend.locked_session_files():
            snapshot = backend.source.load_source_snapshot(key)
            snapshot.events[len(prefix)]["attachments"] = [record.to_dict()]
            snapshot.header["revision"] += 1
            backend.source._write(snapshot.header, snapshot.events)
        assert store.custody.records("pending") == []
        assert await gateway.runtime.acquire_for_gateway()
        archive = next((sessions.sessions_dir / ".session-mode" / "cron-interruptions").glob("*.json"))
        with store.custody.maintenance() as (_gate, evidence):
            assert evidence.complete, evidence.reasons
            assert record.sha256 in evidence.hashes
        assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
        payload = json.loads(archive.read_bytes())
        payload["schema_version"] = 99
        archive.write_text(json.dumps(payload))
        with store.custody.maintenance() as (_gate, evidence):
            assert not evidence.complete
        assert store.path_for(record).exists()
    finally:
        gateway.runtime.scheduler.release_lease()
        await lease.close()


def test_declared_custom_generated_root_retains_source_images_and_reclaims_outputs(tmp_path: Path) -> None:
    backend, store, _record, source_image = orphan(tmp_path)
    artifact = store_generated_image_artifact(PNG, prompt="custom root", model="fixture",
        save_dir="custom/outputs", source_images=[str(source_image)])
    snapshot = publish(backend, "telegram:custom-output", [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "custom", "type": "function", "function": {"name": "image", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "custom", "content": json.dumps({"artifacts": [artifact]})},
    ])
    backend.source.delete_source_session(snapshot.state.session_key, snapshot.state)
    assert store.custody.records("pending") == []
    with store.custody.maintenance() as (_gate, evidence):
        assert evidence.complete, evidence.reasons
        assert source_image in evidence.paths
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
    result = garbage_collect_generated_media(artifact_root=store.media_root / "custom" / "outputs",
        orphan_grace_days=0, max_bytes=1)
    assert result["evidence_complete"], result["reasons"]
    assert result["deleted"] == 1 and not Path(artifact["path"]).exists()
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == [source_image]


def test_unavailable_media_capability_preserves_text_startup_and_refuses_media(tmp_path: Path, monkeypatch) -> None:
    import nanobot.attachments._media as media
    from nanobot.attachments.lifecycle import prune_unreferenced_result

    set_config_path(tmp_path / "config.json")
    monkeypatch.setattr(media, "media_custody_available", lambda: False)
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    snapshot = publish(backend, "cli:text-only", [{"role": "user", "content": "plain text"}])
    assert snapshot.events[0]["content"] == "plain text"
    store = AttachmentStore(tmp_path / "media" / "attachments", media_root=tmp_path / "media")
    assert not store.custody.state.exists()
    staged = store.stage_bytes(b"unavailable managed bytes", mime="text/plain", name="new.txt", source="websocket")
    with pytest.raises(media.MediaRetentionError, match="unavailable"):
        store.commit(staged)
    assert not store.root.exists() and not store.custody.state.exists()
    before = tree(tmp_path)
    result = prune_unreferenced_result(store, backend.sessions_root, grace_seconds=0)
    assert not result.evidence_complete and any("unavailable" in reason for reason in result.reasons)
    generated = garbage_collect_generated_media(artifact_root=store.media_root / "generated", max_bytes=1)
    assert not generated["evidence_complete"] and any("unavailable" in reason for reason in generated["reasons"])
    assert tree(tmp_path) == before


def test_implicit_memory_root_alias_refuses_even_with_empty_target(tmp_path: Path) -> None:
    backend, store, _record, path = orphan(tmp_path)
    outside = tmp_path / "outside-memory"
    outside.mkdir()
    (backend.workspace / "memory").symlink_to(outside, target_is_directory=True)
    with store.custody.maintenance() as (_gate, evidence):
        assert not evidence.complete and evidence.reasons
    assert prune_unreferenced(store, backend.sessions_root, grace_seconds=0) == []
    assert path.exists() and list(outside.iterdir()) == []
