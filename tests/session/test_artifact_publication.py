"""Owned publication binds durable blobs without granting retrieval authority."""

from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from nanobot.session.artifact import parse_artifact_descriptor, published_artifacts
from nanobot.session.artifact_store import ArtifactStorageError, PrivateArtifactStore
from nanobot.session.execution_owner import ExecutionOwnershipError, acquire_execution
from nanobot.session.manager import JsonlSessionStore, Session, SessionWriteConflict
from nanobot.session.source import SourceValidationError, prefix_digest
from nanobot.session.source_store import SourceSessionStore
from nanobot.session.turn_writer import SourceTurnWriter
from tests.session.test_artifact import fixture
from tests.session.test_source_turn_writer import assistant, result


@asynccontextmanager
async def publication(tmp_path):
    backend = JsonlSessionStore(tmp_path / 'workspace')
    session = Session('test:writer')
    backend.save(session)
    backend.source.upgrade_source_session(session.key, expected_revision=session.revision,
        expected_generation=session.generation, expected_incarnation=session.incarnation)
    async with acquire_execution(backend, session.key, run_id='run') as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id='run')
        writer.begin_tool_batch(assistant())
        await writer.record_tool_result_event(result(content='evidence'))
        await writer.record_tool_result_event(result(1, 'second'))
        snapshot = owner.read_snapshot()
        event = snapshot.events[1]
        raw, _ = fixture()
        raw.update(workspace_id=backend.sessions_dir.name, session_key=session.key,
            incarnation=snapshot.state.incarnation, generation=snapshot.state.generation,
            event_id=event['_event_id'], execution_id=event['_execution_id'],
            stream_id=event['_stream_id'])
        descriptor = parse_artifact_descriptor(raw)
        store = PrivateArtifactStore(tmp_path.resolve() / 'artifacts',
            workspace_id=descriptor.workspace_id, session_key=session.key,
            incarnation=descriptor.incarnation, generation=descriptor.generation)
        source = SourceSessionStore(backend, artifact_store=store)
        yield backend, source, store, owner, descriptor


def publish(source, owner, descriptor, transaction, expected=None):
    return source.commit_artifact_descriptor(descriptor.session_key,
        expected or owner.read_snapshot().state, descriptor, owner=owner, blobs=transaction)


async def test_publication_reload_retry_and_stale_cas(tmp_path):
    async with publication(tmp_path) as setup:
        backend, source, store, owner, descriptor = setup
        before = owner.read_snapshot()
        with store.transaction() as transaction:
            transaction.put(descriptor, b'evidence')
            after = publish(source, owner, descriptor, transaction)
            assert after.state.revision == before.state.revision + 1
            assert after.events == before.events and after.checkpoint == before.checkpoint
            assert prefix_digest(after.events, len(after.events)) == prefix_digest(before.events, len(before.events))
            for name in ('updated_at', 'last_archived', 'last_consolidated'):
                assert after.header[name] == before.header[name]
            assert published_artifacts(backend.source.load_source_snapshot(descriptor.session_key),
                workspace_id=store.workspace_id) == (descriptor,)
            assert publish(source, owner, descriptor, transaction) == after
            with pytest.raises(SessionWriteConflict):
                publish(source, owner, descriptor, transaction, before.state)
            assert 'artifact_grants' not in after.header['metadata']


@pytest.mark.parametrize('change', ['id', 'origin'])
async def test_conflicting_immutable_publication(tmp_path, change):
    async with publication(tmp_path) as setup:
        _, source, store, owner, descriptor = setup
        with store.transaction() as transaction:
            transaction.put(descriptor, b'evidence')
            before = publish(source, owner, descriptor, transaction)
            changed = (replace(descriptor, artifact_id='c' * 32) if change == 'id' else
                replace(descriptor, source_access=replace(descriptor.source_access, relative_path='other')))
            transaction.put(changed, b'evidence')
            with pytest.raises(SourceValidationError, match='Conflicting'):
                publish(source, owner, changed, transaction)
            assert owner.read_snapshot() == before


@pytest.mark.parametrize('failure', ['missing', 'corrupt', 'closed', 'wrong_store'])
async def test_blob_failures_do_not_publish(tmp_path, failure):
    async with publication(tmp_path) as setup:
        _, source, store, owner, descriptor = setup
        before = owner.read_snapshot()
        if failure == 'wrong_store':
            store = PrivateArtifactStore(tmp_path.resolve() / 'other', workspace_id=descriptor.workspace_id,
                session_key=descriptor.session_key, incarnation=descriptor.incarnation,
                generation=descriptor.generation)
        with store.transaction() as transaction:
            if failure != 'missing':
                transaction.put(descriptor, b'evidence')
            if failure == 'corrupt':
                next(store.root.rglob('*.blob')).write_bytes(b'changed!')
            if failure != 'closed':
                with pytest.raises((ArtifactStorageError, FileNotFoundError)):
                    publish(source, owner, descriptor, transaction)
        if failure == 'closed':
            with pytest.raises((ArtifactStorageError, FileNotFoundError)):
                publish(source, owner, descriptor, transaction)
        assert owner.read_snapshot() == before


async def test_header_failure_leaves_only_orphan_blob(tmp_path, monkeypatch):
    async with publication(tmp_path) as setup:
        _, source, store, owner, descriptor = setup
        before = owner.read_snapshot()
        with store.transaction() as transaction:
            transaction.put(descriptor, b'evidence')
            def fail(*args):
                raise OSError('injected disk failure')
            monkeypatch.setattr(source, '_write', fail)
            with pytest.raises(OSError):
                publish(source, owner, descriptor, transaction)
            assert transaction.read_range(descriptor) == b'evidence'
        assert owner.read_snapshot() == before


async def test_generic_metadata_cannot_replace_manifest_and_reset_clears_it(tmp_path):
    async with publication(tmp_path) as setup:
        _, source, store, owner, descriptor = setup
        with store.transaction() as transaction:
            transaction.put(descriptor, b'evidence')
            before = publish(source, owner, descriptor, transaction)
            updated = source.update_metadata(descriptor.session_key, {'artifact_manifest': None})
            assert updated.header['metadata']['artifact_manifest'] == before.header['metadata']['artifact_manifest']
            await owner.quiesce()
            with pytest.raises(ExecutionOwnershipError):
                publish(source, owner, descriptor, transaction)
            reset = source.reset_source_session(descriptor.session_key, updated.state, [], owner=owner)
            assert reset.state.generation == updated.state.generation + 1
            assert 'artifact_manifest' not in reset.header['metadata']
            assert transaction.read_range(descriptor) == b'evidence'


@pytest.mark.parametrize('failure', ['manifest', 'namespace', 'owner', 'unconfigured'])
async def test_publication_admission_failures_preserve_source(tmp_path, failure):
    async with publication(tmp_path) as setup:
        backend, source, store, owner, descriptor = setup
        if failure == 'manifest':
            with backend.locked_session_files():
                header, events = source._read_unlocked(descriptor.session_key)
                header['metadata']['artifact_manifest'] = None
                source._write(header, events)
        before = owner.read_snapshot()
        with store.transaction() as transaction:
            transaction.put(descriptor, b'evidence')
            if failure == 'namespace':
                scope = store.root.joinpath(*store._parts)
                scope.rename(scope.with_name(scope.name + '-old'))
                scope.mkdir(mode=0o700)
            if failure == 'unconfigured':
                source = backend.source
            from nanobot.session.artifact import ArtifactValidationError
            with pytest.raises((ArtifactValidationError, ArtifactStorageError,
                                ExecutionOwnershipError, SourceValidationError)):
                source.commit_artifact_descriptor(descriptor.session_key, before.state, descriptor,
                    owner=None if failure == 'owner' else owner, blobs=transaction)
        assert owner.read_snapshot() == before


async def test_deletion_removes_manifest_authority_but_retains_blob(tmp_path):
    async with publication(tmp_path) as setup:
        backend, source, store, owner, descriptor = setup
        with store.transaction() as transaction:
            transaction.put(descriptor, b'evidence')
            committed = publish(source, owner, descriptor, transaction)
            await owner.quiesce()
            source.delete_source_session(descriptor.session_key, committed.state, owner=owner)
            assert not backend.get_session_path(descriptor.session_key).exists()
            assert transaction.read_range(descriptor) == b'evidence'


async def test_stale_legacy_save_cannot_erase_manifest(tmp_path):
    async with publication(tmp_path) as setup:
        backend, source, store, owner, descriptor = setup
        stale = backend.load(descriptor.session_key)
        assert stale is not None
        with store.transaction() as transaction:
            transaction.put(descriptor, b'evidence')
            before = publish(source, owner, descriptor, transaction)
            stale.metadata['artifact_manifest'] = None
            try:
                backend.save(stale)
            except SessionWriteConflict:
                pass
            assert owner.read_snapshot().header['metadata']['artifact_manifest'] == before.header['metadata']['artifact_manifest']


async def test_recovery_required_owner_cannot_publish(tmp_path):
    async with publication(tmp_path) as setup:
        _, source, store, owner, descriptor = setup
        before = owner.read_snapshot()
        owner.recovery_required = True
        with store.transaction() as transaction:
            transaction.put(descriptor, b'evidence')
            with pytest.raises(ExecutionOwnershipError):
                publish(source, owner, descriptor, transaction)
        assert owner.read_snapshot() == before


async def test_namespace_replacement_after_verification_prevents_header_write(tmp_path, monkeypatch):
    async with publication(tmp_path) as setup:
        _, source, store, owner, descriptor = setup
        before = owner.read_snapshot()
        with store.transaction() as transaction:
            transaction.put(descriptor, b'evidence')
            original = transaction.verify_durable
            def replace_after_verify(*args, **kwargs):
                original(*args, **kwargs)
                scope = store.root.joinpath(*store._parts)
                scope.rename(scope.with_name(scope.name + '-old'))
                scope.mkdir(mode=0o700)
            monkeypatch.setattr(transaction, 'verify_durable', replace_after_verify)
            with pytest.raises(ArtifactStorageError):
                publish(source, owner, descriptor, transaction)
        assert owner.read_snapshot() == before
