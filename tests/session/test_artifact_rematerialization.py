"""Canonical artifact recovery uses retained source only and never rewrites it."""

from dataclasses import replace

import pytest

from nanobot.session.artifact_reader import ArtifactReader
from nanobot.session.artifact_rematerialization import (
    ArtifactRematerializationError,
    ArtifactRematerializer,
)
from nanobot.session.artifact_store import PrivateArtifactStore
from tests.session.test_artifact_reader import setup_read


async def test_missing_blob_is_restored_exactly_without_source_mutation(tmp_path):
    async with setup_read(tmp_path, text='retained canonical 中文 🌟') as setup:
        writer, tool, store, descriptor, content = setup
        reader = ArtifactReader(writer, 'root_history', store, tool)
        source_path = writer.store._backend.get_session_path(writer.owner.key)
        before = source_path.read_bytes()
        expected = writer.owner.read_snapshot().state
        next(store.root.rglob('*.blob')).unlink()
        (tool._workspace / 'file').write_text('different current source bytes')

        result = ArtifactRematerializer(reader).rematerialize(
            descriptor.reference, expected=expected
        )

        assert result.status == 'installed'
        assert result.reference == descriptor.reference
        assert result.byte_length == len(content)
        assert source_path.read_bytes() == before
        assert writer.owner.read_snapshot().state == expected
        reopened = PrivateArtifactStore(
            store.root,
            workspace_id=store.workspace_id,
            session_key=store.session_key,
            incarnation=store.incarnation,
            generation=store.generation,
        )
        restarted = ArtifactReader(writer, 'root_history', reopened, tool)
        assert restarted.read(descriptor.reference).text.encode() == content


async def test_repeat_reuses_valid_blob_and_corruption_is_never_repaired(tmp_path):
    async with setup_read(tmp_path) as setup:
        writer, tool, store, descriptor, content = setup
        reader = ArtifactReader(writer, 'root_history', store, tool)
        recovery = ArtifactRematerializer(reader)
        expected = writer.owner.read_snapshot().state
        blob = next(store.root.rglob('*.blob'))
        blob.unlink()
        assert recovery.rematerialize(descriptor.reference, expected=expected).status == 'installed'
        assert recovery.rematerialize(descriptor.reference, expected=expected).status == 'already_present'

        blob.write_bytes(b'x' * len(content))
        with pytest.raises(ArtifactRematerializationError) as caught:
            recovery.rematerialize(descriptor.reference, expected=expected)
        assert caught.value.reason == 'corrupt'
        assert caught.value.__cause__ is None and caught.value.__suppress_context__
        assert blob.read_bytes() == b'x' * len(content)


async def test_stale_state_and_wrong_reference_fail_content_free(tmp_path):
    async with setup_read(tmp_path) as setup:
        writer, tool, store, descriptor, _ = setup
        reader = ArtifactReader(writer, 'root_history', store, tool)
        recovery = ArtifactRematerializer(reader)
        expected = writer.owner.read_snapshot().state
        blob = next(store.root.rglob('*.blob'))
        blob.unlink()
        for reference, state in (
            (descriptor.reference, replace(expected, revision=expected.revision - 1)),
            ('artifact:' + 'c' * 32, expected),
        ):
            with pytest.raises(ArtifactRematerializationError) as caught:
                recovery.rematerialize(reference, expected=state)
            assert caught.value.reason == 'unavailable'
            assert caught.value.__cause__ is None and caught.value.__suppress_context__
            assert descriptor.artifact_id not in str(caught.value)
            assert not blob.exists()


async def test_install_fault_is_retry_safe_and_source_is_unchanged(tmp_path, monkeypatch):
    async with setup_read(tmp_path) as setup:
        writer, tool, store, descriptor, content = setup
        reader = ArtifactReader(writer, 'root_history', store, tool)
        recovery = ArtifactRematerializer(reader)
        expected = writer.owner.read_snapshot().state
        source_path = writer.store._backend.get_session_path(writer.owner.key)
        before = source_path.read_bytes()
        next(store.root.rglob('*.blob')).unlink()

        import nanobot.session.artifact_store as artifact_store

        real_link = artifact_store.os.link
        monkeypatch.setattr(
            artifact_store.os,
            'link',
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError()),
        )
        with pytest.raises(ArtifactRematerializationError) as caught:
            recovery.rematerialize(descriptor.reference, expected=expected)
        assert caught.value.reason == 'io'
        assert source_path.read_bytes() == before
        assert not list(store.root.rglob('*.blob'))

        monkeypatch.setattr(artifact_store.os, 'link', real_link)
        assert recovery.rematerialize(descriptor.reference, expected=expected).status == 'installed'
        assert reader.read(descriptor.reference).text.encode() == content
