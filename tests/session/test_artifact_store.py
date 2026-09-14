"""Private blob installation cannot follow aliases, overwrite evidence, or skip integrity."""

import hashlib
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path

import pytest

from nanobot.session.artifact import ArtifactValidationError, parse_artifact_descriptor
from nanobot.session.artifact_store import ArtifactStorageError, PrivateArtifactStore
from session.test_artifact import WORKSPACE, fixture


def make_store(root):
    return PrivateArtifactStore(root, workspace_id=WORKSPACE, session_key='test:artifact',
                                incarnation='incarnation', generation=2)


def descriptor(content='evidence'):
    raw, _ = fixture(content)
    return parse_artifact_descriptor(raw)


def namespace(store):
    return store.root.joinpath(*store._parts)


def child_put(root, raw, queue):
    try:
        store = make_store(Path(root))
        with store.transaction() as transaction:
            transaction.put(parse_artifact_descriptor(raw), b'evidence')
        queue.put('ok')
    except Exception as exc:
        errors = []
        while exc is not None:
            errors.append((type(exc).__name__, getattr(exc, 'errno', None), str(exc)))
            exc = exc.__context__
        queue.put(errors)


def child_put_with_namespace_quota(root, raw, content, queue):
    try:
        store = PrivateArtifactStore(
            Path(root),
            workspace_id=WORKSPACE,
            session_key='test:artifact',
            incarnation='incarnation',
            generation=2,
            max_blob_bytes=8,
            max_namespace_bytes=9,
        )
        with store.transaction() as transaction:
            transaction.put(parse_artifact_descriptor(raw), content)
        queue.put('ok')
    except Exception as exc:
        queue.put(type(exc).__name__)


@pytest.fixture
def blob_store(tmp_path):
    return make_store(tmp_path.resolve() / 'artifacts')


def test_private_immutable_blob_survives_reopen_and_bounded_byte_reads(blob_store):
    text = 'evidence 中文 🌟'
    value = descriptor(text)
    encoded = text.encode()
    with blob_store.transaction() as transaction:
        transaction.put(value, encoded)
        transaction.put(value, encoded)
        assert transaction.read_range(value, offset=2, limit=5) == encoded[2:7]
        assert transaction.read_range(value, offset=len(encoded) + 10) == b''
    reopened = make_store(blob_store.root)
    with reopened.transaction() as transaction:
        assert transaction.read_range(value) == encoded
    assert {p.name for p in namespace(blob_store).iterdir()} == {'.lock', value.artifact_id + '.blob'}
    for entry in blob_store.root.rglob('*'):
        assert entry.stat().st_mode & 0o077 == 0


def test_same_id_cannot_overwrite_different_bytes(blob_store):
    original = descriptor()
    changed = replace(original, sha256=hashlib.sha256(b'different').hexdigest(), byte_length=9)
    with blob_store.transaction() as transaction:
        transaction.put(original, b'evidence')
        with pytest.raises(ArtifactStorageError):
            transaction.put(changed, b'different')
        assert transaction.read_range(original) == b'evidence'


@pytest.mark.parametrize('field,value', [('workspace_id', 'c' * 32), ('session_key', 'other'),
    ('incarnation', 'other'), ('generation', 3)])
def test_store_namespace_is_not_selected_by_descriptor(blob_store, field, value):
    with blob_store.transaction() as transaction:
        with pytest.raises(ArtifactValidationError, match='namespace'):
            transaction.put(replace(descriptor(), **{field: value}), b'evidence')
    assert not list(blob_store.root.rglob('*.blob'))


@pytest.mark.parametrize('offset,limit', [(-1, 1), (True, 1), (0, True), (0, -1), (0, 16385), ('0', 1)])
def test_byte_ranges_are_strict_and_bounded(blob_store, offset, limit):
    with blob_store.transaction() as transaction:
        transaction.put(descriptor(), b'evidence')
        with pytest.raises(ArtifactValidationError, match='range'):
            transaction.read_range(descriptor(), offset=offset, limit=limit)


def test_transaction_cannot_be_reused_after_lock_release(blob_store):
    with blob_store.transaction() as transaction:
        transaction.put(descriptor(), b'evidence')
    with pytest.raises(ArtifactStorageError, match='ended'):
        transaction.read_range(descriptor())


@pytest.mark.parametrize('change', ['same_length', 'short', 'hardlink', 'mode'])
def test_reads_verify_whole_blob_even_for_empty_range(blob_store, tmp_path, change):
    value = descriptor()
    with blob_store.transaction() as transaction:
        transaction.put(value, b'evidence')
    path = namespace(blob_store) / (value.artifact_id + '.blob')
    if change == 'same_length':
        path.write_bytes(b'changed!')
    elif change == 'short':
        path.write_bytes(b'short')
    elif change == 'hardlink':
        os.link(path, tmp_path / 'alias')
    else:
        path.chmod(0o644)
    with pytest.raises(ArtifactStorageError):
        with blob_store.transaction() as transaction:
            transaction.read_range(value, limit=0)


@pytest.mark.parametrize('target', ['blob', 'lock', 'namespace'])
def test_symlink_components_never_reach_external_files(blob_store, tmp_path, target):
    external = tmp_path / 'outside'
    external.mkdir()
    protected = external / (descriptor().artifact_id + '.blob')
    protected.write_bytes(b'protected')
    with blob_store.transaction():
        pass
    scope = namespace(blob_store)
    if target == 'blob':
        (scope / protected.name).symlink_to(protected)
    elif target == 'lock':
        (scope / '.lock').unlink()
        (scope / '.lock').symlink_to(protected)
    else:
        old = scope.with_name(scope.name + '-old')
        scope.rename(old)
        scope.symlink_to(external, target_is_directory=True)
    with pytest.raises(ArtifactStorageError):
        with blob_store.transaction() as transaction:
            transaction.put(descriptor(), b'evidence')
    assert protected.read_bytes() == b'protected'
    assert {p.name for p in external.iterdir()} == {protected.name}


def test_symlink_in_root_ancestry_is_rejected(tmp_path):
    actual = tmp_path.resolve() / 'actual'
    actual.mkdir()
    alias = tmp_path.resolve() / 'alias'
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ArtifactStorageError):
        make_store(alias / 'artifacts')
    assert not (actual / 'artifacts').exists()


def test_replaced_root_is_not_adopted_by_existing_store(blob_store):
    original = blob_store.root.with_name('old-root')
    blob_store.root.rename(original)
    blob_store.root.mkdir(mode=0o700)
    with pytest.raises(ArtifactStorageError):
        with blob_store.transaction():
            pass
    assert not list(blob_store.root.iterdir())


def test_namespace_replacement_during_operation_stays_on_open_directory(blob_store, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    with blob_store.transaction() as transaction:
        scope = namespace(blob_store)
        old = scope.with_name('old-generation')
        scope.rename(old)
        scope.symlink_to(outside, target_is_directory=True)
        with pytest.raises(ArtifactStorageError):
            transaction.put(descriptor(), b'evidence')
    assert list(outside.iterdir()) == []
    assert not (old / (descriptor().artifact_id + '.blob')).exists()
    with pytest.raises(ArtifactStorageError):
        with blob_store.transaction():
            pass


def test_failed_install_leaves_no_published_blob(blob_store, monkeypatch):
    with blob_store.transaction() as transaction:
        monkeypatch.setattr(os, 'link', lambda *a, **k: (_ for _ in ()).throw(OSError('install failed')))
        with pytest.raises(OSError):
            transaction.put(descriptor(), b'evidence')
    assert {p.name for p in namespace(blob_store).iterdir()} == {'.lock'}


def test_failure_after_install_leaves_only_unpublished_durable_bytes(blob_store, monkeypatch):
    with blob_store.transaction() as transaction:
        sync = os.fsync

        def fail_directory(fd):
            if fd == transaction._fd:
                raise OSError('directory sync failed')
            return sync(fd)

        monkeypatch.setattr(os, 'fsync', fail_directory)
        with pytest.raises(OSError):
            transaction.put(descriptor(), b'evidence')
    monkeypatch.undo()
    with blob_store.transaction() as transaction:
        transaction.put(descriptor(), b'evidence')  # Reuse also fsyncs before success.
        assert transaction.read_range(descriptor()) == b'evidence'


def test_blob_capacity_stops_before_creating_content(tmp_path):
    store = PrivateArtifactStore(tmp_path.resolve() / 'artifacts', workspace_id=WORKSPACE,
        session_key='test:artifact', incarnation='incarnation', generation=2, max_blob_bytes=4)
    with store.transaction() as transaction:
        with pytest.raises(ArtifactStorageError, match='capacity'):
            transaction.put(descriptor(), b'evidence')
    assert not list(store.root.rglob('*.blob'))


def test_namespace_quota_counts_distinct_blobs_but_not_identical_reuse(tmp_path):
    store = PrivateArtifactStore(
        tmp_path.resolve() / 'artifacts',
        workspace_id=WORKSPACE,
        session_key='test:artifact',
        incarnation='incarnation',
        generation=2,
        max_blob_bytes=8,
        max_namespace_bytes=9,
    )
    first = descriptor('first')
    second = replace(descriptor('other'), artifact_id='c' * 32)

    with store.transaction() as transaction:
        transaction.put(first, b'first')
        transaction.put(first, b'first')
        with pytest.raises(ArtifactStorageError, match='quota'):
            transaction.put(second, b'other')
        assert transaction.read_range(first) == b'first'
    assert not (namespace(store) / (second.artifact_id + '.blob')).exists()


def test_namespace_quota_counts_retained_staging_without_deleting_it(tmp_path):
    store = PrivateArtifactStore(
        tmp_path.resolve() / 'artifacts',
        workspace_id=WORKSPACE,
        session_key='test:artifact',
        incarnation='incarnation',
        generation=2,
        max_blob_bytes=8,
        max_namespace_bytes=10,
    )
    first = descriptor('first')
    second = replace(descriptor('other'), artifact_id='c' * 32)

    with store.transaction() as transaction:
        transaction.put(first, b'first')
        staged = namespace(store) / ('.tmp-' + 'd' * 32)
        staged.write_bytes(b'stage')
        staged.chmod(0o600)
        with pytest.raises(ArtifactStorageError, match='quota'):
            transaction.put(second, b'other')
        assert transaction.read_range(first) == b'first'
        assert staged.read_bytes() == b'stage'


def test_lowered_caps_keep_existing_evidence_readable_and_verifiable(tmp_path):
    root = tmp_path.resolve() / 'artifacts'
    original = PrivateArtifactStore(
        root,
        workspace_id=WORKSPACE,
        session_key='test:artifact',
        incarnation='incarnation',
        generation=2,
        max_blob_bytes=8,
        max_namespace_bytes=16,
    )
    first = descriptor('first')
    second = replace(descriptor('other'), artifact_id='c' * 32)
    with original.transaction() as transaction:
        transaction.put(first, b'first')

    lowered = PrivateArtifactStore(
        root,
        workspace_id=WORKSPACE,
        session_key='test:artifact',
        incarnation='incarnation',
        generation=2,
        max_blob_bytes=4,
        max_namespace_bytes=4,
    )
    with lowered.transaction() as transaction:
        assert transaction.read_range(first) == b'first'
        transaction.verify_durable(first, store=lowered)
        transaction.put(first, b'first')
        with pytest.raises(ArtifactStorageError, match='capacity'):
            transaction.put(second, b'other')


def test_concurrent_distinct_blobs_cannot_exceed_namespace_quota(tmp_path):
    root = tmp_path.resolve() / 'artifacts'
    first = descriptor('first')
    second = replace(descriptor('other'), artifact_id='c' * 32)
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    processes = [
        context.Process(target=child_put_with_namespace_quota,
                        args=(str(root), value.to_dict(), content, queue))
        for value, content in ((first, b'first'), (second, b'other'))
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=15)
            assert not process.is_alive() and process.exitcode == 0
        assert sorted(queue.get(timeout=2) for _ in processes) == ['ArtifactStorageError', 'ok']
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
    store = PrivateArtifactStore(
        root,
        workspace_id=WORKSPACE,
        session_key='test:artifact',
        incarnation='incarnation',
        generation=2,
        max_blob_bytes=8,
        max_namespace_bytes=9,
    )
    assert len(list(namespace(store).glob('*.blob'))) == 1


def test_two_processes_reuse_one_complete_immutable_blob(blob_store):
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    processes = [context.Process(target=child_put, args=(str(blob_store.root), descriptor().to_dict(), queue))
                 for _ in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=15)
            assert not process.is_alive() and process.exitcode == 0
        assert [queue.get(timeout=2) for _ in processes] == ['ok', 'ok']
        with blob_store.transaction() as transaction:
            assert transaction.read_range(descriptor()) == b'evidence'
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        queue.close()


def child_crash_after_link(root, raw):
    store = make_store(Path(root))
    link = os.link

    def crash(*args, **kwargs):
        link(*args, **kwargs)
        os._exit(23)

    os.link = crash
    with store.transaction() as transaction:
        transaction.put(parse_artifact_descriptor(raw), b'evidence')


def test_actual_crash_between_link_and_unlink_recovers_only_verified_staging_link(blob_store):
    context = multiprocessing.get_context('spawn')
    process = context.Process(target=child_crash_after_link, args=(str(blob_store.root), descriptor().to_dict()))
    try:
        process.start()
        process.join(timeout=15)
        assert not process.is_alive() and process.exitcode == 23
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
    scope = namespace(blob_store)
    assert len(list(scope.glob('.tmp-*'))) == 1
    with blob_store.transaction() as transaction:
        with pytest.raises(ArtifactStorageError):
            transaction.read_range(descriptor())
        transaction.put(descriptor(), b'evidence')
        assert transaction.read_range(descriptor()) == b'evidence'
    assert not list(scope.glob('.tmp-*'))
    assert (scope / (descriptor().artifact_id + '.blob')).stat().st_nlink == 1


def test_arbitrary_hardlink_is_not_treated_as_recoverable_staging(blob_store):
    with blob_store.transaction() as transaction:
        transaction.put(descriptor(), b'evidence')
        scope = namespace(blob_store)
        os.link(scope / (descriptor().artifact_id + '.blob'), scope / 'not-a-temporary')
        with pytest.raises(ArtifactStorageError):
            transaction.put(descriptor(), b'evidence')
    assert (scope / 'not-a-temporary').read_bytes() == b'evidence'


def test_corrupt_staging_install_is_not_deleted_or_recovered(blob_store):
    with blob_store.transaction() as transaction:
        transaction.put(descriptor(), b'evidence')
        scope = namespace(blob_store)
        path = scope / (descriptor().artifact_id + '.blob')
        temporary = scope / ('.tmp-' + 'd' * 32)
        os.link(path, temporary)
        path.write_bytes(b'changed!')
        with pytest.raises(ArtifactStorageError):
            transaction.put(descriptor(), b'evidence')
    assert temporary.exists() and path.stat().st_nlink == 2


def test_adopted_namespace_ancestors_are_synced_before_use(blob_store, monkeypatch):
    with blob_store.transaction():
        pass
    synced = set()
    sync = os.fsync

    def observe(fd):
        info = os.fstat(fd)
        synced.add((info.st_dev, info.st_ino))
        return sync(fd)

    monkeypatch.setattr(os, 'fsync', observe)
    reopened = make_store(blob_store.root)
    with reopened.transaction() as transaction:
        transaction.put(descriptor(), b'evidence')
    expected = [blob_store.root.parent, blob_store.root]
    current = blob_store.root
    for part in blob_store._parts:
        current = current / part
        expected.append(current)
    assert {(p.stat().st_dev, p.stat().st_ino) for p in expected} <= synced


def test_replacement_during_install_cannot_be_reported_as_reachable(blob_store, tmp_path, monkeypatch):
    outside = tmp_path / 'outside-during-install'
    outside.mkdir()
    link = os.link
    with blob_store.transaction() as transaction:
        scope = namespace(blob_store)
        old = scope.with_name('detached-generation')

        def replace_namespace(*args, **kwargs):
            scope.rename(old)
            scope.symlink_to(outside, target_is_directory=True)
            return link(*args, **kwargs)

        monkeypatch.setattr(os, 'link', replace_namespace)
        with pytest.raises(ArtifactStorageError):
            transaction.put(descriptor(), b'evidence')
    assert not list(outside.iterdir())
    assert (old / (descriptor().artifact_id + '.blob')).read_bytes() == b'evidence'


def test_replacing_lock_inode_invalidates_publication_capability(blob_store):
    with blob_store.transaction() as transaction:
        scope = namespace(blob_store)
        (scope / '.lock').rename(scope / '.old-lock')
        (scope / '.lock').touch(mode=0o600)
        with pytest.raises(ArtifactStorageError):
            transaction.assert_reachable()


def test_unsupported_filesystem_capabilities_fail_before_creation(tmp_path, monkeypatch):
    root = tmp_path.resolve() / 'unsupported'
    monkeypatch.setattr(os, 'supports_dir_fd', set())
    with pytest.raises(ArtifactStorageError, match='unavailable'):
        make_store(root)
    assert not root.exists()
