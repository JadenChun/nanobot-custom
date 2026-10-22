"""Owned artifact lifetime checks preserve real locks, late hooks and cleanup."""

import os
from contextlib import contextmanager

import pytest

from nanobot.session.artifact_store import (
    ArtifactBlobTransaction,
    ArtifactStorageError,
    PrivateArtifactStore,
)
from tests.session.test_artifact_store import descriptor, make_store


@pytest.mark.parametrize("body_fails", [False, True])
def test_cleanup_ends_transaction_before_closing_namespace(tmp_path, monkeypatch, body_fails):
    store = make_store(tmp_path / "artifacts")
    original = ArtifactBlobTransaction.end
    trace = []
    failure = RuntimeError("controlled transaction body failure")

    def end(value):
        trace.append((value, value._active, os.fstat(value._fd)))
        original(value)
        assert not value._active

    def forbidden_override():
        raise AssertionError("instance override must not replace owner cleanup")

    with monkeypatch.context() as patch:
        patch.setattr(ArtifactBlobTransaction, "end", end)
        try:
            with store.transaction() as transaction:
                transaction.put(descriptor(), b"evidence")
                patch.setattr(transaction, "end", forbidden_override)
                if body_fails:
                    raise failure
        except RuntimeError as caught:
            assert body_fails and caught is failure

    assert len(trace) == 1 and trace[0][0] is transaction and trace[0][1]
    with pytest.raises(OSError):
        os.fstat(transaction._fd)
    with pytest.raises(ArtifactStorageError, match="transaction has ended"):
        transaction.read_range(descriptor())
    with store.transaction(blocking=False) as next_transaction:
        assert next_transaction.read_range(descriptor()) == b"evidence"


def test_end_only_revokes_operations_without_changing_evidence(tmp_path, monkeypatch):
    store = make_store(tmp_path / "artifacts")
    with store.transaction() as transaction:
        transaction.put(descriptor(), b"evidence")
        before = {path.relative_to(tmp_path): path.read_bytes()
                  for path in tmp_path.rglob("*") if path.is_file()}
        transaction.end()
        transaction.end()

        def forbidden_root(*args, **kwargs):
            raise AssertionError("ended transaction must fail before root access")

        with monkeypatch.context() as patch:
            patch.setattr(store, "_root_fd", forbidden_root)
            with pytest.raises(ArtifactStorageError, match="transaction has ended"):
                transaction.assert_reachable()
            with pytest.raises(ArtifactStorageError, match="transaction has ended"):
                transaction.put(descriptor(), b"evidence")
        assert transaction.matches_store(store)
        assert {path.relative_to(tmp_path): path.read_bytes()
                for path in tmp_path.rglob("*") if path.is_file()} == before


def test_reachability_uses_owned_reject_only_operation_nonvirtually(tmp_path, monkeypatch):
    store = make_store(tmp_path / "artifacts")
    original = PrivateArtifactStore.assert_namespace_binding
    trace = []

    def validate(value, *, select_store, binding_changed):
        trace.append(value)
        return original(value, select_store=select_store, binding_changed=binding_changed)

    def forbidden_override(**kwargs):
        raise AssertionError("instance override must not bypass namespace checks")

    with store.transaction() as transaction:
        before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
        with monkeypatch.context() as patch:
            patch.setattr(PrivateArtifactStore, "assert_namespace_binding", validate)
            patch.setattr(store, "assert_namespace_binding", forbidden_override)
            assert transaction.assert_reachable() is None
        assert trace == [store]
        assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


def test_namespace_store_is_observed_again_after_root_hook(tmp_path, monkeypatch):
    store = make_store(tmp_path / "artifacts")
    alternate = PrivateArtifactStore(
        store.root, workspace_id=store.workspace_id, session_key=store.session_key,
        incarnation=store.incarnation, generation=store.generation + 1,
    )
    with alternate.transaction():
        pass
    original = store._root_fd
    with store.transaction() as transaction:
        @contextmanager
        def swap_after_root(**kwargs):
            with original(**kwargs) as root:
                transaction._store = alternate
                yield root

        with monkeypatch.context() as patch:
            patch.setattr(store, "_root_fd", swap_after_root)
            with pytest.raises(ArtifactStorageError, match="namespace or lock binding changed"):
                transaction.assert_reachable()
        transaction._store = store
        transaction.assert_reachable()


def test_expected_identities_are_observed_after_each_inode_hook(tmp_path, monkeypatch):
    store = make_store(tmp_path / "artifacts")
    with store.transaction() as transaction:
        original = transaction._inode
        transaction._directory_identity = (-1, -1)
        transaction._lock_identity = (-1, -1)
        trace = []

        def restore_identity(info):
            identity = original(info)
            trace.append(identity)
            if len(trace) == 1:
                transaction._directory_identity = identity
            else:
                transaction._lock_identity = identity
            return identity

        with monkeypatch.context() as patch:
            patch.setattr(transaction, "_inode", restore_identity)
            transaction.assert_reachable()
        assert trace == [transaction._directory_identity, transaction._lock_identity]


def test_directory_identity_mismatch_short_circuits_lock_comparison(tmp_path, monkeypatch):
    store = make_store(tmp_path / "artifacts")
    with store.transaction() as transaction:
        original = transaction._inode
        transaction._directory_identity = (-1, -1)
        trace = []

        def observe_identity(info):
            trace.append(original(info))
            return original(info)

        with monkeypatch.context() as patch:
            patch.setattr(transaction, "_inode", observe_identity)
            with pytest.raises(ArtifactStorageError, match="namespace or lock binding changed"):
                transaction.assert_reachable()
        assert len(trace) == 1


def test_inode_lookup_precedes_directory_stat(tmp_path, monkeypatch):
    store = make_store(tmp_path / "artifacts")
    with store.transaction() as transaction:
        original_inode = transaction._inode
        original_stat = os.fstat
        identity = original_inode(original_stat(transaction._fd))
        trace = []

        def lookup(value):
            assert value is transaction
            trace.append("inode-lookup")
            return original_inode

        def observe_stat(fd):
            info = original_stat(fd)
            if original_inode(info) == identity:
                trace.append("directory-stat")
            return info

        with monkeypatch.context() as patch:
            patch.setattr(ArtifactBlobTransaction, "_inode", property(lookup))
            patch.setattr(os, "fstat", observe_stat)
            transaction.assert_reachable()
        assert trace[-3:] == ["inode-lookup", "directory-stat", "inode-lookup"]


def test_identity_fault_preserves_exception_and_closes_recheck_descriptors(tmp_path, monkeypatch):
    store = make_store(tmp_path / "artifacts")
    failure = RuntimeError("controlled inode hook failure")
    original_close = os.close
    closed = []
    with store.transaction() as transaction:
        def reject(info):
            raise failure

        def close(fd):
            closed.append(fd)
            original_close(fd)

        with monkeypatch.context() as patch:
            patch.setattr(transaction, "_inode", reject)
            patch.setattr(os, "close", close)
            with pytest.raises(RuntimeError) as caught:
                transaction.assert_reachable()
        assert caught.value is failure
        assert len(closed) >= 5
        for fd in closed:
            with pytest.raises(OSError):
                os.fstat(fd)
        assert os.fstat(transaction._fd)
        transaction.assert_reachable()
