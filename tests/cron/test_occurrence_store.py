"""Durability contract for the private scheduled-occurrence ledger."""

from __future__ import annotations

import asyncio
import copy
import multiprocessing
import os
import pickle
import stat
import threading
from dataclasses import replace
from pathlib import Path
from queue import Empty
from unittest.mock import patch

import pytest

import nanobot.cron.occurrence_store as occurrence_store
from nanobot.cron.occurrence_store import (
    CronOccurrenceStore,
    OccurrenceAlreadyCompletedError,
    OccurrenceConflictError,
    OccurrenceCorruptError,
    OccurrenceExecutionBusyError,
    OccurrenceOwnershipError,
    OccurrencePersistenceError,
    OccurrenceReconciliationRequiredError,
)

NAMESPACE = "cron-store:private-test"
SNAPSHOT = {"name": "Morning report", "payload": {"message": "private input"}}
DESTINATION_POLICY = {"destinations": [{"channel": "telegram", "to": "123"}]}


def _claim_in_subprocess(root: str, ready: object, outcomes: object, owner: str) -> None:
    """Race one independent interpreter for the cross-process claim test."""
    store = CronOccurrenceStore(Path(root), namespace=NAMESPACE)
    ready.wait()  # type: ignore[union-attr]
    try:
        store.claim_scheduled(
            job_id="job-a",
            job_revision=7,
            scheduled_fire_time_ms=1_700_000_000_000,
            job_snapshot=SNAPSHOT,
            destination_policy=DESTINATION_POLICY,
            owner=owner,
        )
    except OccurrenceReconciliationRequiredError:
        outcomes.put("reconciliation-required")  # type: ignore[union-attr]
    else:
        outcomes.put("won")  # type: ignore[union-attr]


def _complete_in_fork(store: CronOccurrenceStore, claim: object, outcomes: object) -> None:
    """Try to use a fork-inherited in-memory claim from a new process."""
    try:
        _complete(store, claim, {"status": "forked"})  # type: ignore[arg-type]
    except OccurrenceOwnershipError:
        outcomes.put("rejected")  # type: ignore[union-attr]
    else:
        outcomes.put("accepted")  # type: ignore[union-attr]


def _use_fence_in_fork(store, claim, fence, outcomes, release) -> None:
    try:
        store.begin_execution(claim, fence)
    except OccurrenceOwnershipError:
        outcomes.put("rejected")
    else:
        outcomes.put("accepted")
    release.wait()


def _store(tmp_path: Path) -> CronOccurrenceStore:
    return CronOccurrenceStore(tmp_path / "private-occurrences", namespace=NAMESPACE)


def test_qualified_module_capture_retains_in_place_flock_patches(tmp_path: Path, monkeypatch) -> None:
    store = _store(tmp_path)
    module = occurrence_store._fcntl
    original = module.flock
    calls = []

    def observed(fd, operation):
        calls.append(operation)
        return original(fd, operation)

    monkeypatch.setattr(module, "flock", observed)
    monkeypatch.setattr(occurrence_store, "_fcntl", None)
    _claim(store)
    assert module.LOCK_EX in calls and module.LOCK_UN in calls


def _claim(store: CronOccurrenceStore, *, owner: str = "worker-a", revision: int = 7,
           fire_time_ms: int = 1_700_000_000_000, job_id: str = "job-a"):
    return store.claim_scheduled(
        job_id=job_id,
        job_revision=revision,
        scheduled_fire_time_ms=fire_time_ms,
        job_snapshot=SNAPSHOT,
        destination_policy=DESTINATION_POLICY,
        owner=owner,
    )


def _complete(store: CronOccurrenceStore, claim, completion: dict[str, object]):
    async def execute():
        with store.acquire_execution_fence(claim) as fence:
            store.begin_execution(claim, fence)
            return store.complete(claim, completion, fence)

    return asyncio.run(execute())


def _wire_for_claim(store: CronOccurrenceStore, *, completion: dict[str, object] | None = None,
                    generation: int = 1, reconciliation: dict[str, object] | None = None) -> bytes:
    """Build the deterministic wire shape used for encoded-size boundaries."""
    identity = store._validate_identity(
        namespace=NAMESPACE,
        job_id="job-a",
        job_revision=7,
        scheduled_fire_time_ms=1_700_000_000_000,
    )
    return store._wire_record(
        occurrence_id=occurrence_store._occurrence_id(identity),
        identity=identity,
        state="completed" if completion is not None else "claimed",
        owner="worker-a",
        generation=generation,
        job_snapshot=SNAPSHOT,
        destination_policy=DESTINATION_POLICY,
        completion=completion,
        reconciliation=reconciliation,
    )


def _same_inode(fd: int, path: Path) -> bool:
    info = os.fstat(fd)
    expected = path.stat()
    return (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino)


def _persistent_final_parent_fsync_failure(root: Path):
    """Fail only the final root's parent fsync in every participating thread."""
    original_fsync = os.fsync
    local = threading.local()

    def fail_final_parent(fd: int) -> None:
        if root.exists() and _same_inode(fd, root):
            local.root_was_synced = True
            original_fsync(fd)
            return
        if getattr(local, "root_was_synced", False) and _same_inode(fd, root.parent):
            raise OSError("persistent final parent fsync failure")
        original_fsync(fd)

    return fail_final_parent


def test_scheduled_identity_binds_namespace_revision_and_fire_time(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = _claim(store)
    second = _claim(store, revision=8)
    third = _claim(store, fire_time_ms=1_700_000_060_000)

    other_namespace = CronOccurrenceStore(store.root, namespace="cron-store:other")
    fourth = _claim(other_namespace)

    assert len({first.occurrence_id, second.occurrence_id, third.occurrence_id, fourth.occurrence_id}) == 4
    assert first.identity.store_namespace == NAMESPACE
    assert first.identity.job_revision == 7
    assert first.identity.scheduled_fire_time_ms == 1_700_000_000_000
    assert not hasattr(store, "claim_manual")


def test_claim_freezes_snapshot_and_destination_policy_and_rejects_conflicts(tmp_path: Path) -> None:
    store = _store(tmp_path)
    claim = _claim(store)
    assert claim.job_snapshot == SNAPSHOT
    assert claim.destination_policy == DESTINATION_POLICY

    with pytest.raises(OccurrenceConflictError, match="snapshot"):
        store.claim_scheduled(
            job_id="job-a",
            job_revision=7,
            scheduled_fire_time_ms=1_700_000_000_000,
            job_snapshot={"name": "edited after admission"},
            destination_policy=DESTINATION_POLICY,
            owner="worker-b",
        )

    with pytest.raises(OccurrenceConflictError, match="destination policy"):
        store.claim_scheduled(
            job_id="job-a",
            job_revision=7,
            scheduled_fire_time_ms=1_700_000_000_000,
            job_snapshot=SNAPSHOT,
            destination_policy={"destinations": []},
            owner="worker-b",
        )


def test_path_shaped_job_ids_use_safe_distinct_digest_keys(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = _claim(store, job_id="../../same")
    second = _claim(store, job_id=".._.._same")

    assert first.occurrence_id != second.occurrence_id
    files = sorted(path.name for path in store.root.iterdir() if path.suffix == ".json")
    assert files == sorted([f"{first.occurrence_id}.json", f"{second.occurrence_id}.json"])
    assert not (tmp_path / "same").exists()


def test_only_one_process_wins_the_exclusive_claim(tmp_path: Path) -> None:
    root = tmp_path / "private-occurrences"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    outcomes = context.Queue()
    workers = [
        context.Process(target=_claim_in_subprocess, args=(str(root), ready, outcomes, f"worker-{i}"))
        for i in range(2)
    ]
    for worker in workers:
        worker.start()
    ready.set()
    for worker in workers:
        worker.join(timeout=15)
        assert worker.exitcode == 0

    try:
        results = [outcomes.get(timeout=3) for _ in workers]
    except Empty as exc:  # pragma: no cover - makes a worker failure legible
        raise AssertionError("claim worker did not report an outcome") from exc
    assert results.count("won") == 1
    assert results.count("reconciliation-required") == 1


def test_concurrent_open_does_not_invalidate_live_owner_claim(tmp_path: Path) -> None:
    store = _store(tmp_path)
    claim = _claim(store)
    observer = CronOccurrenceStore(store.root, namespace=NAMESPACE)

    outstanding = observer.reconciliation_required()
    assert [(record.owner, record.generation, record.state) for record in outstanding] == [
        ("worker-a", 1, "claimed")
    ]
    assert observer.get(claim.occurrence_id).state == "claimed"

    completed = _complete(store, claim, {"status": "ok", "result": "computed once"})
    assert completed.state == "completed"
    assert observer.get(claim.occurrence_id).completion == {"result": "computed once", "status": "ok"}


def test_reopen_requires_reconciliation_and_stale_owner_cannot_mutate_successor(tmp_path: Path) -> None:
    original_store = _store(tmp_path)
    original = _claim(original_store)
    reopened = CronOccurrenceStore(original_store.root, namespace=NAMESPACE)

    with pytest.raises(OccurrenceReconciliationRequiredError):
        _claim(reopened, owner="restart-worker")

    successor = reopened.reclaim_after_reconciliation(
        original.occurrence_id,
        owner="recovery-worker",
        expected_owner="worker-a",
        expected_generation=1,
        resolution="confirmed_not_started",
    )
    assert successor.generation == 2
    assert successor.owner == "recovery-worker"

    with pytest.raises(OccurrenceOwnershipError):
        _complete(original_store, original, {"status": "ok"})
    completed = _complete(reopened, successor, {"status": "ok", "result": "reconciled"})
    assert completed.generation == 2
    assert completed.state == "completed"


def test_corrupt_record_fails_closed_and_is_not_replaced(tmp_path: Path) -> None:
    store = _store(tmp_path)
    claim = _claim(store)
    record_path = store.root / f"{claim.occurrence_id}.json"
    record_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(OccurrenceCorruptError):
        CronOccurrenceStore(store.root, namespace=NAMESPACE)
    assert record_path.read_text(encoding="utf-8") == "{not valid json"


async def test_completion_is_idempotent_only_for_identical_canonical_content(tmp_path: Path) -> None:
    store = _store(tmp_path)
    claim = _claim(store)
    with store.acquire_execution_fence(claim) as fence:
        store.begin_execution(claim, fence)
        first = store.complete(claim, {"status": "ok", "result": {"count": 1}}, fence)
        duplicate = store.complete(claim, {"result": {"count": 1}, "status": "ok"}, fence)
        assert duplicate == first

        with pytest.raises(OccurrenceConflictError, match="completion"):
            store.complete(claim, {"status": "error", "result": {"count": 1}}, fence)

    with pytest.raises(OccurrenceAlreadyCompletedError):
        _claim(store, revision=7)


async def test_completed_retry_reverifies_durability_after_replace_directory_fsync_failure(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    claim = _claim(store)
    completion = {"status": "ok", "result": "computed"}

    # The temp file fsync succeeds but the post-replace directory fsync fails.
    # The record may be visible, yet the caller receives no delivery authority.
    with store.acquire_execution_fence(claim) as fence:
        store.begin_execution(claim, fence)
        with patch("nanobot.cron.occurrence_store.os.fsync", side_effect=[None, OSError("directory")]):
            with pytest.raises(OccurrencePersistenceError):
                store.complete(claim, completion, fence)
        assert store.get(claim.occurrence_id).state == "completed"

        # An identical retry must fsync and verify the existing completed record;
        # merely returning the visible JSON would incorrectly authorize delivery.
        with patch("nanobot.cron.occurrence_store.os.fsync", side_effect=OSError("still not durable")):
            with pytest.raises(OccurrencePersistenceError):
                store.complete(claim, completion, fence)

        assert store.complete(claim, completion, fence).state == "completed"
    reopened = CronOccurrenceStore(store.root, namespace=NAMESPACE)
    assert reopened.get(claim.occurrence_id).completion == completion


def test_encoded_record_limit_applies_before_claim_completion_and_reclaim_writes(tmp_path: Path) -> None:
    exact_store = _store(tmp_path / "exact")
    claimed_wire = _wire_for_claim(exact_store)
    with patch("nanobot.cron.occurrence_store._MAX_RECORD_BYTES", len(claimed_wire)):
        claim = _claim(exact_store)
    assert claim.occurrence_id

    claim_store = _store(tmp_path / "claim-oversize")
    with patch("nanobot.cron.occurrence_store._MAX_RECORD_BYTES", len(claimed_wire) - 1):
        with pytest.raises(OccurrenceConflictError, match="encoded record"):
            _claim(claim_store)
    assert not list(claim_store.root.glob("*.json"))

    completion_store = _store(tmp_path / "completion-oversize")
    completion_claim = _claim(completion_store)
    completion = {"status": "ok", "result": "computed"}
    completed_wire = _wire_for_claim(completion_store, completion=completion)
    with patch("nanobot.cron.occurrence_store._MAX_RECORD_BYTES", len(completed_wire) - 1):
        with pytest.raises(OccurrenceConflictError, match="encoded record"):
            _complete(completion_store, completion_claim, completion)
    assert completion_store.get(completion_claim.occurrence_id).state == "claimed"

    reclaimed_wire = _wire_for_claim(
        completion_store,
        generation=2,
        reconciliation={
            "resolution": "confirmed_not_started",
            "replaced_owner": "worker-a",
            "replaced_generation": 1,
        },
    )
    with patch("nanobot.cron.occurrence_store._MAX_RECORD_BYTES", len(reclaimed_wire) - 1):
        with pytest.raises(OccurrenceConflictError, match="encoded record"):
            completion_store.reclaim_after_reconciliation(
                completion_claim.occurrence_id,
                owner="recovery-worker",
                expected_owner="worker-a",
                expected_generation=1,
                resolution="confirmed_not_started",
            )
    assert completion_store.get(completion_claim.occurrence_id).generation == 1


def test_only_the_exact_process_local_issued_claim_can_complete(tmp_path: Path) -> None:
    store = _store(tmp_path)
    claim = _claim(store)
    with pytest.raises(OccurrenceOwnershipError, match="issued"):
        _complete(store, replace(claim), {"status": "forged"})
    assert _complete(store, claim, {"status": "ok"}).state == "completed"


@pytest.mark.skipif(os.name == "nt", reason="fork inheritance is POSIX-only")
def test_fork_inherited_claim_cannot_complete(tmp_path: Path) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork is unavailable on this interpreter")
    store = _store(tmp_path)
    claim = _claim(store)
    context = multiprocessing.get_context("fork")
    outcomes = context.Queue()
    child = context.Process(target=_complete_in_fork, args=(store, claim, outcomes))
    child.start()
    child.join(timeout=15)
    assert child.exitcode == 0
    assert outcomes.get(timeout=3) == "rejected"
    assert _complete(store, claim, {"status": "parent"}).state == "completed"


def test_reclaim_issues_a_new_local_claim_object(tmp_path: Path) -> None:
    store = _store(tmp_path)
    original = _claim(store)
    successor = store.reclaim_after_reconciliation(
        original.occurrence_id,
        owner="recovery-worker",
        expected_owner="worker-a",
        expected_generation=1,
        resolution="confirmed_not_started",
    )
    with pytest.raises(OccurrenceOwnershipError, match="issued"):
        _complete(store, replace(successor), {"status": "forged"})
    assert _complete(store, successor, {"status": "ok"}).generation == 2


async def test_write_and_replace_failures_do_not_return_execution_or_delivery_authority(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    with patch("nanobot.cron.occurrence_store.os.fsync", side_effect=OSError("fsync failed")):
        with pytest.raises(OccurrencePersistenceError):
            _claim(store)

    durable_store = _store(tmp_path / "second")
    claim = _claim(durable_store)
    with durable_store.acquire_execution_fence(claim) as fence:
        durable_store.begin_execution(claim, fence)
        with patch("nanobot.cron.occurrence_store.os.fsync", side_effect=OSError("fsync failed")):
            with pytest.raises(OccurrencePersistenceError):
                durable_store.complete(claim, {"status": "ok"}, fence)
        assert durable_store.get(claim.occurrence_id).state == "claimed"

        with patch("nanobot.cron.occurrence_store.os.rename", side_effect=OSError("replace failed")):
            with pytest.raises(OccurrencePersistenceError):
                durable_store.complete(claim, {"status": "ok"}, fence)
    assert durable_store.get(claim.occurrence_id).state == "claimed"


def test_records_and_lock_are_private(tmp_path: Path) -> None:
    store = _store(tmp_path)
    claim = _claim(store)
    record_path = store.root / f"{claim.occurrence_id}.json"

    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(record_path.stat().st_mode) == 0o600
    assert stat.S_IMODE((store.root / ".lock").stat().st_mode) == 0o600


def test_root_creation_requires_parent_fsync_before_any_claim(tmp_path: Path) -> None:
    root = tmp_path / "new-private-occurrences"
    with patch("nanobot.cron.occurrence_store.os.fsync", side_effect=OSError("parent fsync failed")):
        with pytest.raises(OccurrencePersistenceError):
            CronOccurrenceStore(root, namespace=NAMESPACE)
    assert not list(root.glob("*.json")) if root.exists() else True


def test_persistent_final_parent_fsync_failure_cannot_be_bypassed_by_reopen(
    tmp_path: Path,
) -> None:
    root = tmp_path / "persistent-parent-fsync"
    with patch(
        "nanobot.cron.occurrence_store.os.fsync",
        side_effect=_persistent_final_parent_fsync_failure(root),
    ):
        for _ in range(2):
            with pytest.raises(OccurrencePersistenceError, match="root is unavailable"):
                CronOccurrenceStore(root, namespace=NAMESPACE)
    assert not list(root.glob("*.json")) if root.exists() else True


def test_concurrent_initializers_cannot_bypass_persistent_parent_fsync_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "concurrent-parent-fsync"
    start = threading.Barrier(2)
    failures: list[type[BaseException]] = []

    def initialize() -> None:
        start.wait(timeout=5)
        try:
            CronOccurrenceStore(root, namespace=NAMESPACE)
        except BaseException as exc:  # pragma: no branch - asserted below
            failures.append(type(exc))

    with patch(
        "nanobot.cron.occurrence_store.os.fsync",
        side_effect=_persistent_final_parent_fsync_failure(root),
    ):
        workers = [threading.Thread(target=initialize) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=10)
            assert not worker.is_alive()
    assert failures == [OccurrencePersistenceError, OccurrencePersistenceError]


@pytest.mark.parametrize("boundary", ["fchmod", "child_fsync", "parent_fsync"])
def test_new_root_child_fd_closes_when_initialization_boundary_fails(
    tmp_path: Path, boundary: str
) -> None:
    root = tmp_path / f"close-{boundary}"
    captured: dict[str, int] = {}

    if boundary == "fchmod":
        def fail_fchmod(fd: int, _mode: int) -> None:
            captured["child"] = fd
            raise OSError("fchmod failed")

        patcher = patch("nanobot.cron.occurrence_store.os.fchmod", side_effect=fail_fchmod)
    elif boundary == "child_fsync":
        original_fsync = os.fsync

        def fail_child_fsync(fd: int) -> None:
            if root.exists() and _same_inode(fd, root):
                captured["child"] = fd
                raise OSError("child fsync failed")
            original_fsync(fd)

        patcher = patch("nanobot.cron.occurrence_store.os.fsync", side_effect=fail_child_fsync)
    else:
        original_fsync = os.fsync
        local = threading.local()

        def fail_parent_fsync(fd: int) -> None:
            if root.exists() and _same_inode(fd, root):
                captured["child"] = fd
                local.root_was_synced = True
                original_fsync(fd)
                return
            if getattr(local, "root_was_synced", False) and _same_inode(fd, root.parent):
                raise OSError("parent fsync failed")
            original_fsync(fd)

        patcher = patch("nanobot.cron.occurrence_store.os.fsync", side_effect=fail_parent_fsync)

    with patcher:
        with pytest.raises(OccurrencePersistenceError):
            CronOccurrenceStore(root, namespace=NAMESPACE)
    with pytest.raises(OSError):
        os.fstat(captured["child"])


def test_existing_root_and_lock_permissions_are_rejected_without_repair(tmp_path: Path) -> None:
    root = tmp_path / "private-occurrences"
    root.mkdir(mode=0o700)
    root.chmod(0o755)
    with pytest.raises(OccurrenceCorruptError, match="private"):
        CronOccurrenceStore(root, namespace=NAMESPACE)
    assert stat.S_IMODE(root.stat().st_mode) == 0o755

    root.chmod(0o700)
    store = CronOccurrenceStore(root, namespace=NAMESPACE)
    lock = root / ".lock"
    lock.chmod(0o644)
    with pytest.raises(OccurrenceCorruptError, match="private"):
        CronOccurrenceStore(root, namespace=NAMESPACE)
    assert stat.S_IMODE(lock.stat().st_mode) == 0o644
    assert store.root == root


def test_lock_hardlinks_fifos_and_live_root_or_lock_swaps_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    claim = _claim(store)
    lock = store.root / ".lock"
    hardlink = tmp_path / "lock-alias"
    os.link(lock, hardlink)
    with pytest.raises(OccurrenceCorruptError, match="private"):
        CronOccurrenceStore(store.root, namespace=NAMESPACE)
    hardlink.unlink()

    lock.unlink()
    os.mkfifo(lock)
    with pytest.raises(OccurrenceCorruptError, match="unexpected type"):
        CronOccurrenceStore(store.root, namespace=NAMESPACE)
    lock.unlink()
    lock.write_text("", encoding="utf-8")
    lock.chmod(0o600)

    # A live instance keeps the original root/lock binding and refuses swaps.
    old_lock = store.root / ".old-lock"
    lock.rename(old_lock)
    lock.write_text("", encoding="utf-8")
    lock.chmod(0o600)
    with pytest.raises(OccurrenceCorruptError, match="binding"):
        store.get(claim.occurrence_id)


def test_live_root_swap_fails_against_the_original_root_binding(tmp_path: Path) -> None:
    store = _store(tmp_path)
    claim = _claim(store)
    original = store.root
    original.rename(tmp_path / "moved-private-occurrences")
    original.mkdir(mode=0o700)

    with pytest.raises(OccurrenceCorruptError, match="root binding"):
        store.get(claim.occurrence_id)


def test_root_symlink_and_unqualified_platform_fail_closed(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "ledger-link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(OccurrenceCorruptError):
        CronOccurrenceStore(link, namespace=NAMESPACE)

    with patch.object(occurrence_store, "_fcntl", None, create=True):
        with pytest.raises(OccurrencePersistenceError, match="qualified POSIX"):
            _store(tmp_path / "unsupported")


async def test_execution_fence_blocks_exact_reconciliation_and_other_occurrences_do_not_contend(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    first = _claim(store)
    second = _claim(store, fire_time_ms=1_700_000_060_000)
    with store.acquire_execution_fence(first) as first_fence:
        store.begin_execution(first, first_fence)
        with pytest.raises(OccurrenceExecutionBusyError, match="live execution"):
            store.reclaim_after_reconciliation(
                first.occurrence_id,
                owner="recovery",
                expected_owner=first.owner,
                expected_generation=first.generation,
                resolution="confirmed_not_started",
            )
        with store.acquire_execution_fence(second) as second_fence:
            store.begin_execution(second, second_fence)
            assert second_fence.active and second_fence.started
        unchanged = store.get(first.occurrence_id)
        assert (unchanged.owner, unchanged.generation, unchanged.state) == ("worker-a", 1, "claimed")

    successor = store.reclaim_after_reconciliation(
        first.occurrence_id,
        owner="recovery",
        expected_owner=first.owner,
        expected_generation=first.generation,
        resolution="confirmed_not_started",
    )
    assert successor.generation == 2


async def test_fence_capability_rejects_copy_serialization_wrong_task_store_and_inactive_use(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    claim = _claim(store)
    fence = store.acquire_execution_fence(claim)
    with pytest.raises(TypeError, match="copied"):
        copy.copy(fence)
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(fence)

    async def wrong_task() -> None:
        with pytest.raises(OccurrenceOwnershipError, match="transferred"):
            store.begin_execution(claim, fence)

    await asyncio.create_task(wrong_task())
    foreign = CronOccurrenceStore(store.root, namespace=NAMESPACE)
    with pytest.raises(OccurrenceOwnershipError):
        foreign.begin_execution(claim, fence)
    store.begin_execution(claim, fence)
    store.release_execution_fence(fence)
    assert not fence.active
    with pytest.raises(OccurrenceOwnershipError, match="inactive"):
        store.complete(claim, {"status": "ok"}, fence)


@pytest.mark.parametrize("boundary", ["file", "root"])
async def test_execution_lock_fsync_failure_and_existing_residue_retry_grant_no_fence(
    tmp_path: Path,
    boundary: str,
) -> None:
    store = _store(tmp_path)
    claim = _claim(store)
    lock_path = store.root / f".execution.{claim.occurrence_id}.lock"
    original_fsync = os.fsync
    failures = 0

    def fail_boundary(fd: int) -> None:
        nonlocal failures
        is_root = _same_inode(fd, store.root)
        if (boundary == "root" and is_root) or (
            boundary == "file" and stat.S_ISREG(os.fstat(fd).st_mode)
        ):
            failures += 1
            raise OSError(f"{boundary} fsync failed")
        original_fsync(fd)

    with patch("nanobot.cron.occurrence_store.os.fsync", side_effect=fail_boundary):
        for _ in range(2):
            with pytest.raises(OccurrencePersistenceError, match="execution lock"):
                store.acquire_execution_fence(claim)
            assert store._issued_fences == {}
            assert lock_path.exists()
    assert failures == 2
    assert lock_path.exists()
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
    assert lock_path.stat().st_nlink == 1
    with store.acquire_execution_fence(claim) as fence:
        store.begin_execution(claim, fence)
    assert lock_path.exists()
    assert sorted(path.name for path in store.root.glob(".execution.*.lock")) == [lock_path.name]


@pytest.mark.parametrize("unsafe", ["mode", "hardlink", "fifo", "symlink"])
def test_constructor_rejects_unsafe_execution_lock_entries(tmp_path: Path, unsafe: str) -> None:
    store = _store(tmp_path)
    claim = _claim(store)
    name = f".execution.{claim.occurrence_id}.lock"
    path = store.root / name
    path.write_bytes(b"")
    path.chmod(0o600)
    if unsafe == "mode":
        path.chmod(0o640)
    elif unsafe == "hardlink":
        os.link(path, tmp_path / "execution-lock-alias")
    else:
        path.unlink()
        if unsafe == "fifo":
            os.mkfifo(path, 0o600)
        else:
            target = tmp_path / "execution-lock-target"
            target.write_bytes(b"")
            target.chmod(0o600)
            path.symlink_to(target)
    with pytest.raises(OccurrenceCorruptError, match="execution lock"):
        CronOccurrenceStore(store.root, namespace=NAMESPACE)


@pytest.mark.skipif(os.name == "nt", reason="fork inheritance is POSIX-only")
async def test_fork_child_drops_inherited_fence_without_unlocking_or_retaining_parent_lock(
    tmp_path: Path,
) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork is unavailable on this interpreter")
    store = _store(tmp_path)
    claim = _claim(store)
    context = multiprocessing.get_context("fork")
    outcomes = context.Queue()
    release = context.Event()
    fence = store.acquire_execution_fence(claim)
    store.begin_execution(claim, fence)
    child = context.Process(
        target=_use_fence_in_fork,
        args=(store, claim, fence, outcomes, release),
    )
    child.start()
    assert outcomes.get(timeout=3) == "rejected"
    observer = CronOccurrenceStore(store.root, namespace=NAMESPACE)
    with pytest.raises(OccurrenceExecutionBusyError):
        observer.reclaim_after_reconciliation(
            claim.occurrence_id,
            owner="recovery",
            expected_owner=claim.owner,
            expected_generation=claim.generation,
            resolution="confirmed_compensated",
        )
    store.release_execution_fence(fence)
    successor = observer.reclaim_after_reconciliation(
        claim.occurrence_id,
        owner="recovery",
        expected_owner=claim.owner,
        expected_generation=claim.generation,
        resolution="confirmed_compensated",
    )
    assert successor.generation == 2
    release.set()
    child.join(timeout=10)
    assert child.exitcode == 0
