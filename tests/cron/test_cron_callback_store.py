import asyncio
import os
from pathlib import Path

import pytest

from nanobot.cron.callback_store import (
    CronCallbackConflictError,
    CronCallbackObservation,
    CronCallbackReconciliationRequiredError,
    CronCallbackStore,
)


def test_unlock_failure_still_closes_callback_lock(tmp_path: Path, monkeypatch) -> None:
    import nanobot.cron.callback_store as module

    store = CronCallbackStore(tmp_path / "callbacks", lambda value: None)
    real_flock = module.fcntl.flock
    unlock_descriptors: list[int] = []

    def fail_unlock(descriptor: int, operation: int) -> None:
        if operation == module.fcntl.LOCK_UN:
            unlock_descriptors.append(descriptor)
            raise OSError("injected unlock failure")
        real_flock(descriptor, operation)

    with monkeypatch.context() as patch:
        patch.setattr(module.fcntl, "flock", fail_unlock)
        with pytest.raises(OSError, match="injected unlock failure"):
            with store._locked():
                pass
    assert len(unlock_descriptors) == 1
    try:
        os.fstat(unlock_descriptors[0])
    except OSError:
        closed = True
    else:
        closed = False
        os.close(unlock_descriptors[0])
    assert closed, "unlock failure leaked the callback lock descriptor"


def _observation() -> CronCallbackObservation:
    return CronCallbackObservation(
        occurrence_id="a" * 64,
        run_id="cron-occurrence-a-1",
        job_id="job-1",
        job_revision="revision-1",
        generation=1,
        kind="result",
        result_status="ok",
    )


async def test_callback_is_durable_content_free_and_invoked_once(tmp_path: Path) -> None:
    seen = []
    store = CronCallbackStore(tmp_path / "callbacks", lambda value: seen.append(value))
    result = await store.invoke_once(_observation())
    assert result.state == "succeeded"
    assert seen == [_observation()]
    persisted = store.get(result.callback_id)
    assert persisted == result
    assert "content" not in (store.root / f"{result.callback_id}.json").read_text()
    with pytest.raises(CronCallbackReconciliationRequiredError):
        await store.invoke_once(_observation())
    assert seen == [_observation()]


async def test_failure_is_recorded_and_cancelled_invocation_requires_reconciliation(
    tmp_path: Path,
) -> None:
    def fail(_value):
        raise RuntimeError("secret callback output")

    failed = CronCallbackStore(tmp_path / "failed", fail)
    record = await failed.invoke_once(_observation())
    assert record.state == "failed"
    assert record.failure_code == "callback_failed"
    assert "secret callback output" not in (
        failed.root / f"{record.callback_id}.json"
    ).read_text()

    async def cancel(_value):
        raise __import__("asyncio").CancelledError

    cancelled = CronCallbackStore(tmp_path / "cancelled", cancel)
    with pytest.raises(__import__("asyncio").CancelledError):
        await cancelled.invoke_once(_observation())
    callback_id = cancelled.callback_id(_observation())
    assert cancelled.get(callback_id).state == "invoking"
    with pytest.raises(CronCallbackReconciliationRequiredError):
        await cancelled.invoke_once(_observation())
    assert cancelled.reconcile(callback_id, resolution="abandoned").state == "abandoned"


async def test_leaf_root_substitution_after_success_never_reinvokes_observer(
    tmp_path: Path,
) -> None:
    seen = []
    root = tmp_path / "callbacks"
    store = CronCallbackStore(root, lambda value: seen.append(value))
    await store.invoke_once(_observation())
    original = tmp_path / "callbacks-original"
    root.rename(original)
    replacement = tmp_path / "callbacks-replacement"
    replacement.mkdir(mode=0o700)
    root.symlink_to(replacement, target_is_directory=True)

    with pytest.raises(CronCallbackConflictError, match="root"):
        await store.invoke_once(_observation())
    assert seen == [_observation()]


def test_symlinked_ancestor_is_rejected_at_construction(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(CronCallbackConflictError, match="no-follow"):
        CronCallbackStore(alias / "callbacks", lambda _value: None)


async def test_root_substitution_while_pending_fails_before_observer(tmp_path: Path) -> None:
    seen = []
    root = tmp_path / "callbacks"
    store = CronCallbackStore(root, lambda value: seen.append(value))
    store.publish(_observation())
    root.rename(tmp_path / "callbacks-original")
    root.mkdir(mode=0o700)

    with pytest.raises(CronCallbackConflictError, match="binding"):
        await store.invoke_once(_observation())
    assert seen == []


async def test_root_substitution_after_invoking_never_reinvokes_observer(
    tmp_path: Path,
) -> None:
    calls = 0

    async def cancel(_value):
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError

    root = tmp_path / "callbacks"
    store = CronCallbackStore(root, cancel)
    with pytest.raises(asyncio.CancelledError):
        await store.invoke_once(_observation())
    root.rename(tmp_path / "callbacks-original")
    root.mkdir(mode=0o700)

    with pytest.raises(CronCallbackConflictError, match="binding"):
        await store.invoke_once(_observation())
    assert calls == 1


async def test_root_substitution_at_invoking_boundary_prevents_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []
    root = tmp_path / "callbacks"
    store = CronCallbackStore(root, lambda value: seen.append(value))
    transition = store._transition

    def substitute(callback_id, expected, state, failure_code=None):
        result = transition(callback_id, expected, state, failure_code)
        if state == "invoking":
            root.rename(tmp_path / "callbacks-original")
            root.mkdir(mode=0o700)
        return result

    monkeypatch.setattr(store, "_transition", substitute)
    with pytest.raises(CronCallbackConflictError, match="binding"):
        await store.invoke_once(_observation())
    assert seen == []


@pytest.mark.parametrize("entry_kind", ["symlink", "hardlink"])
def test_lock_swap_fails_closed(tmp_path: Path, entry_kind: str) -> None:
    root = tmp_path / "callbacks"
    store = CronCallbackStore(root, lambda _value: None)
    original = root / ".lock-original"
    (root / ".lock").rename(original)
    if entry_kind == "symlink":
        (root / ".lock").symlink_to(original)
    else:
        os.link(original, root / ".lock")

    with pytest.raises(CronCallbackConflictError, match="lock|private"):
        store.publish(_observation())


@pytest.mark.parametrize("entry_kind", ["symlink", "hardlink"])
def test_record_swap_fails_closed(tmp_path: Path, entry_kind: str) -> None:
    root = tmp_path / "callbacks"
    store = CronCallbackStore(root, lambda _value: None)
    record = store.publish(_observation())
    path = root / f"{record.callback_id}.json"
    original = root / f"{record.callback_id}.original"
    path.rename(original)
    if entry_kind == "symlink":
        path.symlink_to(original)
    else:
        os.link(original, path)

    with pytest.raises(CronCallbackConflictError, match="record|private"):
        store.get(record.callback_id)


@pytest.mark.parametrize(
    "mutation", ["rename", "unlink", "replace", "chmod", "hardlink"],
)
async def test_record_mutation_after_open_never_recreates_or_reinvokes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    seen = []
    root = tmp_path / "callbacks"
    store = CronCallbackStore(root, lambda value: seen.append(value))
    succeeded = await store.invoke_once(_observation())
    path = root / f"{succeeded.callback_id}.json"
    encoded = path.read_bytes()
    real_read = os.read
    mutated = False

    def mutate_after_open(fd: int, size: int) -> bytes:
        nonlocal mutated
        data = real_read(fd, size)
        if data and not mutated:
            mutated = True
            if mutation == "rename":
                path.rename(root / "record-renamed")
            elif mutation == "unlink":
                path.unlink()
            elif mutation == "replace":
                path.rename(root / "record-replaced")
                path.write_bytes(encoded)
                path.chmod(0o600)
            elif mutation == "chmod":
                path.chmod(0o644)
            else:
                os.link(path, root / "record-hardlink")
        return data

    monkeypatch.setattr(os, "read", mutate_after_open)
    with pytest.raises(CronCallbackConflictError, match="record|private"):
        await store.invoke_once(_observation())
    assert mutated
    assert seen == [_observation()]
