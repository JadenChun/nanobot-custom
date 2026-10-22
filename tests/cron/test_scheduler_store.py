import multiprocessing
import os
import shutil

import pytest

from nanobot.cron.occurrence_store import OccurrenceCorruptError
from nanobot.cron.scheduler_store import SchedulerConflictError, SchedulerStore
from tests.cron.stage3_fixtures import POSIX_FORK, offline_denial  # noqa: F401


def store(tmp_path):
    return SchedulerStore(tmp_path / "scheduler", namespace="stage3", runtime_identity="copied")


def test_exact_cas_and_request_sequence_retry(tmp_path):
    authority = store(tmp_path)
    inputs = {"job_id": "j", "job_instance_id": "a" * 64, "definition_revision": "b" * 64}
    first = authority.allocate_request(source="manual", controller=None, request_id="a", inputs=inputs, clock_ms=1234)
    with pytest.raises(SchedulerConflictError):
        authority.allocate_request(source="manual", controller=None, request_id="a", inputs=inputs, clock_ms=9999)
    retry = authority.allocate_request(source="manual", controller=None, request_id="a", inputs=inputs, clock_ms=1234)
    second = authority.allocate_request(source="manual", controller=None, request_id="b", inputs=inputs, clock_ms=1234)
    assert first == retry and first["sequence"] == 1 and second["sequence"] == 2
    with pytest.raises(SchedulerConflictError):
        authority.allocate_request(source="manual", controller=None, request_id="a", inputs={**inputs, "job_id": "changed"}, clock_ms=1234)
    with pytest.raises(SchedulerConflictError):
        authority.compare_and_swap({**authority.read(), "generation": 0}, {**authority.read(), "generation": 3})
    assert store(tmp_path).read()["requests"] == [first, second]


def test_permanent_lease_contention_and_root_swap(tmp_path):
    first = store(tmp_path)
    second = store(tmp_path)
    lease = first.acquire_lease()
    with pytest.raises(SchedulerConflictError, match="busy"):
        second.acquire_lease()
    inode = (first.root / ".lease").stat().st_ino
    lease.release()
    other = second.acquire_lease()
    assert (first.root / ".lease").stat().st_ino == inode
    second.root.rename(tmp_path / "retained")
    (tmp_path / "scheduler").mkdir(mode=0o700)
    with pytest.raises(OccurrenceCorruptError):
        other.reprove()
    # Restore exact path for a validated release, never discard evidence.
    (tmp_path / "scheduler").rmdir()
    (tmp_path / "retained").rename(tmp_path / "scheduler")
    other.release()


@pytest.mark.parametrize("attack", ["hardlink", "mode", "replace", "residue", "symlink"])
def test_unsafe_state_fails_closed(tmp_path, attack):
    authority = store(tmp_path)
    state = authority.root / "state.json"
    if attack == "hardlink":
        os.link(state, tmp_path / "extra-link")
    elif attack == "mode":
        state.chmod(0o644)
    elif attack == "replace":
        copy = authority.root / "copy"
        shutil.copyfile(state, copy)
        copy.chmod(0o600)
        os.replace(copy, state)
    elif attack == "residue":
        (authority.root / ".state.residue.tmp").touch(mode=0o600)
    else:
        state.rename(tmp_path / "retained-state")
        state.symlink_to(tmp_path / "retained-state")
    with pytest.raises((OccurrenceCorruptError, OSError)):
        authority.read()


def test_two_process_contention_and_fork_child_cannot_release(tmp_path, monkeypatch):
    # Narrow contract exception: local POSIX worker, no external program.
    monkeypatch.setattr(os, "fork", POSIX_FORK)
    authority = store(tmp_path)
    lease = authority.acquire_lease()
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)

    def child():
        try:
            lease.release()
        except SchedulerConflictError:
            inherited_rejected = True
        else:
            inherited_rejected = False
        try:
            store(tmp_path).acquire_lease()
        except SchedulerConflictError:
            sender.send((inherited_rejected, "busy"))
        else:
            sender.send((inherited_rejected, "acquired"))
        sender.close()

    process = context.Process(target=child)
    process.start()
    sender.close()
    assert receiver.poll(5)
    assert receiver.recv() == (True, "busy")
    process.join(5)
    assert process.exitcode == 0
    lease.reprove()
    lease.release()


def test_request_bound_is_not_pruned(tmp_path, monkeypatch):
    import nanobot.cron.scheduler_store as module

    monkeypatch.setattr(module, "MAX_REQUESTS", 1)
    authority = store(tmp_path)
    inputs = {"job_id": "j", "job_instance_id": "a" * 64, "definition_revision": "b" * 64}
    authority.allocate_request(source="manual", controller=None, request_id="one", inputs=inputs, clock_ms=1)
    before = (authority.root / "state.json").read_bytes()
    with pytest.raises(OccurrenceCorruptError):
        authority.allocate_request(source="manual", controller=None, request_id="two", inputs=inputs, clock_ms=1)
    assert (authority.root / "state.json").read_bytes() == before


@pytest.mark.parametrize("source,controller", [("manual", None), ("system", "named")])
@pytest.mark.parametrize("clock", [True, 1234.0, -1])
def test_retry_clock_is_exact_integer_before_numeric_comparison(tmp_path, source, controller, clock):
    authority = store(tmp_path)
    inputs = {"job_id": "j", "job_instance_id": "a" * 64, "definition_revision": "b" * 64}
    authority.allocate_request(source=source, controller=controller, request_id="one", inputs=inputs, clock_ms=1234)
    before = (authority.root / "state.json").read_bytes()
    with pytest.raises(OccurrenceCorruptError):
        authority.allocate_request(source=source, controller=controller, request_id="one", inputs=inputs, clock_ms=clock)
    assert (authority.root / "state.json").read_bytes() == before


def test_v1_reads_publish_nothing_and_cannot_accept_bridge_field(tmp_path):
    authority = store(tmp_path)
    path = authority.root / "state.json"
    before = (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
    assert authority.read()["kind"] == "cron_scheduler_state_v1"
    assert (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes()) == before
    with pytest.raises(OccurrenceCorruptError):
        authority._validate({**authority.read(), "bridges": []})


def test_state_read_allows_filesystem_access_time_update(tmp_path):
    authority = store(tmp_path)
    path = authority.root / "state.json"
    before = path.stat()
    # A normal first read on a relatime filesystem updates atime. This is not
    # a competing writer and must not invalidate an otherwise stable snapshot.
    os.utime(path, ns=(0, before.st_mtime_ns))
    assert authority.read()["generation"] == 0
    assert path.stat().st_mtime_ns == before.st_mtime_ns
