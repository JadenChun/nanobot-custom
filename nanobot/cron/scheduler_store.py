"""Private cursor authority and continuously held POSIX scheduler lease.

No definition, execution, completion or delivery authority is persisted here.
The descriptor-chain and short-lock methods are the existing P3 primitive;
state publication has its own strict, bounded schema and CAS.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
import weakref
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generator,
    Literal,
    NoReturn,
    NotRequired,
    Protocol,
    SupportsIndex,
    TypedDict,
    cast,
)

from nanobot.cron.action_store import (
    decode_pinned_scheduler_state_json,
    decode_scheduler_state_json,
)
from nanobot.cron.definition_store import ScheduleCommit
from nanobot.cron.occurrence_store import (
    CronOccurrenceStore,
    OccurrenceConflictError,
    OccurrenceCorruptError,
    OccurrenceNotFoundError,
    OccurrenceRecord,
    TriggeredOccurrenceIdentity,
    scheduler_bridge_identity_conflicts,
)
from nanobot.cron.schedule import is_fire

MAX_STATE_BYTES = 1_048_576
MAX_CURSORS = 1024
MAX_CLASSIFICATIONS = 4096
MAX_EXCLUSIONS = 4096
MAX_REQUESTS = 4096
MAX_BRIDGES = 4096
_LEASE_TOKEN = object()
_SCHEDULER_OWNER_PERMIT = object()
_LEASES: weakref.WeakSet[SchedulerLease] = weakref.WeakSet()


def require_scheduler_owner_permit(permit: object) -> None:
    if permit is not _SCHEDULER_OWNER_PERMIT:
        raise TypeError("scheduler owner collaboration requires internal permit")


class SchedulerHeldRoot(Protocol):
    @property
    def root_fd(self) -> int: ...

    def assert_bound(self) -> None: ...


class SchedulerRootBinding(SchedulerHeldRoot, Protocol):
    def close(self) -> None: ...


def _inode(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _private(info: os.stat_result, *, directory: bool, links: int = 1) -> None:
    """Reject non-private, foreign, linked, or wrong-type filesystem objects."""
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not expected(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
        or (not directory and info.st_nlink != links)
    ):
        raise OccurrenceCorruptError("occurrence ledger object is not private or has an unexpected type")


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError, RecursionError) as exc:
        raise OccurrenceConflictError("ledger values must be finite JSON data") from exc


def _strict_json_loads(data: bytes, *, label: Literal["scheduler state", "pinned scheduler state"]) -> object:
    if label == "pinned scheduler state":
        return decode_pinned_scheduler_state_json(data)
    return decode_scheduler_state_json(data)


class TriggerInputs(TypedDict):
    job_id: str
    job_instance_id: str
    definition_revision: str


class TriggerRequest(TypedDict):
    source: Literal["manual", "system"]
    controller: str | None
    request_id: str
    inputs: TriggerInputs
    clock_ms: int
    sequence: int
    allocation_sequence: int


class ObservedOccurrence(TypedDict):
    occurrence_id: str
    fire: int
    owner: str
    generation: int
    state: Literal["claimed", "completed"]


class ScheduleClassification(TypedDict):
    key: str
    revision: int
    predecessor_revision: int
    lower: int
    upper: int
    kind: Literal["missed", "claimed", "observed"]
    exclusions: list[ObservedOccurrence]


class ScheduleCursor(TypedDict):
    key: str
    job_id: str
    commit: ScheduleCommit
    revision: int
    boundary: int
    next_fire: int | None


class ReconciliationBridge(TypedDict):
    occurrence_id: str
    identity: dict[str, object]
    immutable_digest: str
    predecessor_owner: str
    predecessor_generation: int
    successor_owner: str
    successor_generation: int
    resolution: Literal["confirmed_not_started", "confirmed_compensated"]
    external_evidence: str
    publication_generation: int


class SchedulerState(TypedDict):
    kind: Literal["cron_scheduler_state_v1", "cron_scheduler_state_v2"]
    schema_version: int
    namespace: str
    runtime_identity: str
    generation: int
    cursors: list[ScheduleCursor]
    classifications: list[ScheduleClassification]
    requests: list[TriggerRequest]
    bridges: NotRequired[list[ReconciliationBridge]]


class SchedulerConflictError(RuntimeError):
    """Exact predecessor, lease or retained evidence conflicts."""


def _check_scheduler_file(root_fd: int, name: str, fd: int, inode: tuple[int, int], content: bytes | None = None) -> None:
    before = os.fstat(fd)
    entry = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    _private(before, directory=False)
    _private(entry, directory=False)
    if _inode(before) != inode or _inode(entry) != inode:
        raise OccurrenceCorruptError("scheduler file binding changed")
    if content is not None and (before.st_size != len(content) or os.pread(fd, len(content) + 1, 0) != content):
        raise OccurrenceCorruptError("scheduler file content changed")


def _validate_scheduler_entries(root_fd: int) -> None:
    names = os.listdir(root_fd)
    if len(names) > 4 or set(names) - {".lock", ".lease", "state.json"}:
        raise OccurrenceCorruptError("scheduler unknown entry or staging residue")
    for name in names:
        _private(os.stat(name, dir_fd=root_fd, follow_symlinks=False), directory=False)


class _PrivateFiles(CronOccurrenceStore):
    """Reuse P3 descriptor-chain primitives without exposing occurrence methods."""

    def initialize_scheduler(self, permit: object, store: SchedulerStore) -> None:
        require_scheduler_owner_permit(permit)
        with self._held_lock() as held:
            SchedulerStore.initialize_scheduler_held(store, permit, held)

    def read_scheduler(self, permit: object, store: SchedulerStore) -> SchedulerState:
        require_scheduler_owner_permit(permit)
        with self._held_lock() as held:
            return SchedulerStore.read_scheduler_held(store, permit, held)

    def read_scheduler_acquisition(self, permit: object, store: SchedulerStore,
                                   lease: SchedulerLease, *, refresh: bool) -> SchedulerState:
        require_scheduler_owner_permit(permit)
        with self._held_lock() as held:
            return SchedulerStore.read_acquisition_scheduler_held(store, permit, held, lease, refresh=refresh)

    def compare_scheduler_state(self, permit: object, store: SchedulerStore,
                                expected: SchedulerState, successor: SchedulerState) -> SchedulerState:
        require_scheduler_owner_permit(permit)
        with self._held_lock() as held:
            return SchedulerStore.compare_scheduler_held(store, permit, held, expected, successor)

    def write_scheduler_staging(self, permit: object, fd: int, data: bytes) -> None:
        require_scheduler_owner_permit(permit)
        CronOccurrenceStore._write_all(fd, data)

    def acquire_scheduler_lease(self, permit: object, store: SchedulerStore) -> SchedulerLease:
        require_scheduler_owner_permit(permit)
        binding = self._open_root_binding()
        fd = None
        try:
            flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
            try:
                fd = os.open(".lease", flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=binding.root_fd)
                os.fchmod(fd, 0o600)
            except FileExistsError:
                fd = os.open(".lease", flags, dir_fd=binding.root_fd)
            inode = _inode(os.fstat(fd))
            SchedulerStore.check_scheduler_lease_file(store, permit, binding, fd, inode)
            os.fsync(fd)
            os.fsync(binding.root_fd)
            binding.assert_bound()
            SchedulerStore.check_scheduler_lease_file(store, permit, binding, fd, inode)
            try:
                CronOccurrenceStore.acquire_scheduler_permanent_lock(permit, fd)
            except BlockingIOError as exc:
                raise SchedulerConflictError("scheduler lease busy") from exc
            return SchedulerStore.install_scheduler_lease(store, permit, binding, fd)
        except BaseException:
            if fd is not None:
                os.close(fd)
            binding.close()
            raise

    @staticmethod
    def read_existing_scheduler_evidence(
        root: Path, validate_content: Callable[[object], SchedulerState]
    ) -> SchedulerState:
        files = _PrivateFiles.__new__(_PrivateFiles)
        files.root, files._root_identity, files._lock_identity = root, None, None
        with files._root_binding() as held:
            files._root_identity = _inode(os.fstat(held.root_fd))
            held.assert_bound()
            _validate_scheduler_entries(held.root_fd)
            try:
                fd = os.open("state.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=held.root_fd)
            except FileNotFoundError as exc:
                raise OccurrenceNotFoundError("scheduler state missing") from exc
            try:
                info = os.fstat(fd)
                inode = _inode(info)
                _check_scheduler_file(held.root_fd, "state.json", fd, inode)
                if not 0 < info.st_size <= MAX_STATE_BYTES:
                    raise OccurrenceCorruptError("scheduler state size invalid")
                data = b""
                while len(data) <= MAX_STATE_BYTES:
                    part = os.read(fd, min(65536, MAX_STATE_BYTES + 1 - len(data)))
                    if not part:
                        break
                    data += part
                if len(data) != info.st_size or os.fstat(fd) != info:
                    raise OccurrenceCorruptError("scheduler state mutated during read")
                _check_scheduler_file(held.root_fd, "state.json", fd, inode, data)
                wire = _strict_json_loads(data, label="scheduler state")
                wire = _fields(wire, {"state", "integrity"})
                if wire["integrity"] != hashlib.sha256(_canonical_json(wire["state"]).encode()).hexdigest():
                    raise OccurrenceCorruptError("scheduler state integrity mismatch")
                value = validate_content(wire["state"])
                held.assert_bound()
                _check_scheduler_file(held.root_fd, "state.json", fd, inode, data)
            finally:
                os.close(fd)
            held.assert_bound()
        return value

    def __init__(self, root: Path):
        self._flock = CronOccurrenceStore._require_qualified_platform()
        if not root.is_absolute() or root == Path(root.anchor) or ".." in root.parts:
            raise ValueError("scheduler root must be lexical absolute and non-root")
        self.root = root
        self._root_identity = None
        self._lock_identity = None
        with self._root_binding(create=True) as binding:
            self._root_identity = _inode(os.fstat(binding.root_fd))
        with self._held_lock():
            pass


class SchedulerLease:
    """Sealed process-local capability; permanent lock file is never unlinked."""

    def __init__(self, token: object, store: SchedulerStore, binding: SchedulerRootBinding, fd: int):
        if token is not _LEASE_TOKEN:
            raise TypeError("lease can be issued only by scheduler store")
        self._store, self._binding, self._fd = store, binding, fd
        self._pid, self._inode = os.getpid(), _inode(os.fstat(fd))
        self._active = True
        self._acquisition_read = False
        _LEASES.add(self)

    @property
    def is_active(self) -> bool:
        """Observe the local flag only; admission still requires reprove()."""
        return self._active

    def reprove(self) -> None:
        if not self._active or self._pid != os.getpid() or not SchedulerStore.matches_issued_scheduler_lease(self._store, self):
            raise SchedulerConflictError("scheduler lease is inactive or foreign")
        self._binding.assert_bound()
        SchedulerStore.check_scheduler_lease_file(self._store, _SCHEDULER_OWNER_PERMIT, self._binding, self._fd, self._inode)

    def release(self) -> None:
        if not self._active or self._pid != os.getpid() or not SchedulerStore.matches_issued_scheduler_lease(self._store, self):
            raise SchedulerConflictError("scheduler lease is inactive or foreign")
        error: BaseException | None = None
        try:
            self.reprove()
        except BaseException as exc:
            error = exc
        try:
            # The runtime calls release only after owned effects have joined.
            # A binding attack still reports failure, but cannot leak the exact
            # original process descriptor indefinitely or unlink its evidence.
            CronOccurrenceStore.release_scheduler_permanent_lock(_SCHEDULER_OWNER_PERMIT, self._fd)
        finally:
            os.close(self._fd)
            self._binding.close()
            self._active = False
            SchedulerStore.retire_scheduler_lease(self._store, _SCHEDULER_OWNER_PERMIT)
            _LEASES.discard(self)
        if error is not None:
            raise error

    def consume_acquisition_refresh(self, permit: object, store: SchedulerStore) -> bool:
        require_scheduler_owner_permit(permit)
        if self._store is not store:
            raise SchedulerConflictError("scheduler acquisition lease foreign")
        refresh = not self._acquisition_read
        self._acquisition_read = True
        return refresh

    def close_inherited_scheduler_descriptors(self) -> None:
        os.close(self._fd)
        self._binding.close()
        self._active = False

    def __reduce_ex__(self, _protocol: SupportsIndex) -> NoReturn:
        raise TypeError("scheduler lease cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("scheduler lease cannot be copied")

    def __deepcopy__(self, _memo: object) -> NoReturn:
        raise TypeError("scheduler lease cannot be copied")


def _fork_child() -> None:
    for lease in tuple(_LEASES):
        # Do NOT LOCK_UN: flock ownership is shared with the parent descriptor.
        SchedulerLease.close_inherited_scheduler_descriptors(lease)
    _LEASES.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_fork_child)


def _exact_int(value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise OccurrenceCorruptError("scheduler integer invalid")
    return value


def _fields(value: object, names: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(cast(dict[object, object], value)) != names:
        raise OccurrenceCorruptError("scheduler schema invalid")
    return cast(dict[str, Any], value)  # Exact string-key field set checked above.


def immutable_occurrence_digest(record: OccurrenceRecord) -> str:
    """Reprove the frozen definition and delivery binding, without effect authority."""
    from nanobot.cron.definition_store import CronDefinition
    from nanobot.cron.occurrence import scheduler_destination_policy_conflicts

    definition = CronDefinition.decode_wire(record.job_snapshot)
    if (definition.job_id != record.identity.job_id
            or definition.definition_revision != record.identity.job_revision
            or scheduler_destination_policy_conflicts(definition, record)):
        raise SchedulerConflictError("scheduler occurrence immutable inputs conflict")
    return hashlib.sha256(_canonical_json({"job_snapshot": record.job_snapshot,
        "destination_policy": record.destination_policy}).encode()).hexdigest()


def bridge_reconciliation(bridge: ReconciliationBridge) -> dict[str, object]:
    return {"resolution": bridge["resolution"], "replaced_owner": bridge["predecessor_owner"],
            "replaced_generation": bridge["predecessor_generation"]}


def matches_bridge_reconciliation(record: OccurrenceRecord, bridge: ReconciliationBridge) -> bool:
    # Canonical comparison distinguishes True/1 and 1.0/1 in the existing
    # occurrence mapping seam; Python dictionary equality does not.
    return _canonical_json(record.reconciliation) == _canonical_json(bridge_reconciliation(bridge))


def prove_trigger_request(state: SchedulerState, record: OccurrenceRecord) -> None:
    if not isinstance(record.identity, TriggeredOccurrenceIdentity):
        return
    identity = record.identity
    request = next((item for item in state["requests"] if item["source"] == identity.source
        and item["controller"] == identity.controller and item["sequence"] == identity.sequence), None)
    if request is None or request["clock_ms"] != identity.scheduled_fire_time_ms or request["inputs"] != {
            "job_id": identity.job_id, "job_instance_id": record.job_snapshot["jobInstanceId"],
            "definition_revision": identity.job_revision}:
        raise SchedulerConflictError("reconciliation trigger allocation binding conflict")


def prove_bridge_record(state: SchedulerState, record: OccurrenceRecord) -> list[ReconciliationBridge]:
    bridges = [item for item in state.get("bridges", []) if item["occurrence_id"] == record.occurrence_id]
    digest = immutable_occurrence_digest(record)
    prove_trigger_request(state, record)
    if bridges:
        final = bridges[-1]
        if (any(item["identity"] != record.identity.as_dict() or item["immutable_digest"] != digest for item in bridges)
                or (final["successor_owner"], final["successor_generation"]) != (record.owner, record.generation)
                or not matches_bridge_reconciliation(record, final)):
            raise SchedulerConflictError("scheduler reconciliation bridge conflicts with occurrence")
    return bridges


def prove_observation(state: SchedulerState, evidence: ObservedOccurrence, record: OccurrenceRecord) -> None:
    bridges = prove_bridge_record(state, record)
    if evidence["state"] == "completed" and record.state != "completed":
        raise SchedulerConflictError("scheduler completed observation became claimed")
    owner, generation = evidence["owner"], evidence["generation"]
    if (owner, generation) == (record.owner, record.generation):
        return
    for bridge in bridges:
        if bridge["predecessor_generation"] < generation:
            continue
        if (bridge["predecessor_owner"], bridge["predecessor_generation"]) != (owner, generation):
            break
        if evidence["state"] == "completed":
            break  # Completed observations cannot be reclaimed.
        owner, generation = bridge["successor_owner"], bridge["successor_generation"]
    if (owner, generation) != (record.owner, record.generation):
        raise SchedulerConflictError("scheduler observed claim absent or conflicting")


def verify_occurrence_evidence(state: SchedulerState, records: list[OccurrenceRecord]) -> None:
    by_id = {record.occurrence_id: record for record in records}
    for bridge in state.get("bridges", []):
        record = by_id.get(bridge["occurrence_id"])
        if record is None:
            raise SchedulerConflictError("scheduler bridge occurrence absent")
        prove_bridge_record(state, record)
    for cursor in state["cursors"]:
        commit = cursor["commit"]
        for classification in (item for item in state["classifications"] if item["key"] == cursor["key"]):
            excluded = {item["occurrence_id"] for item in classification["exclusions"]}
            for evidence in classification["exclusions"]:
                record = by_id.get(evidence["occurrence_id"])
                if record is None or isinstance(record.identity, TriggeredOccurrenceIdentity) or record.identity.store_namespace != state["namespace"] or record.identity.job_id != cursor["job_id"] or record.identity.job_revision != commit["definition_revision"] or record.job_snapshot.get("jobInstanceId") != commit["job_instance_id"] or record.identity.scheduled_fire_time_ms != evidence["fire"] or record.job_snapshot.get("schedule") != commit["schedule"]:
                    raise SchedulerConflictError("scheduler observed claim absent or conflicting")
                prove_observation(state, evidence, record)
            for record in records:
                if isinstance(record.identity, TriggeredOccurrenceIdentity):
                    continue
                fire = record.identity.scheduled_fire_time_ms
                if record.identity.job_revision == commit["definition_revision"] and record.job_snapshot.get("jobInstanceId") == commit["job_instance_id"] and classification["lower"] < fire <= classification["upper"] and is_fire(commit["schedule"], commit["commit_clock_ms"], fire) and record.occurrence_id not in excluded:
                    raise SchedulerConflictError("independent claim intersects committed missed evidence")


class SchedulerStore:
    @staticmethod
    def inspect_existing_evidence(root: Path, *, namespace: str, runtime_identity: str) -> dict[str, object]:
        root, namespace = CronOccurrenceStore.admit_existing_inspection(root, namespace=namespace)
        store = SchedulerStore.__new__(SchedulerStore)
        store.namespace = namespace
        store.runtime_identity = runtime_identity
        state = _PrivateFiles.read_existing_scheduler_evidence(root, lambda content: store._validate(content))
        return {"classification": "scheduler_evidence", "generation": state["generation"],
                "cursors": [{"key": item["key"], "revision": item["revision"],
                             "boundary": item["boundary"], "next_fire": item["next_fire"]}
                            for item in state["cursors"]],
                "classifications": state["classifications"],
                "bridges": state.get("bridges", []),
                "request_count": len(state["requests"])}

    def __init__(self, root: Path, *, namespace: str, runtime_identity: str):
        self.namespace = CronOccurrenceStore.validate_scheduler_namespace(namespace)
        if not isinstance(cast(object, runtime_identity), str) or not runtime_identity:
            raise ValueError("runtime identity required")
        self.runtime_identity = runtime_identity
        self._files = _PrivateFiles(Path(root))
        self.root = self._files.root
        self._lease: SchedulerLease | None = None
        self._state_inode: tuple[int, int] | None = None
        self._state_content: bytes | None = None
        self._allocated_requests: dict[tuple[str, str | None, str], TriggerRequest] = {}
        _PrivateFiles.initialize_scheduler(self._files, _SCHEDULER_OWNER_PERMIT, self)

    def matches_issued_scheduler_lease(self, lease: SchedulerLease) -> bool:
        return self._lease is lease

    def check_scheduler_lease_file(self, permit: object, binding: SchedulerRootBinding,
                                  fd: int, inode: tuple[int, int]) -> None:
        require_scheduler_owner_permit(permit)
        self._check_file(binding.root_fd, ".lease", fd, inode)

    def install_scheduler_lease(self, permit: object, binding: SchedulerRootBinding,
                               fd: int) -> SchedulerLease:
        require_scheduler_owner_permit(permit)
        self._lease = SchedulerLease(_LEASE_TOKEN, self, binding, fd)
        self._lease.reprove()
        return self._lease

    def retire_scheduler_lease(self, permit: object) -> None:
        require_scheduler_owner_permit(permit)
        self._lease = None

    def initialize_scheduler_held(self, permit: object, held: SchedulerHeldRoot) -> None:
        require_scheduler_owner_permit(permit)
        self._validate_entries(held.root_fd)
        try:
            self._read(held)
        except OccurrenceNotFoundError:
            self._publish(held, self._empty(), create=True)

    def read_scheduler_held(self, permit: object, held: SchedulerHeldRoot) -> SchedulerState:
        require_scheduler_owner_permit(permit)
        return self._read(held)

    def read_acquisition_scheduler_held(self, permit: object, held: SchedulerHeldRoot,
                                      lease: SchedulerLease, *, refresh: bool) -> SchedulerState:
        require_scheduler_owner_permit(permit)
        lease.reprove()
        state = self._read(held, reconciliation_refresh=refresh)
        lease.reprove()
        return state

    def _empty(self) -> SchedulerState:
        return {"kind": "cron_scheduler_state_v1", "schema_version": 1,
                "namespace": self.namespace, "runtime_identity": self.runtime_identity,
                "generation": 0, "cursors": [], "classifications": [], "requests": []}

    @staticmethod
    def _check_file(root_fd: int, name: str, fd: int, inode: tuple[int, int], content: bytes | None = None) -> None:
        _check_scheduler_file(root_fd, name, fd, inode, content)

    def _validate_entries(self, root_fd: int) -> None:
        _validate_scheduler_entries(root_fd)

    def _validate(self, state: object) -> SchedulerState:
        from nanobot.cron.definition_store import parse_persisted_scheduler_commit
        from nanobot.cron.schedule import is_fire, next_fire

        v2 = isinstance(state, dict) and cast(dict[object, object], state).get("kind") == "cron_scheduler_state_v2"
        value = _fields(cast(object, state), set(self._empty()) | ({"bridges"} if v2 else set()))
        if value["kind"] != ("cron_scheduler_state_v2" if v2 else "cron_scheduler_state_v1") or type(value["schema_version"]) is not int or value["schema_version"] != (2 if v2 else 1) or value["namespace"] != self.namespace or value["runtime_identity"] != self.runtime_identity:
            raise OccurrenceCorruptError("scheduler authority binding mismatch")
        _exact_int(value["generation"])
        for name, limit in (("cursors", MAX_CURSORS), ("classifications", MAX_CLASSIFICATIONS), ("requests", MAX_REQUESTS)):
            if not isinstance(value[name], list) or len(value[name]) > limit:
                raise OccurrenceCorruptError("scheduler evidence bound exceeded")
        keys: set[str] = set()
        for cursor in value["cursors"]:
            _fields(cursor, {"key", "job_id", "commit", "revision", "boundary", "next_fire"})
            commit = parse_persisted_scheduler_commit(cursor["commit"])
            key = commit["job_instance_id"] + ":" + commit["definition_revision"]
            if key != cursor["key"] or key in keys or commit["store_namespace"] != self.namespace:
                raise OccurrenceCorruptError("scheduler cursor commit binding invalid")
            keys.add(key)
            _exact_int(cursor["revision"], 1)
            _exact_int(cursor["boundary"], -1)
            if cursor["next_fire"] != next_fire(commit["schedule"], commit["commit_clock_ms"], cursor["boundary"]):
                raise OccurrenceCorruptError("scheduler successor does not match exact schedule")
            history = [item for item in value["classifications"] if item.get("key") == key]
            if len(history) != cursor["revision"] or not history or history[-1]["upper"] != cursor["boundary"]:
                raise OccurrenceCorruptError("scheduler cursor lost predecessor classification")
            prior = None
            for revision, item in enumerate(history, 1):
                _fields(item, {"key", "revision", "predecessor_revision", "lower", "upper", "kind", "exclusions"})
                if item["revision"] != revision or type(item["revision"]) is not int or item["predecessor_revision"] != revision - 1 or type(item["predecessor_revision"]) is not int:
                    raise OccurrenceCorruptError("scheduler classification revision mismatch")
                _exact_int(item["lower"], -1)
                _exact_int(item["upper"], -1)
                if item["upper"] < item["lower"] or (prior is not None and item["lower"] != prior):
                    raise OccurrenceCorruptError("scheduler classification boundary mismatch")
                prior = item["upper"]
                if item["kind"] not in {"missed", "claimed", "observed"} or not isinstance(item["exclusions"], list) or len(item["exclusions"]) > MAX_EXCLUSIONS:
                    raise OccurrenceCorruptError("scheduler classification invalid")
                ids: set[str] = set()
                for exclusion in item["exclusions"]:
                    _fields(exclusion, {"occurrence_id", "fire", "owner", "generation", "state"})
                    if exclusion["occurrence_id"] in ids or not isinstance(exclusion["occurrence_id"], str) or len(exclusion["occurrence_id"]) != 64 or any(char not in "0123456789abcdef" for char in exclusion["occurrence_id"]):
                        raise OccurrenceCorruptError("scheduler exclusion identity invalid")
                    ids.add(exclusion["occurrence_id"])
                    _exact_int(exclusion["fire"])
                    _exact_int(exclusion["generation"], 1)
                    if not isinstance(exclusion["owner"], str) or not exclusion["owner"] or exclusion["state"] not in {"claimed", "completed"} or not item["lower"] < exclusion["fire"] <= item["upper"] or not is_fire(commit["schedule"], commit["commit_clock_ms"], exclusion["fire"]):
                        raise OccurrenceCorruptError("scheduler exclusion not in represented fire set")
                if item["kind"] != "missed" and (len(item["exclusions"]) != 1 or next_fire(commit["schedule"], commit["commit_clock_ms"], item["lower"]) != item["upper"]):
                    raise OccurrenceCorruptError("scheduler claimed classification not exact")
        if any(item.get("key") not in keys for item in value["classifications"]):
            raise OccurrenceCorruptError("scheduler orphan classification")
        request_keys: set[tuple[str, str | None, str]] = set()
        sequences: dict[tuple[str, str | None], int] = {}
        for ordinal, request in enumerate(value["requests"], 1):
            _fields(request, {"source", "controller", "request_id", "inputs", "clock_ms", "sequence", "allocation_sequence"})
            key = (request["source"], request["controller"], request["request_id"])
            domain = key[:2]
            expected_sequence = sequences.get(domain, 0) + 1
            if key in request_keys or request["source"] not in {"manual", "system"} or not isinstance(request["request_id"], str) or not request["request_id"] or len(request["request_id"].encode()) > 512 or request["sequence"] != expected_sequence or type(request["sequence"]) is not int or type(request["allocation_sequence"]) is not int or request["allocation_sequence"] != ordinal:
                raise OccurrenceCorruptError("scheduler request binding invalid")
            if (request["source"] == "manual" and request["controller"] is not None) or (request["source"] == "system" and (not isinstance(request["controller"], str) or not request["controller"].strip())):
                raise OccurrenceCorruptError("scheduler controller binding invalid")
            _fields(request["inputs"], {"job_id", "job_instance_id", "definition_revision"})
            if not isinstance(request["inputs"]["job_id"], str) or not request["inputs"]["job_id"]:
                raise OccurrenceCorruptError("scheduler request job invalid")
            for name in ("job_instance_id", "definition_revision"):
                digest = request["inputs"][name]
                if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                    raise OccurrenceCorruptError("scheduler request definition token invalid")
            _exact_int(request["clock_ms"])
            request_keys.add(key)
            sequences[domain] = expected_sequence
        if v2:
            if not isinstance(value["bridges"], list) or not 0 < len(cast(list[object], value["bridges"])) <= MAX_BRIDGES:
                raise OccurrenceCorruptError("scheduler bridge bound exceeded")
            tails: dict[str, dict[str, Any]] = {}
            publication = 0
            for raw_bridge in cast(list[object], value["bridges"]):
                bridge = _fields(raw_bridge, {"occurrence_id", "identity", "immutable_digest", "predecessor_owner",
                    "predecessor_generation", "successor_owner", "successor_generation", "resolution",
                    "external_evidence", "publication_generation"})
                identity = CronOccurrenceStore.parse_persisted_identity(bridge["identity"])
                _exact_int(identity.scheduled_fire_time_ms)
                if scheduler_bridge_identity_conflicts(identity, self.namespace, bridge):
                    raise OccurrenceCorruptError("scheduler bridge identity mismatch")
                digest = bridge["immutable_digest"]
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise OccurrenceCorruptError("scheduler bridge immutable digest invalid")
                for name in ("predecessor_owner", "successor_owner"):
                    if not isinstance(bridge[name], str) or not bridge[name] or len(bridge[name].encode()) > 512:
                        raise OccurrenceCorruptError("scheduler bridge owner invalid")
                predecessor = _exact_int(bridge["predecessor_generation"], 1)
                if _exact_int(bridge["successor_generation"], 2) != predecessor + 1:
                    raise OccurrenceCorruptError("scheduler bridge generation skipped")
                if bridge["resolution"] not in {"confirmed_not_started", "confirmed_compensated"} or bridge["external_evidence"] != "EXTERNAL-EVIDENCE-VERIFIED":
                    raise OccurrenceCorruptError("scheduler bridge evidence invalid")
                ordinal = _exact_int(bridge["publication_generation"], 1)
                if not publication < ordinal <= value["generation"]:
                    raise OccurrenceCorruptError("scheduler bridge publication generation invalid")
                publication = ordinal
                tail = tails.get(bridge["occurrence_id"])
                if tail is not None and (tail["successor_owner"] != bridge["predecessor_owner"]
                        or tail["successor_generation"] != predecessor or tail["identity"] != bridge["identity"]
                        or tail["immutable_digest"] != digest):
                    raise OccurrenceCorruptError("scheduler bridge history gap or fork")
                tails[bridge["occurrence_id"]] = bridge
        # Complete schema, nested commit, exact scalar and relationship checks
        # above establish this stable first-party persisted shape.
        return cast(SchedulerState, value)

    def _read(self, held: SchedulerHeldRoot, *, durable: bool = True,
              reconciliation_refresh: bool = False) -> SchedulerState:
        held.assert_bound()
        self._validate_entries(held.root_fd)
        try:
            fd = os.open("state.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=held.root_fd)
        except FileNotFoundError as exc:
            raise OccurrenceNotFoundError("scheduler state missing") from exc
        try:
            info = os.fstat(fd)
            inode = _inode(info)
            self._check_file(held.root_fd, "state.json", fd, inode)
            if not reconciliation_refresh and self._state_inode is not None and inode != self._state_inode:
                raise OccurrenceCorruptError("scheduler state replaced")
            if not 0 < info.st_size <= MAX_STATE_BYTES:
                raise OccurrenceCorruptError("scheduler state size invalid")
            data = b""
            while len(data) <= MAX_STATE_BYTES:
                part = os.read(fd, min(65536, MAX_STATE_BYTES + 1 - len(data)))
                if not part:
                    break
                data += part
            if len(data) != info.st_size or os.fstat(fd) != info:
                raise OccurrenceCorruptError("scheduler state mutated during read")
            if not reconciliation_refresh and self._state_content is not None and data != self._state_content:
                raise OccurrenceCorruptError("scheduler state content replaced")
            self._check_file(held.root_fd, "state.json", fd, inode, data)
            wire = _strict_json_loads(data, label="scheduler state")
            wire = _fields(wire, {"state", "integrity"})
            if wire["integrity"] != hashlib.sha256(_canonical_json(wire["state"]).encode()).hexdigest():
                raise OccurrenceCorruptError("scheduler state integrity mismatch")
            value = self._validate(wire["state"])
            if reconciliation_refresh and self._state_inode is not None and (inode != self._state_inode or data != self._state_content):
                self._prove_reconciliation_extension(value, inode, data)
            # Durable reproof is mandatory for every runtime admission/read.
            if durable:
                os.fsync(fd)
                held.assert_bound()
                self._check_file(held.root_fd, "state.json", fd, inode, data)
                os.fsync(held.root_fd)
            held.assert_bound()
            self._check_file(held.root_fd, "state.json", fd, inode, data)
            self._state_inode, self._state_content = inode, data
            return value
        finally:
            os.close(fd)

    def _prove_reconciliation_extension(self, value: SchedulerState, inode: tuple[int, int], data: bytes) -> None:
        if self._state_content is None:
            raise OccurrenceCorruptError("scheduler acquisition predecessor proof missing")
        if inode == self._state_inode or data == self._state_content:
            raise OccurrenceCorruptError("scheduler state replaced without reconciliation publication")
        # The pinned bytes came from _read's exact envelope validation.
        prior_wire = cast(dict[str, object], _strict_json_loads(self._state_content, label="pinned scheduler state"))
        prior = self._validate(prior_wire["state"])
        for name in ("namespace", "runtime_identity", "cursors", "classifications", "requests"):
            if value[name] != prior[name]:
                raise OccurrenceCorruptError("scheduler acquisition changed non-reconciliation evidence")
        prefix = prior.get("bridges", [])
        bridges = value.get("bridges", [])
        extension = bridges[len(prefix):]
        if (bridges[:len(prefix)] != prefix or not extension
                or value["generation"] != prior["generation"] + len(extension)
                or [item["publication_generation"] for item in extension] != list(range(prior["generation"] + 1, value["generation"] + 1))):
            raise OccurrenceCorruptError("scheduler acquisition reconciliation extension invalid")

    def read_for_acquisition(self, lease: SchedulerLease) -> SchedulerState:
        """Refresh only once on a newly acquired permanent lease, never held ownership."""
        lease.reprove()
        refresh = SchedulerLease.consume_acquisition_refresh(lease, _SCHEDULER_OWNER_PERMIT, self)
        return _PrivateFiles.read_scheduler_acquisition(self._files, _SCHEDULER_OWNER_PERMIT, self, lease, refresh=refresh)

    def prepare_bridge(self, state: SchedulerState, bridge: ReconciliationBridge) -> SchedulerState:
        """Bound/validate the entire successor before occurrence authority is mutated."""
        successor = deepcopy(state)
        successor.update(kind="cron_scheduler_state_v2", schema_version=2,
                         generation=state["generation"] + 1)
        successor["bridges"] = [*state.get("bridges", []), deepcopy(bridge)]
        self._encode(successor)
        return successor

    def _encode(self, state: SchedulerState) -> bytes:
        state = self._validate(state)
        data = _canonical_json({"state": state, "integrity": hashlib.sha256(_canonical_json(state).encode()).hexdigest()}).encode()
        if len(data) > MAX_STATE_BYTES:
            raise SchedulerConflictError("scheduler state byte bound exceeded")
        return data

    def _publish(self, held: SchedulerHeldRoot, state: SchedulerState, *, create: bool = False) -> None:
        data = self._encode(state)
        name = "state.json" if create else ".state." + uuid.uuid4().hex + ".tmp"
        held.assert_bound()
        fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=held.root_fd)
        try:
            os.fchmod(fd, 0o600)
            inode = _inode(os.fstat(fd))
            _PrivateFiles.write_scheduler_staging(self._files, _SCHEDULER_OWNER_PERMIT, fd, data)
            self._check_file(held.root_fd, name, fd, inode, data)
            os.fsync(fd)
            held.assert_bound()
            self._check_file(held.root_fd, name, fd, inode, data)
            if not create:
                predecessor_fd = os.open("state.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=held.root_fd)
                try:
                    # Non-create writes follow _read, which pins this exact inode.
                    self._check_file(held.root_fd, "state.json", predecessor_fd, cast(tuple[int, int], self._state_inode), self._state_content)
                finally:
                    os.close(predecessor_fd)
                os.rename(name, "state.json", src_dir_fd=held.root_fd, dst_dir_fd=held.root_fd)
            self._check_file(held.root_fd, "state.json", fd, inode, data)
            # Remember published evidence even when the final ack fails. An
            # exact retry can then reprove it; it cannot regress the cursor.
            self._state_inode, self._state_content = inode, data
            os.fsync(held.root_fd)
            held.assert_bound()
            self._check_file(held.root_fd, "state.json", fd, inode, data)
        finally:
            os.close(fd)
        # Failed staging is retained as forensic residue, never auto-repaired.

    def read(self) -> SchedulerState:
        return _PrivateFiles.read_scheduler(self._files, _SCHEDULER_OWNER_PERMIT, self)

    def compare_and_swap(self, expected: SchedulerState, successor: SchedulerState) -> SchedulerState:
        return _PrivateFiles.compare_scheduler_state(self._files, _SCHEDULER_OWNER_PERMIT, self, expected, successor)

    def compare_scheduler_held(self, permit: object, held: SchedulerHeldRoot,
                               expected: SchedulerState, successor: SchedulerState) -> SchedulerState:
        require_scheduler_owner_permit(permit)
        current = self._read(held)
        if current == successor:
            return current
        if current != expected or successor["generation"] != expected["generation"] + 1:
            raise SchedulerConflictError("scheduler state predecessor stale")
        # Retained proof is append-only, including first-commit cursors.
        for name in ("classifications", "requests"):
            if successor[name][:len(expected[name])] != expected[name]:
                raise SchedulerConflictError("scheduler evidence cannot be rewritten")
        if successor.get("bridges", [])[:len(expected.get("bridges", []))] != expected.get("bridges", []):
            raise SchedulerConflictError("scheduler bridge evidence cannot be rewritten")
        additions = successor.get("bridges", [])[len(expected.get("bridges", [])):]
        if additions and (len(additions) != 1 or additions[0]["publication_generation"] != successor["generation"]
                or any(successor[name] != expected[name] for name in ("cursors", "classifications", "requests"))):
            raise SchedulerConflictError("scheduler bridge publication must be reconciliation-only")
        for prior in expected["cursors"]:
            current_cursor = next((item for item in successor["cursors"] if item["key"] == prior["key"]), None)
            if current_cursor is None or any(current_cursor[name] != prior[name] for name in ("key", "job_id", "commit")):
                raise SchedulerConflictError("scheduler immutable cursor evidence changed")
        self._publish(held, successor)
        return successor

    def acquire_lease(self) -> SchedulerLease:
        if self._lease is not None:
            self._lease.reprove()
            return self._lease
        return _PrivateFiles.acquire_scheduler_lease(self._files, _SCHEDULER_OWNER_PERMIT, self)

    @contextmanager
    def stopped_writer(self) -> Generator[SchedulerLease, None, None]:
        """Exclusive stopped-writer scope; no timer graph can own this store."""
        if self._lease is not None:
            raise SchedulerConflictError("scheduler owner active")
        lease = self.acquire_lease()
        try:
            yield lease
        finally:
            lease.release()

    def allocate_request(self, *, source: Literal["manual", "system"], controller: str | None, request_id: str,
                         inputs: TriggerInputs, clock_ms: int) -> TriggerRequest:
        _exact_int(clock_ms)
        state = self.read()
        key = (source, controller, request_id)
        for request in state["requests"]:
            if (request["source"], request["controller"], request["request_id"]) == (source, controller, request_id):
                if request["inputs"] != inputs or request["clock_ms"] != clock_ms:
                    raise SchedulerConflictError("authenticated request inputs changed")
                if key not in self._allocated_requests or self._allocated_requests[key] != request:
                    raise SchedulerConflictError("retained request has no process-local pre-effect authority")
                return request
        request: TriggerRequest = {"source": source, "controller": controller, "request_id": request_id,
                   "inputs": inputs, "clock_ms": clock_ms,
                   "sequence": 1 + sum(item["source"] == source and item["controller"] == controller for item in state["requests"]),
                   "allocation_sequence": len(state["requests"]) + 1}
        successor: SchedulerState = {**state, "generation": state["generation"] + 1, "requests": [*state["requests"], request]}
        self._allocated_requests[key] = request
        self.compare_and_swap(state, successor)
        return request
