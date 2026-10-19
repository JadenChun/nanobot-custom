"""Durable per-session mode records for the guarded P5 runtime."""

from __future__ import annotations

import asyncio
import base64
import ctypes
import errno
import hashlib
import json
import os
import secrets
import stat
import threading
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Generator,
    Literal,
    Mapping,
    NotRequired,
    ParamSpec,
    TypedDict,
    TypeVar,
    cast,
)

from nanobot.runtime_context import public_history_message
from nanobot.session.history_visibility import is_hidden_history_message
from nanobot.session.manager import JsonlSessionStore, Session, SessionStorageError
from nanobot.session.mode_storage import (
    PinnedDirectoryPath,
    SessionModeError,
    SessionModeStorage,
)
from nanobot.session.source_store import (
    SourceSessionStore,
    SourceValidationError,
    select_legacy_source_event_upgrader,
    validate_source_session_bytes,
)

_upgrade_events = select_legacy_source_event_upgrader()

if TYPE_CHECKING:
    from nanobot.session.execution_owner import ExecutionOwner as _ExecutionOwner

Mode = Literal["legacy", "canonical"]
SessionModeBinding = tuple[str, int, Mode, str, int]

_OperationOwner = TypeVar(
    "_OperationOwner", bound="SessionModeResolver | SessionExportTransaction"
)
_OperationParams = ParamSpec("_OperationParams")
_OperationResult = TypeVar("_OperationResult")


class _JournalCommon(TypedDict):
    operation: str
    registry_id: str
    source_key: str | None
    target_key: str | None
    prior_record_digest: str | None
    target_binding: dict[str, object] | None
    prior_record: NotRequired[str | None]
    target_record: NotRequired[str | None]
    input_digest: NotRequired[str | None]
    input_size: NotRequired[int | None]
    output_digest: NotRequired[str | None]
    output_size: NotRequired[int | None]
    staging_name: NotRequired[str | None]
    rollback_name: NotRequired[str | None]
    sidecar_digest: NotRequired[str | None]
    sidecar_size: NotRequired[int | None]
    sidecar_bytes: NotRequired[str | None]
    source_record: NotRequired[str | None]
    source_record_digest: NotRequired[str | None]
    source_binding: NotRequired[dict[str, object] | None]
    source_input_digest: NotRequired[str | None]
    source_input_size: NotRequired[int | None]

_MODE_ERROR = "session mode evidence is invalid; offline recovery is required"
_JOURNAL_PHASES = {
    "prepared",
    "target_installed",
    "mode_committed",
    "published",
    "complete",
    "aborted",
}
_JOURNAL_FIELDS = {
    "schema_version",
    "registry_id",
    "operation_id",
    "sequence",
    "phase",
    "operation",
    "source_key",
    "target_key",
    "prior_record_digest",
    "prior_record",
    "target_record",
    "target_binding",
    "input_digest",
    "input_size",
    "output_digest",
    "output_size",
    "staging_name",
    "rollback_name",
    "sidecar_digest",
    "sidecar_size",
    "sidecar_bytes",
    "source_record",
    "source_record_digest",
    "source_binding",
    "source_input_digest",
    "source_input_size",
    "timestamp",
    "previous_record_digest",
}
_OPERATIONS = {"bootstrap", "create", "convert", "reset", "delete", "fork", "export"}


def _locked_operation(*, migration: bool = False, execution: bool = False) -> Callable[
    [Callable[Concatenate[_OperationOwner, _OperationParams], _OperationResult]],
    Callable[Concatenate[_OperationOwner, _OperationParams], _OperationResult],
]:
    def decorate(
        method: Callable[Concatenate[_OperationOwner, _OperationParams], _OperationResult],
    ) -> Callable[Concatenate[_OperationOwner, _OperationParams], _OperationResult]:
        @wraps(method)
        def locked(
            instance: _OperationOwner,
            *args: _OperationParams.args,
            **kwargs: _OperationParams.kwargs,
        ) -> _OperationResult:
            with instance.locked_mode_operation(
                migration=migration, execution=execution, arguments=args, keywords=kwargs
            ):
                return method(instance, *args, **kwargs)

        return locked

    return decorate


@dataclass(frozen=True, slots=True)
class ResolvedSessionMode:
    state: Literal["absent", "active", "deleted", "quarantined"]
    mode: Mode | None
    registry_id: str | None
    record_revision: int
    incarnation: str | None
    generation: int | None

    @property
    def binding(self) -> SessionModeBinding | None:
        if (
            self.state != "active"
            or self.registry_id is None
            or self.mode is None
            or self.incarnation is None
            or self.generation is None
        ):
            return None
        return (
            self.registry_id,
            self.record_revision,
            self.mode,
            self.incarnation,
            self.generation,
        )


@dataclass(frozen=True, slots=True)
class LockedSessionSnapshot:
    binding: SessionModeBinding
    mode_record_bytes: bytes
    source_bytes: bytes


@dataclass(frozen=True, slots=True)
class _SourceInventory:
    key: str
    storage_key: str
    source_name: str
    source_bytes: bytes
    source_digest: str
    source_size: int
    sidecar_name: str
    sidecar_bytes: bytes | None

    @property
    def sidecar_digest(self) -> str | None:
        return _digest(self.sidecar_bytes) if self.sidecar_bytes is not None else None

    @property
    def sidecar_size(self) -> int | None:
        return len(self.sidecar_bytes) if self.sidecar_bytes is not None else None

    @property
    def sidecar_text(self) -> str | None:
        return (
            base64.b64encode(self.sidecar_bytes).decode("ascii")
            if self.sidecar_bytes is not None
            else None
        )


@dataclass(frozen=True, slots=True)
class SessionExportFilePlan:
    name: str
    digest: str
    size: int


@dataclass(frozen=True, slots=True)
class SessionExportPlan:
    destination: str
    files: tuple[SessionExportFilePlan, ...]


class SessionExportTransaction:
    """Public operation-owned export stage and publication journal."""

    def __init__(
        self,
        resolver: SessionModeResolver,
        operation_id: str,
        key: str,
        destination: Path,
        binding: dict[str, object],
        plan: SessionExportPlan,
        parent_pin: PinnedDirectoryPath,
    ) -> None:
        self._resolver = resolver
        self.operation_id = operation_id
        self.key = key
        self.destination = destination
        self.binding = dict(binding)
        self.plan = plan
        self._parent_pin = parent_pin
        self._published = False
        self._closed = False

    def belongs_to_resolver(self, resolver: SessionModeResolver) -> bool:
        """Observe owner identity only; this is not journal or lock admission."""
        return self._resolver is resolver

    @contextmanager
    def locked_mode_operation(
        self,
        *,
        migration: bool,
        execution: bool,
        arguments: tuple[object, ...],
        keywords: Mapping[str, object],
    ) -> Generator[None, None, None]:
        """Run an export call under its resolver's original ordered lock scope."""
        with self._resolver.locked_mode_operation(
            migration=migration,
            execution=execution,
            arguments=arguments,
            keywords=keywords,
            transaction=self,
        ):
            yield

    @property
    def staging_name(self) -> str:
        return f"exports/{self.operation_id}"

    @property
    def publication_stage_name(self) -> str:
        return f".nanobot-export-{self.operation_id}"

    def read_staged(self, name: str) -> bytes:
        if self._closed or name not in {item.name for item in self.plan.files}:
            raise _fail()
        return self._resolver.storage.read(
            f".session-mode/{self.staging_name}/{name}", limit=67_108_864
        )

    @_locked_operation(execution=True)
    def mark_published(
        self,
        *,
        fault_after_stage_file: bool = False,
        fault_after_install: bool = False,
    ) -> None:
        if self._closed or self._published or not self.destination.is_absolute():
            raise _fail()
        self._parent_pin.validate()
        destination_name = self.destination.name
        stage_name = self.publication_stage_name
        try:
            os.stat(destination_name, dir_fd=self._parent_pin.fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise _fail()
        try:
            os.mkdir(stage_name, 0o700, dir_fd=self._parent_pin.fd)
        except FileExistsError:
            raise _fail() from None
        destination_fd: int | None = None
        try:
            destination_fd = os.open(
                stage_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=self._parent_pin.fd,
            )
            for index, item in enumerate(self.plan.files):
                data = self.read_staged(item.name)
                if len(data) != item.size or _digest(data) != item.digest:
                    raise _fail()
                descriptor = os.open(
                    item.name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=destination_fd,
                )
                try:
                    os.fchmod(descriptor, 0o600)
                    offset = 0
                    while offset < len(data):
                        offset += os.write(descriptor, data[offset:])
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                if fault_after_stage_file and index == 0:
                    raise RuntimeError("injected session-mode fault")
            os.fsync(destination_fd)
        except BaseException:
            # Preserve operation-owned staging as crash evidence; final remains absent.
            raise
        finally:
            if destination_fd is not None:
                os.close(destination_fd)
        self._resolver.require_export_stage(
            self._parent_pin,
            stage_name,
            self.binding.get("files"),
        )
        os.fsync(self._parent_pin.fd)
        _rename_directory_no_replace(
            self._parent_pin.fd,
            stage_name,
            destination_name,
        )
        os.fsync(self._parent_pin.fd)
        if fault_after_install:
            raise RuntimeError("injected session-mode fault")
        self._resolver.record_export_phase(self, phase="published")
        self._published = True

    @_locked_operation(execution=True)
    def complete(self) -> None:
        if self._closed or not self._published:
            raise _fail()
        self._resolver.record_export_phase(self, phase="complete")
        self._closed = True
        self._parent_pin.close()

    @_locked_operation(execution=True)
    def abort(self) -> None:
        if self._closed or self._published:
            raise _fail()
        self._resolver.record_export_phase(self, phase="aborted")
        self._closed = True
        self._parent_pin.close()


def _fail(message: str = _MODE_ERROR) -> SessionModeError:
    return SessionModeError(message)


def _rename_directory_no_replace(parent_fd: int, source: str, destination: str) -> None:
    """Atomically install one sibling directory without replacing any entry."""

    library = ctypes.CDLL(None, use_errno=True)
    renameatx_np = getattr(library, "renameatx_np", None)
    if renameatx_np is not None:
        renameatx_np.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            parent_fd,
            os.fsencode(source),
            parent_fd,
            os.fsencode(destination),
            0x00000004,  # RENAME_EXCL
        )
    else:
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise _fail("exclusive export publication is unavailable")
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            parent_fd,
            os.fsencode(source),
            parent_fd,
            os.fsencode(destination),
            1,  # RENAME_NOREPLACE
        )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise _fail()
    raise _fail("exclusive export publication is unavailable")


def _duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _parse_json(raw: bytes) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
        if not isinstance(value, dict):
            raise ValueError
        result = cast(dict[str, object], value)
        if raw != _canonical_json(result):
            raise ValueError
        return result
    except (UnicodeError, ValueError, TypeError, RecursionError):
        raise _fail() from None


def _canonical_json(value: dict[str, object]) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    except (TypeError, ValueError, RecursionError):
        raise _fail() from None


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _aware_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _digest_or_none(value: object) -> bool:
    return value is None or (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _operation_name(value: object, operation_id: str, prefixes: tuple[str, ...]) -> bool:
    if value is None:
        return True
    if not isinstance(value, str) or value.startswith("/"):
        return False
    parts = value.split("/")
    return (
        len(parts) >= 2
        and parts[0] in prefixes
        and parts.count(operation_id) == 1
        and all(part and part not in {".", ".."} for part in parts)
    )


def _valid_journal_keys(value: dict[str, object]) -> bool:
    operation = value.get("operation")
    source = value.get("source_key")
    target = value.get("target_key")
    if source is not None and (not isinstance(source, str) or not source):
        return False
    if target is not None and (not isinstance(target, str) or not target):
        return False
    if not isinstance(value.get("target_binding"), dict):
        return False
    if operation == "bootstrap":
        return (source is None and target is None) or (
            isinstance(source, str) and source == target
        )
    if operation == "create":
        return source is None and isinstance(target, str)
    if operation in {"convert", "reset"}:
        return isinstance(source, str) and source == target
    if operation == "fork":
        return isinstance(source, str) and isinstance(target, str) and source != target
    if operation in {"delete", "export"}:
        return isinstance(source, str) and target is None
    return False


class SessionModeResolver:
    """Resolve and journal the sole authority for one managed namespace."""

    def __init__(self, backend: JsonlSessionStore) -> None:
        self.backend = backend
        self.workspace = backend.workspace
        self.namespace = backend.sessions_dir
        self.root = self.namespace / ".session-mode"
        self.registry_path = self.root / "registry.json"
        self.records = self.root / "records"
        self.journals = self.root / "journals"
        self.rollback = self.root / "rollback"
        self.workspace_witness = self.workspace / ".nanobot" / "session-mode-binding.json"
        self.storage = SessionModeStorage(backend)
        self._closed = False
        self._guard_local = threading.local()

    def _reject_files_held(self) -> None:
        if self.backend.session_files_locked:
            raise _fail()

    @contextmanager
    def locked_mode_operation(
        self,
        *,
        migration: bool,
        execution: bool,
        arguments: tuple[object, ...],
        keywords: Mapping[str, object],
        transaction: SessionExportTransaction | None = None,
    ) -> Generator[None, None, None]:
        """Own migration -> execution -> files order for the existing call scope."""
        if transaction is not None and not transaction.belongs_to_resolver(self):
            raise _fail()
        backend = self.backend
        with ExitStack() as stack:
            if migration:
                self._reject_files_held()
                stack.enter_context(backend.locked_namespace_migration())
            if execution:
                key = transaction.key if transaction is not None else arguments[0] if arguments else keywords.get("key")
                if not isinstance(key, str) or not key:
                    raise _fail()
                stack.enter_context(self._execution_guards((key,)))
            stack.enter_context(backend.locked_session_files())
            yield

    @staticmethod
    def _execution_identity() -> tuple[int, int, asyncio.Task[Any] | None]:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        return os.getpid(), threading.get_ident(), task

    @contextmanager
    def _execution_guards(self, keys: tuple[str, ...]) -> Generator[None, None, None]:
        """Reuse only this synchronous call scope's exact, already-held guard set."""
        from nanobot.session.execution_owner import execution_lock_guard

        self._require_open()
        identity = self._execution_identity()
        ordered = tuple(sorted(set(keys), key=self.backend.storage_key))
        scope = getattr(self._guard_local, "scope", None)
        if scope is not None:
            if scope[0] != identity or not set(ordered).issubset(scope[1]):
                raise _fail()
            yield
            return
        self._reject_files_held()
        with ExitStack() as stack:
            for key in ordered:
                stack.enter_context(execution_lock_guard(self.backend, key))
            self._guard_local.scope = (identity, ordered)
            try:
                yield
            finally:
                del self._guard_local.scope

    def _inventory_key_hint(self) -> tuple[str, ...]:
        keys: set[str] = set()
        for name in self.storage.list_namespace():
            if name.endswith(".jsonl") or name.endswith(".checkpoint.json"):
                stem = name.removesuffix(".checkpoint.json") if name.endswith(".checkpoint.json") else name.removesuffix(".jsonl")
                key = self.backend.decode_storage_key(stem)
                if key is None or self.backend.storage_key(key) != stem:
                    raise _fail()
                keys.add(key)
        return tuple(sorted(keys, key=self.backend.storage_key))

    def _recovery_hint(self, key: str) -> tuple[tuple[str, ...], bytes]:
        keys = {key}
        rows: dict[str, object] = {}
        candidates: dict[str, list[dict[str, object]]] = {}
        for op in self.storage.list(".session-mode/journals"):
            journal = self._journal_records(op)
            if journal[-1]["phase"] not in {"complete", "aborted"}:
                candidates[op] = journal
        while True:
            prior = len(keys)
            for op, journal in candidates.items():
                last = journal[-1]
                implicated = {item for item in (last["source_key"], last["target_key"]) if isinstance(item, str)}
                if implicated & keys:
                    rows[op] = journal
                    keys.update(implicated)
            if len(keys) == prior:
                break
        return tuple(sorted(keys, key=self.backend.storage_key)), _canonical_json(rows)

    def close(self) -> None:
        self.storage.close()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise _fail("session authority is closed")

    @property
    def workspace_id(self) -> str:
        value = self.namespace.name
        if len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
            raise _fail()
        return value

    def _witnesses(self) -> dict[str, object] | None:
        registry_exists = self.storage.exists(".session-mode/registry.json")
        workspace_exists = self.storage.exists("session-mode-binding.json", witness=True)
        residue = self.storage.exists(".session-mode")
        if not registry_exists and not workspace_exists:
            if residue:
                raise _fail()
            return None
        if not registry_exists or not workspace_exists:
            raise _fail()
        registry = _parse_json(self.storage.read(".session-mode/registry.json"))
        witness = _parse_json(self.storage.read("session-mode-binding.json", witness=True))
        expected_fields = {
            "schema_version",
            "registry_id",
            "workspace_id",
            "namespace",
            "state",
            "bootstrap_operation_id",
        }
        if set(registry) != expected_fields or registry != witness:
            raise _fail()
        if (
            type(registry.get("schema_version")) is not int
            or registry.get("schema_version") != 1
            or registry.get("workspace_id") != self.workspace_id
            or registry.get("namespace") != self.workspace_id
            or registry.get("state") not in {"preparing", "active"}
            or not self._hex32(registry.get("registry_id"))
            or (
                registry.get("bootstrap_operation_id") is not None
                and not self._hex32(registry.get("bootstrap_operation_id"))
            )
        ):
            raise _fail()
        if registry["state"] != "active":
            raise _fail()
        return registry

    @staticmethod
    def _hex32(value: object) -> bool:
        return isinstance(value, str) and len(value) == 32 and all(
            character in "0123456789abcdef" for character in value
        )

    def is_registered(self) -> bool:
        return self._witnesses() is not None

    def _registry(self) -> dict[str, object]:
        registry = self._witnesses()
        if registry is None:
            raise _fail("session namespace is not registered")
        return registry

    def _new_registry(self, operation_id: str) -> dict[str, object]:
        return {
            "schema_version": 1,
            "registry_id": secrets.token_hex(16),
            "workspace_id": self.workspace_id,
            "namespace": self.workspace_id,
            "state": "preparing",
            "bootstrap_operation_id": operation_id,
        }

    def _record_name(self, key: str) -> str:
        return f".session-mode/records/{self.backend.storage_key(key)}.json"

    def _source_inventory(self) -> tuple[_SourceInventory, ...]:
        if not self.backend.session_files_locked:
            raise _fail()
        names = self.storage.list_namespace()
        allowed = {".workspace", ".session-files.lock", ".execution-locks", ".session-mode"}
        pairs: dict[str, dict[str, str]] = {}
        for name in names:
            if name in allowed:
                continue
            stem = (
                name.removesuffix(".checkpoint.json")
                if name.endswith(".checkpoint.json")
                else name.removesuffix(".jsonl")
                if name.endswith(".jsonl")
                else None
            )
            key = self.backend.decode_storage_key(stem) if stem is not None else None
            if key is None or self.backend.storage_key(key) != stem:
                raise _fail()
            assert stem is not None
            kind = "sidecar" if name.endswith(".checkpoint.json") else "source"
            entry = pairs.setdefault(stem, {})
            if kind in entry:
                raise _fail()
            entry[kind] = name
        result: list[_SourceInventory] = []
        for storage_key, pair in sorted(pairs.items()):
            source_name = pair.get("source")
            if source_name is None:
                raise _fail()
            key = self.backend.decode_storage_key(storage_key)
            if key is None or self.backend.storage_key(key) != storage_key:
                raise _fail()
            source = self.storage.read(source_name, limit=67_108_864)
            sidecar_name = f"{storage_key}.checkpoint.json"
            sidecar = self.storage.read(sidecar_name) if "sidecar" in pair else None
            metadata = self._read_header(key).get("metadata")
            if not isinstance(metadata, dict):
                raise _fail()
            canonical = cast(dict[object, object], metadata).get("source_schema_version") == 1
            try:
                if canonical:
                    if sidecar is not None:
                        raise _fail()
                    validate_source_session_bytes(source, expected_key=key)
                else:
                    session = self.backend.parse_legacy_bytes(source, expected_key=key)
                    if sidecar is not None:
                        self.backend.overlay_managed_checkpoint_bytes(session, sidecar)
            except (SessionStorageError, SourceValidationError):
                raise _fail() from None
            result.append(
                _SourceInventory(
                    key,
                    storage_key,
                    source_name,
                    source,
                    _digest(source),
                    len(source),
                    sidecar_name,
                    sidecar,
                )
            )
        return tuple(result)

    def _source_names(self) -> tuple[str, ...]:
        return tuple(item.source_name for item in self._source_inventory())

    def _inventory_for_key(self, key: str) -> _SourceInventory:
        matches = [item for item in self._source_inventory() if item.key == key]
        if len(matches) != 1:
            raise _fail()
        return matches[0]

    def _verify_pair_exact(self, item: _SourceInventory) -> None:
        if self.storage.read(item.source_name, limit=67_108_864) != item.source_bytes:
            raise _fail()
        if self.storage.exists(item.sidecar_name) != (item.sidecar_bytes is not None):
            raise _fail()
        if item.sidecar_bytes is not None and self.storage.read(item.sidecar_name) != item.sidecar_bytes:
            raise _fail()

    def _verify_canonical_output(self, key: str, output: bytes) -> None:
        if self.storage.exists(f"{self.backend.storage_key(key)}.checkpoint.json") or self.storage.read(
            f"{self.backend.storage_key(key)}.jsonl", limit=67_108_864
        ) != output:
            raise _fail()

    def _read_header(self, key: str) -> dict[str, object]:
        try:
            data = self.storage.read(
                f"{self.backend.storage_key(key)}.jsonl", limit=67_108_864
            )
            first = data.splitlines()[0]
            value = json.loads(first, object_pairs_hook=_duplicates)
            if not isinstance(value, dict):
                raise ValueError
            return cast(dict[str, object], value)
        except (OSError, IndexError, ValueError, TypeError, RecursionError):
            raise _fail() from None

    def _valid_standard_binding(self, binding: object) -> bool:
        if not isinstance(binding, dict) or set(cast(dict[object, object], binding)) != {
            "session_key",
            "storage_key",
            "state",
            "mode",
            "incarnation",
            "generation",
            "transition_revision",
            "transition_digest",
            "record_revision",
        }:
            return False
        session_key = cast(dict[object, object], binding).get("session_key")
        storage_key = cast(dict[object, object], binding).get("storage_key")
        state = cast(dict[object, object], binding).get("state")
        mode = cast(dict[object, object], binding).get("mode")
        if (
            not isinstance(session_key, str)
            or not session_key
            or not isinstance(storage_key, str)
            or self.backend.decode_storage_key(storage_key) != session_key
            or self.backend.storage_key(session_key) != storage_key
            or state not in {"active", "deleted", "quarantined"}
            or not isinstance(cast(dict[object, object], binding).get("incarnation"), str)
            or not cast(dict[object, object], binding).get("incarnation")
            or type(cast(dict[object, object], binding).get("generation")) is not int
            or cast(int, cast(dict[object, object], binding)["generation"]) < 0
            or type(cast(dict[object, object], binding).get("transition_revision")) is not int
            or cast(int, cast(dict[object, object], binding)["transition_revision"]) < 0
            or type(cast(dict[object, object], binding).get("record_revision")) is not int
            or cast(int, cast(dict[object, object], binding)["record_revision"]) < 1
            or not _digest_or_none(cast(dict[object, object], binding).get("transition_digest"))
        ):
            return False
        if state == "active":
            return mode in {"legacy", "canonical"} and cast(dict[object, object], binding)["transition_digest"] is not None
        return mode is None and cast(dict[object, object], binding)["transition_digest"] is None

    def _valid_journal_binding(self, value: dict[str, object]) -> bool:
        binding = value.get("target_binding")
        operation = value.get("operation")
        source = value.get("source_key")
        target = value.get("target_key")
        if operation == "bootstrap" and source is None and target is None:
            if not isinstance(binding, dict) or set(cast(dict[object, object], binding)) != {
                "registry",
                "inventory_digest",
            }:
                return False
            registry = cast(dict[object, object], binding).get("registry")
            return (
                isinstance(registry, dict)
                and set(cast(dict[object, object], registry))
                == {
                    "schema_version",
                    "registry_id",
                    "workspace_id",
                    "namespace",
                    "state",
                    "bootstrap_operation_id",
                }
                and type(cast(dict[object, object], registry).get("schema_version")) is int
                and cast(dict[object, object], registry).get("schema_version") == 1
                and self._hex32(cast(dict[object, object], registry).get("registry_id"))
                and self._hex32(cast(dict[object, object], registry).get("workspace_id"))
                and cast(dict[object, object], registry).get("namespace") == cast(dict[object, object], registry).get("workspace_id")
                and cast(dict[object, object], registry).get("state") == "preparing"
                and cast(dict[object, object], registry).get("bootstrap_operation_id") == value.get("operation_id")
                and isinstance(cast(dict[object, object], binding).get("inventory_digest"), str)
                and _digest_or_none(cast(dict[object, object], binding).get("inventory_digest"))
            )
        if operation == "export":
            if not isinstance(binding, dict) or set(cast(dict[object, object], binding)) != {
                "session",
                "destination",
                "destination_name",
                "parent_device",
                "parent_inode",
                "files",
            }:
                return False
            files = cast(dict[object, object], binding).get("files")
            return (
                self._valid_standard_binding(cast(dict[object, object], binding).get("session"))
                and isinstance(cast(dict[object, object], binding).get("destination"), str)
                and Path(cast(str, cast(dict[object, object], binding)["destination"])).is_absolute()
                and isinstance(cast(dict[object, object], binding).get("destination_name"), str)
                and Path(cast(str, cast(dict[object, object], binding)["destination"])).name
                == cast(dict[object, object], binding).get("destination_name")
                and type(cast(dict[object, object], binding).get("parent_device")) is int
                and cast(int, cast(dict[object, object], binding)["parent_device"]) >= 0
                and type(cast(dict[object, object], binding).get("parent_inode")) is int
                and cast(int, cast(dict[object, object], binding)["parent_inode"]) >= 0
                and isinstance(files, list)
                and 0 < len(cast(list[object], files)) <= 64
                and all(
                    isinstance(item, dict)
                    and set(cast(dict[object, object], item)) == {"name", "digest", "size"}
                    and isinstance(cast(dict[object, object], item).get("name"), str)
                    and cast(str, cast(dict[object, object], item)["name"]) not in {"", ".", ".."}
                    and "/" not in cast(str, cast(dict[object, object], item)["name"])
                    and "\\" not in cast(str, cast(dict[object, object], item)["name"])
                    and isinstance(cast(dict[object, object], item).get("digest"), str)
                    and _digest_or_none(cast(dict[object, object], item).get("digest"))
                    and type(cast(dict[object, object], item).get("size")) is int
                    and 0 <= cast(int, cast(dict[object, object], item)["size"]) <= 67_108_864
                    for item in cast(list[object], files)
                )
            )
        return self._valid_standard_binding(binding)

    def _valid_planned_record(
        self,
        record: dict[str, object],
        journal: dict[str, object],
    ) -> bool:
        if set(record) != {
            "schema_version",
            "registry_id",
            "record_revision",
            "session_key",
            "storage_key",
            "state",
            "mode",
            "source_schema_version",
            "incarnation",
            "generation",
            "transition_revision",
            "transition_digest",
            "committed_operation_id",
            "committed_at",
        }:
            return False
        binding = journal.get("target_binding")
        mode = record.get("mode")
        return (
            type(record.get("schema_version")) is int
            and record.get("schema_version") == 1
            and record.get("registry_id") == journal.get("registry_id")
            and record.get("committed_operation_id") == journal.get("operation_id")
            and _aware_timestamp(record.get("committed_at"))
            and self._valid_standard_binding(self._binding(record))
            and record.get("source_schema_version")
            == (1 if mode == "canonical" else None)
            and (mode != "canonical" or type(record.get("source_schema_version")) is int)
            and self._binding(record) == binding
        )

    def _journal_records(self, operation_id: str) -> list[dict[str, object]]:
        directory = f".session-mode/journals/{operation_id}"
        entries = self.storage.list(directory, suffix=".json")
        records: list[dict[str, object]] = []
        previous: str | None = None
        for sequence, name in enumerate(entries, 1):
            value = _parse_json(self.storage.read(f"{directory}/{name}"))
            if (
                set(value) != _JOURNAL_FIELDS
                or name != f"{sequence:04d}-{value.get('phase')}.json"
                or type(value.get("schema_version")) is not int
                or value.get("schema_version") != 1
                or value.get("operation_id") != operation_id
                or type(value.get("sequence")) is not int
                or value.get("sequence") != sequence
                or value.get("phase") not in _JOURNAL_PHASES
                or value.get("operation") not in _OPERATIONS
                or not _valid_journal_keys(value)
                or not self._valid_journal_binding(value)
                or not self._hex32(value.get("registry_id"))
                or value.get("previous_record_digest") != previous
                or not _digest_or_none(value.get("prior_record_digest"))
                or (
                    value.get("prior_record") is not None
                    and not isinstance(value.get("prior_record"), str)
                )
                or (
                    value.get("target_record") is not None
                    and not isinstance(value.get("target_record"), str)
                )
                or not _digest_or_none(value.get("input_digest"))
                or not _digest_or_none(value.get("output_digest"))
                or any(
                    item is not None and (type(item) is not int or item < 0)
                    for item in (
                        value.get("input_size"),
                        value.get("output_size"),
                        value.get("sidecar_size"),
                    )
                )
                or not _digest_or_none(value.get("sidecar_digest"))
                or not _digest_or_none(value.get("source_record_digest"))
                or not _digest_or_none(value.get("source_input_digest"))
                or (
                    value.get("sidecar_bytes") is not None
                    and not isinstance(value.get("sidecar_bytes"), str)
                )
                or (
                    value.get("source_record") is not None
                    and not isinstance(value.get("source_record"), str)
                )
                or (
                    value.get("source_binding") is not None
                    and not self._valid_standard_binding(value.get("source_binding"))
                )
                or (
                    value.get("source_input_size") is not None
                    and (
                        type(value.get("source_input_size")) is not int
                        or cast(int, value["source_input_size"]) < 0
                    )
                )
                or not _aware_timestamp(value.get("timestamp"))
                or not _operation_name(
                    value.get("staging_name"), operation_id, ("staging", "exports")
                )
                or not _operation_name(value.get("rollback_name"), operation_id, ("rollback",))
            ):
                raise _fail()
            prior_text = value.get("prior_record")
            if isinstance(prior_text, str) and _digest(prior_text.encode("utf-8")) != value.get(
                "prior_record_digest"
            ):
                raise _fail()
            source_text = value.get("source_record")
            if isinstance(source_text, str):
                source_value = _parse_json(source_text.encode("utf-8"))
                if (
                    _digest(source_text.encode("utf-8")) != value.get("source_record_digest")
                    or self._binding(source_value) != value.get("source_binding")
                    or source_value.get("session_key") != value.get("source_key")
                    or source_value.get("state") != "active"
                    or not self._valid_planned_record(source_value, {
                        "registry_id": value["registry_id"],
                        "operation_id": source_value.get("committed_operation_id"),
                        "target_binding": value["source_binding"],
                    })
                    or not self._hex32(source_value.get("committed_operation_id"))
                ):
                    raise _fail()
            target_text = value.get("target_record")
            if isinstance(target_text, str):
                target_value = _parse_json(target_text.encode("utf-8"))
                if not self._valid_planned_record(target_value, value):
                    raise _fail()
            elif value.get("operation") in {"create", "convert", "reset", "delete", "fork"}:
                raise _fail()
            sidecar_text = value.get("sidecar_bytes")
            if sidecar_text is None:
                if value.get("sidecar_digest") is not None or value.get("sidecar_size") is not None:
                    raise _fail()
            else:
                try:
                    sidecar = base64.b64decode(cast(str, sidecar_text), validate=True)
                except (ValueError, TypeError):
                    raise _fail() from None
                if (
                    _digest(sidecar) != value.get("sidecar_digest")
                    or len(sidecar) != value.get("sidecar_size")
                ):
                    raise _fail()
            if value.get("operation") == "fork":
                if (
                    not isinstance(source_text, str)
                    or value.get("source_input_digest") is None
                    or value.get("source_input_size") is None
                ):
                    raise _fail()
            elif any(
                value.get(name) is not None
                for name in (
                    "source_record",
                    "source_record_digest",
                    "source_binding",
                    "source_input_digest",
                    "source_input_size",
                )
            ):
                raise _fail()
            raw = _canonical_json(value)
            previous = _digest(raw)
            records.append(value)
        if not records:
            raise _fail()
        operation = cast(str, records[0]["operation"])
        immutable = _JOURNAL_FIELDS - {
            "sequence",
            "phase",
            "timestamp",
            "previous_record_digest",
        }
        first = records[0]
        if any(
            item["operation"] != operation
            or any(item[name] != first[name] for name in immutable)
            for item in records
        ):
            raise _fail()
        phases = [cast(str, item["phase"]) for item in records]
        full = (
            ["prepared", "mode_committed", "complete"]
            if operation == "delete"
            else ["prepared", "published", "complete"]
            if operation == "export"
            else ["prepared", "target_installed", "mode_committed", "complete"]
        )
        valid = phases == full[: len(phases)]
        if phases[-1] == "aborted":
            valid = (
                phases == ["prepared", "aborted"]
                if operation == "export"
                else phases[:-1] == full[: len(phases) - 1]
                and "mode_committed" not in phases
            )
        if not valid:
            raise _fail()
        return records

    def _append_journal(
        self,
        operation_id: str,
        *,
        phase: str,
        operation: str,
        registry_id: str,
        source_key: str | None,
        target_key: str | None,
        prior_record_digest: str | None,
        target_binding: dict[str, object] | None,
        prior_record: str | None = None,
        target_record: str | None = None,
        input_digest: str | None = None,
        input_size: int | None = None,
        output_digest: str | None = None,
        output_size: int | None = None,
        staging_name: str | None = None,
        rollback_name: str | None = None,
        sidecar_digest: str | None = None,
        sidecar_size: int | None = None,
        sidecar_bytes: str | None = None,
        source_record: str | None = None,
        source_record_digest: str | None = None,
        source_binding: dict[str, object] | None = None,
        source_input_digest: str | None = None,
        source_input_size: int | None = None,
    ) -> None:
        directory = f".session-mode/journals/{operation_id}"
        existing = self._journal_records(operation_id) if self.storage.exists(directory) else []
        if existing:
            first = existing[0]
            source_key = cast(str | None, first["source_key"])
            target_key = cast(str | None, first["target_key"])
            prior_record_digest = cast(str | None, first["prior_record_digest"])
            prior_record = cast(str | None, first["prior_record"])
            target_record = cast(str | None, first["target_record"])
            target_binding = cast(dict[str, object] | None, first["target_binding"])
            input_digest = cast(str | None, first["input_digest"])
            input_size = cast(int | None, first["input_size"])
            output_digest = cast(str | None, first["output_digest"])
            output_size = cast(int | None, first["output_size"])
            staging_name = cast(str | None, first["staging_name"])
            rollback_name = cast(str | None, first["rollback_name"])
            sidecar_digest = cast(str | None, first["sidecar_digest"])
            sidecar_size = cast(int | None, first["sidecar_size"])
            sidecar_bytes = cast(str | None, first["sidecar_bytes"])
            source_record = cast(str | None, first["source_record"])
            source_record_digest = cast(str | None, first["source_record_digest"])
            source_binding = cast(dict[str, object] | None, first["source_binding"])
            source_input_digest = cast(str | None, first["source_input_digest"])
            source_input_size = cast(int | None, first["source_input_size"])
        sequence = len(existing) + 1
        previous = _digest(_canonical_json(existing[-1])) if existing else None
        record: dict[str, object] = {
            "schema_version": 1,
            "registry_id": registry_id,
            "operation_id": operation_id,
            "sequence": sequence,
            "phase": phase,
            "operation": operation,
            "source_key": source_key,
            "target_key": target_key,
            "prior_record_digest": prior_record_digest,
            "prior_record": prior_record,
            "target_record": target_record,
            "target_binding": target_binding,
            "input_digest": input_digest,
            "input_size": input_size,
            "output_digest": output_digest,
            "output_size": output_size,
            "staging_name": staging_name,
            "rollback_name": rollback_name,
            "sidecar_digest": sidecar_digest,
            "sidecar_size": sidecar_size,
            "sidecar_bytes": sidecar_bytes,
            "source_record": source_record,
            "source_record_digest": source_record_digest,
            "source_binding": source_binding,
            "source_input_digest": source_input_digest,
            "source_input_size": source_input_size,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "previous_record_digest": previous,
        }
        self.storage.write(
            f"{directory}/{sequence:04d}-{phase}.json",
            _canonical_json(record),
            replace=False,
        )

    def _record(
        self,
        key: str,
        *,
        registry_id: str,
        operation_id: str,
        mode: Mode | None,
        state: str,
        revision: int,
        generation: int,
        incarnation: str,
        transition_revision: int,
        transition_digest: str | None,
        record_revision: int = 1,
        committed_at: str | None = None,
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "registry_id": registry_id,
            "record_revision": record_revision,
            "session_key": key,
            "storage_key": self.backend.storage_key(key),
            "state": state,
            "mode": mode,
            "source_schema_version": 1 if mode == "canonical" else None,
            "incarnation": incarnation,
            "generation": generation,
            "transition_revision": transition_revision,
            "transition_digest": transition_digest,
            "committed_operation_id": operation_id,
            "committed_at": committed_at or datetime.now(timezone.utc).isoformat(),
        }

    def _binding(self, record: dict[str, object]) -> dict[str, object]:
        return {
            key: record[key]
            for key in (
                "session_key",
                "storage_key",
                "state",
                "mode",
                "incarnation",
                "generation",
                "transition_revision",
                "transition_digest",
                "record_revision",
            )
        }

    @staticmethod
    def _render_canonical(session: Session) -> tuple[bytes, int]:
        events = _upgrade_events(session.messages)
        revision = session.revision + 1
        metadata = dict(session.metadata)
        metadata.pop("runtime_checkpoint", None)
        metadata["source_schema_version"] = 1
        header: dict[str, object] = {
            "_type": "metadata",
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": metadata,
            "revision": revision,
            "generation": session.generation,
            "incarnation": session.incarnation,
            "last_archived": session.last_archived,
            "last_consolidated": session.last_archived,
        }
        rows: list[dict[str, object]] = [header, *events]
        return b"".join(
            (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            for row in rows
        ), revision

    def _stage_source(self, operation_id: str, key: str, data: bytes) -> str:
        relative = f"staging/{operation_id}/canonical-source.jsonl"
        self.storage.write(f".session-mode/{relative}", data, replace=False)
        installed = self.storage.read(f".session-mode/{relative}")
        validate_source_session_bytes(installed, expected_key=key)
        if installed != data:
            raise _fail()
        return relative

    def _install_stage(self, relative: str, key: str) -> None:
        self.storage.install_session_file(
            f".session-mode/{relative}",
            f"{self.backend.storage_key(key)}.jsonl",
        )

    def _validate_record(
        self, key: str, registry: dict[str, object],
        *, inventory: Mapping[str, _SourceInventory] | None = None,
    ) -> dict[str, object] | None:
        name = self._record_name(key)
        sidecar_name = f"{self.backend.storage_key(key)}.checkpoint.json"
        if not self.storage.exists(name):
            if self.storage.exists(f"{self.backend.storage_key(key)}.jsonl") or self.storage.exists(sidecar_name):
                raise _fail()
            return None
        record = _parse_json(self.storage.read(name))
        expected = {
            "schema_version", "registry_id", "record_revision", "session_key", "storage_key",
            "state", "mode", "source_schema_version", "incarnation", "generation",
            "transition_revision", "transition_digest", "committed_operation_id", "committed_at",
        }
        if (
            set(record) != expected
            or type(record.get("schema_version")) is not int
            or record.get("schema_version") != 1
            or record.get("registry_id") != registry["registry_id"]
            or record.get("session_key") != key
            or record.get("storage_key") != self.backend.storage_key(key)
            or type(record.get("record_revision")) is not int
            or cast(int, record["record_revision"]) < 1
            or type(record.get("generation")) is not int
            or cast(int, record["generation"]) < 0
            or type(record.get("transition_revision")) is not int
            or cast(int, record["transition_revision"]) < 0
            or not isinstance(record.get("incarnation"), str)
            or not cast(str, record["incarnation"])
            or not _digest_or_none(record.get("transition_digest"))
            or not _aware_timestamp(record.get("committed_at"))
            or record.get("state") not in {"active", "deleted", "quarantined"}
            or not self._hex32(record.get("committed_operation_id"))
        ):
            raise _fail()
        if record["state"] != "active":
            if self.storage.exists(f"{self.backend.storage_key(key)}.jsonl") or self.storage.exists(sidecar_name):
                raise _fail()
            if (
                record["mode"] is not None
                or record["source_schema_version"] is not None
                or record["transition_digest"] is not None
            ):
                raise _fail()
            journal = self._journal_records(cast(str, record["committed_operation_id"]))
            if any(entry["registry_id"] != registry["registry_id"] for entry in journal):
                raise _fail()
            committed = [entry for entry in journal if entry["phase"] == "mode_committed"]
            if (
                record["state"] != "deleted"
                or journal[-1]["phase"] != "complete"
                or journal[-1]["operation"] != "delete"
                or not committed
                or committed[-1].get("target_binding") != self._binding(record)
            ):
                raise _fail()
            return record
        mode = record.get("mode")
        if mode not in {"legacy", "canonical"}:
            raise _fail()
        if (
            record.get("source_schema_version") != (1 if mode == "canonical" else None)
            or (
                mode == "canonical"
                and type(record.get("source_schema_version")) is not int
            )
        ):
            raise _fail()
        if record.get("transition_digest") is None:
            raise _fail()
        journal = self._journal_records(cast(str, record["committed_operation_id"]))
        if any(entry["registry_id"] != registry["registry_id"] for entry in journal):
            raise _fail()
        committed = [entry for entry in journal if entry["phase"] == "mode_committed"]
        if not committed or committed[-1].get("target_binding") != self._binding(record):
            raise _fail()
        data = self.storage.read(f"{self.backend.storage_key(key)}.jsonl", limit=67_108_864)
        header = self._read_header(key)
        metadata = header.get("metadata")
        if not isinstance(metadata, dict):
            raise _fail()
        revision = header.get("revision")
        generation = header.get("generation")
        if (
            header.get("key") != key
            or header.get("incarnation") != record.get("incarnation")
            or generation != record.get("generation")
            or type(revision) is not int
            or revision < cast(int, record["transition_revision"])
            or cast(dict[object, object], metadata).get("source_schema_version") != (1 if mode == "canonical" else None)
        ):
            raise _fail()
        if revision == record["transition_revision"] and _digest(data) != record["transition_digest"]:
            raise _fail()
        if inventory is None:
            self._inventory_for_key(key)
        elif key not in inventory:
            raise _fail()
        return record

    def require_export_stage(
        self,
        parent: PinnedDirectoryPath,
        name: str,
        files: object,
    ) -> None:
        """Prove the staged bytes through the original pinned-directory reader."""
        if self._export_directory_status(parent, name, files) != "exact":
            raise _fail()

    def record_export_phase(
        self,
        transaction: SessionExportTransaction,
        *,
        phase: Literal["published", "complete", "aborted"],
    ) -> None:
        """Advance only this export journal inside its real held operation scope."""
        if (
            type(transaction) is not SessionExportTransaction
            or not transaction.belongs_to_resolver(self)
            or phase not in {"published", "complete", "aborted"}
        ):
            raise _fail()
        self._require_open()
        scope = getattr(self._guard_local, "scope", None)
        if (
            not self.backend.session_files_locked
            or scope is None
            or scope[0] != self._execution_identity()
            or transaction.key not in scope[1]
        ):
            raise _fail()
        expected_destination = "absent" if phase == "aborted" else "exact"
        if self._export_destination_status(transaction.binding) != expected_destination:
            raise _fail()
        last = self._journal_records(transaction.operation_id)[-1]
        if (
            last["operation"] != "export"
            or last["source_key"] != transaction.key
            or last["target_key"] is not None
            or last["target_binding"] != transaction.binding
            or last["staging_name"] != transaction.staging_name
            or last["phase"] != ("published" if phase == "complete" else "prepared")
        ):
            raise _fail()
        self._append_journal(
            transaction.operation_id,
            phase=phase,
            operation="export",
            registry_id=cast(str, last["registry_id"]),
            source_key=transaction.key,
            target_key=None,
            prior_record_digest=cast(str | None, last["prior_record_digest"]),
            target_binding=transaction.binding,
            staging_name=transaction.staging_name,
        )

    @staticmethod
    def _export_directory_status(
        parent: PinnedDirectoryPath,
        name: str,
        files: object,
    ) -> Literal["absent", "exact", "conflict"]:
        if not isinstance(files, list):
            raise _fail()
        try:
            value = os.stat(name, dir_fd=parent.fd, follow_symlinks=False)
        except FileNotFoundError:
            return "absent"
        if (
            not stat.S_ISDIR(value.st_mode)
            or value.st_uid != os.geteuid()
            or value.st_mode & 0o777 != 0o700
        ):
            return "conflict"
        directory = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent.fd,
        )
        try:
            expected_names: set[object] = {cast(dict[object, object], item).get("name") for item in cast(list[object], files) if isinstance(item, dict)}
            if len(expected_names) != len(cast(list[object], files)) or set(os.listdir(directory)) != expected_names:
                return "conflict"
            for item in cast(list[object], files):
                assert isinstance(item, dict)
                file_name = cast(dict[object, object], item).get("name")
                digest = cast(dict[object, object], item).get("digest")
                size = cast(dict[object, object], item).get("size")
                if (
                    not isinstance(file_name, str)
                    or not isinstance(digest, str)
                    or type(size) is not int
                    or size < 0
                    or size > 67_108_864
                ):
                    raise _fail()
                descriptor = os.open(
                    file_name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=directory,
                )
                try:
                    opened = os.fstat(descriptor)
                    if (
                        not stat.S_ISREG(opened.st_mode)
                        or opened.st_uid != os.geteuid()
                        or opened.st_nlink != 1
                        or opened.st_mode & 0o777 != 0o600
                        or opened.st_size != size
                    ):
                        return "conflict"
                    data = b""
                    while len(data) < size:
                        chunk = os.read(descriptor, min(65536, size - len(data)))
                        if not chunk:
                            return "conflict"
                        data += chunk
                    if _digest(data) != digest:
                        return "conflict"
                finally:
                    os.close(descriptor)
            return "exact"
        finally:
            os.close(directory)

    @staticmethod
    def _export_destination_status(binding: object) -> Literal["absent", "exact", "conflict"]:
        if not isinstance(binding, dict):
            raise _fail()
        destination_value = cast(dict[object, object], binding).get("destination")
        name = cast(dict[object, object], binding).get("destination_name")
        files = cast(dict[object, object], binding).get("files")
        if (
            not isinstance(destination_value, str)
            or not isinstance(name, str)
            or not isinstance(files, list)
        ):
            raise _fail()
        destination = Path(destination_value)
        if not destination.is_absolute() or destination.name != name:
            raise _fail()
        parent = PinnedDirectoryPath(destination.parent)
        try:
            parent_stat = os.fstat(parent.fd)
            if (
                parent_stat.st_dev != cast(dict[object, object], binding).get("parent_device")
                or parent_stat.st_ino != cast(dict[object, object], binding).get("parent_inode")
            ):
                return "conflict"
            return SessionModeResolver._export_directory_status(parent, name, cast(list[object], files))
        finally:
            parent.close()

    def _verify_fork_source(self, row: dict[str, object]) -> None:
        source_key = cast(str, row["source_key"])
        source_record = cast(str, row["source_record"]).encode("utf-8")
        if self.storage.read(self._record_name(source_key)) != source_record:
            raise _fail()
        source_data = self.storage.read(
            f"{self.backend.storage_key(source_key)}.jsonl", limit=67_108_864
        )
        if _digest(source_data) != row["source_input_digest"] or len(source_data) != row["source_input_size"]:
            raise _fail()
        source_sidecar = f"{self.backend.storage_key(source_key)}.checkpoint.json"
        expected = (
            base64.b64decode(cast(str, row["sidecar_bytes"]), validate=True)
            if row["sidecar_bytes"] is not None else None
        )
        if self.storage.exists(source_sidecar) != (expected is not None):
            raise _fail()
        if expected is not None and self.storage.read(source_sidecar) != expected:
            raise _fail()

    def _verify_reset_sidecar(
        self, key: str, row: dict[str, object], target_record: dict[str, object]
    ) -> None:
        source = self.storage.read(f"{self.backend.storage_key(key)}.jsonl", limit=67_108_864)
        predecessor = _digest(source) == row["input_digest"] and len(source) == row["input_size"]
        target = _digest(source) == row["output_digest"] and len(source) == row["output_size"]
        if not predecessor and not target:
            raise _fail()
        sidecar_name = f"{self.backend.storage_key(key)}.checkpoint.json"
        expected = (
            base64.b64decode(cast(str, row["sidecar_bytes"]), validate=True)
            if row["sidecar_bytes"] is not None else None
        )
        present = self.storage.exists(sidecar_name)
        if present:
            if target_record["mode"] == "canonical" or expected is None or self.storage.read(sidecar_name) != expected:
                raise _fail()
            if target:
                self.storage.unlink(sidecar_name)
        elif predecessor and expected is not None:
            raise _fail()

    def _reconcile_incomplete(self, key: str, registry: dict[str, object]) -> None:
        for operation_id in self.storage.list(".session-mode/journals"):
            if not self._hex32(operation_id):
                raise _fail()
            journal = self._journal_records(operation_id)
            if any(entry["registry_id"] != registry["registry_id"] for entry in journal):
                raise _fail()
            last = journal[-1]
            if last["phase"] in {"complete", "aborted"}:
                continue
            if key not in {last.get("source_key"), last.get("target_key")}:
                continue
            operation = last.get("operation")
            if operation == "fork" and key == last.get("source_key"):
                self._reconcile_incomplete(cast(str, last["target_key"]), registry)
                continue
            if operation == "fork":
                self._verify_fork_source(last)
            if operation in {"create", "convert", "fork"}:
                stage = last.get("staging_name")
                if isinstance(stage, str) and self.storage.exists(f".session-mode/{stage}"):
                    staged = self.storage.read(f".session-mode/{stage}", limit=67_108_864)
                    if _digest(staged) != last["output_digest"] or len(staged) != last["output_size"]:
                        raise _fail()
                if operation == "convert" and isinstance(last["rollback_name"], str):
                    rollback = f".session-mode/{last['rollback_name']}"
                    if self.storage.exists(rollback):
                        saved = self.storage.read(rollback, limit=67_108_864)
                        if _digest(saved) != last["input_digest"] or len(saved) != last["input_size"]:
                            raise _fail()
            if operation in {"create", "convert", "fork", "export"} and self.storage.exists(
                f"{self.backend.storage_key(key)}.checkpoint.json"
            ):
                raise _fail()
            committed = (
                _parse_json(self.storage.read(self._record_name(key)))
                if self.storage.exists(self._record_name(key))
                else None
            )
            if operation == "delete":
                first = journal[0]
                source_name = f"{self.backend.storage_key(key)}.jsonl"
                sidecar_name = f"{self.backend.storage_key(key)}.checkpoint.json"
                sidecar_text = first.get("sidecar_bytes")
                expected_sidecar = (
                    base64.b64decode(cast(str, sidecar_text), validate=True)
                    if sidecar_text is not None
                    else None
                )
                if committed is None or committed.get("committed_operation_id") != operation_id:
                    if last["phase"] != "prepared":
                        raise _fail()
                    if self.storage.read(self._record_name(key)) != cast(str, first["prior_record"]).encode("utf-8"):
                        raise _fail()
                    if not self.storage.exists(source_name):
                        raise _fail()
                    predecessor = self.storage.read(source_name, limit=67_108_864)
                    if (
                        _digest(predecessor) != first.get("input_digest")
                        or len(predecessor) != first.get("input_size")
                    ):
                        raise _fail()
                    if self.storage.exists(sidecar_name):
                        if expected_sidecar is None or self.storage.read(sidecar_name) != expected_sidecar:
                            raise _fail()
                    elif expected_sidecar is not None:
                        raise _fail()
                    self._append_journal(
                        operation_id,
                        phase="aborted",
                        operation="delete",
                        registry_id=cast(str, registry["registry_id"]),
                        source_key=key,
                        target_key=None,
                        prior_record_digest=cast(str | None, first["prior_record_digest"]),
                        prior_record=cast(str | None, first["prior_record"]),
                        target_record=cast(str | None, first["target_record"]),
                        target_binding=None,
                        input_digest=cast(str | None, first["input_digest"]),
                        input_size=cast(int | None, first["input_size"]),
                        sidecar_digest=cast(str | None, first["sidecar_digest"]),
                        sidecar_size=cast(int | None, first["sidecar_size"]),
                        sidecar_bytes=cast(str | None, first["sidecar_bytes"]),
                    )
                    continue
                target_record = first.get("target_record")
                if not isinstance(target_record, str) or self.storage.read(
                    self._record_name(key)
                ) != target_record.encode("utf-8"):
                    raise _fail()
                common_delete: _JournalCommon = {
                    "operation": "delete", "registry_id": cast(str, registry["registry_id"]),
                    "source_key": key, "target_key": None,
                    "prior_record_digest": cast(str | None, first["prior_record_digest"]),
                    "prior_record": cast(str | None, first["prior_record"]),
                    "target_record": target_record,
                    "target_binding": cast(dict[str, object], first["target_binding"]),
                    "input_digest": cast(str | None, first["input_digest"]),
                    "input_size": cast(int | None, first["input_size"]),
                    "sidecar_digest": cast(str | None, first["sidecar_digest"]),
                    "sidecar_size": cast(int | None, first["sidecar_size"]),
                    "sidecar_bytes": cast(str | None, first["sidecar_bytes"]),
                }
                phases = [entry["phase"] for entry in journal]
                if "mode_committed" not in phases:
                    if not self.storage.exists(source_name):
                        raise _fail()
                    source = self.storage.read(source_name, limit=67_108_864)
                    if _digest(source) != first["input_digest"] or len(source) != first["input_size"]:
                        raise _fail()
                    if self.storage.exists(sidecar_name) != (expected_sidecar is not None):
                        raise _fail()
                    if expected_sidecar is not None and self.storage.read(sidecar_name) != expected_sidecar:
                        raise _fail()
                    self._append_journal(operation_id, phase="mode_committed", **common_delete)
                source_exists = self.storage.exists(source_name)
                sidecar_exists = self.storage.exists(sidecar_name)
                if sidecar_exists:
                    if expected_sidecar is None or self.storage.read(sidecar_name) != expected_sidecar:
                        raise _fail()
                elif expected_sidecar is not None and source_exists:
                    raise _fail()
                if source_exists:
                    source_bytes = self.storage.read(source_name, limit=67_108_864)
                    if (
                        _digest(source_bytes) != first.get("input_digest")
                        or len(source_bytes) != first.get("input_size")
                    ):
                        raise _fail()
                    self.storage.unlink(source_name)
                if sidecar_exists:
                    self.storage.unlink(sidecar_name)
                self._append_journal(operation_id, phase="complete", **common_delete)
                continue
            if operation == "export" and last["phase"] in {"prepared", "published"}:
                first = journal[0]
                status = self._export_destination_status(first.get("target_binding"))
                if status == "conflict":
                    raise _fail()
                common_export: _JournalCommon = {
                    "operation": "export", "registry_id": cast(str, registry["registry_id"]),
                    "source_key": key, "target_key": None,
                    "prior_record_digest": cast(str | None, first["prior_record_digest"]),
                    "target_binding": cast(dict[str, object], first["target_binding"]),
                    "input_digest": cast(str | None, first["input_digest"]),
                    "input_size": cast(int | None, first["input_size"]),
                    "output_digest": cast(str | None, first["output_digest"]),
                    "output_size": cast(int | None, first["output_size"]),
                    "staging_name": cast(str | None, first["staging_name"]),
                }
                if status == "absent" and last["phase"] == "prepared":
                    self._append_journal(operation_id, phase="aborted", **common_export)
                    continue
                if status != "exact":
                    raise _fail()
                if last["phase"] == "prepared":
                    self._append_journal(operation_id, phase="published", **common_export)
                self._append_journal(operation_id, phase="complete", **common_export)
                continue
            if operation == "reset":
                first = journal[0]
                target_record = cast(str, first["target_record"])
                target_value = _parse_json(target_record.encode("utf-8"))
                self._verify_reset_sidecar(key, first, target_value)
            if committed is not None and committed.get("committed_operation_id") == operation_id:
                if self.storage.read(self._record_name(key)) != cast(str, last["target_record"]).encode("utf-8"):
                    raise _fail()
                self._finish_postcommit_journal(committed)
                continue
            if operation in {"create", "convert", "reset", "fork"}:
                expected_prior = last["prior_record"]
                if (committed is None) != (expected_prior is None):
                    raise _fail()
                if expected_prior is not None and self.storage.read(self._record_name(key)) != cast(str, expected_prior).encode("utf-8"):
                    raise _fail()
            if operation == "reset" and last["phase"] in {"prepared", "target_installed"}:
                first = journal[0]
                current_bytes = self.storage.read(
                    f"{self.backend.storage_key(key)}.jsonl", limit=67_108_864
                )
                current_digest = _digest(current_bytes)
                if current_digest == first.get("input_digest") and len(current_bytes) == first.get("input_size") and last["phase"] == "prepared":
                    if self.storage.read(self._record_name(key)) != cast(str, first["prior_record"]).encode("utf-8"):
                        raise _fail()
                    self._append_journal(
                        operation_id,
                        phase="aborted",
                        operation="reset",
                        registry_id=cast(str, registry["registry_id"]),
                        source_key=key,
                        target_key=key,
                        prior_record_digest=cast(str | None, first["prior_record_digest"]),
                        prior_record=cast(str | None, first["prior_record"]),
                        target_record=cast(str | None, first["target_record"]),
                        target_binding=None,
                        input_digest=cast(str | None, first["input_digest"]),
                        input_size=cast(int | None, first["input_size"]),
                        output_digest=cast(str | None, first["output_digest"]),
                        output_size=cast(int | None, first["output_size"]),
                    )
                    continue
                if current_digest != first.get("output_digest") or len(current_bytes) != first.get("output_size"):
                    raise _fail()
                target_record = first.get("target_record")
                if not isinstance(target_record, str):
                    raise _fail()
                record_bytes = target_record.encode("utf-8")
                _parse_json(record_bytes)
                common_reset: _JournalCommon = {
                    "operation": "reset", "registry_id": cast(str, registry["registry_id"]),
                    "source_key": key, "target_key": key,
                    "prior_record_digest": cast(str | None, first["prior_record_digest"]),
                    "prior_record": cast(str | None, first["prior_record"]),
                    "target_record": target_record,
                    "target_binding": cast(dict[str, object], first["target_binding"]),
                    "input_digest": cast(str | None, first["input_digest"]),
                    "input_size": cast(int | None, first["input_size"]),
                    "output_digest": cast(str | None, first["output_digest"]),
                    "output_size": cast(int | None, first["output_size"]),
                }
                if last["phase"] == "prepared":
                    self._append_journal(operation_id, phase="target_installed", **common_reset)
                self.storage.write(self._record_name(key), record_bytes)
                self._append_journal(operation_id, phase="mode_committed", **common_reset)
                self._append_journal(operation_id, phase="complete", **common_reset)
                continue
            if operation not in {"create", "fork", "convert"} or last["phase"] not in {
                "prepared", "target_installed"
            }:
                raise _fail()
            target_name = f"{self.backend.storage_key(key)}.jsonl"
            installed = last["phase"] == "target_installed"
            if last["phase"] == "prepared" and self.storage.exists(target_name):
                candidate = self.storage.read(
                    target_name, limit=67_108_864
                )
                if (
                    _digest(candidate) == last.get("input_digest")
                    and len(candidate) == last.get("input_size")
                ):
                    installed = False
                elif (
                    _digest(candidate) == last.get("output_digest")
                    and len(candidate) == last.get("output_size")
                ):
                    installed = True
                else:
                    raise _fail()
            if not installed and last["phase"] == "prepared":
                staging_name = last.get("staging_name")
                if isinstance(staging_name, str):
                    stage_name = f".session-mode/{staging_name}"
                    if self.storage.exists(stage_name):
                        staged = self.storage.read(stage_name)
                        if _digest(staged) != last.get("output_digest") or len(staged) != last.get("output_size"):
                            raise _fail()
                        self.storage.unlink(stage_name)
            if installed:
                if not self.storage.exists(target_name):
                    raise _fail()
                target_bytes = self.storage.read(
                    target_name, limit=67_108_864
                )
                if (
                    _digest(target_bytes) != last.get("output_digest")
                    or len(target_bytes) != last.get("output_size")
                ):
                    raise _fail()
                if operation == "convert":
                    rollback_name = last.get("rollback_name")
                    if not isinstance(rollback_name, str):
                        raise _fail()
                    predecessor = self.storage.read(f".session-mode/{rollback_name}")
                    if (
                        _digest(predecessor) != last.get("input_digest")
                        or len(predecessor) != last.get("input_size")
                    ):
                        raise _fail()
                    self.storage.write(target_name, predecessor)
                else:
                    self.storage.unlink(target_name)
            self._append_journal(operation_id, phase="aborted", operation=cast(str, operation),
                registry_id=cast(str, registry["registry_id"]),
                source_key=cast(str | None, last["source_key"]), target_key=cast(str | None, last["target_key"]),
                prior_record_digest=cast(str | None, last["prior_record_digest"]), target_binding=None,
                input_digest=cast(str | None, last["input_digest"]), input_size=cast(int | None, last["input_size"]),
                output_digest=cast(str | None, last["output_digest"]), output_size=cast(int | None, last["output_size"]),
                rollback_name=cast(str | None, last["rollback_name"]))

    def _assert_no_incomplete(self, key: str, registry: dict[str, object]) -> None:
        for operation_id in self.storage.list(".session-mode/journals"):
            if not self._hex32(operation_id):
                raise _fail()
            journal = self._journal_records(operation_id)
            if any(entry["registry_id"] != registry["registry_id"] for entry in journal):
                raise _fail()
            last = journal[-1]
            if (
                last["phase"] not in {"complete", "aborted"}
                and (
                    key in {last.get("source_key"), last.get("target_key")}
                    or (last.get("source_key") is None and last.get("target_key") is None)
                )
            ):
                raise _fail()

    def _finish_postcommit_journal(self, record: dict[str, object]) -> None:
        operation_id = cast(str, record["committed_operation_id"])
        journal = self._journal_records(operation_id)
        first = journal[0]
        source = self.storage.read(
            f"{self.backend.storage_key(cast(str, record['session_key']))}.jsonl", limit=67_108_864
        )
        if _digest(source) != first["output_digest"] or len(source) != first["output_size"]:
            raise _fail()
        phases = [entry["phase"] for entry in journal]
        if "mode_committed" in phases:
            if phases[-1] not in {"complete", "aborted"}:
                last = journal[-1]
                self._append_journal(operation_id, phase="complete", operation=cast(str, last["operation"]),
                    registry_id=cast(str, record["registry_id"]), source_key=cast(str | None, last["source_key"]),
                    target_key=cast(str | None, last["target_key"]), prior_record_digest=cast(str | None, last["prior_record_digest"]),
                    target_binding=self._binding(record), output_digest=cast(str | None, last["output_digest"]),
                    output_size=cast(int | None, last["output_size"]), rollback_name=cast(str | None, last["rollback_name"]))
            return
        # Atomic mode publication is the authority commit. A crash before its
        # journal witness must complete forward, never restore legacy bytes.
        last = journal[-1]
        if last["phase"] != "target_installed" or last.get("output_digest") != record["transition_digest"]:
            raise _fail()
        data = self.storage.read(
            f"{self.backend.storage_key(cast(str, record['session_key']))}.jsonl",
            limit=67_108_864,
        )
        if _digest(data) != record["transition_digest"] or len(data) != last.get("output_size"):
            raise _fail()
        self._append_journal(operation_id, phase="mode_committed", operation=cast(str, last["operation"]),
            registry_id=cast(str, record["registry_id"]), source_key=cast(str | None, last["source_key"]),
            target_key=cast(str | None, last["target_key"]), prior_record_digest=cast(str | None, last["prior_record_digest"]),
            target_binding=self._binding(record), output_digest=_digest(data), output_size=len(data),
            rollback_name=cast(str | None, last["rollback_name"]))
        self._append_journal(operation_id, phase="complete", operation=cast(str, last["operation"]),
            registry_id=cast(str, record["registry_id"]), source_key=cast(str | None, last["source_key"]),
            target_key=cast(str | None, last["target_key"]), prior_record_digest=cast(str | None, last["prior_record_digest"]),
            target_binding=self._binding(record), output_digest=_digest(data), output_size=len(data),
            rollback_name=cast(str | None, last["rollback_name"]))

    @_locked_operation()
    def resolve(self, key: str) -> ResolvedSessionMode:
        self._require_open()
        registry = self._witnesses()
        if registry is None:
            if self.storage.exists(f"{self.backend.storage_key(key)}.jsonl") or self.storage.exists(
                f"{self.backend.storage_key(key)}.checkpoint.json"
            ):
                raise _fail("session namespace requires explicit offline bootstrap")
            return ResolvedSessionMode("absent", None, None, 0, None, None)
        self._assert_no_incomplete(key, registry)
        record = self._validate_record(key, registry)
        if record is None:
            return ResolvedSessionMode("absent", None, cast(str, registry["registry_id"]), 0, None, None)
        state = cast(Literal["active", "deleted", "quarantined"], record["state"])
        return ResolvedSessionMode(
            state, cast(Mode | None, record["mode"]), cast(str, registry["registry_id"]),
            cast(int, record["record_revision"]), cast(str, record["incarnation"]),
            cast(int, record["generation"]),
        )

    def recover(self, key: str) -> ResolvedSessionMode:
        """Explicit offline reconciliation; ordinary resolution is read-only."""
        self._reject_files_held()
        hint = self._recovery_hint(key)
        with self._execution_guards(hint[0]), self.backend.locked_session_files():
            if self._recovery_hint(key) != hint:
                raise _fail()
            registry = self._registry()
            self._reconcile_incomplete(key, registry)
            return self.resolve(key)

    def _ensure_registry_for_first_write(self) -> dict[str, object]:
        registry = self._witnesses()
        if registry is not None:
            return registry
        files = self._source_names()
        if files:
            raise _fail("session namespace requires explicit offline bootstrap")
        self._bootstrap_locked(offline_copy=True)
        return self._registry()

    @_locked_operation(migration=True, execution=True)
    def ensure_canonical(self, key: str, *, fault_after: str | None = None) -> ResolvedSessionMode:
        self._require_open()
        current = self.resolve(key)
        if current.state == "active":
            return current
        registry = self._ensure_registry_for_first_write()
        prior_bytes: bytes | None = None
        prior_record: dict[str, object] | None = None
        if current.state == "deleted":
            prior_bytes = self.storage.read(self._record_name(key))
            prior_record = _parse_json(prior_bytes)
        source_name = f"{self.backend.storage_key(key)}.jsonl"
        sidecar_name = f"{self.backend.storage_key(key)}.checkpoint.json"
        if self.storage.exists(source_name) or self.storage.exists(sidecar_name):
            raise _fail()
        operation_id = secrets.token_hex(16)
        registry_id = cast(str, registry["registry_id"])
        session = Session(key)
        output, revision = self._render_canonical(session)
        staging_name = f"staging/{operation_id}/canonical-source.jsonl"
        record = self._record(key, registry_id=registry_id, operation_id=operation_id,
            mode="canonical", state="active", revision=revision,
            generation=session.generation, incarnation=session.incarnation,
            transition_revision=revision, transition_digest=_digest(output),
            record_revision=(
                cast(int, prior_record["record_revision"]) + 1
                if prior_record is not None
                else 1
            ))
        binding = self._binding(record)
        target_record = _canonical_json(record).decode("utf-8")
        self._append_journal(operation_id, phase="prepared", operation="create", registry_id=registry_id,
            source_key=None, target_key=key,
            prior_record_digest=_digest(prior_bytes) if prior_bytes is not None else None,
            prior_record=prior_bytes.decode("utf-8") if prior_bytes is not None else None,
            target_binding=binding,
            target_record=target_record,
            output_digest=_digest(output), output_size=len(output), staging_name=staging_name)
        if fault_after == "prepared":
            raise RuntimeError("injected session-mode fault")
        if self.storage.exists(sidecar_name):
            raise _fail()
        self._stage_source(operation_id, key, output)
        if self.storage.exists(source_name) or self.storage.exists(sidecar_name):
            raise _fail()
        self._install_stage(staging_name, key)
        self._append_journal(operation_id, phase="target_installed", operation="create", registry_id=registry_id,
            source_key=None, target_key=key,
            prior_record_digest=_digest(prior_bytes) if prior_bytes is not None else None,
            prior_record=prior_bytes.decode("utf-8") if prior_bytes is not None else None,
            target_binding=binding,
            target_record=target_record,
            output_digest=_digest(output), output_size=len(output), staging_name=staging_name)
        if fault_after == "target_installed":
            raise RuntimeError("injected session-mode fault")
        self._verify_canonical_output(key, output)
        self.storage.write(self._record_name(key), _canonical_json(record))
        if fault_after == "mode_record_published":
            raise RuntimeError("injected session-mode fault")
        self._append_journal(operation_id, phase="mode_committed", operation="create", registry_id=registry_id,
            source_key=None, target_key=key,
            prior_record_digest=_digest(prior_bytes) if prior_bytes is not None else None,
            prior_record=prior_bytes.decode("utf-8") if prior_bytes is not None else None,
            target_binding=binding,
            target_record=target_record,
            output_digest=_digest(output), output_size=len(output))
        self._append_journal(operation_id, phase="complete", operation="create", registry_id=registry_id,
            source_key=None, target_key=key,
            prior_record_digest=_digest(prior_bytes) if prior_bytes is not None else None,
            prior_record=prior_bytes.decode("utf-8") if prior_bytes is not None else None,
            target_binding=binding,
            target_record=target_record,
            output_digest=_digest(output), output_size=len(output))
        return self.resolve(key)

    def bootstrap(self, *, offline_copy: bool, fault_after: str | None = None) -> int:
        self._reject_files_held()
        with self.backend.locked_namespace_migration():
            hint = self._inventory_key_hint()
            with self._execution_guards(hint), self.backend.locked_session_files():
                if self._inventory_key_hint() != hint:
                    raise _fail()
                return self._bootstrap_locked(offline_copy=offline_copy, fault_after=fault_after)

    def _bootstrap_locked(self, *, offline_copy: bool, fault_after: str | None = None) -> int:
        if offline_copy is not True:
            raise _fail("offline copied-state assertion is required; older binaries do not honor this lease")
        items = self._source_inventory()
        inventory_rows: list[dict[str, object]] = []
        for item in items:
            inventory_rows.append(
                {
                    "key": item.key,
                    "source_name": item.source_name,
                    "source_digest": item.source_digest,
                    "source_size": item.source_size,
                    "sidecar_name": item.sidecar_name,
                    "sidecar_state": "present" if item.sidecar_bytes is not None else "absent",
                    "sidecar_digest": item.sidecar_digest,
                    "sidecar_size": item.sidecar_size,
                }
            )
        inventory = _canonical_json({"items": inventory_rows})

        registry_exists = self.storage.exists(".session-mode/registry.json")
        witness_exists = self.storage.exists("session-mode-binding.json", witness=True)
        active_match = False
        if registry_exists and witness_exists:
            registry_value = _parse_json(self.storage.read(".session-mode/registry.json"))
            witness_value = _parse_json(
                self.storage.read("session-mode-binding.json", witness=True)
            )
            if registry_value == witness_value and registry_value.get("state") == "active":
                self._witnesses()
                active_match = True

        masters: list[tuple[str, dict[str, object]]] = []
        for candidate in self.storage.list(".session-mode/journals"):
            if not self._hex32(candidate):
                raise _fail()
            journal = self._journal_records(candidate)
            first = journal[0]
            target = first.get("target_binding")
            if (
                first.get("operation") == "bootstrap"
                and first.get("source_key") is None
                and first.get("target_key") is None
                and isinstance(target, dict)
                and "registry" in target
                and journal[-1].get("phase") not in {"complete", "aborted"}
            ):
                masters.append((candidate, first))
        if len(masters) > 1:
            raise _fail()
        if active_match and not masters:
            return len(self.storage.list(".session-mode/records", suffix=".json"))
        if masters:
            operation_id, prepared_record = masters[0]
            target_binding = prepared_record.get("target_binding")
            if not isinstance(target_binding, dict):
                raise _fail()
            registry_value = cast(dict[object, object], target_binding).get("registry")
            if (
                not isinstance(registry_value, dict)
                or cast(dict[object, object], target_binding).get("inventory_digest") != _digest(inventory)
                or prepared_record.get("input_size") != len(inventory)
            ):
                raise _fail()
            registry = cast(dict[str, object], registry_value)
            committed_at = cast(str, prepared_record["timestamp"])
        else:
            if registry_exists or witness_exists or self.storage.exists(".session-mode"):
                raise _fail()
            operation_id = secrets.token_hex(16)
            registry = self._new_registry(operation_id)
            master_binding: dict[str, object] = {
                "registry": registry,
                "inventory_digest": _digest(inventory),
            }
            self._append_journal(
                operation_id,
                phase="prepared",
                operation="bootstrap",
                registry_id=cast(str, registry["registry_id"]),
                source_key=None,
                target_key=None,
                prior_record_digest=None,
                target_binding=master_binding,
                input_digest=_digest(inventory),
                input_size=len(inventory),
            )
            committed_at = cast(str, self._journal_records(operation_id)[0]["timestamp"])
            if fault_after == "master_prepared":
                raise RuntimeError("injected session-mode fault")

        registry_id = cast(str, registry["registry_id"])
        preparing = _canonical_json(registry)
        active_registry = dict(registry)
        active_registry["state"] = "active"
        active = _canonical_json(active_registry)

        if registry_exists:
            existing = self.storage.read(".session-mode/registry.json")
            if existing not in {preparing, active}:
                raise _fail()
        else:
            self.storage.write(".session-mode/registry.json", preparing)
        if fault_after == "registry_preparing":
            raise RuntimeError("injected session-mode fault")
        if witness_exists:
            existing = self.storage.read("session-mode-binding.json", witness=True)
            if existing not in {preparing, active}:
                raise _fail()
        else:
            self.storage.write("session-mode-binding.json", preparing, witness=True)
        if fault_after == "witness_preparing":
            raise RuntimeError("injected session-mode fault")

        for item in items:
            key = item.key
            data = item.source_bytes
            header = self._read_header(key)
            metadata = header.get("metadata")
            if not isinstance(metadata, dict):
                raise _fail()
            mode: Mode = "canonical" if cast(dict[object, object], metadata).get("source_schema_version") == 1 else "legacy"
            if "source_schema_version" in metadata and mode != "canonical":
                raise _fail()
            if mode == "canonical":
                snapshot = self.backend.source.load_source_snapshot(key)
                revision, generation, incarnation = snapshot.state.revision, snapshot.state.generation, snapshot.state.incarnation
            else:
                session = self.backend.parse_legacy_bytes(data, expected_key=key)
                revision, generation, incarnation = session.revision, session.generation, session.incarnation
            item_operation = hashlib.sha256(
                f"{operation_id}:{self.backend.storage_key(key)}".encode("ascii")
            ).hexdigest()[:32]
            record = self._record(key, registry_id=registry_id, operation_id=item_operation,
                mode=mode, state="active", revision=revision, generation=generation,
                incarnation=incarnation, transition_revision=revision,
                transition_digest=_digest(data), committed_at=committed_at)
            binding = self._binding(record)
            phases = (
                [entry["phase"] for entry in self._journal_records(item_operation)]
                if self.storage.exists(f".session-mode/journals/{item_operation}")
                else []
            )
            common: _JournalCommon = {
                "operation": "bootstrap", "registry_id": registry_id, "source_key": key,
                "target_key": key, "prior_record_digest": None, "target_binding": binding,
                "input_digest": _digest(data), "input_size": len(data), "output_digest": _digest(data),
                "output_size": len(data), "sidecar_digest": item.sidecar_digest,
                "sidecar_size": item.sidecar_size, "sidecar_bytes": item.sidecar_text,
            }
            if not phases:
                self._append_journal(item_operation, phase="prepared", **common)
                phases.append("prepared")
            record_bytes = _canonical_json(record)
            if self.storage.exists(self._record_name(key)):
                if self.storage.read(self._record_name(key)) != record_bytes:
                    raise _fail()
            else:
                self.storage.write(self._record_name(key), record_bytes)
            if "target_installed" not in phases:
                self._append_journal(item_operation, phase="target_installed", **common)
                phases.append("target_installed")
            if "mode_committed" not in phases:
                self._append_journal(item_operation, phase="mode_committed", **common)
                phases.append("mode_committed")
            if "complete" not in phases:
                self._append_journal(item_operation, phase="complete", **common)
        if fault_after == "records_committed":
            raise RuntimeError("injected session-mode fault")

        master = self._journal_records(operation_id)
        if master[-1]["phase"] == "prepared":
            first = master[0]
            self._append_journal(
                operation_id,
                phase="target_installed",
                operation="bootstrap",
                registry_id=registry_id,
                source_key=None,
                target_key=None,
                prior_record_digest=None,
                target_binding=cast(dict[str, object], first["target_binding"]),
                input_digest=_digest(inventory),
                input_size=len(inventory),
            )
        if fault_after == "target_installed":
            raise RuntimeError("injected session-mode fault")
        self.storage.write(".session-mode/registry.json", active)
        if fault_after == "registry_active":
            raise RuntimeError("injected session-mode fault")
        self.storage.write("session-mode-binding.json", active, witness=True)
        master = self._journal_records(operation_id)
        master_phases = [entry["phase"] for entry in master]
        if "mode_committed" not in master_phases:
            first = master[0]
            common_master: _JournalCommon = {
                "operation": "bootstrap", "registry_id": registry_id, "source_key": None,
                "target_key": None, "prior_record_digest": None,
                "target_binding": cast(dict[str, object], first["target_binding"]),
                "input_digest": _digest(inventory), "input_size": len(inventory),
            }
            self._append_journal(operation_id, phase="mode_committed", **common_master)
            if fault_after == "mode_committed":
                raise RuntimeError("injected session-mode fault")
            self._append_journal(operation_id, phase="complete", **common_master)
        elif "complete" not in master_phases:
            first = master[0]
            self._append_journal(
                operation_id,
                phase="complete",
                operation="bootstrap",
                registry_id=registry_id,
                source_key=None,
                target_key=None,
                prior_record_digest=None,
                target_binding=cast(dict[str, object], first["target_binding"]),
                input_digest=_digest(inventory),
                input_size=len(inventory),
            )
        self._witnesses()
        return len(items)

    @_locked_operation(execution=True)
    def convert(self, key: str, *, offline_copy: bool, fault_after: str | None = None) -> ResolvedSessionMode:
        if offline_copy is not True:
            raise _fail("offline copied-state assertion is required; older binaries do not honor this lease")
        if self.storage.exists(f"{self.backend.storage_key(key)}.checkpoint.json"):
            raise _fail()
        current = self.resolve(key)
        if current.state != "active" or current.mode != "legacy":
            raise _fail()
        registry = self._registry()
        prior_bytes = self.storage.read(self._record_name(key))
        prior_record = _parse_json(prior_bytes)
        original = self.storage.read(
            f"{self.backend.storage_key(key)}.jsonl", limit=67_108_864
        )
        session = self.backend.parse_legacy_bytes(original, expected_key=key)
        from nanobot.session.busy_inbox import require_busy_inbox_disposition

        require_busy_inbox_disposition(session.metadata)
        forbidden = {
            "runtime_checkpoint",
            "pending_user_turn",
            "pending_user_followups",
            "execution_lease",
            "execution_recovery_required",
            "artifact_conflict",
            "source_conflict",
            "recovery_required",
            "receipt_recovery_required",
        }
        if forbidden.intersection(session.metadata):
            raise _fail()
        operation_id = secrets.token_hex(16)
        rollback_relative = f"rollback/{self.backend.storage_key(key)}/{operation_id}/legacy.jsonl"
        output, revision = self._render_canonical(session)
        staging_name = f"staging/{operation_id}/canonical-source.jsonl"
        record = self._record(key, registry_id=cast(str, registry["registry_id"]), operation_id=operation_id,
            mode="canonical", state="active", revision=revision,
            generation=session.generation, incarnation=session.incarnation,
            transition_revision=revision, transition_digest=_digest(output),
            record_revision=cast(int, prior_record["record_revision"]) + 1)
        binding = self._binding(record)
        target_record = _canonical_json(record).decode("utf-8")
        self._append_journal(operation_id, phase="prepared", operation="convert",
            registry_id=cast(str, registry["registry_id"]), source_key=key, target_key=key,
            prior_record_digest=_digest(prior_bytes), target_binding=binding,
            prior_record=prior_bytes.decode("utf-8"), target_record=target_record,
            input_digest=_digest(original), input_size=len(original), output_digest=_digest(output),
            output_size=len(output), staging_name=staging_name, rollback_name=rollback_relative)
        if fault_after == "prepared":
            raise RuntimeError("injected session-mode fault")
        if self.storage.exists(f"{self.backend.storage_key(key)}.checkpoint.json"):
            raise _fail()
        self.storage.write(f".session-mode/{rollback_relative}", original, replace=False)
        self._stage_source(operation_id, key, output)
        if self.storage.read(f"{self.backend.storage_key(key)}.jsonl", limit=67_108_864) != original:
            raise _fail()
        if self.storage.exists(f"{self.backend.storage_key(key)}.checkpoint.json"):
            raise _fail()
        self._install_stage(staging_name, key)
        self._append_journal(operation_id, phase="target_installed", operation="convert",
            registry_id=cast(str, registry["registry_id"]), source_key=key, target_key=key,
            prior_record_digest=_digest(prior_bytes), target_binding=binding,
            prior_record=prior_bytes.decode("utf-8"), target_record=target_record,
            input_digest=_digest(original), input_size=len(original), output_digest=_digest(output),
            output_size=len(output), staging_name=staging_name, rollback_name=rollback_relative)
        if fault_after == "target_installed":
            raise RuntimeError("injected session-mode fault")
        self._verify_canonical_output(key, output)
        self.storage.write(self._record_name(key), _canonical_json(record))
        if fault_after == "mode_record_published":
            raise RuntimeError("injected session-mode fault")
        self._append_journal(operation_id, phase="mode_committed", operation="convert",
            registry_id=cast(str, registry["registry_id"]), source_key=key, target_key=key,
            prior_record_digest=_digest(prior_bytes), target_binding=binding,
            prior_record=prior_bytes.decode("utf-8"), target_record=target_record,
            input_digest=_digest(original), input_size=len(original), output_digest=_digest(output),
            output_size=len(output), rollback_name=rollback_relative)
        self._append_journal(operation_id, phase="complete", operation="convert",
            registry_id=cast(str, registry["registry_id"]), source_key=key, target_key=key,
            prior_record_digest=_digest(prior_bytes), target_binding=binding,
            prior_record=prior_bytes.decode("utf-8"), target_record=target_record,
            input_digest=_digest(original), input_size=len(original), output_digest=_digest(output),
            output_size=len(output), rollback_name=rollback_relative)
        return self.resolve(key)

    def reset(
        self,
        key: str,
        replacement_events: list[dict[str, object]],
        *,
        owner: object | None = None,
        fault_after: str | None = None,
    ) -> ResolvedSessionMode:
        """Journal and install one frozen reset under execution-before-files order."""

        from nanobot.session.execution_owner import ExecutionOwner, execution_transaction

        self._reject_files_held()
        with execution_transaction(
            self.backend,
            key,
            cast("ExecutionOwner | None", owner),
            destructive=True,
            allow_legacy=True,
        ):
            # The transaction admitted None or an exact ExecutionOwner before
            # entering this sole call to the internal reset implementation.
            return self._reset_unlocked(
                key, replacement_events, owner=cast("ExecutionOwner | None", owner), fault_after=fault_after
            )

    def _reset_unlocked(
        self,
        key: str,
        replacement_events: list[dict[str, object]],
        *,
        owner: _ExecutionOwner | None,
        fault_after: str | None,
    ) -> ResolvedSessionMode:

        current = self.resolve(key)
        if current.state != "active":
            raise _fail()
        if current.mode == "legacy":
            if owner is not None:
                raise _fail()
            item = self._inventory_for_key(key)
            original = item.source_bytes
            session = self.backend.parse_legacy_bytes(original, expected_key=key)
            if item.sidecar_bytes is not None:
                session = self.backend.overlay_managed_checkpoint_bytes(session, item.sidecar_bytes)
            from nanobot.session.busy_inbox import require_busy_inbox_disposition

            require_busy_inbox_disposition(session.metadata)
            session.replace_messages(replacement_events)
            session.metadata.pop("context_checkpoint", None)
            session.metadata.pop("runtime_checkpoint", None)
            session.metadata.pop("_last_summary", None)
            session.provider_state = None
            session.revision += 1
            session.generation += 1
            session.updated_at = datetime.now(timezone.utc)
            output = self.backend.render_legacy_bytes(session)
            prior_bytes = self.storage.read(self._record_name(key))
            prior = _parse_json(prior_bytes)
            operation_id = secrets.token_hex(16)
            record = self._record(
                key, registry_id=cast(str, prior["registry_id"]), operation_id=operation_id,
                mode="legacy", state="active", revision=session.revision,
                generation=session.generation, incarnation=session.incarnation,
                transition_revision=session.revision, transition_digest=_digest(output),
                record_revision=cast(int, prior["record_revision"]) + 1,
                committed_at=session.updated_at.isoformat(),
            )
            common: _JournalCommon = {
                "operation": "reset", "registry_id": cast(str, prior["registry_id"]),
                "source_key": key, "target_key": key, "prior_record_digest": _digest(prior_bytes),
                "target_binding": self._binding(record), "prior_record": prior_bytes.decode("utf-8"),
                "target_record": _canonical_json(record).decode("utf-8"),
                "input_digest": item.source_digest, "input_size": item.source_size,
                "output_digest": _digest(output), "output_size": len(output),
                "sidecar_digest": item.sidecar_digest, "sidecar_size": item.sidecar_size,
                "sidecar_bytes": item.sidecar_text,
            }
            self._append_journal(operation_id, phase="prepared", **common)
            if fault_after == "prepared":
                raise RuntimeError("injected session-mode fault")
            self._verify_pair_exact(item)
            if self.storage.read(self._record_name(key)) != prior_bytes:
                raise _fail()
            self.storage.write(item.source_name, output)
            if fault_after == "source_installed":
                raise RuntimeError("injected session-mode fault")
            if item.sidecar_bytes is not None:
                if self.storage.read(item.sidecar_name) != item.sidecar_bytes:
                    raise _fail()
                self.storage.unlink(item.sidecar_name)
            if fault_after == "sidecar_removed":
                raise RuntimeError("injected session-mode fault")
            self._append_journal(operation_id, phase="target_installed", **common)
            if fault_after == "target_installed":
                raise RuntimeError("injected session-mode fault")
            if self.storage.read(item.source_name, limit=67_108_864) != output or self.storage.exists(item.sidecar_name):
                raise _fail()
            self.storage.write(self._record_name(key), _canonical_json(record))
            if fault_after == "mode_record_published":
                raise RuntimeError("injected session-mode fault")
            self._append_journal(operation_id, phase="mode_committed", **common)
            if fault_after == "mode_committed":
                raise RuntimeError("injected session-mode fault")
            self._append_journal(operation_id, phase="complete", **common)
            return self.resolve(key)
        if current.mode != "canonical":
            raise _fail()
        from nanobot.session.execution_owner import ExecutionOwnershipError
        from nanobot.session.source import validate_execution_links

        self._inventory_for_key(key)
        reset_input = SourceSessionStore.read_reset_input(self.backend.source, key)
        if owner is None and (
            validate_execution_links(reset_input.events)
            or reset_input.has_runtime_checkpoint()
        ):
            raise ExecutionOwnershipError()
        snapshot = reset_input.snapshot(self.backend.source)
        planned_at = datetime.now(timezone.utc).isoformat()
        plan = self.backend.source.plan_source_reset(
            snapshot,
            replacement_events,
            updated_at=planned_at,
        )
        prior_bytes = self.storage.read(self._record_name(key))
        prior = _parse_json(prior_bytes)
        operation_id = secrets.token_hex(16)
        registry_id = cast(str, prior["registry_id"])
        record = self._record(
            key,
            registry_id=registry_id,
            operation_id=operation_id,
            mode="canonical",
            state="active",
            revision=plan.target_state.revision,
            generation=plan.target_state.generation,
            incarnation=plan.target_state.incarnation,
            transition_revision=plan.target_state.revision,
            transition_digest=_digest(plan.target_bytes),
            record_revision=cast(int, prior["record_revision"]) + 1,
            committed_at=planned_at,
        )
        binding = self._binding(record)
        input_bytes = self.storage.read(
            f"{self.backend.storage_key(key)}.jsonl", limit=67_108_864
        )
        common = {
            "operation": "reset", "registry_id": registry_id, "source_key": key, "target_key": key,
            "prior_record_digest": _digest(prior_bytes), "target_binding": binding,
            "prior_record": prior_bytes.decode("utf-8"),
            "target_record": _canonical_json(record).decode("utf-8"),
            "input_digest": _digest(input_bytes), "input_size": len(input_bytes),
            "output_digest": _digest(plan.target_bytes), "output_size": len(plan.target_bytes),
        }
        self._append_journal(operation_id, phase="prepared", **common)
        if fault_after == "prepared":
            raise RuntimeError("injected session-mode fault")
        if self.storage.exists(f"{self.backend.storage_key(key)}.checkpoint.json") or self.storage.read(
            f"{self.backend.storage_key(key)}.jsonl", limit=67_108_864
        ) != input_bytes or self.storage.read(self._record_name(key)) != prior_bytes:
            raise _fail()
        SourceSessionStore.install_prepared_mode_reset(
            self.backend,
            key,
            replacement_events,
            owner=owner,
            plan=plan,
        )
        if fault_after == "source_installed":
            raise RuntimeError("injected session-mode fault")
        self._append_journal(operation_id, phase="target_installed", **common)
        if fault_after == "target_installed":
            raise RuntimeError("injected session-mode fault")
        self._verify_canonical_output(key, plan.target_bytes)
        self.storage.write(self._record_name(key), _canonical_json(record))
        if fault_after == "mode_record_published":
            raise RuntimeError("injected session-mode fault")
        self._append_journal(operation_id, phase="mode_committed", **common)
        if fault_after == "mode_committed":
            raise RuntimeError("injected session-mode fault")
        self._append_journal(operation_id, phase="complete", **common)
        return self.resolve(key)

    def fork(
        self,
        source_key: str,
        target_key: str,
        before_user_index: int,
        *,
        fault_after: str | None = None,
    ) -> ResolvedSessionMode | None:
        """Snapshot and install one fork while retaining sorted source/target locks."""

        if source_key == target_key or type(before_user_index) is not int or before_user_index < 0:
            raise _fail()
        with self._execution_guards((source_key, target_key)), self.backend.locked_session_files():
            return self._fork_unlocked(
                source_key, target_key, before_user_index, fault_after=fault_after
            )

    def _fork_unlocked(
        self,
        source_key: str,
        target_key: str,
        before_user_index: int,
        *,
        fault_after: str | None,
    ) -> ResolvedSessionMode | None:
        source = self.resolve(source_key)
        target = self.resolve(target_key)
        if source.state != "active" or target.state not in {"absent", "deleted"}:
            raise _fail()
        source_item = self._inventory_for_key(source_key)
        source_record_bytes = self.storage.read(self._record_name(source_key))
        source_record_value = _parse_json(source_record_bytes)
        source_binding = self._binding(source_record_value)
        if source.mode == "canonical":
            header, source_events = validate_source_session_bytes(
                source_item.source_bytes, expected_key=source_key
            )
            if source_item.sidecar_bytes is not None:
                raise _fail()
            metadata = cast(dict[str, Any], header["metadata"])
            from nanobot.session.execution_owner import ExecutionOwnershipError
            from nanobot.session.source import validate_execution_links

            if validate_execution_links(source_events) or any(
                name in metadata for name in ("execution_lease", "execution_recovery_required")
            ):
                raise ExecutionOwnershipError()
            source_messages = source_events
            source_session = None
        else:
            source_session = self.backend.parse_legacy_bytes(
                source_item.source_bytes, expected_key=source_key
            )
            if source_item.sidecar_bytes is not None:
                source_session = self.backend.overlay_managed_checkpoint_bytes(
                    source_session, source_item.sidecar_bytes
                )
            header = None
            source_messages = source_session.messages
        copied: list[dict[str, Any]] = []
        user_index = 0
        for message in source_messages:
            if message.get("role") == "user" and not is_hidden_history_message(message):
                if user_index == before_user_index:
                    break
                user_index += 1
            copied.append(
                deepcopy(message) if source.mode == "canonical" else public_history_message(message)
            )
        if user_index != before_user_index:
            return None
        source_metadata = (
            cast(dict[str, Any], header["metadata"])
            if header is not None
            else cast(Session, source_session).metadata
        )
        metadata = deepcopy(source_metadata)
        for name in {
            "goal_state", "pending_user_turn", "pending_user_followups", "runtime_checkpoint",
            "session_handle", "webui_recovery", "thread_goal", "title", "title_user_edited",
            "artifact_manifest", "context_checkpoint",
        }:
            metadata.pop(name, None)
        source_archive = (
            cast(int, header.get("last_archived", 0))
            if header is not None
            else cast(Session, source_session).last_archived
        )
        last_consolidated = min(source_archive, len(copied))
        if source_archive > len(copied):
            metadata.pop("_last_summary", None)
            last_consolidated = 0
        now = datetime.now(timezone.utc)
        target_session = Session(
            key=target_key,
            messages=copied,
            created_at=now,
            updated_at=now,
            metadata=metadata,
            last_consolidated=last_consolidated,
        )
        if header is not None:
            target_header = deepcopy(header)
            target_header.update({
                "key": target_key,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "metadata": metadata,
                "revision": 1,
                "generation": 0,
                "incarnation": target_session.incarnation,
                "last_archived": last_consolidated,
                "last_consolidated": last_consolidated,
            })
            target_bytes = b"".join(
                (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                for row in (target_header, *copied)
            )
        else:
            target_bytes, _ = self._render_canonical(target_session)
        target_header, _events = validate_source_session_bytes(target_bytes, expected_key=target_key)
        target_source = f"{self.backend.storage_key(target_key)}.jsonl"
        target_sidecar = f"{self.backend.storage_key(target_key)}.checkpoint.json"
        if self.storage.exists(target_source) or self.storage.exists(target_sidecar):
            raise _fail()
        registry = self._registry()
        operation_id = secrets.token_hex(16)
        staging_name = f"staging/{operation_id}/canonical-source.jsonl"
        prior_bytes = (
            self.storage.read(self._record_name(target_key))
            if self.storage.exists(self._record_name(target_key))
            else None
        )
        record = self._record(
            target_key,
            registry_id=cast(str, registry["registry_id"]),
            operation_id=operation_id,
            mode="canonical",
            state="active",
            revision=cast(int, target_header["revision"]),
            generation=cast(int, target_header["generation"]),
            incarnation=cast(str, target_header["incarnation"]),
            transition_revision=cast(int, target_header["revision"]),
            transition_digest=_digest(target_bytes),
            record_revision=(
                cast(int, _parse_json(prior_bytes)["record_revision"]) + 1
                if prior_bytes is not None
                else 1
            ),
        )
        binding = self._binding(record)
        target_record = _canonical_json(record).decode("utf-8")
        common: _JournalCommon = {
            "operation": "fork", "registry_id": cast(str, registry["registry_id"]),
            "source_key": source_key, "target_key": target_key,
            "prior_record_digest": _digest(prior_bytes) if prior_bytes is not None else None,
            "prior_record": prior_bytes.decode("utf-8") if prior_bytes is not None else None,
            "target_record": target_record, "target_binding": binding,
            "output_digest": _digest(target_bytes), "output_size": len(target_bytes),
            "staging_name": staging_name,
            "source_record": source_record_bytes.decode("utf-8"),
            "source_record_digest": _digest(source_record_bytes),
            "source_binding": source_binding,
            "source_input_digest": source_item.source_digest,
            "source_input_size": source_item.source_size,
            "sidecar_digest": source_item.sidecar_digest,
            "sidecar_size": source_item.sidecar_size,
            "sidecar_bytes": source_item.sidecar_text,
        }
        self._append_journal(operation_id, phase="prepared", **common)
        if fault_after == "prepared":
            raise RuntimeError("injected session-mode fault")
        self._stage_source(operation_id, target_key, target_bytes)
        if fault_after == "staged":
            raise RuntimeError("injected session-mode fault")
        if (
            self.storage.read(self._record_name(source_key)) != source_record_bytes
            or self.storage.read(source_item.source_name, limit=67_108_864)
            != source_item.source_bytes
            or self.storage.exists(source_item.sidecar_name)
            != (source_item.sidecar_bytes is not None)
            or (
                source_item.sidecar_bytes is not None
                and self.storage.read(source_item.sidecar_name) != source_item.sidecar_bytes
            )
            or self.storage.exists(target_sidecar)
            or self.storage.exists(target_source)
        ):
            raise _fail()
        self._install_stage(staging_name, target_key)
        if fault_after == "source_installed":
            raise RuntimeError("injected session-mode fault")
        self._append_journal(operation_id, phase="target_installed", **common)
        if fault_after == "target_installed":
            raise RuntimeError("injected session-mode fault")
        self._verify_canonical_output(target_key, target_bytes)
        self.storage.write(self._record_name(target_key), target_record.encode("utf-8"))
        if fault_after == "mode_record_published":
            raise RuntimeError("injected session-mode fault")
        self._append_journal(operation_id, phase="mode_committed", **common)
        if fault_after == "mode_committed":
            raise RuntimeError("injected session-mode fault")
        self._append_journal(operation_id, phase="complete", **common)
        return self.resolve(target_key)

    @_locked_operation()
    def list_modes(self) -> list[ResolvedSessionMode]:
        registry = self._witnesses()
        if registry is None:
            if self._source_names():
                raise _fail("session namespace requires explicit offline bootstrap")
            return []
        self._assert_no_incomplete("", registry)
        inventory = {item.key: item for item in self._source_inventory()}
        results: list[ResolvedSessionMode] = []
        keys: set[str] = set()
        active_keys: set[str] = set()
        for name in self.storage.list(".session-mode/records", suffix=".json"):
            key = self.backend.decode_storage_key(name.removesuffix(".json"))
            if (
                key is None
                or self.backend.storage_key(key) != name.removesuffix(".json")
                or key in keys
            ):
                raise _fail()
            keys.add(key)
            record = self._validate_record(key, registry, inventory=inventory)
            if record is None:
                raise _fail()
            if record["state"] == "active":
                active_keys.add(key)
                results.append(ResolvedSessionMode(
                    "active", cast(Mode, record["mode"]), cast(str, registry["registry_id"]),
                    cast(int, record["record_revision"]), cast(str, record["incarnation"]),
                    cast(int, record["generation"]),
                ))
        if set(inventory) != active_keys:
            raise _fail()
        return results

    @_locked_operation(execution=True)
    def begin_export(
        self,
        key: str,
        destination: Path,
        files: Mapping[str, bytes],
    ) -> SessionExportTransaction:
        """Create a journaled export stage without changing session authority."""

        current = self.resolve(key)
        if current.state != "active" or current.mode != "canonical" or current.binding is None:
            raise _fail()
        destination = Path(destination)
        if not destination.is_absolute() or destination.exists() or destination.is_symlink():
            raise _fail()
        parent = destination.parent
        if (
            destination.name in {"", ".", ".."}
            or not parent.is_dir()
            or parent.is_symlink()
            or parent.resolve(strict=True) != parent
        ):
            raise _fail()
        if not files or len(files) > 64:
            raise _fail()
        detached: dict[str, bytes] = {}
        plans: list[SessionExportFilePlan] = []
        for name, data in sorted(files.items()):
            if (
                not isinstance(cast(object, name), str)
                or name in {"", ".", ".."}
                or "/" in name
                or "\\" in name
                or not isinstance(cast(object, data), bytes)
                or len(data) > 67_108_864
            ):
                raise _fail()
            detached[name] = bytes(data)
            plans.append(SessionExportFilePlan(name, _digest(data), len(data)))
        plan = SessionExportPlan(str(destination), tuple(plans))
        parent_pin = PinnedDirectoryPath(parent)
        registry = self._registry()
        record_bytes = self.storage.read(self._record_name(key))
        record = _parse_json(record_bytes)
        binding = self._binding(record)
        operation_id = secrets.token_hex(16)
        parent_stat = os.fstat(parent_pin.fd)
        export_binding: dict[str, object] = {
            "session": binding,
            "destination": str(destination),
            "destination_name": destination.name,
            "parent_device": parent_stat.st_dev,
            "parent_inode": parent_stat.st_ino,
            "files": [
                {"name": item.name, "digest": item.digest, "size": item.size}
                for item in plan.files
            ],
        }
        plan_bytes = _canonical_json({"files": export_binding["files"]})
        self._append_journal(
            operation_id,
            phase="prepared",
            operation="export",
            registry_id=cast(str, registry["registry_id"]),
            source_key=key,
            target_key=None,
            prior_record_digest=_digest(record_bytes),
            target_binding=export_binding,
            input_digest=_digest(record_bytes),
            input_size=len(record_bytes),
            output_digest=_digest(plan_bytes),
            output_size=sum(item.size for item in plan.files),
            staging_name=f"exports/{operation_id}",
        )
        for name, data in detached.items():
            self.storage.write(
                f".session-mode/exports/{operation_id}/{name}",
                data, replace=False, limit=67_108_864,
            )
        return SessionExportTransaction(
            self,
            operation_id,
            key,
            destination,
            export_binding,
            plan,
            parent_pin,
        )

    @contextmanager
    def locked_snapshot(self, key: str) -> Generator[LockedSessionSnapshot, None, None]:
        """Yield exact record/source bytes while the existing storage lock is held."""

        with self._execution_guards((key,)), self.backend.locked_session_files():
            scope = self._guard_local.scope
            current = self.resolve(key)
            if current.state != "active" or current.binding is None:
                raise _fail()
            record = self.storage.read(self._record_name(key))
            source = self.storage.read(
                f"{self.backend.storage_key(key)}.jsonl", limit=67_108_864
            )
            yield LockedSessionSnapshot(current.binding, record, source)
            self._require_open()
            if self._execution_identity() != scope[0] or getattr(self._guard_local, "scope", None) is not scope:
                raise _fail()
            if self.storage.read(self._record_name(key)) != record:
                raise _fail()
            if self.storage.read(
                f"{self.backend.storage_key(key)}.jsonl", limit=67_108_864
            ) != source:
                raise _fail()
            if current.binding[2] == "canonical" and self.storage.exists(
                f"{self.backend.storage_key(key)}.checkpoint.json"
            ):
                raise _fail()

    def delete(self, key: str, *, fault_after: str | None = None) -> bool:
        self._reject_files_held()
        current = self.resolve(key)
        if current.state != "active":
            return False
        from nanobot.session.execution_owner import execution_transaction

        with execution_transaction(
            self.backend, key, owner=None, allow_legacy=True, destructive=True
        ):
            return self._delete_unlocked(key, fault_after=fault_after)

    def _delete_unlocked(self, key: str, *, fault_after: str | None) -> bool:
        current = self.resolve(key)
        if current.state != "active":
            return False
        registry = self._registry()
        old_bytes = self.storage.read(self._record_name(key))
        old = _parse_json(old_bytes)
        item = self._inventory_for_key(key)
        data = item.source_bytes
        from nanobot.session.busy_inbox import require_busy_inbox_disposition

        require_busy_inbox_disposition(self._read_header(key)["metadata"])
        if current.mode == "canonical":
            from nanobot.session.execution_owner import ExecutionOwnershipError
            from nanobot.session.source import validate_execution_links

            _header, events = validate_source_session_bytes(data, expected_key=key)
            if validate_execution_links(events):
                raise ExecutionOwnershipError()
        sidecar_name = item.sidecar_name
        sidecar = item.sidecar_bytes
        operation_id = secrets.token_hex(16)
        tombstone = self._record(
            key,
            registry_id=cast(str, registry["registry_id"]),
            operation_id=operation_id,
            mode=None,
            state="deleted",
            revision=cast(int, old["transition_revision"]),
            generation=cast(int, old["generation"]),
            incarnation=cast(str, old["incarnation"]),
            transition_revision=cast(int, old["transition_revision"]),
            transition_digest=None,
            record_revision=cast(int, old["record_revision"]) + 1,
        )
        binding = self._binding(tombstone)
        target_record = _canonical_json(tombstone)
        common: _JournalCommon = {
            "operation": "delete",
            "registry_id": cast(str, registry["registry_id"]),
            "source_key": key,
            "target_key": None,
            "prior_record_digest": _digest(old_bytes),
            "target_binding": binding,
            "prior_record": old_bytes.decode("utf-8"),
            "target_record": target_record.decode("utf-8"),
            "input_digest": _digest(data),
            "input_size": len(data),
            "sidecar_digest": _digest(sidecar) if sidecar is not None else None,
            "sidecar_size": len(sidecar) if sidecar is not None else None,
            "sidecar_bytes": (base64.b64encode(sidecar).decode("ascii") if sidecar is not None else None),
        }
        self._append_journal(operation_id, phase="prepared", **common)
        if fault_after == "prepared":
            raise RuntimeError("injected session-mode fault")
        self._verify_pair_exact(item)
        if self.storage.read(self._record_name(key)) != old_bytes:
            raise _fail()
        self.storage.write(self._record_name(key), target_record)
        if fault_after == "mode_record_published":
            raise RuntimeError("injected session-mode fault")
        self._append_journal(operation_id, phase="mode_committed", **common)
        if fault_after == "mode_committed":
            raise RuntimeError("injected session-mode fault")
        self._verify_pair_exact(item)
        self.storage.unlink(f"{self.backend.storage_key(key)}.jsonl")
        if fault_after == "source_unlinked":
            raise RuntimeError("injected session-mode fault")
        self.storage.unlink(sidecar_name)
        if fault_after == "source_deleted":
            raise RuntimeError("injected session-mode fault")
        self._append_journal(operation_id, phase="complete", **common)
        return True


__all__ = [
    "ResolvedSessionMode",
    "LockedSessionSnapshot",
    "SessionModeBinding",
    "SessionModeError",
    "SessionModeResolver",
    "SessionExportTransaction",
]
