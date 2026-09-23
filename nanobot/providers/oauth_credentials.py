"""Dormant, provider-specific persistence for OAuth credentials.

This module deliberately has no default path and performs no network work. The
injected importer must return Codex refresh credentials or a Copilot GitHub
token; cached Copilot bearer tokens and endpoint data are not accepted.
"""

from __future__ import annotations

import errno
import json
import os
import re
import secrets
import stat
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, TypeAlias

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised by unsupported platforms
    _fcntl = None


_LONG_LIVED_TOKEN_SECONDS = 315_360_000
_MAX_JSON_BYTES = 65_536
_MISSING = object()
_INVALID_ERROR = "invalid credential store"
_UNAVAILABLE_ERROR = "credential store unavailable"
_CREDENTIAL_ERROR = "invalid credential"
_IMPORT_ERROR = "credential import failed"
_STALE_ERROR = "login attempt is stale"
_NEEDS_REFRESH_ERROR = "credential needs refresh"
_IMPORTER_UNAVAILABLE_ERROR = "credential importer unavailable"
_TOO_LARGE_ERROR = "credential record exceeds size limit"
_GENERATION_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_REQUIRED_JSON_KEYS = frozenset({"access", "refresh", "expires_at_ms"})
_OPTIONAL_JSON_KEYS = frozenset({"account_id"})


class CredentialError(RuntimeError):
    """A fixed, content-free credential persistence error."""


class CredentialNeedsRefreshError(CredentialError):
    """The canonical credential is expired and must be refreshed by its owner."""


class CredentialFlavor(str, Enum):
    CODEX = "codex"
    COPILOT = "copilot"


class CredentialStatus(str, Enum):
    ABSENT = "absent"
    AVAILABLE = "available"
    NEEDS_REFRESH = "needs_refresh"
    LOGGED_OUT = "logged_out"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class OAuthToken:
    """A validated canonical token record; token values are hidden from repr."""

    access: str = field(repr=False)
    refresh: str = field(repr=False)
    expires_at_ms: int
    account_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.access, str)
            or not self.access
            or not isinstance(self.refresh, str)
            or not isinstance(self.expires_at_ms, int)
            or isinstance(self.expires_at_ms, bool)
            or self.expires_at_ms <= 0
            or (
                self.account_id is not None
                and (not isinstance(self.account_id, str) or not self.account_id)
            )
        ):
            raise CredentialError(_CREDENTIAL_ERROR)

    def to_dict(self) -> dict[str, str | int]:
        result: dict[str, str | int] = {
            "access": self.access,
            "refresh": self.refresh,
            "expires_at_ms": self.expires_at_ms,
        }
        if self.account_id is not None:
            result["account_id"] = self.account_id
        return result

    @classmethod
    def from_json(cls, raw: str | bytes) -> OAuthToken:
        try:
            value = json.loads(raw, object_pairs_hook=_object_without_duplicates)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
            raise CredentialError(_CREDENTIAL_ERROR) from None
        return cls.from_mapping(value)

    @classmethod
    def from_mapping(cls, value: object) -> OAuthToken:
        if not isinstance(value, Mapping):
            raise CredentialError(_CREDENTIAL_ERROR)
        try:
            keys = set(value.keys())
        except Exception:
            raise CredentialError(_CREDENTIAL_ERROR) from None
        if not _REQUIRED_JSON_KEYS.issubset(keys) or keys - (
            _REQUIRED_JSON_KEYS | _OPTIONAL_JSON_KEYS
        ):
            raise CredentialError(_CREDENTIAL_ERROR)
        try:
            access = value["access"]
            refresh = value["refresh"]
            expires_at_ms = value["expires_at_ms"]
            account_id = value.get("account_id")
        except Exception:
            raise CredentialError(_CREDENTIAL_ERROR) from None
        if not isinstance(access, str) or not isinstance(refresh, str):
            raise CredentialError(_CREDENTIAL_ERROR)
        if not isinstance(expires_at_ms, int) or isinstance(expires_at_ms, bool):
            raise CredentialError(_CREDENTIAL_ERROR)
        if account_id is not None and not isinstance(account_id, str):
            raise CredentialError(_CREDENTIAL_ERROR)
        return cls(access, refresh, expires_at_ms, account_id)


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class CredentialStorePaths:
    canonical: Path
    state: Path
    lock: Path


def resolve_credential_store_paths(canonical_path: str | Path) -> CredentialStorePaths:
    """Resolve an explicit canonical path and reserve its two sidecar names."""
    raw_path = Path(canonical_path)
    if not raw_path.is_absolute() or not raw_path.name:
        raise ValueError("absolute credential path required")
    if raw_path.name.endswith((".state.json", ".lock")):
        raise ValueError("reserved credential path")
    try:
        parent = raw_path.parent.resolve(strict=True)
    except OSError:
        raise ValueError("credential parent directory required") from None
    if not parent.is_dir():
        raise ValueError("credential parent directory required")
    canonical = parent / raw_path.name
    return CredentialStorePaths(
        canonical=canonical,
        state=parent / f"{raw_path.name}.state.json",
        lock=parent / f"{raw_path.name}.lock",
    )


class _LoginGeneration:
    __slots__ = ("__owner", "__value")

    def __init__(self, owner: object, value: str) -> None:
        self.__owner = owner
        self.__value = value

    def _value_for(self, owner: object) -> str | None:
        return self.__value if self.__owner is owner else None

    def __repr__(self) -> str:
        return "<LoginGeneration>"

    def __copy__(self) -> _LoginGeneration:
        raise TypeError("login generation cannot be copied")

    def __deepcopy__(self, memo: object) -> _LoginGeneration:
        del memo
        raise TypeError("login generation cannot be copied")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("login generation cannot be pickled")


@dataclass(frozen=True, slots=True)
class _StoreState:
    generation: str
    logged_out: bool

    def to_dict(self, flavor: CredentialFlavor) -> dict[str, bool | int | str]:
        return {
            "version": 1,
            "generation": self.generation,
            "logged_out": self.logged_out,
            "flavor": flavor.value,
        }

    @classmethod
    def from_value(cls, value: object, flavor: CredentialFlavor) -> _StoreState:
        if not isinstance(value, Mapping):
            raise _InvalidStoreError
        expected = {"version", "generation", "logged_out", "flavor"}
        try:
            if set(value.keys()) != expected:
                raise _InvalidStoreError
            version = value["version"]
            generation = value["generation"]
            logged_out = value["logged_out"]
            stored_flavor = value["flavor"]
        except _InvalidStoreError:
            raise
        except Exception:
            raise _InvalidStoreError from None
        if (
            version != 1
            or not isinstance(generation, str)
            or _GENERATION_PATTERN.fullmatch(generation) is None
            or not isinstance(logged_out, bool)
            or stored_flavor != flavor.value
        ):
            raise _InvalidStoreError
        return cls(generation, logged_out)


class _InvalidStoreError(Exception):
    pass


class _UnavailableError(Exception):
    pass


CredentialInput: TypeAlias = OAuthToken | Mapping[str, object] | str
Importer: TypeAlias = Callable[[], CredentialInput | None]


class OAuthCredentialCoordinator:
    """Serialize and durably persist one explicitly selected provider credential."""

    def __init__(
        self,
        canonical_path: str | Path,
        flavor: CredentialFlavor | str,
        importer: Importer | None = None,
        *,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(flavor, CredentialFlavor):
            try:
                flavor = CredentialFlavor(flavor)
            except (ValueError, TypeError):
                raise ValueError("credential flavor required") from None
        self.paths = resolve_credential_store_paths(canonical_path)
        self.flavor = flavor
        self._importer = importer
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._generation_owner = object()

    def _now_ms(self) -> int:
        value = self._clock_ms()
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CredentialError(_UNAVAILABLE_ERROR)
        return value

    def _validate_token(self, token: OAuthToken) -> OAuthToken:
        if self.flavor is CredentialFlavor.CODEX:
            if not token.refresh or not token.account_id:
                raise CredentialError(_CREDENTIAL_ERROR)
        elif token.refresh != "":
            raise CredentialError(_CREDENTIAL_ERROR)
        return token

    def _coerce_input(self, value: object) -> OAuthToken:
        if self.flavor is CredentialFlavor.COPILOT:
            if not isinstance(value, str) or not value:
                raise CredentialError(_CREDENTIAL_ERROR)
            token = OAuthToken(
                access=value,
                refresh="",
                expires_at_ms=self._now_ms() + _LONG_LIVED_TOKEN_SECONDS * 1000,
            )
        elif isinstance(value, OAuthToken):
            token = value
        elif isinstance(value, Mapping):
            token = OAuthToken.from_mapping(value)
        else:
            raise CredentialError(_CREDENTIAL_ERROR)
        return self._validate_token(token)

    @contextmanager
    def _locked(self) -> Iterator[int]:
        if _fcntl is None or not hasattr(os, "O_NOFOLLOW"):
            raise _UnavailableError
        directory_fd = -1
        lock_fd = -1
        try:
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_flags |= getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
            directory_fd = os.open(self.paths.canonical.parent, directory_flags)
            directory_info = os.fstat(directory_fd)
            if not stat.S_ISDIR(directory_info.st_mode):
                raise _UnavailableError
            try:
                lock_fd = os.open(
                    self.paths.lock.name,
                    os.O_RDWR
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=directory_fd,
                )
                os.fchmod(lock_fd, 0o600)
            except FileExistsError:
                lock_fd = os.open(
                    self.paths.lock.name,
                    os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=directory_fd,
                )
            lock_info = os.fstat(lock_fd)
            if (
                not stat.S_ISREG(lock_info.st_mode)
                or lock_info.st_nlink != 1
                or lock_info.st_uid != os.getuid()
                or stat.S_IMODE(lock_info.st_mode) != 0o600
            ):
                raise _UnavailableError
            _fcntl.flock(lock_fd, _fcntl.LOCK_EX)
            yield directory_fd
        except (_InvalidStoreError, _UnavailableError):
            raise
        except OSError:
            raise _UnavailableError from None
        finally:
            if lock_fd >= 0:
                if _fcntl is not None:
                    try:
                        _fcntl.flock(lock_fd, _fcntl.LOCK_UN)
                    except OSError:
                        pass
                try:
                    os.close(lock_fd)
                except OSError:
                    pass
            if directory_fd >= 0:
                try:
                    os.close(directory_fd)
                except OSError:
                    pass

    def _read_private_json(self, directory_fd: int, name: str) -> object:
        try:
            fd = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            return _MISSING
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise _InvalidStoreError from None
            raise _UnavailableError from None
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_size > _MAX_JSON_BYTES
            ):
                raise _InvalidStoreError
            chunks: list[bytes] = []
            remaining = _MAX_JSON_BYTES + 1
            while remaining > 0:
                chunk = os.read(fd, min(remaining, 8192))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if len(raw) > _MAX_JSON_BYTES:
                raise _InvalidStoreError
            try:
                return json.loads(raw, object_pairs_hook=_object_without_duplicates)
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
                raise _InvalidStoreError from None
        except OSError:
            raise _UnavailableError from None
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def _read_state(self, directory_fd: int) -> _StoreState | None:
        value = self._read_private_json(directory_fd, self.paths.state.name)
        if value is _MISSING:
            return None
        return _StoreState.from_value(value, self.flavor)

    def _check_replaceable(self, directory_fd: int, name: str) -> bool:
        try:
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        except OSError:
            raise _UnavailableError from None
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise _InvalidStoreError
        return True

    def _atomic_write(self, directory_fd: int, name: str, value: Mapping[str, object]) -> None:
        self._check_replaceable(directory_fd, name)
        try:
            payload = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError):
            raise CredentialError(_UNAVAILABLE_ERROR) from None
        if len(payload) > _MAX_JSON_BYTES:
            raise CredentialError(_TOO_LARGE_ERROR)
        temporary_name = f".{self.paths.canonical.name}.{secrets.token_hex(12)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0)
        fd = -1
        try:
            fd = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
            os.fchmod(fd, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short credential write")
                view = view[written:]
            os.fsync(fd)
        except OSError:
            raise _UnavailableError from None
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        # Never remove a temp file on error: replace/fsync failures can be ambiguous.
        self._check_replaceable(directory_fd, name)
        try:
            os.replace(
                temporary_name,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.fsync(directory_fd)
        except OSError:
            raise _UnavailableError from None

    def _write_state(self, directory_fd: int, state: _StoreState) -> None:
        self._atomic_write(directory_fd, self.paths.state.name, state.to_dict(self.flavor))

    def _validate_state_for_canonical(
        self, state: _StoreState | None, canonical_exists: bool
    ) -> _StoreState | None:
        if state is None and canonical_exists:
            raise _InvalidStoreError
        return state

    def _check_expiry(self, token: OAuthToken) -> None:
        if token.expires_at_ms <= self._now_ms():
            raise CredentialNeedsRefreshError(_NEEDS_REFRESH_ERROR)

    def peek_status(self) -> CredentialStatus:
        try:
            with self._locked() as directory_fd:
                state = self._read_state(directory_fd)
                if state is not None and state.logged_out:
                    return CredentialStatus.LOGGED_OUT
                value = self._read_private_json(directory_fd, self.paths.canonical.name)
                if value is _MISSING:
                    return CredentialStatus.ABSENT
                self._validate_state_for_canonical(state, canonical_exists=True)
                try:
                    token = self._validate_token(OAuthToken.from_mapping(value))
                except CredentialError:
                    raise _InvalidStoreError from None
                if token.expires_at_ms <= self._now_ms():
                    return CredentialStatus.NEEDS_REFRESH
                return CredentialStatus.AVAILABLE
        except _InvalidStoreError:
            return CredentialStatus.INVALID
        except _UnavailableError:
            raise CredentialError(_UNAVAILABLE_ERROR) from None
        except CredentialNeedsRefreshError:
            return CredentialStatus.NEEDS_REFRESH
        except CredentialError as exc:
            if str(exc) == _CREDENTIAL_ERROR:
                return CredentialStatus.INVALID
            raise

    def get_cached_or_import(self) -> OAuthToken | None:
        try:
            with self._locked() as directory_fd:
                state = self._read_state(directory_fd)
                if state is not None and state.logged_out:
                    return None
                value = self._read_private_json(directory_fd, self.paths.canonical.name)
                if value is not _MISSING:
                    self._validate_state_for_canonical(state, canonical_exists=True)
                    try:
                        token = self._validate_token(OAuthToken.from_mapping(value))
                    except CredentialError:
                        raise _InvalidStoreError from None
                    self._check_expiry(token)
                    return token
                if self._importer is None:
                    return None
                token = self._imported_token()
                if token is None:
                    return None
                self._check_expiry(token)
                self._atomic_write(directory_fd, self.paths.canonical.name, token.to_dict())
                self._write_state(
                    directory_fd,
                    _StoreState(secrets.token_hex(16), logged_out=False),
                )
                return token
        except _InvalidStoreError:
            raise CredentialError(_INVALID_ERROR) from None
        except _UnavailableError:
            raise CredentialError(_UNAVAILABLE_ERROR) from None

    def begin_login(self) -> _LoginGeneration:
        try:
            with self._locked() as directory_fd:
                state = self._read_state(directory_fd)
                canonical_exists = self._check_replaceable(
                    directory_fd, self.paths.canonical.name
                )
                self._validate_state_for_canonical(state, canonical_exists)
                generation = secrets.token_hex(16)
                self._write_state(
                    directory_fd,
                    _StoreState(
                        generation,
                        logged_out=state.logged_out if state is not None else False,
                    ),
                )
                return _LoginGeneration(self._generation_owner, generation)
        except _InvalidStoreError:
            raise CredentialError(_INVALID_ERROR) from None
        except _UnavailableError:
            raise CredentialError(_UNAVAILABLE_ERROR) from None

    def _generation_value(
        self, generation: _LoginGeneration, state: _StoreState | None
    ) -> str:
        value = generation._value_for(self._generation_owner)
        if value is None or state is None or value != state.generation:
            raise CredentialError(_STALE_ERROR)
        return value

    def _imported_token(self) -> OAuthToken | None:
        if self._importer is None:
            return None
        try:
            imported = self._importer()
        except Exception:
            raise CredentialError(_IMPORT_ERROR) from None
        if imported is None:
            return None
        try:
            return self._coerce_input(imported)
        except CredentialError:
            raise CredentialError(_IMPORT_ERROR) from None

    def commit_login(self, token: CredentialInput, generation: _LoginGeneration) -> OAuthToken:
        try:
            with self._locked() as directory_fd:
                state = self._read_state(directory_fd)
                self._generation_value(generation, state)
                self._check_replaceable(directory_fd, self.paths.canonical.name)
                validated = self._coerce_input(token)
                self._check_expiry(validated)
                self._atomic_write(directory_fd, self.paths.canonical.name, validated.to_dict())
                self._write_state(
                    directory_fd,
                    _StoreState(secrets.token_hex(16), logged_out=False),
                )
                return validated
        except _InvalidStoreError:
            raise CredentialError(_INVALID_ERROR) from None
        except _UnavailableError:
            raise CredentialError(_UNAVAILABLE_ERROR) from None

    def explicit_reimport(self, generation: _LoginGeneration) -> OAuthToken | None:
        if self._importer is None:
            raise CredentialError(_IMPORTER_UNAVAILABLE_ERROR)
        try:
            with self._locked() as directory_fd:
                state = self._read_state(directory_fd)
                self._generation_value(generation, state)
                self._check_replaceable(directory_fd, self.paths.canonical.name)
                token = self._imported_token()
                if token is None:
                    return None
                self._check_expiry(token)
                self._atomic_write(directory_fd, self.paths.canonical.name, token.to_dict())
                self._write_state(
                    directory_fd,
                    _StoreState(secrets.token_hex(16), logged_out=False),
                )
                return token
        except _InvalidStoreError:
            raise CredentialError(_INVALID_ERROR) from None
        except _UnavailableError:
            raise CredentialError(_UNAVAILABLE_ERROR) from None

    def logout(self) -> None:
        try:
            with self._locked() as directory_fd:
                self._read_state(directory_fd)
                self._write_state(
                    directory_fd,
                    _StoreState(secrets.token_hex(16), logged_out=True),
                )
                canonical_exists = self._check_replaceable(
                    directory_fd, self.paths.canonical.name
                )
                if canonical_exists:
                    try:
                        os.unlink(self.paths.canonical.name, dir_fd=directory_fd)
                        os.fsync(directory_fd)
                    except OSError:
                        raise _UnavailableError from None
        except _InvalidStoreError:
            raise CredentialError(_INVALID_ERROR) from None
        except _UnavailableError:
            raise CredentialError(_UNAVAILABLE_ERROR) from None
