"""Dormant, provider-specific persistence for OAuth credentials.

This module deliberately has no default path and performs no network work. The
injected importer must return Codex refresh credentials or a Copilot GitHub
token; cached Copilot bearer tokens and endpoint data are not accepted.
"""

from __future__ import annotations

import errno
import hashlib
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
from typing import TYPE_CHECKING, Generator, NoReturn, SupportsIndex, TypeAlias, cast

if TYPE_CHECKING:
    from ctypes import c_void_p

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
_ENVELOPE_KEYS = frozenset({"version", "flavor", "generation", "token"})


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
            not isinstance(cast(object, self.access), str)
            or not self.access
            or not isinstance(cast(object, self.refresh), str)
            or not isinstance(cast(object, self.expires_at_ms), int)
            or isinstance(self.expires_at_ms, bool)
            or self.expires_at_ms <= 0
            or (
                self.account_id is not None
                and (not isinstance(cast(object, self.account_id), str) or not self.account_id)
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
        value = cast(Mapping[object, object], value)
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


def _windows_current_user_sid() -> str:
    """Return the current process user's SID without pywin32."""
    import ctypes  # pragma: no cover - native Windows only
    from ctypes import wintypes

    class _SidAndAttributes(ctypes.Structure):
        _fields_ = [("sid", wintypes.LPVOID), ("attributes", wintypes.DWORD)]

    class _TokenUser(ctypes.Structure):
        _fields_ = [("user", _SidAndAttributes)]

    token_handle = wintypes.HANDLE()
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    open_process_token = ctypes.windll.advapi32.OpenProcessToken
    open_process_token.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    if not open_process_token(get_current_process(), 0x8, ctypes.byref(token_handle)):
        raise OSError("credential owner lookup failed")
    try:
        required = wintypes.DWORD()
        get_token_information = ctypes.windll.advapi32.GetTokenInformation
        get_token_information.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        get_token_information(token_handle, 1, None, 0, ctypes.byref(required))
        token_buffer = ctypes.create_string_buffer(required.value)
        if not get_token_information(
            token_handle,
            1,
            token_buffer,
            required.value,
            ctypes.byref(required),
        ):
            raise OSError("credential owner lookup failed")
        token_user = ctypes.cast(token_buffer, ctypes.POINTER(_TokenUser)).contents
        sid_text = wintypes.LPWSTR()
        convert_sid = ctypes.windll.advapi32.ConvertSidToStringSidW
        convert_sid.argtypes = [wintypes.LPVOID, ctypes.POINTER(wintypes.LPWSTR)]
        if not convert_sid(
            token_user.user.sid, ctypes.byref(sid_text)
        ):
            raise OSError("credential owner lookup failed")
        try:
            current_user_sid = sid_text.value
        finally:
            local_free = ctypes.windll.kernel32.LocalFree
            local_free.argtypes = [wintypes.LPVOID]
            local_free(ctypes.cast(sid_text, wintypes.LPVOID))
    finally:
        close_handle = ctypes.windll.kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle(token_handle)
    if not current_user_sid:
        raise OSError("credential owner lookup failed")
    return current_user_sid


@contextmanager
def _windows_security_descriptor() -> Generator[c_void_p, None, None]:
    """Yield the protected credential DACL's security descriptor."""
    import ctypes  # pragma: no cover - native Windows only
    from ctypes import wintypes

    descriptor = wintypes.LPVOID()
    convert = ctypes.windll.advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.ULONG),
    ]
    if not convert(
        f"D:P(A;;FA;;;{_windows_current_user_sid()})(A;;FA;;;SY)(A;;FA;;;BA)",
        1,
        ctypes.byref(descriptor),
        None,
    ):
        raise OSError("credential ACL creation failed")
    try:
        yield descriptor
    finally:
        local_free = ctypes.windll.kernel32.LocalFree
        local_free.argtypes = [wintypes.LPVOID]
        local_free(descriptor)


def _protect_windows_path(path: Path) -> None:
    """Apply a protected owner/SYSTEM/Administrators DACL without pywin32."""
    if os.name != "nt":
        return
    import ctypes  # pragma: no cover - native Windows only
    from ctypes import wintypes

    with _windows_security_descriptor() as descriptor:
        set_security = ctypes.windll.advapi32.SetFileSecurityW
        set_security.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.LPVOID]
        if not set_security(str(path), 0x4 | 0x80000000, descriptor):
            raise OSError("credential ACL publication failed")


def _protect_windows_handle(fd: int) -> None:
    """Apply and verify the credential DACL on one stable file handle."""
    import ctypes  # pragma: no cover - native Windows only
    import msvcrt
    from ctypes import wintypes

    handle = wintypes.HANDLE(msvcrt.get_osfhandle(fd))
    security_information = 0x4 | 0x80000000  # DACL | PROTECTED_DACL
    with _windows_security_descriptor() as descriptor:
        dacl_present = wintypes.BOOL()
        dacl_defaulted = wintypes.BOOL()
        dacl = wintypes.LPVOID()
        get_dacl = ctypes.windll.advapi32.GetSecurityDescriptorDacl
        get_dacl.argtypes = [
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.BOOL),
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.BOOL),
        ]
        if not get_dacl(
            descriptor,
            ctypes.byref(dacl_present),
            ctypes.byref(dacl),
            ctypes.byref(dacl_defaulted),
        ) or not dacl_present:
            raise OSError("credential ACL creation failed")

        set_security = ctypes.windll.advapi32.SetSecurityInfo
        set_security.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.LPVOID,
        ]
        if set_security(handle, 1, security_information, None, None, dacl, None) != 0:
            raise OSError("credential ACL publication failed")

        needed = wintypes.DWORD()
        get_security = ctypes.windll.advapi32.GetKernelObjectSecurity
        get_security.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        get_security(handle, 0x4, None, 0, ctypes.byref(needed))
        if not needed.value:
            raise OSError("credential ACL verification failed")
        actual = ctypes.create_string_buffer(needed.value)
        if not get_security(handle, 0x4, actual, needed.value, ctypes.byref(needed)):
            raise OSError("credential ACL verification failed")

        control = wintypes.WORD()
        revision = wintypes.DWORD()
        get_control = ctypes.windll.advapi32.GetSecurityDescriptorControl
        get_control.argtypes = [
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.WORD),
            ctypes.POINTER(wintypes.DWORD),
        ]
        if not get_control(actual, ctypes.byref(control), ctypes.byref(revision)):
            raise OSError("credential ACL verification failed")
        if not control.value & 0x1000:  # SE_DACL_PROTECTED
            raise OSError("credential ACL verification failed")

        get_length = ctypes.windll.advapi32.GetSecurityDescriptorLength
        get_length.argtypes = [wintypes.LPVOID]
        get_length.restype = wintypes.DWORD
        expected_length = get_length(descriptor)
        actual_length = get_length(actual)
        if not expected_length or expected_length != actual_length:
            raise OSError("credential ACL verification failed")
        if ctypes.string_at(descriptor, expected_length) != ctypes.string_at(actual, actual_length):
            raise OSError("credential ACL verification failed")


def _rename_windows_verified(fd: int, target: Path) -> None:
    """Atomically publish an already verified file by its still-open handle."""
    import ctypes  # pragma: no cover - native Windows only
    import msvcrt
    from ctypes import wintypes

    class _FileRenameInfoHeader(ctypes.Structure):
        _fields_ = [
            ("flags", wintypes.DWORD),
            ("root_directory", wintypes.HANDLE),
            ("file_name_length", wintypes.DWORD),
        ]

    encoded_target = str(target).encode("utf-16-le")
    filename_offset = _FileRenameInfoHeader.file_name_length.offset + ctypes.sizeof(
        wintypes.DWORD
    )
    buffer = ctypes.create_string_buffer(filename_offset + len(encoded_target))
    header = _FileRenameInfoHeader.from_buffer(buffer)
    header.flags = 0x1 | 0x2  # REPLACE_IF_EXISTS | POSIX_SEMANTICS
    header.root_directory = None
    header.file_name_length = len(encoded_target)
    ctypes.memmove(ctypes.addressof(buffer) + filename_offset, encoded_target, len(encoded_target))

    set_information = ctypes.windll.kernel32.SetFileInformationByHandle
    set_information.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    if not set_information(
        msvcrt.get_osfhandle(fd),
        22,  # FileRenameInfoEx
        buffer,
        len(buffer),
    ):
        raise OSError("durable credential replace failed")


def _write_windows_verified(target: Path, temporary: Path, payload: bytes) -> None:
    """Write and publish bytes while retaining sole authority over one handle."""
    fd = -1
    try:
        fd = _open_windows_verified(
            temporary,
            access=0x80000000 | 0x40000000 | 0x00010000 | 0x00040000,
            creation=1,  # CREATE_NEW
            share=0,
        )
        _protect_windows_handle(fd)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short credential write")
            view = view[written:]
        os.fsync(fd)
        _rename_windows_verified(fd, target)
        _protect_windows_handle(fd)
        os.fsync(fd)
    finally:
        if fd >= 0:
            os.close(fd)


def _open_windows_verified(
    path: Path,
    *,
    access: int,
    creation: int,
    share: int,
) -> int:
    """Open a stable non-reparse regular-file handle and return a CRT fd."""
    import ctypes  # pragma: no cover - native Windows only
    import msvcrt
    from ctypes import wintypes

    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("creation_time", wintypes.FILETIME),
            ("last_access_time", wintypes.FILETIME),
            ("last_write_time", wintypes.FILETIME),
            ("volume_serial_number", wintypes.DWORD),
            ("file_size_high", wintypes.DWORD),
            ("file_size_low", wintypes.DWORD),
            ("number_of_links", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        ]

    create_file = ctypes.windll.kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        access,
        share,
        None,
        creation,
        0x00200000 | 0x08000000,  # OPEN_REPARSE_POINT | SEQUENTIAL_SCAN
        None,
    )
    invalid_handle = wintypes.HANDLE(-1).value
    if handle == invalid_handle:
        error = ctypes.windll.kernel32.GetLastError()
        if error in {2, 3}:
            raise FileNotFoundError(path)
        raise OSError(error, "secure credential open failed")
    try:
        set_handle_information = ctypes.windll.kernel32.SetHandleInformation
        set_handle_information.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD]
        if not set_handle_information(handle, 0x1, 0):
            raise OSError("credential handle protection failed")
        get_file_type = ctypes.windll.kernel32.GetFileType
        get_file_type.argtypes = [wintypes.HANDLE]
        if get_file_type(handle) != 0x1:
            raise _InvalidStoreError
        info = _ByHandleFileInformation()
        get_information = ctypes.windll.kernel32.GetFileInformationByHandle
        get_information.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ByHandleFileInformation)]
        if not get_information(handle, ctypes.byref(info)):
            raise OSError("credential metadata unavailable")
        if (
            info.file_attributes & 0x400  # FILE_ATTRIBUTE_REPARSE_POINT
            or info.file_attributes & 0x10  # FILE_ATTRIBUTE_DIRECTORY
            or info.number_of_links != 1
        ):
            raise _InvalidStoreError
        flags = getattr(os, "O_BINARY", 0)
        if access & 0x40000000:  # GENERIC_WRITE
            flags |= os.O_RDWR
        else:
            flags |= os.O_RDONLY
        fd = msvcrt.open_osfhandle(handle, flags)
        handle = invalid_handle
        os.set_inheritable(fd, False)
        return fd
    finally:
        if handle != invalid_handle:
            close_handle = ctypes.windll.kernel32.CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle(handle)


def _unlink_windows_verified(path: Path) -> None:
    """Delete the exact verified file handle, not a subsequently swapped path."""
    import ctypes  # pragma: no cover - native Windows only
    import msvcrt
    from ctypes import wintypes

    fd = _open_windows_verified(
        path,
        access=0x80000000 | 0x00010000,  # GENERIC_READ | DELETE
        creation=3,
        share=0x1 | 0x2,
    )
    try:
        disposition = wintypes.BOOL(True)
        set_information = ctypes.windll.kernel32.SetFileInformationByHandle
        set_information.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        if not set_information(
            msvcrt.get_osfhandle(fd),
            13,  # FileDispositionInfo
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            raise OSError("credential deletion failed")
    finally:
        os.close(fd)


@dataclass(frozen=True, slots=True)
class CredentialStorePaths:
    canonical: Path
    state: Path
    lock: Path
    derived: Path


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
        derived=parent / f"{raw_path.name}.derived.json",
    )


class _LoginGeneration:
    __slots__ = ("__owner", "__value")

    def __init__(self, owner: object, value: str) -> None:
        self.__owner = owner
        self.__value = value

    def _value_for(self, owner: object) -> str | None:
        return self.__value if self.__owner is owner else None

    def require_current_intent(
        self, observe_owner: Callable[[], object], state: _StoreState | None,
    ) -> str:
        """Validate this generation's owner and intent without publishing credentials."""
        value = self._value_for(observe_owner())
        if value is None or state is None or value != state.intent_generation:
            raise CredentialError(_STALE_ERROR)
        return value

    def __repr__(self) -> str:
        return "<LoginGeneration>"

    def __copy__(self) -> NoReturn:
        raise TypeError("login generation cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("login generation cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("login generation cannot be pickled")


@dataclass(frozen=True, slots=True)
class _StoreState:
    active_generation: str | None
    logged_out: bool
    intent_generation: str | None = None
    publication: _Publication | None = None

    def to_dict(self, flavor: CredentialFlavor) -> dict[str, object]:
        result: dict[str, object] = {
            "version": 2,
            "active_generation": self.active_generation,
            "logged_out": self.logged_out,
            "flavor": flavor.value,
        }
        if self.intent_generation is not None:
            result["intent_generation"] = self.intent_generation
        if self.publication is not None:
            result["publication"] = self.publication.to_dict()
        return result

    @classmethod
    def from_value(cls, value: object, flavor: CredentialFlavor) -> _StoreState:
        if not isinstance(value, Mapping):
            raise _InvalidStoreError
        value = cast(Mapping[object, object], value)
        try:
            version = value["version"]
        except Exception:
            raise _InvalidStoreError from None
        if version == 1:
            expected = {"version", "generation", "logged_out", "flavor"}
            try:
                if set(value.keys()) != expected:
                    raise _InvalidStoreError
                generation = value["generation"]
                logged_out = value["logged_out"]
                stored_flavor = value["flavor"]
            except _InvalidStoreError:
                raise
            except Exception:
                raise _InvalidStoreError from None
            if (
                not isinstance(generation, str)
                or _GENERATION_PATTERN.fullmatch(generation) is None
                or not isinstance(logged_out, bool)
                or stored_flavor != flavor.value
            ):
                raise _InvalidStoreError
            return cls(None if logged_out else generation, logged_out)

        allowed = {
            "version",
            "active_generation",
            "logged_out",
            "flavor",
            "intent_generation",
            "publication",
        }
        try:
            keys = set(value.keys())
            if not {"version", "active_generation", "logged_out", "flavor"}.issubset(keys):
                raise _InvalidStoreError
            if keys - allowed:
                raise _InvalidStoreError
            active_generation = value["active_generation"]
            logged_out = value["logged_out"]
            stored_flavor = value["flavor"]
            intent_generation = value.get("intent_generation")
            publication_value = value.get("publication")
        except _InvalidStoreError:
            raise
        except Exception:
            raise _InvalidStoreError from None
        if (
            version != 2
            or not _valid_optional_generation(active_generation)
            or not isinstance(logged_out, bool)
            or stored_flavor != flavor.value
            or not _valid_optional_generation(intent_generation)
            or (logged_out and active_generation is not None)
        ):
            raise _InvalidStoreError
        publication = (
            None
            if publication_value is None
            else _Publication.from_value(publication_value)
        )
        # Both optional generations passed _valid_optional_generation above.
        return cls(cast(str | None, active_generation), logged_out,
                   cast(str | None, intent_generation), publication)


def _valid_optional_generation(value: object) -> bool:
    return value is None or (
        isinstance(value, str) and _GENERATION_PATTERN.fullmatch(value) is not None
    )


@dataclass(frozen=True, slots=True)
class _Publication:
    previous_generation: str | None
    target_generation: str
    digest: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "previous_generation": self.previous_generation,
            "target_generation": self.target_generation,
            "digest": self.digest,
        }

    @classmethod
    def from_value(cls, value: object) -> _Publication:
        if not isinstance(value, Mapping) or set(cast(Mapping[object, object], value).keys()) != {
            "previous_generation",
            "target_generation",
            "digest",
        }:
            raise _InvalidStoreError
        value = cast(Mapping[object, object], value)
        previous = value["previous_generation"]
        target = value["target_generation"]
        digest = value["digest"]
        if (
            not _valid_optional_generation(previous)
            or not isinstance(target, str)
            or _GENERATION_PATTERN.fullmatch(target) is None
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise _InvalidStoreError
        # The optional previous generation passed _valid_optional_generation above.
        return cls(cast(str | None, previous), target, digest)


@dataclass(frozen=True, slots=True)
class CredentialView:
    """An active credential plus its non-secret cache generation."""

    token: OAuthToken = field(repr=False)
    generation: str


class _InvalidStoreError(Exception):
    pass


class _UnavailableError(Exception):
    pass


CredentialInput: TypeAlias = OAuthToken | Mapping[str, object] | str
Importer: TypeAlias = Callable[[], CredentialInput | None]
RefreshCallback: TypeAlias = Callable[[OAuthToken], CredentialInput]


@dataclass(frozen=True, slots=True)
class DerivedBearer:
    """A generation-bound Copilot bearer; the secret is hidden from repr."""

    bearer: str = field(repr=False)
    expires_at_ms: int
    generation: str
    endpoint: str | None = None


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
        if os.name == "nt":  # pragma: no cover - native Windows only
            _protect_windows_path(self.paths.canonical.parent)
        self.flavor = flavor
        self._importer = importer
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._generation_owner = object()

    def _now_ms(self) -> int:
        value = cast(object, self._clock_ms())
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
            if isinstance(value, str) and value:
                token = OAuthToken(
                    access=value,
                    refresh="",
                    expires_at_ms=self._now_ms() + _LONG_LIVED_TOKEN_SECONDS * 1000,
                )
            elif isinstance(value, OAuthToken) and value.refresh == "":
                token = value
            else:
                raise CredentialError(_CREDENTIAL_ERROR)
        elif isinstance(value, OAuthToken):
            token = value
        elif isinstance(value, Mapping):
            token = OAuthToken.from_mapping(cast(object, value))
        else:
            raise CredentialError(_CREDENTIAL_ERROR)
        return self._validate_token(token)

    @contextmanager
    def _locked(self) -> Generator[int, None, None]:
        if os.name == "nt":  # pragma: no cover - exercised by native Windows CI
            import msvcrt

            lock_fd = -1
            try:
                lock_fd = _open_windows_verified(
                    self.paths.lock,
                    access=0x80000000 | 0x40000000,
                    creation=4,  # OPEN_ALWAYS
                    share=0x1 | 0x2,  # deny delete/rename while locked
                )
                _protect_windows_path(self.paths.lock)
                if os.fstat(lock_fd).st_size == 0:
                    os.write(lock_fd, b"\0")
                    os.fsync(lock_fd)
                os.lseek(lock_fd, 0, os.SEEK_SET)
                msvcrt.locking(lock_fd, msvcrt.LK_LOCK, 1)
                yield -1
            except (_InvalidStoreError, _UnavailableError):
                raise
            except OSError:
                raise _UnavailableError from None
            finally:
                if lock_fd >= 0:
                    try:
                        os.lseek(lock_fd, 0, os.SEEK_SET)
                        msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                    os.close(lock_fd)
            return
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
        if os.name == "nt":  # pragma: no cover - exercised by native Windows CI
            path = self.paths.canonical.parent / name
            fd = -1
            try:
                fd = _open_windows_verified(
                    path,
                    access=0x80000000,
                    creation=3,  # OPEN_EXISTING
                    share=0x1 | 0x2,  # deny delete/rename during validation/read
                )
                if os.fstat(fd).st_size > _MAX_JSON_BYTES:
                    raise _InvalidStoreError
                raw = os.read(fd, _MAX_JSON_BYTES + 1)
                if len(raw) > _MAX_JSON_BYTES:
                    raise _InvalidStoreError
                return json.loads(raw, object_pairs_hook=_object_without_duplicates)
            except FileNotFoundError:
                return _MISSING
            except _InvalidStoreError:
                raise
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
                raise _InvalidStoreError from None
            except OSError:
                raise _UnavailableError from None
            finally:
                if fd >= 0:
                    os.close(fd)
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
        if os.name == "nt":  # pragma: no cover - exercised by native Windows CI
            path = self.paths.canonical.parent / name
            fd = -1
            try:
                fd = _open_windows_verified(
                    path,
                    access=0x80000000,
                    creation=3,
                    share=0x1 | 0x2,
                )
            except FileNotFoundError:
                return False
            except _InvalidStoreError:
                raise
            except OSError:
                raise _UnavailableError from None
            finally:
                if fd >= 0:
                    os.close(fd)
            return True
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
        if os.name == "nt":  # pragma: no cover - exercised by native Windows CI
            target = self.paths.canonical.parent / name
            temporary = target.with_name(f".{self.paths.canonical.name}.{secrets.token_hex(12)}.tmp")
            try:
                self._check_replaceable(directory_fd, name)
                _write_windows_verified(target, temporary, payload)
            except OSError:
                raise _UnavailableError from None
            return
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

    def _envelope(
        self, token: OAuthToken, generation: str
    ) -> dict[str, object]:
        return {
            "version": 2,
            "flavor": self.flavor.value,
            "generation": generation,
            "token": token.to_dict(),
        }

    def _envelope_digest(self, envelope: Mapping[str, object]) -> str:
        try:
            payload = json.dumps(
                envelope,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError):
            raise _InvalidStoreError from None
        return hashlib.sha256(payload).hexdigest()

    def _parse_envelope(self, value: object) -> tuple[OAuthToken, str]:
        if not isinstance(value, Mapping) or cast(object, set(cast(Mapping[object, object], value).keys())) != _ENVELOPE_KEYS:
            raise _InvalidStoreError
        value = cast(Mapping[object, object], value)
        try:
            version = value["version"]
            flavor = value["flavor"]
            generation = value["generation"]
            token_value = value["token"]
        except Exception:
            raise _InvalidStoreError from None
        if (
            version != 2
            or flavor != self.flavor.value
            or not isinstance(generation, str)
            or _GENERATION_PATTERN.fullmatch(generation) is None
        ):
            raise _InvalidStoreError
        try:
            token = self._validate_token(OAuthToken.from_mapping(token_value))
        except CredentialError:
            raise _InvalidStoreError from None
        return token, generation

    def _read_consistent(
        self, directory_fd: int
    ) -> tuple[_StoreState | None, CredentialView | None]:
        state_value = self._read_private_json(directory_fd, self.paths.state.name)
        state = None if state_value is _MISSING else _StoreState.from_value(state_value, self.flavor)
        canonical_value = self._read_private_json(directory_fd, self.paths.canonical.name)
        if state is None:
            if canonical_value is not _MISSING:
                legacy = self._legacy_canonical(canonical_value)
                view = self._publish(directory_fd, None, legacy)
                return _StoreState(view.generation, False), view
            return None, None

        # Bootstrap the strict v1 Nanobot pair in place. A lone old canonical is invalid.
        if canonical_value is not _MISSING and isinstance(canonical_value, Mapping):
            if "version" not in canonical_value:
                try:
                    old_token = self._validate_token(OAuthToken.from_mapping(cast(object, canonical_value)))
                except CredentialError:
                    raise _InvalidStoreError from None
                if state.logged_out or state.active_generation is None:
                    raise _InvalidStoreError
                envelope = self._envelope(old_token, state.active_generation)
                self._atomic_write(directory_fd, self.paths.canonical.name, envelope)
                state = _StoreState(state.active_generation, False, state.intent_generation)
                self._write_state(directory_fd, state)
                canonical_value = envelope

        view: CredentialView | None = None
        if canonical_value is not _MISSING:
            token, generation = self._parse_envelope(cast(object, canonical_value))
            view = CredentialView(token, generation)

        if state.publication is not None:
            publication = state.publication
            if view is not None and view.generation == publication.target_generation:
                envelope = self._envelope(view.token, view.generation)
                if self._envelope_digest(envelope) != publication.digest:
                    raise _InvalidStoreError
                state = _StoreState(view.generation, False, None)
                self._write_state(directory_fd, state)
            elif (
                publication.previous_generation is None and view is None
            ) or (
                view is not None and view.generation == publication.previous_generation
            ):
                state = _StoreState(
                    publication.previous_generation,
                    state.logged_out,
                    state.intent_generation,
                )
                self._write_state(directory_fd, state)
            else:
                raise _InvalidStoreError

        if state.logged_out:
            return state, None
        if state.active_generation is None:
            if view is not None:
                raise _InvalidStoreError
            return state, None
        if view is None or view.generation != state.active_generation:
            raise _InvalidStoreError
        return state, view

    def _legacy_canonical(self, value: object) -> OAuthToken:
        """Parse only the two historical Nanobot canonical record shapes."""
        if not isinstance(value, Mapping):
            raise _InvalidStoreError
        value = cast(Mapping[object, object], value)
        try:
            if self.flavor is CredentialFlavor.COPILOT:
                token = value.get("github_token")
                if not isinstance(token, str) or not token:
                    raise _InvalidStoreError
                return self._coerce_input(token)
            if set(value.keys()) - {"access", "refresh", "expires", "account_id"}:
                raise _InvalidStoreError
            # This fixed constructor immediately validates every field in __post_init__;
            # no invalid token can return. Keep the original mapping access order.
            return self._validate_token(
                OAuthToken(
                    cast(str, value["access"]),
                    cast(str, value["refresh"]),
                    cast(int, value["expires"]),
                    cast(str | None, value.get("account_id")),
                )
            )
        except (_InvalidStoreError, CredentialError):
            raise _InvalidStoreError from None
        except Exception:
            raise _InvalidStoreError from None

    def _publish(
        self,
        directory_fd: int,
        state: _StoreState | None,
        token: OAuthToken,
        *,
        target_generation: str | None = None,
    ) -> CredentialView:
        generation = target_generation or secrets.token_hex(16)
        envelope = self._envelope(token, generation)
        publication = _Publication(
            state.active_generation if state is not None and not state.logged_out else None,
            generation,
            self._envelope_digest(envelope),
        )
        prepared = _StoreState(
            state.active_generation if state is not None and not state.logged_out else None,
            state.logged_out if state is not None else False,
            state.intent_generation if state is not None else None,
            publication,
        )
        self._write_state(directory_fd, prepared)
        self._atomic_write(directory_fd, self.paths.canonical.name, envelope)
        self._write_state(directory_fd, _StoreState(generation, False))
        return CredentialView(token, generation)

    def _check_expiry(self, token: OAuthToken) -> None:
        if token.expires_at_ms <= self._now_ms():
            raise CredentialNeedsRefreshError(_NEEDS_REFRESH_ERROR)

    def peek_status(self) -> CredentialStatus:
        try:
            with self._locked() as directory_fd:
                state, view = self._read_consistent(directory_fd)
                if state is not None and state.logged_out:
                    return CredentialStatus.LOGGED_OUT
                if view is None:
                    return CredentialStatus.ABSENT
                if view.token.expires_at_ms <= self._now_ms():
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
        view = self.get_view_or_import()
        return None if view is None else view.token

    def get_view_or_import(self) -> CredentialView | None:
        try:
            with self._locked() as directory_fd:
                state, view = self._read_consistent(directory_fd)
                if state is not None and state.logged_out:
                    return None
                if view is not None:
                    self._check_expiry(view.token)
                    return view
                if self._importer is None:
                    return None
                token = self._imported_token()
                if token is None:
                    return None
                self._check_expiry(token)
                return self._publish(directory_fd, state, token)
        except _InvalidStoreError:
            raise CredentialError(_INVALID_ERROR) from None
        except _UnavailableError:
            raise CredentialError(_UNAVAILABLE_ERROR) from None

    def refresh_codex(
        self,
        min_ttl_seconds: int,
        refresh: RefreshCallback,
    ) -> CredentialView:
        """Return a min-TTL Codex view, serializing refresh across processes."""
        if self.flavor is not CredentialFlavor.CODEX or min_ttl_seconds < 0:
            raise ValueError("Codex refresh parameters required")
        try:
            with self._locked() as directory_fd:
                state, view = self._read_consistent(directory_fd)
                if state is not None and state.logged_out:
                    raise CredentialError(_NEEDS_REFRESH_ERROR)
                if view is None:
                    token = self._imported_token()
                    if token is None:
                        raise CredentialError(_NEEDS_REFRESH_ERROR)
                    view = self._publish(directory_fd, state, token)
                    state = _StoreState(view.generation, False)
                if view.token.expires_at_ms - self._now_ms() > min_ttl_seconds * 1000:
                    return view
                try:
                    refreshed = self._coerce_input(refresh(view.token))
                except CredentialError:
                    raise
                except Exception:
                    raise CredentialError("credential refresh failed") from None
                if refreshed.account_id != view.token.account_id:
                    raise CredentialError("credential account mismatch")
                self._check_expiry(refreshed)
                return self._publish(directory_fd, state, refreshed)
        except _InvalidStoreError:
            raise CredentialError(_INVALID_ERROR) from None
        except _UnavailableError:
            raise CredentialError(_UNAVAILABLE_ERROR) from None

    def get_copilot_bearer(
        self,
        min_ttl_seconds: int,
        exchange: Callable[[OAuthToken], Mapping[str, object]],
        *,
        expected_generation: str | None = None,
    ) -> DerivedBearer:
        """Return a generation-bound derived bearer under the provider lock."""
        if self.flavor is not CredentialFlavor.COPILOT or min_ttl_seconds < 0:
            raise ValueError("Copilot exchange parameters required")
        try:
            with self._locked() as directory_fd:
                state, view = self._read_consistent(directory_fd)
                if state is not None and state.logged_out:
                    raise CredentialError(_NEEDS_REFRESH_ERROR)
                if view is None:
                    token = self._imported_token()
                    if token is None:
                        raise CredentialError(_NEEDS_REFRESH_ERROR)
                    view = self._publish(directory_fd, state, token)
                if (
                    expected_generation is not None
                    and view.generation != expected_generation
                ):
                    raise CredentialError(_STALE_ERROR)
                cached = self._read_private_json(directory_fd, self.paths.derived.name)
                derived = self._parse_derived(cached, view.generation)
                if (
                    derived is not None
                    and derived.expires_at_ms - self._now_ms() > min_ttl_seconds * 1000
                ):
                    return derived
                try:
                    payload = exchange(view.token)
                    bearer = payload.get("bearer")
                    expires_at_ms = payload.get("expires_at_ms")
                    endpoint = payload.get("endpoint")
                except Exception:
                    raise CredentialError("credential exchange failed") from None
                if (
                    not isinstance(bearer, str)
                    or not bearer
                    or not isinstance(expires_at_ms, int)
                    or isinstance(expires_at_ms, bool)
                    or expires_at_ms <= self._now_ms()
                    or (endpoint is not None and not isinstance(endpoint, str))
                ):
                    raise CredentialError("credential exchange failed")
                value: dict[str, object] = {
                    "version": 1,
                    "flavor": self.flavor.value,
                    "generation": view.generation,
                    "bearer": bearer,
                    "expires_at_ms": expires_at_ms,
                }
                if endpoint:
                    value["endpoint"] = endpoint
                self._atomic_write(directory_fd, self.paths.derived.name, value)
                return DerivedBearer(bearer, expires_at_ms, view.generation, endpoint or None)
        except _InvalidStoreError:
            raise CredentialError(_INVALID_ERROR) from None
        except _UnavailableError:
            raise CredentialError(_UNAVAILABLE_ERROR) from None

    def _parse_derived(self, value: object, generation: str) -> DerivedBearer | None:
        if value is _MISSING:
            return None
        if not isinstance(value, Mapping):
            raise _InvalidStoreError
        value = cast(Mapping[object, object], value)
        allowed = {"version", "flavor", "generation", "bearer", "expires_at_ms", "endpoint"}
        if set(value.keys()) - allowed or not {
            "version", "flavor", "generation", "bearer", "expires_at_ms"
        }.issubset(value.keys()):
            raise _InvalidStoreError
        bearer = value["bearer"]
        expires = value["expires_at_ms"]
        endpoint = value.get("endpoint")
        if (
            value["version"] != 1
            or value["flavor"] != self.flavor.value
            or value["generation"] != generation
        ):
            return None
        if (
            not isinstance(bearer, str)
            or not bearer
            or not isinstance(expires, int)
            or isinstance(expires, bool)
            or (endpoint is not None and not isinstance(endpoint, str))
        ):
            raise _InvalidStoreError
        return DerivedBearer(bearer, expires, generation, endpoint)

    def begin_login(self) -> _LoginGeneration:
        try:
            with self._locked() as directory_fd:
                state, _view = self._read_consistent(directory_fd)
                generation = secrets.token_hex(16)
                self._write_state(
                    directory_fd,
                    _StoreState(
                        state.active_generation if state is not None else None,
                        logged_out=state.logged_out if state is not None else False,
                        intent_generation=generation,
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
        return _LoginGeneration.require_current_intent(
            generation, lambda: self._generation_owner, state,
        )

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
                state, _view = self._read_consistent(directory_fd)
                self._generation_value(generation, state)
                validated = self._coerce_input(token)
                self._check_expiry(validated)
                return self._publish(directory_fd, state, validated).token
        except _InvalidStoreError:
            raise CredentialError(_INVALID_ERROR) from None
        except _UnavailableError:
            raise CredentialError(_UNAVAILABLE_ERROR) from None

    def explicit_reimport(self, generation: _LoginGeneration) -> OAuthToken | None:
        if self._importer is None:
            raise CredentialError(_IMPORTER_UNAVAILABLE_ERROR)
        try:
            with self._locked() as directory_fd:
                state, _view = self._read_consistent(directory_fd)
                self._generation_value(generation, state)
                token = self._imported_token()
                if token is None:
                    return None
                self._check_expiry(token)
                return self._publish(directory_fd, state, token).token
        except _InvalidStoreError:
            raise CredentialError(_INVALID_ERROR) from None
        except _UnavailableError:
            raise CredentialError(_UNAVAILABLE_ERROR) from None

    def logout(self) -> None:
        try:
            with self._locked() as directory_fd:
                _state, _view = self._read_consistent(directory_fd)
                self._write_state(
                    directory_fd,
                    _StoreState(None, logged_out=True),
                )
                canonical_exists = self._check_replaceable(
                    directory_fd, self.paths.canonical.name
                )
                if canonical_exists:
                    try:
                        if os.name == "nt":
                            _unlink_windows_verified(self.paths.canonical)
                        else:
                            os.unlink(self.paths.canonical.name, dir_fd=directory_fd)
                            os.fsync(directory_fd)
                    except OSError:
                        raise _UnavailableError from None
                derived_exists = self._check_replaceable(directory_fd, self.paths.derived.name)
                if derived_exists:
                    try:
                        if os.name == "nt":
                            _unlink_windows_verified(self.paths.derived)
                        else:
                            os.unlink(self.paths.derived.name, dir_fd=directory_fd)
                            os.fsync(directory_fd)
                    except OSError:
                        raise _UnavailableError from None
        except _InvalidStoreError:
            raise CredentialError(_INVALID_ERROR) from None
        except _UnavailableError:
            raise CredentialError(_UNAVAILABLE_ERROR) from None
