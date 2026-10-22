"""Dormant I3b runtime binding and lifecycle authority.

This module is deliberately a runtime-only substrate.  It binds already
admitted runtimes to an exact source-store object and a sealed snapshot loaded
by session key from that store, and exposes no scanner, summarizer, provider
factory, network call, or storage write.  Later integrations may use the
binding as admission evidence after I1 durable foreground activity and I2
checkpoint validation have completed.

The public values in this module are intentionally content-free in their
representations.  Session keys, paths, provider handles, credentials, and
provider signatures stay in private in-memory authority state; fingerprints
are keyed digests that cannot be serialized or used as persisted identities.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import math
import os
import secrets
import struct
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence, Set
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NoReturn, Protocol, TypeVar, cast

from nanobot.agent.idle_compaction import IdleCompactionPolicy
from nanobot.config.schema import IdleCompactionConfig
from nanobot.providers.factory import ProviderSnapshot
from nanobot.session.idle_checkpoint import IdlePolicyBinding
from nanobot.session.model_selection import model_preset_from_metadata
from nanobot.session.source_root import SourceRootAuthority
from nanobot.session.source_store import SourceSessionStore, SourceSnapshot, SourceState
from nanobot.utils.llm_runtime import LLMRuntime, runtime_from_provider_snapshot

_ResultT = TypeVar("_ResultT")


class _IdleCoordinatorLifecycle(Protocol):
    """Only the lifecycle operations consumed by this registry."""

    def foreground_admitted(self, session_key: str) -> Awaitable[None] | None: ...

    def aclose(self) -> Awaitable[None] | None: ...


_PROCESS_KEY = secrets.token_bytes(32)


class IdleRuntimeError(RuntimeError):
    """Base error for rejected dormant runtime operations."""


class IdleRuntimeBindingError(IdleRuntimeError):
    """The supplied binding or source evidence is not authoritative."""


class IdleRuntimeIneligibleError(IdleRuntimeError):
    """The source session cannot be admitted for idle runtime work."""


class IdleRuntimeResolutionError(IdleRuntimeError):
    """The authoritative source-selected preset cannot be resolved."""


class IdleRuntimeStaleError(IdleRuntimeError):
    """A binding no longer matches current runtime or source evidence."""


class IdleRuntimeClosedError(IdleRuntimeError):
    """The dormant service has stopped accepting new admissions."""


class IdleRuntimeDormantError(IdleRuntimeError):
    """Production activation is intentionally unavailable in I3b."""


class IdleRuntimeCrossProcessError(IdleRuntimeError):
    """This process-local runtime authority cannot be used after a fork."""


class IdleRuntimeCrossContextError(IdleRuntimeError):
    """The registry was called from a different thread or event loop."""


class IdlePresetSnapshotResolver(Protocol):
    """Narrow resolver for an already supplied provider snapshot."""

    def resolve_qualified_preset(
        self,
        name: str,
        snapshot: ProviderSnapshot,
    ) -> LLMRuntime: ...


# Kept as a source-compatible spelling for dormant callers.  The protocol no
# longer permits ordinary ``resolve_preset`` calls or arbitrary callables.
IdlePresetResolver = IdlePresetSnapshotResolver


@dataclass(frozen=True, slots=True, repr=False)
class OpaqueFingerprint:
    """A process-local keyed digest with no printable or serializable value."""

    _digest: bytes

    def __post_init__(self) -> None:
        if type(self._digest) is not bytes or len(self._digest) != hashlib.sha256().digest_size:
            raise ValueError("invalid opaque fingerprint")

    def __repr__(self) -> str:
        return "<opaque-fingerprint>"

    def __str__(self) -> str:
        return "<opaque-fingerprint>"

    def __reduce__(self) -> NoReturn:
        raise TypeError("opaque fingerprints cannot be serialized")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("opaque fingerprints cannot be copied")

    def __copy__(self) -> NoReturn:
        raise TypeError("opaque fingerprints cannot be copied")

    @staticmethod
    def combine_binding(
        source_fingerprint: OpaqueFingerprint,
        runtime_fingerprint: OpaqueFingerprint,
        config_fingerprint: OpaqueFingerprint,
    ) -> OpaqueFingerprint:
        """Compose existing binding material without returning raw digest bytes."""
        return _opaque_digest(
            (
                "binding",
                source_fingerprint._digest,
                runtime_fingerprint._digest,
                config_fingerprint._digest,
            )
        )


@dataclass(frozen=True, slots=True, repr=False)
class IdleSummaryProjectionConfig:
    """Versioned summary/projection inputs captured by one binding."""

    summary_version: int = 1
    projection_version: int = 1
    projection_scope: str = "root"
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_chunks: int | None = None

    def __post_init__(self) -> None:
        if type(self.summary_version) is not int or self.summary_version <= 0:
            raise ValueError("summary_version must be a positive integer")
        if type(self.projection_version) is not int or self.projection_version <= 0:
            raise ValueError("projection_version must be a positive integer")
        if not isinstance(cast(object, self.projection_scope), str) or not self.projection_scope.strip():
            raise ValueError("projection_scope must be a non-empty string")
        for name in ("max_input_tokens", "max_output_tokens", "max_chunks"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")

    def __repr__(self) -> str:
        return "<idle-summary-projection-config>"


@dataclass(frozen=True, slots=True, repr=False)
class IdleRuntimeLimits:
    """Explicit I3 limits copied from the dormant configuration contract."""

    enabled: bool = False
    idle_ttl_seconds: float = 0.0
    scan_interval_seconds: float | None = None
    queue_capacity: int | None = None
    max_concurrent_jobs: int | None = None
    max_auxiliary_calls: int | None = None
    max_auxiliary_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_chunks: int | None = None
    max_auxiliary_wall_seconds: float | None = None
    initial_backoff_seconds: float | None = None
    max_backoff_seconds: float | None = None
    summary_model_preset: str | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        _finite_non_negative(self.idle_ttl_seconds, "idle_ttl_seconds")
        for name in (
            "scan_interval_seconds",
            "max_auxiliary_wall_seconds",
            "initial_backoff_seconds",
            "max_backoff_seconds",
        ):
            _optional_positive_number(getattr(self, name), name)
        for name in (
            "queue_capacity",
            "max_concurrent_jobs",
            "max_auxiliary_calls",
            "max_auxiliary_tokens",
            "max_input_tokens",
            "max_output_tokens",
            "max_chunks",
        ):
            _optional_positive_int(getattr(self, name), name)
        if (
            self.initial_backoff_seconds is not None
            and self.max_backoff_seconds is not None
            and self.initial_backoff_seconds > self.max_backoff_seconds
        ):
            raise ValueError("initial_backoff_seconds cannot exceed max_backoff_seconds")
        if self.summary_model_preset is not None and (
            not isinstance(cast(object, self.summary_model_preset), str) or not self.summary_model_preset.strip()
        ):
            raise ValueError("summary_model_preset must be a non-empty preset name or None")

    @classmethod
    def from_config(cls, config: object) -> IdleRuntimeLimits:
        """Copy the strict I3a values without retaining a mutable config model."""
        if isinstance(config, cls):
            return config
        if isinstance(config, IdleCompactionPolicy):
            return cls(
                enabled=config.enabled,
                idle_ttl_seconds=float(config.idle_ttl_seconds),
                queue_capacity=config.queue_capacity,
                max_concurrent_jobs=config.max_concurrent_jobs,
                max_auxiliary_calls=config.max_auxiliary_calls,
                max_auxiliary_tokens=config.max_auxiliary_tokens,
                max_auxiliary_wall_seconds=config.max_auxiliary_wall_seconds,
                initial_backoff_seconds=config.initial_backoff_seconds,
                max_backoff_seconds=config.max_backoff_seconds,
            )
        if not isinstance(config, IdleCompactionConfig):
            raise TypeError("config must be IdleRuntimeLimits or IdleCompactionConfig")
        values = {
            name: getattr(config, name)
            for name in (
                "enabled",
                "idle_ttl_seconds",
                "scan_interval_seconds",
                "queue_capacity",
                "max_concurrent_jobs",
                "max_auxiliary_calls",
                "max_auxiliary_tokens",
                "max_input_tokens",
                "max_output_tokens",
                "max_chunks",
                "max_auxiliary_wall_seconds",
                "initial_backoff_seconds",
                "max_backoff_seconds",
                "summary_model_preset",
            )
        }
        return cls(**values)

    def __repr__(self) -> str:
        return "<idle-runtime-limits>"


@dataclass(frozen=True, slots=True, repr=False)
class IdleWorkspaceBinding:
    """Workspace and artifact namespace identity held privately by a binding."""

    workspace: Path | None = None
    artifact_namespace: object | None = None

    def __post_init__(self) -> None:
        if self.workspace is not None and not isinstance(cast(object, self.workspace), Path):
            raise TypeError("workspace must be a pathlib.Path or None")

    def __repr__(self) -> str:
        return "<idle-workspace-binding>"

    def __reduce__(self) -> NoReturn:
        raise TypeError("workspace bindings cannot be serialized")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("workspace bindings cannot be copied")


def _finite_non_negative(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _optional_positive_int(value: int | None, name: str) -> None:
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError(f"{name} must be a positive integer or None")


def _optional_positive_number(value: float | int | None, name: str) -> None:
    if value is None:
        return
    _finite_non_negative(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive or None")


_MAX_MATERIAL_NODES = 4096
_MAX_MATERIAL_DEPTH = 64
_MAX_MATERIAL_BYTES = 256 * 1024
_MAX_MATERIAL_STRING_BYTES = 64 * 1024
_MAX_MATERIAL_BLOB_BYTES = 64 * 1024


class _MaterialEncodingError(ValueError):
    """Internal deterministic-material rejection without user data."""


def _length_prefixed(tag: bytes, payload: bytes) -> bytes:
    size = len(tag) + 8 + len(payload)
    if size > _MAX_MATERIAL_BYTES:
        raise _MaterialEncodingError("fingerprint material is too large")
    return tag + struct.pack(">Q", len(payload)) + payload


def _bounded_text(value: str) -> bytes:
    if len(value) > _MAX_MATERIAL_STRING_BYTES:
        raise _MaterialEncodingError("fingerprint material text is too large")
    try:
        payload = value.encode("utf-8", "surrogatepass")
    except UnicodeError:
        raise _MaterialEncodingError("fingerprint material has invalid text") from None
    if len(payload) > _MAX_MATERIAL_STRING_BYTES:
        raise _MaterialEncodingError("fingerprint material text is too large")
    return payload


def _join_material(tag: bytes, parts: list[bytes]) -> bytes:
    payload_size = sum(len(part) for part in parts)
    if payload_size + len(tag) + 8 > _MAX_MATERIAL_BYTES:
        raise _MaterialEncodingError("fingerprint material is too large")
    return _length_prefixed(tag, b"".join(parts))


def _encode_material(
    value: object,
    *,
    active: set[int],
    nodes: list[int],
    depth: int,
) -> bytes:
    """Encode supported signature/config values deterministically.

    The encoder deliberately has no object traversal, repr fallback, or
    truncation.  It accepts the primitive/container material emitted by the
    provider factory and rejects cycles, unsupported values, excessive size,
    and non-finite floats before any bytes are hashed.
    """
    if depth > _MAX_MATERIAL_DEPTH:
        raise _MaterialEncodingError("fingerprint material is too deeply nested")
    nodes[0] += 1
    if nodes[0] > _MAX_MATERIAL_NODES:
        raise _MaterialEncodingError("fingerprint material is too large")
    if value is None:
        return b"n"
    if type(value) is bool:
        return b"b1" if value else b"b0"
    if type(value) is int:
        if value.bit_length() > _MAX_MATERIAL_BYTES * 8:
            raise _MaterialEncodingError("fingerprint material integer is too large")
        return _length_prefixed(b"i", str(value).encode("ascii"))
    if type(value) is float:
        if not math.isfinite(value):
            raise _MaterialEncodingError("fingerprint material has a non-finite number")
        return _length_prefixed(b"f", value.hex().encode("ascii"))
    if type(value) is str:
        return _length_prefixed(b"s", _bounded_text(value))
    if type(value) is bytes:
        if len(value) > _MAX_MATERIAL_BLOB_BYTES:
            raise _MaterialEncodingError("fingerprint material bytes are too large")
        return _length_prefixed(b"x", value)
    if isinstance(value, Path):
        return _length_prefixed(b"p", _bounded_text(str(value)))

    identity = id(value)
    if identity in active:
        raise _MaterialEncodingError("fingerprint material contains a cycle")
    if isinstance(value, Mapping):
        active.add(identity)
        try:
            encoded_items: list[tuple[bytes, bytes]] = []
            payload_size = 0
            for key, item in cast(Mapping[object, object], value).items():
                encoded_key = _encode_material(
                    key,
                    active=active,
                    nodes=nodes,
                    depth=depth + 1,
                )
                encoded_item = _encode_material(
                    item,
                    active=active,
                    nodes=nodes,
                    depth=depth + 1,
                )
                pair_size = len(encoded_key) + len(encoded_item) + 18
                if pair_size > _MAX_MATERIAL_BYTES or payload_size > _MAX_MATERIAL_BYTES - pair_size:
                    raise _MaterialEncodingError("fingerprint material is too large")
                encoded_items.append((encoded_key, encoded_item))
                payload_size += pair_size
            encoded_items.sort(key=lambda item: item[0])
            return _join_material(
                b"m",
                [
                    _length_prefixed(b"k", key) + _length_prefixed(b"v", item)
                    for key, item in encoded_items
                ],
            )
        finally:
            active.remove(identity)
    if isinstance(value, (tuple, list)):
        active.add(identity)
        try:
            encoded_sequence: list[bytes] = []
            payload_size = 0
            for item in cast(Sequence[object], value):
                encoded = _encode_material(
                    item,
                    active=active,
                    nodes=nodes,
                    depth=depth + 1,
                )
                item_size = len(encoded) + 9
                if item_size > _MAX_MATERIAL_BYTES or payload_size > _MAX_MATERIAL_BYTES - item_size:
                    raise _MaterialEncodingError("fingerprint material is too large")
                encoded_sequence.append(encoded)
                payload_size += item_size
            return _join_material(
                b"t" if type(cast(object, value)) is tuple else b"l",
                [_length_prefixed(b"e", encoded) for encoded in encoded_sequence],
            )
        finally:
            active.remove(identity)
    if isinstance(value, (set, frozenset)):
        active.add(identity)
        try:
            encoded_set: list[bytes] = []
            payload_size = 0
            for item in cast(Set[object], value):
                encoded = _encode_material(
                    item,
                    active=active,
                    nodes=nodes,
                    depth=depth + 1,
                )
                item_size = len(encoded) + 9
                if item_size > _MAX_MATERIAL_BYTES or payload_size > _MAX_MATERIAL_BYTES - item_size:
                    raise _MaterialEncodingError("fingerprint material is too large")
                encoded_set.append(encoded)
                payload_size += item_size
            encoded_set.sort()
            return _join_material(
                b"u" if type(cast(object, value)) is set else b"v",
                [_length_prefixed(b"e", encoded) for encoded in encoded_set],
            )
        finally:
            active.remove(identity)
    raise _MaterialEncodingError("unsupported fingerprint material")


def _opaque_digest(material: object) -> OpaqueFingerprint:
    encoded: bytes | None = None
    rejected = False
    try:
        encoded = _encode_material(material, active=set(), nodes=[0], depth=0)
    except _MaterialEncodingError:
        rejected = True
    except Exception:
        # Never expose arbitrary mapping/provider error text in authority
        # failures; the digest is intentionally content-free.
        rejected = True
    if rejected or encoded is None:
        raise IdleRuntimeBindingError("unsupported fingerprint material")
    return OpaqueFingerprint(hmac.new(_PROCESS_KEY, encoded, hashlib.sha256).digest())


def _runtime_material(runtime: LLMRuntime) -> object:
    """Material from explicit immutable runtime fields only.

    The provider object is intentionally absent.  Provider configuration,
    credentials, headers, bodies, queries, proxies, fallback settings, and
    account identity are represented by the captured factory signature.
    """
    signature = runtime.snapshot_signature
    if signature is None:
        raise IdleRuntimeBindingError("runtime is missing its captured provider signature")
    generation = runtime.generation
    return (
        "llm-runtime",
        runtime.model,
        (
            "generation",
            generation.temperature,
            generation.max_tokens,
            generation.reasoning_effort,
        ),
        runtime.context_window_tokens,
        runtime.model_preset,
        ("provider-signature", signature),
    )


def _source_material(source: SourceSessionStore, snapshot: SourceSnapshot) -> tuple[str, str, int, int, str, str, str | None, str]:
    state = getattr(snapshot, "state", None)
    if type(state) is not SourceState:
        raise IdleRuntimeBindingError("source snapshot has invalid state")
    key = _required_string(getattr(state, "session_key", None), "session_key")
    header = getattr(snapshot, "header", None)
    if isinstance(header, Mapping):
        header_key = cast(Mapping[object, object], header).get("key")
        if header_key is not None and header_key != key:
            raise IdleRuntimeBindingError("source snapshot header identity changed")
    incarnation = _required_string(getattr(state, "incarnation", None), "incarnation")
    revision = getattr(state, "revision", None)
    if type(revision) is not int or revision < 0:
        raise IdleRuntimeBindingError("source snapshot has invalid revision")
    generation = getattr(state, "generation", None)
    if type(generation) is not int or generation < 0:
        raise IdleRuntimeBindingError("source snapshot has invalid generation")
    store_identity = _required_string(getattr(state, "store_identity", None), "store_identity")
    if source.store_identity_differs(store_identity):
        raise IdleRuntimeBindingError("source snapshot belongs to a different source store")
    root_identity = source.resolved_runtime_source_root()
    checkpoint_id = state.checkpoint_id
    checkpoint_fingerprint = _required_string(
        getattr(state, "checkpoint_fingerprint", None), "checkpoint_fingerprint"
    )
    return (
        key,
        incarnation,
        revision,
        generation,
        store_identity,
        root_identity,
        checkpoint_id,
        checkpoint_fingerprint,
    )


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise IdleRuntimeBindingError(f"source snapshot has invalid {name}")
    return value


def _snapshot_metadata(snapshot: object) -> Mapping[object, object]:
    header = getattr(snapshot, "header", None)
    if not isinstance(header, Mapping):
        raise IdleRuntimeBindingError("source snapshot has no authoritative header")
    metadata = cast(Mapping[object, object], header).get("metadata")
    if not isinstance(metadata, Mapping):
        raise IdleRuntimeBindingError("source snapshot has no authoritative metadata")
    return cast(Mapping[object, object], metadata)


def _snapshot_identity(source: SourceSessionStore, snapshot: SourceSnapshot) -> tuple[str, str, int, int, str, str, str | None, str]:
    material = _source_material(source, snapshot)
    return material


def _snapshot_runtime_selected(
    snapshot: SourceSnapshot,
    *,
    foreground_runtime: LLMRuntime,
    preset_runtimes: Mapping[str, LLMRuntime | ProviderSnapshot] | None,
    preset_resolver: IdlePresetSnapshotResolver | None,
    preset_snapshots: Mapping[str, ProviderSnapshot] | None,
) -> tuple[str | None, LLMRuntime]:
    try:
        name = model_preset_from_metadata(_snapshot_metadata(snapshot))
    except (TypeError, ValueError):
        raise IdleRuntimeIneligibleError("invalid authoritative source preset") from None
    if name is None:
        return None, foreground_runtime
    if preset_runtimes is not None:
        candidate = preset_runtimes.get(name)
        if candidate is None:
            matches = [key for key in preset_runtimes if key.casefold() == name.casefold()]
            candidate = preset_runtimes[matches[0]] if len(matches) == 1 else None
        if candidate is None:
            raise IdleRuntimeResolutionError("authoritative source preset is unavailable")
        try:
            runtime = _coerce_runtime(candidate, name)
        except (TypeError, ValueError, KeyError):
            raise IdleRuntimeResolutionError("authoritative source preset is invalid") from None
        return name, runtime
    if preset_resolver is None:
        raise IdleRuntimeResolutionError("authoritative source preset is unavailable")
    if preset_snapshots is None:
        raise IdleRuntimeResolutionError("an injected resolver requires a qualified provider snapshot")
    snapshot_value = preset_snapshots.get(name)
    if snapshot_value is None:
        matches = [key for key in preset_snapshots if key.casefold() == name.casefold()]
        snapshot_value = preset_snapshots[matches[0]] if len(matches) == 1 else None
    if type(snapshot_value) is not ProviderSnapshot:
        raise IdleRuntimeResolutionError("authoritative source preset is unavailable")
    try:
        candidate = preset_resolver.resolve_qualified_preset(name, snapshot_value)
        runtime = _coerce_runtime(candidate, name)
    except IdleRuntimeError:
        raise
    except Exception:
        # A resolver may wrap provider/configuration details in its exception;
        # dormant authority reports only a content-free admission failure.
        pass
    else:
        return name, runtime
    raise IdleRuntimeResolutionError(
        "injected preset resolver rejected the qualified snapshot"
    )


def _coerce_runtime(candidate: object, name: str) -> LLMRuntime:
    if isinstance(candidate, LLMRuntime):
        if candidate.model_preset not in (None, name):
            raise IdleRuntimeResolutionError("resolved runtime preset does not match source selection")
        return candidate if candidate.model_preset == name else replace(candidate, model_preset=name)
    if type(candidate) is ProviderSnapshot:
        snapshot = candidate
        if snapshot.model_preset not in (None, name):
            raise IdleRuntimeResolutionError("resolved snapshot preset does not match source selection")
        return runtime_from_provider_snapshot(replace(snapshot, model_preset=name))
    raise TypeError("preset resolver must return LLMRuntime or ProviderSnapshot")


@dataclass(frozen=True, slots=True)
class _BindingMaterial:
    source: SourceSessionStore
    identity: tuple[str, str, int, int, str, str, str | None, str]
    selected_preset: str | None
    foreground_runtime: LLMRuntime
    auxiliary_runtime: LLMRuntime
    summary_projection: IdleSummaryProjectionConfig
    limits: IdleRuntimeLimits
    workspace_binding: IdleWorkspaceBinding
    policy_binding: IdlePolicyBinding
    runtime_epoch: int
    config_epoch: int
    source_fingerprint: OpaqueFingerprint
    runtime_fingerprint: OpaqueFingerprint
    config_fingerprint: OpaqueFingerprint
    fingerprint: OpaqueFingerprint


@dataclass(frozen=True, slots=True, init=False, repr=False, eq=False)
class IdleRuntimeBinding:
    """Frozen authority for one exact source/session runtime admission."""

    _authority: object
    _pid: int
    _source: SourceSessionStore
    _session_key: str
    _incarnation: str
    _revision: int
    _generation: int
    _store_identity: str
    _root_identity: object
    _checkpoint_id: str | None
    _checkpoint_fingerprint: str
    _selected_preset: str | None
    _foreground_runtime: LLMRuntime
    _auxiliary_runtime: LLMRuntime
    _summary_projection: IdleSummaryProjectionConfig
    _limits: IdleRuntimeLimits
    _workspace_binding: IdleWorkspaceBinding
    _policy_binding: IdlePolicyBinding
    _runtime_epoch: int
    _config_epoch: int
    _source_fingerprint: OpaqueFingerprint
    _runtime_fingerprint: OpaqueFingerprint
    _config_fingerprint: OpaqueFingerprint
    _fingerprint: OpaqueFingerprint

    def __init__(
        self,
        authority: object,
        pid: int,
        source: SourceSessionStore,
        session_key: str,
        incarnation: str,
        revision: int,
        generation: int,
        store_identity: str,
        root_identity: object,
        checkpoint_id: str | None,
        checkpoint_fingerprint: str,
        selected_preset: str | None,
        foreground_runtime: LLMRuntime,
        auxiliary_runtime: LLMRuntime,
        summary_projection: IdleSummaryProjectionConfig,
        limits: IdleRuntimeLimits,
        workspace_binding: IdleWorkspaceBinding,
        policy_binding: IdlePolicyBinding,
        runtime_epoch: int,
        config_epoch: int,
        source_fingerprint: OpaqueFingerprint,
        runtime_fingerprint: OpaqueFingerprint,
        config_fingerprint: OpaqueFingerprint,
        fingerprint: OpaqueFingerprint,
    ) -> None:
        object.__setattr__(self, "_authority", authority)
        object.__setattr__(self, "_pid", pid)
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_session_key", session_key)
        object.__setattr__(self, "_incarnation", incarnation)
        object.__setattr__(self, "_revision", revision)
        object.__setattr__(self, "_generation", generation)
        object.__setattr__(self, "_store_identity", store_identity)
        object.__setattr__(self, "_root_identity", root_identity)
        object.__setattr__(self, "_checkpoint_id", checkpoint_id)
        object.__setattr__(self, "_checkpoint_fingerprint", checkpoint_fingerprint)
        object.__setattr__(self, "_selected_preset", selected_preset)
        object.__setattr__(self, "_foreground_runtime", foreground_runtime)
        object.__setattr__(self, "_auxiliary_runtime", auxiliary_runtime)
        object.__setattr__(self, "_summary_projection", summary_projection)
        object.__setattr__(self, "_limits", limits)
        object.__setattr__(self, "_workspace_binding", workspace_binding)
        object.__setattr__(self, "_policy_binding", policy_binding)
        object.__setattr__(self, "_runtime_epoch", runtime_epoch)
        object.__setattr__(self, "_config_epoch", config_epoch)
        object.__setattr__(self, "_source_fingerprint", source_fingerprint)
        object.__setattr__(self, "_runtime_fingerprint", runtime_fingerprint)
        object.__setattr__(self, "_config_fingerprint", config_fingerprint)
        object.__setattr__(self, "_fingerprint", fingerprint)

    @classmethod
    def _issue(
        cls,
        *,
        authority: object,
        registry_pid: int,
        material: _BindingMaterial,
    ) -> IdleRuntimeBinding:
        (
            key,
            incarnation,
            revision,
            generation,
            store_identity,
            root_identity,
            checkpoint_id,
            checkpoint_fingerprint,
        ) = material.identity
        return cls(
            authority,
            registry_pid,
            material.source,
            key,
            incarnation,
            revision,
            generation,
            store_identity,
            root_identity,
            checkpoint_id,
            checkpoint_fingerprint,
            material.selected_preset,
            material.foreground_runtime,
            material.auxiliary_runtime,
            material.summary_projection,
            material.limits,
            material.workspace_binding,
            material.policy_binding,
            material.runtime_epoch,
            material.config_epoch,
            material.source_fingerprint,
            material.runtime_fingerprint,
            material.config_fingerprint,
            material.fingerprint,
        )

    @classmethod
    def from_registry_material(
        cls,
        arguments: Callable[[], tuple[object, int, _BindingMaterial]],
    ) -> IdleRuntimeBinding:
        """Construct, but do not register, using the registry's live arguments."""
        # Lookup precedes owner argument reads; keep the original private seam.
        issue = cls._issue
        authority, registry_pid, material = arguments()
        return issue(authority=authority, registry_pid=registry_pid, material=material)

    def matches_registry_material(
        self,
        *,
        authority: Callable[[], object],
        registry_pid: Callable[[], int],
        current_lookup: Callable[[], Callable[[str], IdleRuntimeBinding | None]],
        material: _BindingMaterial,
    ) -> bool:
        """Compare owned fields; lazy suppliers retain live registry read order."""
        return (
            self._authority is authority()
            and self._pid == registry_pid()
            # Obtain the live .get before reading the binding's key, as before.
            and current_lookup()(self._session_key) is self
            and self._source is material.source
            and (
                self._session_key,
                self._incarnation,
                self._revision,
                self._generation,
                self._store_identity,
                self._root_identity,
                self._checkpoint_id,
                self._checkpoint_fingerprint,
            )
            == material.identity
            and self._selected_preset == material.selected_preset
            and self._runtime_epoch == material.runtime_epoch
            and self._config_epoch == material.config_epoch
            and self._source_fingerprint == material.source_fingerprint
            and self._runtime_fingerprint == material.runtime_fingerprint
            and self._config_fingerprint == material.config_fingerprint
            and self._fingerprint == material.fingerprint
        )

    def validate_registry_identity(
        self,
        *,
        authority: Callable[[], object],
        registry_pid: Callable[[], int],
    ) -> None:
        """Retain identity refusals; current-table admission remains the registry's job."""
        if self._authority is not authority():
            raise IdleRuntimeBindingError("foreign or reconstructed idle runtime binding")
        if self._pid != registry_pid():
            raise IdleRuntimeStaleError("idle runtime binding is stale")

    def __repr__(self) -> str:
        return "<IdleRuntimeBinding opaque>"

    def __reduce__(self) -> NoReturn:
        raise TypeError("idle runtime bindings cannot be serialized")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("idle runtime bindings cannot be copied")

    def __copy__(self) -> NoReturn:
        raise TypeError("idle runtime bindings cannot be copied")

    @property
    def session_key(self) -> str:
        """Return the bound key for functional dispatch; observability redacts it."""
        return self._session_key

    @property
    def source_store(self) -> object:
        """Return the exact source-store object used for this authority."""
        return self._source

    @property
    def store_identity(self) -> str:
        return self._store_identity

    @property
    def root_identity(self) -> object:
        return self._root_identity

    @property
    def incarnation(self) -> str:
        return self._incarnation

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def checkpoint_id(self) -> str | None:
        return self._checkpoint_id

    @property
    def checkpoint_fingerprint(self) -> str:
        return self._checkpoint_fingerprint

    @property
    def selected_preset(self) -> str | None:
        return self._selected_preset

    @property
    def foreground_runtime(self) -> LLMRuntime:
        return self._foreground_runtime

    @property
    def runtime(self) -> LLMRuntime:
        """Alias for the immutable foreground runtime used by later consumers."""
        return self._foreground_runtime

    @property
    def foreground(self) -> LLMRuntime:
        return self._foreground_runtime

    @property
    def auxiliary_runtime(self) -> LLMRuntime:
        return self._auxiliary_runtime

    @property
    def auxiliary(self) -> LLMRuntime:
        return self._auxiliary_runtime

    @property
    def summary_projection(self) -> IdleSummaryProjectionConfig:
        return self._summary_projection

    @property
    def summary_config(self) -> IdleSummaryProjectionConfig:
        return self._summary_projection

    @property
    def projection_config(self) -> IdleSummaryProjectionConfig:
        return self._summary_projection

    @property
    def limits(self) -> IdleRuntimeLimits:
        return self._limits

    @property
    def workspace_binding(self) -> IdleWorkspaceBinding:
        return self._workspace_binding

    @property
    def workspace(self) -> Path | None:
        return self._workspace_binding.workspace

    @property
    def artifact_namespace(self) -> object | None:
        return self._workspace_binding.artifact_namespace

    @property
    def policy_binding(self) -> IdlePolicyBinding:
        return self._policy_binding

    @property
    def epoch(self) -> int:
        return self._config_epoch

    @property
    def runtime_epoch(self) -> int:
        return self._runtime_epoch

    @property
    def config_epoch(self) -> int:
        return self._config_epoch

    @property
    def source_fingerprint(self) -> OpaqueFingerprint:
        return self._source_fingerprint

    @property
    def runtime_fingerprint(self) -> OpaqueFingerprint:
        return self._runtime_fingerprint

    @property
    def config_fingerprint(self) -> OpaqueFingerprint:
        return self._config_fingerprint

    @property
    def fingerprint(self) -> OpaqueFingerprint:
        return self._fingerprint



@dataclass(frozen=True, slots=True)
class _SourceCapture:
    """A registry-sealed snapshot loaded synchronously from its exact store."""

    authority: object
    source: SourceSessionStore
    session_key: str
    snapshot: SourceSnapshot

    def __post_init__(self) -> None:
        if self.authority is None or not isinstance(cast(object, self.source), SourceSessionStore):
            raise IdleRuntimeBindingError("invalid sealed source capture")
        if type(self.snapshot) is not SourceSnapshot:
            raise IdleRuntimeBindingError("invalid sealed source capture")
        if self.snapshot.state.session_key != self.session_key:
            raise IdleRuntimeBindingError("sealed source capture identity changed")


class _EventLoopGate:
    """One logical gate for synchronous authority and async publication."""

    def __init__(self) -> None:
        self.thread_id = threading.get_ident()
        try:
            self.loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None
        self.lock = asyncio.Lock()
        self.sync_lock = threading.RLock()
        self.owner_task: asyncio.Task[object] | None = None
        self.sync_active = False

    def check_context(self) -> None:
        if threading.get_ident() != self.thread_id:
            raise IdleRuntimeCrossContextError(
                "idle runtime authority is bound to its creating thread"
            )
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is None:
            return
        if self.loop is None:
            self.loop = running
        elif running is not self.loop:
            raise IdleRuntimeCrossContextError(
                "idle runtime authority is bound to its creating event loop"
            )

    def sync(self, callback: Callable[[], _ResultT]) -> _ResultT:
        self.check_context()
        if self.owner_task is not None:
            raise IdleRuntimeCrossContextError(
                "idle runtime authority is busy in its publication gate"
            )
        with self.sync_lock:
            if self.sync_active:
                raise IdleRuntimeCrossContextError(
                    "idle runtime authority rejects nested synchronous mutation"
                )
            self.sync_active = True
            try:
                return callback()
            finally:
                self.sync_active = False

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[None, None]:
        self.check_context()
        async with self.lock:
            with self.sync_lock:
                self.check_context()
                self.owner_task = asyncio.current_task()
                try:
                    yield
                finally:
                    self.owner_task = None


class IdleRuntimeBindingRegistry:
    """Issue and validate dormant, process-local idle runtime bindings.

    The registry requires an exact SourceSessionStore and loads every
    authoritative snapshot synchronously by session key. Detached snapshots
    are never accepted as caller authority. All mutable authority changes and
    the future synchronous publication callback use one event-loop/thread
    gate; callers must stay on the creating thread and event loop once one is
    captured. No scanner, provider factory, network call, or storage write is
    started by this dormant substrate.
    """

    def __init__(
        self,
        *,
        source_store: SourceSessionStore | None = None,
        source: SourceSessionStore | None = None,
        foreground_runtime: LLMRuntime | None = None,
        runtime: LLMRuntime | None = None,
        foreground: LLMRuntime | None = None,
        auxiliary_runtime: LLMRuntime | None = None,
        auxiliary: LLMRuntime | None = None,
        preset_resolver: IdlePresetSnapshotResolver | None = None,
        preset_snapshots: Mapping[str, ProviderSnapshot] | None = None,
        preset_runtimes: Mapping[str, LLMRuntime | ProviderSnapshot] | None = None,
        summary_projection: IdleSummaryProjectionConfig | None = None,
        summary_config: IdleSummaryProjectionConfig | None = None,
        limits: IdleRuntimeLimits | IdleCompactionConfig | IdleCompactionPolicy | None = None,
        config: IdleRuntimeLimits | IdleCompactionConfig | IdleCompactionPolicy | None = None,
        runtime_config: object | None = None,
        provider_config: object | None = None,
        workspace_binding: IdleWorkspaceBinding | None = None,
        namespace: IdleWorkspaceBinding | None = None,
        workspace: Path | str | None = None,
        artifact_namespace: object | None = None,
        artifact_store: object | None = None,
        policy_binding: IdlePolicyBinding | None = None,
        idle_policy_binding: IdlePolicyBinding | None = None,
        policy: IdlePolicyBinding | None = None,
        coordinator: _IdleCoordinatorLifecycle | None = None,
        owns_coordinator: bool | None = None,
        own_coordinator: bool | None = None,
        runtime_epoch: int = 0,
        config_epoch: int = 0,
    ) -> None:
        if source_store is not None and source is not None and source_store is not source:
            raise ValueError("source_store and source must refer to the same object")
        actual_source = source_store if source_store is not None else source
        if not isinstance(actual_source, SourceSessionStore):
            raise TypeError("source_store must be an exact SourceSessionStore")
        self._source = actual_source

        supplied_foregrounds = [
            value for value in (foreground_runtime, runtime, foreground) if value is not None
        ]
        if supplied_foregrounds and any(
            value is not supplied_foregrounds[0] for value in supplied_foregrounds[1:]
        ):
            raise ValueError("foreground runtime aliases must refer to the same object")
        chosen_foreground = supplied_foregrounds[0] if supplied_foregrounds else None
        if not isinstance(chosen_foreground, LLMRuntime):
            raise TypeError("foreground_runtime must be an LLMRuntime")

        supplied_auxiliaries = [
            value for value in (auxiliary_runtime, auxiliary) if value is not None
        ]
        if supplied_auxiliaries and any(
            value is not supplied_auxiliaries[0] for value in supplied_auxiliaries[1:]
        ):
            raise ValueError("auxiliary runtime aliases must refer to the same object")
        chosen_auxiliary = supplied_auxiliaries[0] if supplied_auxiliaries else None
        if not isinstance(chosen_auxiliary, LLMRuntime):
            raise TypeError("an explicitly supplied auxiliary_runtime must be an LLMRuntime")
        self._foreground_runtime = chosen_foreground
        self._auxiliary_runtime = chosen_auxiliary

        if summary_projection is not None and summary_config is not None:
            if summary_projection != summary_config:
                raise ValueError("summary_projection and summary_config must agree")
        self._summary_projection = (
            summary_projection or summary_config or IdleSummaryProjectionConfig()
        )
        if limits is not None and config is not None:
            if IdleRuntimeLimits.from_config(limits) != IdleRuntimeLimits.from_config(config):
                raise ValueError("limits and config must agree")
        self._limits = IdleRuntimeLimits.from_config(limits or config or IdleRuntimeLimits())

        # Runtime/config objects are accepted only as an epoch-admission API.
        # Their mutable provider contents are intentionally never retained or
        # traversed; a refresh must supply a new runtime and/or epoch.
        if runtime_config is not None and provider_config is not None:
            if runtime_config is not provider_config:
                raise ValueError("runtime_config and provider_config must refer to the same object")

        if workspace_binding is not None and namespace is not None:
            if workspace_binding is not namespace:
                raise ValueError("workspace_binding and namespace must refer to the same object")
        chosen_workspace = workspace_binding or namespace
        if artifact_namespace is not None and artifact_store is not None:
            if artifact_namespace is not artifact_store:
                raise ValueError("artifact_namespace and artifact_store must refer to the same object")
        chosen_artifact = artifact_namespace if artifact_namespace is not None else artifact_store
        if chosen_workspace is not None and (
            workspace is not None or chosen_artifact is not None
        ):
            raise ValueError("workspace_binding cannot be combined with workspace arguments")
        workspace_path = Path(workspace).expanduser().resolve(strict=False) if workspace is not None else None
        self._workspace_binding = chosen_workspace or IdleWorkspaceBinding(
            workspace_path, chosen_artifact
        )

        if policy_binding is not None and idle_policy_binding is not None:
            if policy_binding is not idle_policy_binding:
                raise ValueError("policy_binding and idle_policy_binding must refer to the same object")
        chosen_policy = policy_binding or idle_policy_binding
        if chosen_policy is not None and policy is not None and chosen_policy is not policy:
            raise ValueError("policy_binding and policy must refer to the same object")
        self._policy_binding = chosen_policy or policy
        if self._policy_binding is not None and not isinstance(
            cast(object, self._policy_binding), IdlePolicyBinding
        ):
            raise TypeError("policy_binding must be an IdlePolicyBinding")

        if preset_runtimes is not None and preset_resolver is not None:
            raise ValueError("choose preset_runtimes or preset_resolver, not both")
        if preset_resolver is not None:
            if not callable(getattr(preset_resolver, "resolve_qualified_preset", None)):
                raise TypeError(
                    "preset_resolver must implement resolve_qualified_preset(name, snapshot)"
                )
            if preset_snapshots is None:
                raise TypeError("preset_resolver requires preset_snapshots")
        if preset_snapshots is not None:
            for key, value in preset_snapshots.items():
                if not isinstance(cast(object, key), str) or not key.strip() or type(value) is not ProviderSnapshot:
                    raise TypeError("preset_snapshots must map names to ProviderSnapshot values")
            self._preset_snapshots: dict[str, ProviderSnapshot] = dict(preset_snapshots)
        else:
            self._preset_snapshots = {}
        if preset_runtimes is not None:
            for key, value in preset_runtimes.items():
                if not isinstance(cast(object, key), str) or not key.strip():
                    raise TypeError("preset_runtimes keys must be non-empty strings")
                if not isinstance(value, LLMRuntime) and type(value) is not ProviderSnapshot:
                    raise TypeError("preset_runtimes values must be LLMRuntime or ProviderSnapshot")
            self._preset_runtimes: dict[str, LLMRuntime | ProviderSnapshot] | None = dict(
                preset_runtimes
            )
        else:
            self._preset_runtimes = None
        self._preset_resolver = preset_resolver

        self._coordinator = coordinator
        if owns_coordinator is not None and own_coordinator is not None:
            if owns_coordinator != own_coordinator:
                raise ValueError("owns_coordinator and own_coordinator must agree")
        self._owns_coordinator = (
            True
            if owns_coordinator is None and own_coordinator is None
            else bool(owns_coordinator if owns_coordinator is not None else own_coordinator)
        )

        if type(runtime_epoch) is not int or runtime_epoch < 0:
            raise ValueError("runtime_epoch must be a non-negative integer")
        if type(config_epoch) is not int or config_epoch < 0:
            raise ValueError("config_epoch must be a non-negative integer")
        self._runtime_epoch = runtime_epoch
        self._config_epoch = config_epoch
        self._authority = object()
        self._capture_authority = object()
        self._pid = os.getpid()
        self._gate = _EventLoopGate()
        self._current: dict[str, IdleRuntimeBinding] = {}
        self._active_callback: IdleRuntimeBinding | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False
        self._close_error: BaseException | None = None
        self._issued_count = 0
        self._invalidation_count = 0

    def __repr__(self) -> str:
        return "<IdleRuntimeBindingRegistry opaque>"

    def __reduce__(self) -> NoReturn:
        raise TypeError("idle runtime registries cannot be serialized")

    @property
    def config_epoch(self) -> int:
        return self._config_epoch

    @property
    def runtime_epoch(self) -> int:
        self._gate.check_context()
        return self._runtime_epoch

    @property
    def source_store(self) -> SourceSessionStore:
        self._gate.check_context()
        return self._source

    @property
    def foreground_runtime(self) -> LLMRuntime:
        self._gate.check_context()
        return self._foreground_runtime

    @property
    def auxiliary_runtime(self) -> LLMRuntime:
        self._gate.check_context()
        return self._auxiliary_runtime

    @property
    def summary_projection(self) -> IdleSummaryProjectionConfig:
        self._gate.check_context()
        return self._summary_projection

    @property
    def limits(self) -> IdleRuntimeLimits:
        self._gate.check_context()
        return self._limits

    @property
    def policy_binding(self) -> IdlePolicyBinding | None:
        self._gate.check_context()
        return self._policy_binding

    @property
    def closed(self) -> bool:
        self._gate.check_context()
        return self._closed

    @property
    def binding_count(self) -> int:
        self._gate.check_context()
        return len(self._current)

    @property
    def admission_count(self) -> int:
        self._gate.check_context()
        return len(self._current)

    @property
    def issued_count(self) -> int:
        self._gate.check_context()
        return self._issued_count

    @property
    def invalidation_count(self) -> int:
        self._gate.check_context()
        return self._invalidation_count

    def observability(self) -> dict[str, object]:
        """Return only opaque state, counters, and epochs."""
        self._gate.check_context()
        return {
            "state": "closed" if self._closed else "open",
            "runtime_epoch": self._runtime_epoch,
            "config_epoch": self._config_epoch,
            "binding_count": len(self._current),
            "admission_count": len(self._current),
            "issued_count": self._issued_count,
            "invalidation_count": self._invalidation_count,
            "active_callback": self._active_callback is not None,
        }

    def _ensure_process(self) -> None:
        if os.getpid() != self._pid:
            self._closed = True
            self._current.clear()
            self._active_callback = None
            raise IdleRuntimeCrossProcessError("idle runtime authority is process-local")

    def _ensure_open(self) -> None:
        self._gate.check_context()
        self._ensure_process()
        if self._closed or self._closing:
            raise IdleRuntimeClosedError("idle runtime service is closed")

    @staticmethod
    def _require_session_key(session_key: object) -> str:
        if type(session_key) is not str or not session_key.strip():
            raise TypeError("session_key must be a non-empty string")
        return session_key

    def _capture_source(
        self,
        session_key: str,
        *,
        root_authority: SourceRootAuthority | None = None,
    ) -> _SourceCapture:
        key = self._require_session_key(session_key)
        try:
            snapshot = self._source.load_source_snapshot(key, root_authority=root_authority)
        except Exception:
            raise IdleRuntimeIneligibleError("authoritative source snapshot is unavailable") from None
        if type(snapshot) is not SourceSnapshot:
            raise IdleRuntimeIneligibleError("authoritative source snapshot is invalid")
        if snapshot.state.session_key != key:
            raise IdleRuntimeIneligibleError("authoritative source identity changed")
        try:
            _source_material(self._source, snapshot)
        except IdleRuntimeError:
            raise
        except Exception:
            raise IdleRuntimeIneligibleError("authoritative source snapshot is invalid") from None
        return _SourceCapture(self._capture_authority, self._source, key, snapshot)

    @staticmethod
    def _summary_material(config: IdleSummaryProjectionConfig) -> tuple[object, ...]:
        return (
            config.summary_version,
            config.projection_version,
            config.projection_scope,
            config.max_input_tokens,
            config.max_output_tokens,
            config.max_chunks,
        )

    @staticmethod
    def _limits_material(config: IdleRuntimeLimits) -> tuple[object, ...]:
        return (
            config.enabled,
            config.idle_ttl_seconds,
            config.scan_interval_seconds,
            config.queue_capacity,
            config.max_concurrent_jobs,
            config.max_auxiliary_calls,
            config.max_auxiliary_tokens,
            config.max_input_tokens,
            config.max_output_tokens,
            config.max_chunks,
            config.max_auxiliary_wall_seconds,
            config.initial_backoff_seconds,
            config.max_backoff_seconds,
            config.summary_model_preset,
        )

    @staticmethod
    def _workspace_material(binding: IdleWorkspaceBinding) -> tuple[object, ...]:
        return (
            str(binding.workspace) if binding.workspace is not None else None,
            binding.artifact_namespace,
        )

    @staticmethod
    def _policy_material(binding: IdlePolicyBinding) -> tuple[object, ...]:
        policy = binding.resource_policy
        policy_type = (
            None
            if policy is None
            else (type(policy).__module__, type(policy).__qualname__)
        )
        return (
            binding.fingerprint,
            str(binding.workspace) if binding.workspace is not None else None,
            policy_type,
        )

    def _build_material(
        self,
        capture: _SourceCapture,
    ) -> _BindingMaterial:
        chosen_foreground = self._foreground_runtime
        chosen_auxiliary = self._auxiliary_runtime
        if not isinstance(cast(object, chosen_foreground), LLMRuntime):
            raise TypeError("foreground_runtime must be an LLMRuntime")
        if not isinstance(cast(object, chosen_auxiliary), LLMRuntime):
            raise TypeError("auxiliary_runtime must be an LLMRuntime")
        chosen_summary = self._summary_projection
        chosen_limits = self._limits
        chosen_policy = self._policy_binding
        if chosen_policy is None:
            raise IdleRuntimeBindingError("an explicit IdlePolicyBinding is required")
        if not isinstance(cast(object, chosen_policy), IdlePolicyBinding):
            raise TypeError("policy_binding must be an IdlePolicyBinding")
        selected, resolved_foreground = _snapshot_runtime_selected(
            capture.snapshot,
            foreground_runtime=chosen_foreground,
            preset_runtimes=self._preset_runtimes,
            preset_resolver=self._preset_resolver,
            preset_snapshots=self._preset_snapshots or None,
        )
        identity = _snapshot_identity(self._source, capture.snapshot)
        source_fingerprint = _opaque_digest(("source", identity))
        runtime_fingerprint = _opaque_digest(
            (
                "runtime",
                self._runtime_epoch,
                selected,
                _runtime_material(resolved_foreground),
                _runtime_material(chosen_auxiliary),
            )
        )
        config_fingerprint = _opaque_digest(
            (
                "config",
                self._config_epoch,
                self._summary_material(chosen_summary),
                self._limits_material(chosen_limits),
                self._workspace_material(self._workspace_binding),
                self._policy_material(chosen_policy),
            )
        )
        fingerprint = OpaqueFingerprint.combine_binding(
            source_fingerprint, runtime_fingerprint, config_fingerprint,
        )
        return _BindingMaterial(
            self._source,
            identity,
            selected,
            resolved_foreground,
            chosen_auxiliary,
            chosen_summary,
            chosen_limits,
            self._workspace_binding,
            chosen_policy,
            self._runtime_epoch,
            self._config_epoch,
            source_fingerprint,
            runtime_fingerprint,
            config_fingerprint,
            fingerprint,
        )

    def _binding_matches(self, binding: IdleRuntimeBinding, material: _BindingMaterial) -> bool:
        return IdleRuntimeBinding.matches_registry_material(
            binding,
            authority=lambda: self._authority,
            registry_pid=lambda: self._pid,
            current_lookup=lambda: self._current.get,
            material=material,
        )

    def _validate_binding(self, binding: IdleRuntimeBinding) -> None:
        if type(binding) is not IdleRuntimeBinding:
            raise IdleRuntimeBindingError("invalid idle runtime binding")
        IdleRuntimeBinding.validate_registry_identity(
            binding, authority=lambda: self._authority, registry_pid=lambda: self._pid,
        )

    def _retire_current_locked(self, session_key: str | None = None) -> None:
        if session_key is None:
            count = len(self._current)
            self._current.clear()
        else:
            count = 1 if self._current.pop(session_key, None) is not None else 0
        self._invalidation_count += count

    def _issue_locked(
        self,
        session_key: str,
        *,
        root_authority: SourceRootAuthority | None = None,
    ) -> IdleRuntimeBinding:
        self._ensure_open()
        capture = self._capture_source(session_key, root_authority=root_authority)
        material = self._build_material(capture)
        current = self._current.get(session_key)
        if current is not None and self._binding_matches(current, material):
            return current
        if current is not None:
            self._retire_current_locked(session_key)
        binding = IdleRuntimeBinding.from_registry_material(
            lambda: (self._authority, self._pid, material),
        )
        self._current[session_key] = binding
        self._issued_count += 1
        return binding

    def issue(
        self,
        session_key: str,
    ) -> IdleRuntimeBinding:
        """Load the current source by key and issue or reuse one binding."""
        return self._gate.sync(
            lambda: self._issue_locked(self._require_session_key(session_key))
        )

    def _revalidate_locked(
        self,
        binding: IdleRuntimeBinding,
        session_key: str,
        *,
        root_authority: SourceRootAuthority | None = None,
    ) -> bool:
        self._ensure_open()
        self._validate_binding(binding)
        if binding.session_key != session_key or self._current.get(session_key) is not binding:
            return False
        try:
            capture = self._capture_source(session_key, root_authority=root_authority)
            material = self._build_material(capture)
        except IdleRuntimeError:
            self._retire_current_locked(session_key)
            return False
        if self._binding_matches(binding, material):
            return True
        self._retire_current_locked(session_key)
        return False

    def revalidate(self, binding: IdleRuntimeBinding, session_key: str) -> bool:
        """Reload current source authority and compare one registered binding."""
        return bool(
            self._gate.sync(
                lambda: self._revalidate_locked(
                    binding,
                    self._require_session_key(session_key),
                )
            )
        )

    def require_current(self, binding: IdleRuntimeBinding, session_key: str) -> None:
        if not self.revalidate(binding, session_key):
            raise IdleRuntimeStaleError("idle runtime binding no longer matches current evidence")

    def resolve_current(self, session_key: str) -> IdleRuntimeBinding:
        """Return the one current binding for a source session."""
        return self.issue(session_key)

    async def resolve_current_async(
        self,
        session_key: str,
        *,
        root_authority: SourceRootAuthority | None = None,
    ) -> IdleRuntimeBinding:
        async with self.publication_gate():
            return self._issue_locked(
                self._require_session_key(session_key),
                root_authority=root_authority,
            )

    async def revalidate_async(
        self,
        binding: IdleRuntimeBinding,
        session_key: str,
        *,
        root_authority: SourceRootAuthority | None = None,
    ) -> bool:
        async with self.publication_gate():
            return self._revalidate_locked(
                binding,
                self._require_session_key(session_key),
                root_authority=root_authority,
            )

    def _invalidate_locked(
        self,
        binding: IdleRuntimeBinding | None = None,
        *,
        session_key: str | None = None,
    ) -> None:
        self._ensure_process()
        if binding is not None:
            self._validate_binding(binding)
            if session_key is not None and binding.session_key != session_key:
                raise IdleRuntimeBindingError("binding and session key disagree")
            if self._current.get(binding.session_key) is binding:
                self._retire_current_locked(binding.session_key)
            return
        if session_key is None:
            self._retire_current_locked()
        else:
            self._retire_current_locked(session_key)

    def invalidate(
        self,
        binding: IdleRuntimeBinding | None = None,
        *,
        session_key: str | None = None,
    ) -> None:
        """Retire one current binding, one session, or all current bindings."""
        self._gate.sync(
            lambda: self._invalidate_locked(binding, session_key=session_key)
        )

    def replace_runtimes(
        self,
        *,
        foreground_runtime: object,
        auxiliary_runtime: object,
    ) -> int:
        """Replace explicitly captured runtimes and retire the old epoch."""
        if not isinstance(foreground_runtime, LLMRuntime):
            raise TypeError("foreground_runtime must be an LLMRuntime")
        if not isinstance(auxiliary_runtime, LLMRuntime):
            raise TypeError("auxiliary_runtime must be an LLMRuntime")

        def update() -> int:
            self._ensure_open()
            self._foreground_runtime = foreground_runtime
            self._auxiliary_runtime = auxiliary_runtime
            self._runtime_epoch += 1
            self._retire_current_locked()
            return self._runtime_epoch

        return self._gate.sync(update)

    def replace_config(
        self,
        *,
        limits: IdleRuntimeLimits | IdleCompactionConfig | IdleCompactionPolicy | None = None,
        config: IdleRuntimeLimits | IdleCompactionConfig | IdleCompactionPolicy | None = None,
        summary_projection: IdleSummaryProjectionConfig | None = None,
        summary_config: IdleSummaryProjectionConfig | None = None,
        runtime_config: object | None = None,
        provider_config: object | None = None,
        workspace_binding: IdleWorkspaceBinding | None = None,
        policy_binding: IdlePolicyBinding | None = None,
    ) -> int:
        """Replace explicit config inputs, advance its epoch, and retire bindings."""
        if limits is not None and config is not None:
            if IdleRuntimeLimits.from_config(limits) != IdleRuntimeLimits.from_config(config):
                raise ValueError("limits and config must agree")
        if summary_projection is not None and summary_config is not None:
            if summary_projection != summary_config:
                raise ValueError("summary_projection and summary_config must agree")
        if runtime_config is not None and provider_config is not None:
            if runtime_config is not provider_config:
                raise ValueError("runtime_config and provider_config must refer to the same object")

        def update() -> int:
            self._ensure_process()
            self._limits = IdleRuntimeLimits.from_config(limits or config or self._limits)
            self._summary_projection = summary_projection or summary_config or self._summary_projection
            if workspace_binding is not None:
                self._workspace_binding = workspace_binding
            if policy_binding is not None:
                if not isinstance(cast(object, policy_binding), IdlePolicyBinding):
                    raise TypeError("policy_binding must be an IdlePolicyBinding")
                self._policy_binding = policy_binding
            self._config_epoch += 1
            self._retire_current_locked()
            return self._config_epoch

        return self._gate.sync(update)

    async def foreground_admitted(self, session_key: str) -> None:
        """Invalidate idle admission then notify an injected coordinator."""
        key = self._require_session_key(session_key)
        self.invalidate(session_key=key)
        coordinator = self._coordinator
        if coordinator is None:
            return
        callback = getattr(coordinator, "foreground_admitted", None)
        if not callable(callback):
            raise IdleRuntimeBindingError("injected coordinator lacks foreground_admitted")
        try:
            result = callback(key)
            if inspect.isawaitable(result):
                await cast(Awaitable[object], result)
        except IdleRuntimeError:
            raise
        except Exception:
            raise IdleRuntimeBindingError("foreground admission callback failed") from None

    async def publish_if_current(
        self,
        binding: IdleRuntimeBinding,
        sync_callback: Callable[[IdleRuntimeBinding], _ResultT],
        *,
        session_key: str | None = None,
        root_authority: SourceRootAuthority | None = None,
    ) -> _ResultT:
        """Reload/revalidate and invoke one non-awaitable callback under the gate."""
        key = self._require_session_key(
            binding.session_key if session_key is None else session_key
        )
        if not callable(sync_callback):
            raise TypeError("sync_callback must be callable")
        async with self.publication_gate():
            if not self._revalidate_locked(binding, key, root_authority=root_authority):
                raise IdleRuntimeStaleError(
                    "idle runtime binding no longer matches current evidence"
                )
            self._active_callback = binding
            try:
                result = sync_callback(binding)
                if inspect.isawaitable(result):
                    raise IdleRuntimeBindingError(
                        "publication callback must return without awaiting"
                    )
                return result
            finally:
                self._active_callback = None

    def start(self) -> None:
        """Keep scanner activation unavailable until a later integration gate."""
        def dormant() -> object:
            self._ensure_open()
            raise IdleRuntimeDormantError("idle runtime activation is dormant in I3b")
        self._gate.sync(dormant)

    @asynccontextmanager
    async def publication_gate(self) -> AsyncGenerator[IdleRuntimeBindingRegistry, None]:
        """Serialize short current-resolution/publication sections."""
        self._ensure_open()
        async with self._gate.acquire():
            self._ensure_open()
            yield self

    async def _shutdown(self) -> None:
        coordinator = self._coordinator
        if coordinator is None or not self._owns_coordinator:
            return
        callback = getattr(coordinator, "aclose", None)
        if not callable(callback):
            raise IdleRuntimeBindingError("injected coordinator lacks aclose")
        try:
            result = callback()
            if inspect.isawaitable(result):
                await cast(Awaitable[object], result)
        except IdleRuntimeError:
            raise
        except Exception:
            raise IdleRuntimeBindingError("coordinator close failed") from None

    async def _close_operation(self) -> None:
        async with self._gate.acquire():
            self._closed = True
            self._closing = True
            self._retire_current_locked()
        await self._shutdown()

    async def aclose(self) -> None:
        """Close once, retire authority, and join shared cleanup despite cancellation."""
        self._gate.check_context()
        self._ensure_process()
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close_operation())
        task = self._close_task
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
                continue
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        if task.cancelled():
            self._close_error = asyncio.CancelledError()
        else:
            self._close_error = task.exception()
        if cancelled:
            raise asyncio.CancelledError
        if self._close_error is not None:
            raise self._close_error

    async def close(self) -> None:
        """Compatibility alias for asynchronous lifecycle owners."""
        await self.aclose()


IdleRuntimeService = IdleRuntimeBindingRegistry


__all__ = [
    "IdlePresetResolver",
    "IdlePresetSnapshotResolver",
    "IdleRuntimeBinding",
    "IdleRuntimeBindingError",
    "IdleRuntimeBindingRegistry",
    "IdleRuntimeClosedError",
    "IdleRuntimeCrossProcessError",
    "IdleRuntimeCrossContextError",
    "IdleRuntimeDormantError",
    "IdleRuntimeError",
    "IdleRuntimeIneligibleError",
    "IdleRuntimeLimits",
    "IdleRuntimeResolutionError",
    "IdleRuntimeService",
    "IdleRuntimeStaleError",
    "IdleSummaryProjectionConfig",
    "IdleWorkspaceBinding",
    "OpaqueFingerprint",
]
