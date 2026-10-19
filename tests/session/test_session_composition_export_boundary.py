"""Real session composition/export keeps private storage and exact lifecycle order."""
from contextlib import contextmanager
from pathlib import Path

import pytest

from nanobot.session.manager import SessionManager
from nanobot.session.mode_storage import LifetimeWriterLease
from nanobot.session.runtime_authority import DurableSessionAuthority, OfflineSessionOperator
from nanobot.session.session_export import export_canonical_session
from nanobot.session.session_mode import SessionModeError, SessionModeResolver
from tests.session.test_session_mode_export import _rich_manager


@contextmanager
def manager_writer(tmp_path: Path, *, operator: bool):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    writer = LifetimeWriterLease(workspace, root)
    writer.acquire()
    try:
        manager = SessionManager(
            workspace, sessions_root=root,
            _managed_access=writer.issue_access(operator=operator),
        )
        yield manager
    finally:
        writer.close()


@pytest.mark.parametrize("operator", [False, True])
@pytest.mark.parametrize("verify", [False, True])
def test_composition_retains_only_close_before_install_and_optional_verify(
    tmp_path, monkeypatch, operator, verify,
):
    compose = SessionManager.compose_session_modes
    trace = []
    closers = []
    with manager_writer(tmp_path, operator=operator) as manager:
        backend = manager._jsonl_store
        original_install = manager._install_mode_resolver

        class TracedResolver(SessionModeResolver):
            def __init__(self, selected):
                assert selected is backend
                trace.append("construct")
                super().__init__(selected)

            def list_modes(self):
                trace.append("verify")
                return super().list_modes()

        def select():
            trace.append("select")
            return TracedResolver

        def observe_backend(value):
            assert value is manager
            trace.append("backend")
            return backend

        def retain(close):
            trace.append("retain")
            closers.append(close)

        def install(resolver, *, operator):
            trace.append("install")
            original_install(resolver, operator=operator)

        try:
            with monkeypatch.context() as patch:
                patch.setattr(SessionManager, "_jsonl_store", property(observe_backend), raising=False)
                patch.setattr(manager, "_install_mode_resolver", install)
                assert compose(
                    manager, select, retain, operator=operator, verify_registered_modes=verify,
                ) is None
            assert trace == ["select", "backend", "construct", "retain", "install"] + (
                ["verify"] if verify else []
            )
            assert manager._mode_operator is operator
            assert len(closers) == 1
            resolver = manager._mode_resolver
            original_close = resolver.close

            def late_close():
                trace.append("late-close")
                original_close()

            with monkeypatch.context() as patch:
                patch.setattr(resolver, "close", late_close)
                closers[0]()
            assert trace[-1] == "late-close"
        finally:
            for close in closers:
                close()


@pytest.mark.parametrize("stage", ["construct", "install", "verify"])
def test_composition_failure_keeps_original_exception_and_cleanup_available(
    tmp_path, monkeypatch, stage,
):
    compose = SessionManager.compose_session_modes
    failure = RuntimeError("injected composition failure")
    trace = []
    closers = []
    with manager_writer(tmp_path, operator=True) as manager:
        original_init = SessionModeResolver.__init__

        def construct(value, backend):
            trace.append("construct")
            if stage == "construct":
                raise failure
            original_init(value, backend)

        def retain(close):
            trace.append("retain")
            closers.append(close)

        def install(*args, **kwargs):
            trace.append("install")
            raise failure

        def verify(value):
            trace.append("verify")
            raise failure

        try:
            with monkeypatch.context() as patch:
                patch.setattr(SessionModeResolver, "__init__", construct)
                if stage == "install":
                    patch.setattr(manager, "_install_mode_resolver", install)
                if stage == "verify":
                    patch.setattr(SessionModeResolver, "list_modes", verify)
                with pytest.raises(RuntimeError) as caught:
                    compose(
                        manager, lambda: SessionModeResolver, retain,
                        operator=True, verify_registered_modes=True,
                    )
            assert caught.value is failure
            assert trace == {
                "construct": ["construct"],
                "install": ["construct", "retain", "install"],
                "verify": ["construct", "retain", "verify"],
            }[stage]
            assert len(closers) == (0 if stage == "construct" else 1)
        finally:
            for close in closers:
                close()


@pytest.mark.parametrize("offline", [False, True])
async def test_runtime_uses_owner_local_composition_nonvirtually(tmp_path, monkeypatch, offline):
    original_compose = SessionManager.compose_session_modes
    original_init = SessionManager.__init__
    trace = []

    def forbidden(*args, **kwargs):
        raise AssertionError("instance composition override must not receive the backend")

    def initialize(value, *args, **kwargs):
        original_init(value, *args, **kwargs)
        value.compose_session_modes = forbidden

    def compose(value, *args, **kwargs):
        trace.append((kwargs["operator"], kwargs["verify_registered_modes"]))
        return original_compose(value, *args, **kwargs)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "sessions"
    authority = (
        OfflineSessionOperator(workspace, copied_sessions_root=root)
        if offline else DurableSessionAuthority(workspace, sessions_root=root)
    )
    with monkeypatch.context() as patch:
        patch.setattr(SessionManager, "__init__", initialize)
        patch.setattr(SessionManager, "compose_session_modes", compose)
        lease = await authority.acquire()
    try:
        assert trace == [(offline, not offline)]
        assert lease.manager.has_managed_authority
        assert lease.close is not None
        await lease.close()
        await lease.close()
    finally:
        if lease.close is not None:
            await lease.close()


async def test_cleanup_exception_still_precedes_writer_close(tmp_path, monkeypatch):
    verification_failure = RuntimeError("injected verification failure")
    close_failure = RuntimeError("injected resolver close failure")
    original_acquire = LifetimeWriterLease.acquire
    original_writer_close = LifetimeWriterLease.close
    original_resolver_init = SessionModeResolver.__init__
    original_resolver_close = SessionModeResolver.close
    writers = []
    resolvers = []
    trace = []

    def acquire(value):
        original_acquire(value)
        writers.append(value)

    def initialize(value, *args, **kwargs):
        original_resolver_init(value, *args, **kwargs)
        resolvers.append(value)

    def verify(value):
        trace.append("verify")
        raise verification_failure

    def close_resolver(value):
        trace.append("resolver-close")
        raise close_failure

    def close_writer(value):
        trace.append("writer-close")
        original_writer_close(value)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(LifetimeWriterLease, "acquire", acquire)
            patch.setattr(LifetimeWriterLease, "close", close_writer)
            patch.setattr(SessionModeResolver, "__init__", initialize)
            patch.setattr(SessionModeResolver, "list_modes", verify)
            patch.setattr(SessionModeResolver, "close", close_resolver)
            with pytest.raises(RuntimeError) as caught:
                await DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions").acquire()
        assert caught.value is close_failure
        assert trace == ["verify", "resolver-close"]
    finally:
        for resolver in resolvers:
            original_resolver_close(resolver)
        for writer in writers:
            original_writer_close(writer)


def test_public_export_dispatches_nonvirtually_and_refuses_before_target(tmp_path, monkeypatch):
    original = SessionManager.canonical_export_operations
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = SessionManager(workspace, sessions_root=tmp_path / "sessions")
    trace = []

    def command(value, *args, **kwargs):
        trace.append("command")
        return original(value, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("unmanaged refusal must precede target handling")

    from nanobot.session import session_export

    with monkeypatch.context() as patch:
        patch.setattr(SessionManager, "canonical_export_operations", command)
        patch.setattr(manager, "canonical_export_operations", forbidden)
        patch.setattr(session_export, "Path", forbidden)
        with pytest.raises(SessionModeError, match="offline session operator authority is required"):
            export_canonical_session(manager, "cli:private", tmp_path / "bundle")
    assert trace == ["command"]
    assert not (tmp_path / "bundle").exists()


async def test_live_manager_export_refuses_before_target(tmp_path, monkeypatch):
    command = SessionManager.canonical_export_operations
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions").acquire()

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime operator refusal must precede target handling")

    from nanobot.session import session_export

    try:
        with monkeypatch.context() as patch:
            patch.setattr(session_export, "Path", forbidden)
            with pytest.raises(SessionModeError, match="offline session operator authority is required"):
                command(lease.manager)
        assert not (tmp_path / "bundle").exists()
    finally:
        if lease.close is not None:
            await lease.close()


async def test_compatibility_parser_selection_precedes_staged_argument_observation(
    tmp_path, monkeypatch,
):
    lease = await _rich_manager(tmp_path)
    manager = lease.manager
    backend = manager._jsonl_store
    original_parse = backend.parse_legacy_bytes
    from nanobot.session.session_mode import SessionExportTransaction

    original_read = SessionExportTransaction.read_staged
    trace = []
    compatibility_reads = []

    def observe_backend(value):
        assert value is manager
        trace.append("backend")
        return backend

    def parse(*args, **kwargs):
        trace.append("parse")
        return original_parse(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("staged read cannot replace the already selected parser")

    def read(value, name):
        if name == "compatibility-history.jsonl":
            compatibility_reads.append(name)
            trace.append("read")
            if len(compatibility_reads) == 2:
                backend.parse_legacy_bytes = forbidden
        return original_read(value, name)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(backend, "parse_legacy_bytes", parse)
            patch.setattr(SessionManager, "_jsonl_store", property(observe_backend), raising=False)
            patch.setattr(SessionExportTransaction, "read_staged", read)
            result = export_canonical_session(
                manager, "cli:export", tmp_path / "bundle", include_compatibility=True,
            )
        assert trace == ["read", "backend", "read", "parse", "read"]
        assert len(compatibility_reads) == 3
        assert result.compatibility_included
        assert (result.destination / "canonical-source.jsonl").is_file()
    finally:
        if lease.close is not None:
            await lease.close()
