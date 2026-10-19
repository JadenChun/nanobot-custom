"""Copied-artifact reads retain real locks, late observations and lease cleanup."""

from contextlib import asynccontextmanager, contextmanager
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import pytest

from nanobot.session.artifact_operator import ArtifactOperatorError
from nanobot.session.manager import SessionManager
from nanobot.session.runtime_authority import OfflineSessionOperator
from nanobot.session.session_mode import SessionModeError, SessionModeResolver
from tests.session.test_artifact_operator import copied_artifact_fixture
from tests.session.test_session_composition_export_boundary import manager_writer


def read_operation():
    operation = getattr(SessionManager, "copied_artifact_read_operations", None)
    assert callable(operation), "manager-owned copied artifact read operation is missing"
    return operation


@asynccontextmanager
async def real_lease(fixture):
    lease = await fixture["operator"]._sessions.acquire()
    try:
        yield lease
    finally:
        assert lease.close is not None
        await lease.close()


async def run_operation(fixture, operation):
    if operation == "inspect":
        return await fixture["operator"].inspect(fixture["key"])
    return await fixture["operator"].rematerialize(
        fixture["key"],
        incarnation=fixture["incarnation"],
        generation=fixture["generation"],
        reference=fixture["descriptor"].reference,
    )


def test_absent_resolver_returns_none_after_one_selection(tmp_path, monkeypatch):
    operation = read_operation()
    observed = []

    def selecting(manager):
        observed.append(manager)
        return None

    with manager_writer(tmp_path, operator=False) as manager:
        with monkeypatch.context() as patch:
            patch.setattr(SessionManager, "_mode_resolver", property(selecting), raising=False)
            assert operation(manager) is None
        assert observed == [manager]


async def test_handoff_is_immutable_and_retains_only_selected_read_operations(
    tmp_path, monkeypatch,
):
    operation = read_operation()

    def forbidden_guard(_manager):
        pytest.fail("read handoff added an unrelated offline export guard")

    async with copied_artifact_fixture(tmp_path) as fixture:
        async with real_lease(fixture) as lease:
            resolver = lease.manager._mode_resolver
            assert resolver is not None
            with monkeypatch.context() as patch:
                patch.setattr(
                    SessionManager, "_mode_operator", property(forbidden_guard), raising=False,
                )
                operations = operation(lease.manager)
            assert operations is not None
            assert [field.name for field in fields(operations)] == [
                "locked_snapshot", "read_workspace_id", "read_workspace", "read_namespace",
            ]
            assert not any(hasattr(operations, name) for name in (
                "__dict__", "manager", "resolver", "backend", "owner", "fd", "write",
            ))
            for field in fields(operations):
                callback = getattr(operations, field.name)
                assert callback.__code__.co_freevars == ("resolver",)
                assert callback.__closure__[0].cell_contents is resolver
            with pytest.raises(FrozenInstanceError):
                operations.read_workspace_id = lambda: "not a workspace identity"
            assert operations.read_workspace_id() == resolver.workspace_id
            assert operations.read_workspace() is resolver.workspace
            assert operations.read_namespace() is resolver.namespace


@pytest.mark.parametrize("name,callback", [
    ("workspace_id", "read_workspace_id"),
    ("workspace", "read_workspace"),
    ("namespace", "read_namespace"),
])
async def test_metadata_is_late_repeated_and_uses_single_selected_resolver(
    tmp_path, monkeypatch, name, callback,
):
    operation = read_operation()
    trace = []
    async with copied_artifact_fixture(tmp_path) as fixture:
        async with real_lease(fixture) as lease:
            resolver = lease.manager._mode_resolver
            assert resolver is not None
            selected = resolver
            current = getattr(resolver, name)

            def selecting(manager):
                assert manager is lease.manager
                trace.append("select")
                return selected

            def observing(value):
                assert value is resolver
                trace.append(name)
                return current

            with monkeypatch.context() as patch:
                patch.setattr(SessionManager, "_mode_resolver", property(selecting), raising=False)
                patch.setattr(SessionModeResolver, name, property(observing), raising=False)
                operations = operation(lease.manager)
                assert operations is not None and trace == ["select"]
                selected = None
                first = "first-identity" if name == "workspace_id" else Path("/first-read-only")
                second = "second-identity" if name == "workspace_id" else Path("/second-read-only")
                current = first
                assert getattr(operations, callback)() is first
                current = second
                assert getattr(operations, callback)() is second
                assert trace == ["select", name, name]


@pytest.mark.parametrize("name,callback", [
    ("workspace_id", "read_workspace_id"),
    ("workspace", "read_workspace"),
    ("namespace", "read_namespace"),
])
async def test_metadata_failure_is_lazy_and_keeps_exact_first_error(
    tmp_path, monkeypatch, name, callback,
):
    operation = read_operation()
    failure = OSError("injected metadata read fault")
    trace = []
    async with copied_artifact_fixture(tmp_path) as fixture:
        async with real_lease(fixture) as lease:
            resolver = lease.manager._mode_resolver

            def refusing(value):
                assert value is resolver
                trace.append(name)
                raise failure

            with monkeypatch.context() as patch:
                patch.setattr(SessionModeResolver, name, property(refusing), raising=False)
                operations = operation(lease.manager)
                assert operations is not None and trace == []
                with pytest.raises(OSError) as caught:
                    getattr(operations, callback)()
                assert caught.value is failure and trace == [name]


async def test_snapshot_hook_is_late_and_invokes_original_real_locked_reader(
    tmp_path, monkeypatch,
):
    operation = read_operation()
    trace = []
    async with copied_artifact_fixture(tmp_path) as fixture:
        async with real_lease(fixture) as lease:
            resolver = lease.manager._mode_resolver
            assert resolver is not None
            original = resolver.locked_snapshot
            operations = operation(lease.manager)
            assert operations is not None
            source_before = fixture["source_path"].read_bytes()
            record_before = fixture["record_path"].read_bytes()

            @contextmanager
            def late_snapshot(key):
                trace.append("enter")
                with original(key) as locked:
                    trace.append("yield")
                    yield locked
                    trace.append("returned")
                trace.append("exit")

            with monkeypatch.context() as patch:
                patch.setattr(lease.manager, "_mode_resolver", None)
                patch.setattr(resolver, "locked_snapshot", late_snapshot)
                with operations.locked_snapshot(fixture["key"]) as locked:
                    assert locked.source_bytes == source_before
                    assert locked.mode_record_bytes == record_before
                    assert locked.binding[2:] == (
                        "canonical", fixture["incarnation"], fixture["generation"],
                    )
                    assert trace == ["enter", "yield"]
            assert trace == ["enter", "yield", "returned", "exit"]
            assert fixture["source_path"].read_bytes() == source_before
            assert fixture["record_path"].read_bytes() == record_before


async def test_retained_handoff_does_not_admit_reads_after_real_lease_closes(tmp_path):
    operation = read_operation()
    async with copied_artifact_fixture(tmp_path) as fixture:
        async with real_lease(fixture) as lease:
            operations = operation(lease.manager)
            assert operations is not None
        with pytest.raises(SessionModeError):
            with operations.locked_snapshot(fixture["key"]):
                pytest.fail("closed real resolver admitted a snapshot")


@pytest.mark.parametrize("kind", ["inspect", "rematerialize"])
async def test_consumers_use_nonvirtual_handoff_with_original_observation_order(
    tmp_path, monkeypatch, kind,
):
    operation = read_operation()
    original_acquire = OfflineSessionOperator.acquire
    trace = []

    def forbidden_override():
        pytest.fail("consumer dispatched a manager instance override")

    def observing(manager):
        trace.append("command")
        operations = operation(manager)
        assert operations is not None

        @contextmanager
        def snapshot(key):
            trace.append("snapshot-enter")
            with operations.locked_snapshot(key) as locked:
                yield locked
            trace.append("snapshot-exit")

        def metadata(label, read):
            trace.append(label)
            return read()

        return replace(
            operations,
            locked_snapshot=snapshot,
            read_workspace_id=lambda: metadata("workspace-id", operations.read_workspace_id),
            read_workspace=lambda: metadata("workspace", operations.read_workspace),
            read_namespace=lambda: metadata("namespace", operations.read_namespace),
        )

    async with copied_artifact_fixture(tmp_path) as fixture:
        async def acquire(sessions):
            lease = await original_acquire(sessions)
            assert lease.close is not None
            close = lease.close
            monkeypatch.setattr(lease.manager, "copied_artifact_read_operations", forbidden_override)

            async def closing():
                trace.append("close")
                await close()

            return replace(lease, close=closing)

        with monkeypatch.context() as patch:
            patch.setattr(SessionManager, "copied_artifact_read_operations", observing)
            patch.setattr(OfflineSessionOperator, "acquire", acquire)
            await run_operation(fixture, kind)
        assert trace == {
            "inspect": [
                "command", "snapshot-enter", "snapshot-exit", "workspace-id", "workspace-id",
                "workspace", "namespace", "close",
            ],
            "rematerialize": [
                "command", "snapshot-enter", "workspace-id", "snapshot-exit", "workspace-id",
                "workspace", "namespace", "workspace-id", "snapshot-enter", "snapshot-exit",
                "close",
            ],
        }[kind]


@pytest.mark.parametrize("kind", ["inspect", "rematerialize"])
async def test_missing_resolver_is_content_free_and_still_closes_real_lease(
    tmp_path, monkeypatch, kind,
):
    operation = read_operation()
    original_acquire = OfflineSessionOperator.acquire
    trace = []

    def absent(manager):
        trace.append("select")
        with monkeypatch.context() as patch:
            patch.setattr(manager, "_mode_resolver", None)
            return operation(manager)

    async with copied_artifact_fixture(tmp_path) as fixture:
        async def acquire(sessions):
            lease = await original_acquire(sessions)
            assert lease.close is not None
            close = lease.close

            async def closing():
                trace.append("close")
                await close()

            return replace(lease, close=closing)

        with monkeypatch.context() as patch:
            patch.setattr(SessionManager, "copied_artifact_read_operations", absent)
            patch.setattr(OfflineSessionOperator, "acquire", acquire)
            with pytest.raises(ArtifactOperatorError) as caught:
                await run_operation(fixture, kind)
        assert str(caught.value) == "Copied artifact operation is unavailable"
        assert fixture["key"] not in str(caught.value)
        assert fixture["descriptor"].reference not in str(caught.value)
        assert trace == ["select", "close"]
        async with real_lease(fixture):
            pass


@pytest.mark.parametrize("kind", ["inspect", "rematerialize"])
async def test_close_fault_is_content_free_after_real_cleanup(tmp_path, monkeypatch, kind):
    read_operation()
    original_acquire = OfflineSessionOperator.acquire
    trace = []
    async with copied_artifact_fixture(tmp_path) as fixture:
        async def acquire(sessions):
            lease = await original_acquire(sessions)
            assert lease.close is not None
            close = lease.close

            async def closing():
                await close()
                trace.append("closed")
                raise OSError("injected private close fault")

            return replace(lease, close=closing)

        with monkeypatch.context() as patch:
            patch.setattr(OfflineSessionOperator, "acquire", acquire)
            with pytest.raises(ArtifactOperatorError) as caught:
                await run_operation(fixture, kind)
        assert str(caught.value) == "Copied artifact operation is unavailable"
        assert "private close fault" not in str(caught.value)
        assert trace == ["closed"]
        async with real_lease(fixture):
            pass
