"""Requirements test: native process counting across shared Full-access references.

Related references must count their shared native-process identity once; distinct
identities must add. Counts, conflicts and handle cleanup are observed only
through ``disable_full``.
"""

from __future__ import annotations

from collections import Counter

import pytest

from kohaku.security.desktop_permissions import (
    DesktopPermissionOwner,
    WorkspacePermissionConflictError,
)

CHAT = "chat-1"
BINDING = {"store_identity": "store-req", "incarnation": "inc-b", "generation": 1}


def _enable(owner, binding):
    return owner.enable_full(
        session_key=CHAT, source_binding=dict(binding), persistent=True, existing=True,
        local=True, idle=True, configured_sandbox=False, supervisor_available=True,
        stores_healthy=True,
    )


def _request(owner):
    reference = owner.full_reference(
        session_key=CHAT, source_binding=dict(BINDING), request_turn_id="turn-2"
    )
    assert reference is not None
    return reference


def _register(owner, reference, name, value, closed=None, fail=False):
    def close():
        if closed is not None:
            closed.append(name)
        return True

    def provider():
        if fail:
            raise RuntimeError("count provider unavailable")
        return (value, 0)

    assert owner.register_native_stdin(
        reference=reference, process_handle=object(), close_stdin=close,
        process_counts=provider,
    ) is True


def _retire(owner, binding=BINDING):
    return owner.disable_full(
        session_key=CHAT, source_binding=dict(binding), active_turn=False,
        active_processes=0, unknown_processes=0,
    )


def _assert_conflict(exc, token):
    code = str(getattr(exc.value, "code", "") or "")
    assert token in code or token in str(exc.value)


def _register_with(owner, reference, provider):
    assert owner.register_native_stdin(
        reference=reference, process_handle=object(),
        close_stdin=lambda: True, process_counts=provider,
    ) is True


def _retirement(owner, binding=BINDING):
    view = owner.projection(session_key=CHAT, source_binding=dict(binding))
    retirement = view.get("retirement")
    assert isinstance(retirement, dict)
    return retirement


def test_shared_identity_reports_one_active_process() -> None:
    owner = DesktopPermissionOwner()
    first, second = _enable(owner, BINDING), _request(owner)
    _register(owner, first, "first", 1)
    _register(owner, second, "second", 1)

    with pytest.raises(WorkspacePermissionConflictError) as exc:
        _retire(owner)
    _assert_conflict(exc, "native_processes_active")


def test_distinct_identities_add_to_two_active_processes() -> None:
    owner = DesktopPermissionOwner()
    other = {**BINDING, "incarnation": "inc-other", "generation": 2}
    _register(owner, _enable(owner, BINDING), "first", 1)
    _register(owner, _enable(owner, other), "second", 1)

    with pytest.raises(WorkspacePermissionConflictError) as exc:
        _retire(owner)
    _assert_conflict(exc, "native_processes_active")


def test_retirement_closes_each_shared_handle_once() -> None:
    owner = DesktopPermissionOwner()
    first, second = _enable(owner, BINDING), _request(owner)
    closed: list[str] = []
    _register(owner, first, "first", 0, closed)
    _register(owner, second, "second", 0, closed)

    assert _retire(owner) is True
    assert Counter(closed) == Counter({"first": 1, "second": 1})


def test_provider_error_is_unknown_not_clean_zero() -> None:
    owner = DesktopPermissionOwner()
    _register(owner, _enable(owner, BINDING), "first", 0, fail=True)

    with pytest.raises(WorkspacePermissionConflictError) as exc:
        _retire(owner)
    _assert_conflict(exc, "native_processes_unknown")


def test_success_path_closes_single_handle_once() -> None:
    owner = DesktopPermissionOwner()
    closed: list[str] = []
    _register(owner, _enable(owner, BINDING), "only", 0, closed)

    assert _retire(owner) is True
    assert Counter(closed) == Counter({"only": 1})


def test_one_shared_launch_reports_exactly_one_active_process() -> None:
    owner = DesktopPermissionOwner()
    first, second = _enable(owner, BINDING), _request(owner)

    def shared_provider():
        return (1, 0)

    _register_with(owner, first, shared_provider)
    _register_with(owner, second, shared_provider)

    with pytest.raises(WorkspacePermissionConflictError) as exc:
        _retire(owner)
    _assert_conflict(exc, "native_processes_active")

    retirement = _retirement(owner)
    assert retirement["active_processes"] == 1
    assert retirement["active_processes"] != 2
    assert retirement["unknown_processes"] == 0


def test_two_shared_launches_report_exactly_two_active_processes() -> None:
    owner = DesktopPermissionOwner()
    first, second = _enable(owner, BINDING), _request(owner)

    def shared_provider():
        return (2, 0)

    _register_with(owner, first, shared_provider)
    _register_with(owner, second, shared_provider)

    with pytest.raises(WorkspacePermissionConflictError) as exc:
        _retire(owner)
    _assert_conflict(exc, "native_processes_active")

    retirement = _retirement(owner)
    assert retirement["active_processes"] == 2
    assert retirement["active_processes"] != 4
    assert retirement["unknown_processes"] == 0
