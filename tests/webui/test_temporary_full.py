from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.security.folder_grants import FolderGrantStore
from nanobot.session.manager import SessionManager
from nanobot.webui.temporary_chats import WebUITemporaryChats
from nanobot.webui.workspaces import WebUIWorkspaceController


def _fixture(tmp_path: Path):
    home = tmp_path / "home"
    instance = tmp_path / "instance"
    home.mkdir()
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    controller = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=home,
        default_restrict_to_workspace=True,
        grant_store=FolderGrantStore(
            agent_home=home,
            instance_root=instance,
            path=instance / "webui" / "folder-grants.json",
        ),
        native_supervisor_available=True,
    )
    chats = WebUITemporaryChats(
        bus=MessageBus(),
        session_manager=sessions,
        workspaces=controller,
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )
    controller.set_temporary_owners(
        binding=lambda chat_id: chats.binding(None, chat_id),
        set_mode=lambda chat_id, mode: chats.set_access_mode(chat_id, mode),
    )
    return sessions, controller, chats


def test_temporary_full_is_ephemeral_and_request_bound(tmp_path: Path) -> None:
    sessions, controller, chats = _fixture(tmp_path)
    owner = object()
    chat_id = chats.create(owner, trusted_webui=True)
    binding = chats.binding(owner, chat_id)
    assert binding is not None
    assert binding["store_identity"].startswith("temporary:")
    assert sessions.read_session_metadata(f"websocket:{chat_id}") is None

    prepared = controller.prepare_full({"chat_id": chat_id}, local_ui=True)
    permissions = prepared["workspace_permissions"]
    assert permissions["can_enable_full"] is True
    assert permissions["mode"] == "restricted"

    enabled = controller.set_permission_mode(
        {
            "chat_id": chat_id,
            "mode": "full",
            "expected_runtime_id": permissions["runtime_id"],
            "expected_permission_revision": permissions["revision"],
        },
        local_ui=True,
    )
    assert "error" not in enabled
    assert enabled["workspace_scope"]["access_mode"] == "full"

    policy = chats.message_policy(owner, chat_id, "use full access")
    assert policy is not None and policy.workspace_scope.access_mode == "full"
    reference = controller.temporary_full_reference(
        chat_id,
        policy.source_binding,
        request_turn_id="turn-1",
    )
    assert reference is not None
    assert controller.permission_owner.authorizes_full(
        reference,
        session_key=f"websocket:{chat_id}",
        source_binding=policy.source_binding,
        request_turn_id="turn-1",
    )
    assert not controller.permission_owner.authorizes_full(
        reference,
        session_key=f"websocket:{chat_id}",
        source_binding=policy.source_binding,
        request_turn_id="turn-2",
    )
    assert sessions.read_session_metadata(f"websocket:{chat_id}") is None


def test_temporary_full_does_not_authorize_another_chat_or_source(tmp_path: Path) -> None:
    _sessions, controller, chats = _fixture(tmp_path)
    owner = object()
    first = chats.create(owner, trusted_webui=True)
    second = chats.create(owner, trusted_webui=True)
    first_binding = chats.binding(owner, first)
    second_binding = chats.binding(owner, second)
    assert first_binding is not None and second_binding is not None

    prepared = controller.prepare_full({"chat_id": first}, local_ui=True)
    controller.set_permission_mode(
        {
            "chat_id": first,
            "mode": "full",
            "expected_runtime_id": prepared["workspace_permissions"]["runtime_id"],
            "expected_permission_revision": prepared["workspace_permissions"]["revision"],
        },
        local_ui=True,
    )

    assert controller.temporary_full_reference(
        first, second_binding, request_turn_id="turn-1",
    ) is None
    assert controller.temporary_full_reference(
        second, first_binding, request_turn_id="turn-1",
    ) is None
    reference = controller.temporary_full_reference(
        first, first_binding, request_turn_id="turn-1",
    )
    assert reference is not None

    projection = controller.permission_owner.projection(
        session_key=f"websocket:{first}",
        source_binding=first_binding,
    )
    restricted = controller.set_permission_mode(
        {
            "chat_id": first,
            "mode": "restricted",
            "expected_runtime_id": projection["runtime_id"],
            "expected_permission_revision": projection["revision"],
        },
        local_ui=True,
    )
    assert "error" not in restricted
    assert restricted["workspace_scope"]["access_mode"] == "restricted"
    assert controller.temporary_full_reference(
        first, first_binding, request_turn_id="turn-1",
    ) is None


@pytest.mark.asyncio
async def test_ephemeral_full_execution_admits_without_a_canonical_source(
    tmp_path: Path,
) -> None:
    from nanobot.agent.loop import _TurnExecutionOptions
    from nanobot.agent.turn import TurnRequest, TurnSource
    from nanobot.providers.base import LLMResponse
    from tests.agent.conftest import make_loop
    from tests.agent.test_migration_engine import ScriptedProvider

    provider = ScriptedProvider([LLMResponse(content="temporary full ok")])
    loop = make_loop(tmp_path, provider=provider)
    key = "websocket:temporary-full"
    loop.sessions.get_or_create_transient(key, disabled_tools=frozenset(), retain=True)
    binding = {
        "store_identity": "temporary:temporary-full",
        "incarnation": "incarnation-temporary-full",
        "generation": 1,
    }
    owner = loop.sessions.desktop_permissions
    owner.enable_full(
        session_key=key,
        source_binding=binding,
        persistent=False,
        existing=True,
        local=True,
        idle=True,
        configured_sandbox=False,
        supervisor_available=True,
        stores_healthy=True,
    )
    reference = owner.full_reference(
        session_key=key,
        source_binding=binding,
        request_turn_id="turn-1",
    )
    assert reference is not None
    try:
        result = await loop.execute_turn(
            TurnRequest(
                content="answer with full access",
                session_key=key,
                source=TurnSource.GATEWAY,
                metadata={"webui": True, "webui_turn_id": "turn-1"},
            ),
            _options=_TurnExecutionOptions(
                ephemeral=True,
                workspace_full_reference=reference,
                raise_errors=True,
            ),
        )
    finally:
        await loop.aclose()

    assert result.error is None
    assert result.content == "temporary full ok"


@pytest.mark.asyncio
async def test_discarding_a_temporary_chat_retires_full_and_consent(
    tmp_path: Path,
) -> None:
    _sessions, controller, chats = _fixture(tmp_path)
    owner = object()
    chat_id = chats.create(owner, trusted_webui=True)
    prepared = controller.prepare_full({"chat_id": chat_id}, local_ui=True)
    controller.set_permission_mode(
        {
            "chat_id": chat_id,
            "mode": "full",
            "expected_runtime_id": prepared["workspace_permissions"]["runtime_id"],
            "expected_permission_revision": prepared["workspace_permissions"]["revision"],
        },
        local_ui=True,
        owner_authorized=True,
        owner_installation_id="install-phone",
    )
    binding = chats.binding(owner, chat_id)
    assert binding is not None
    assert controller.permission_owner.full_reference(
        session_key=f"websocket:{chat_id}",
        source_binding=binding,
    ) is not None

    await chats.discard(owner, chat_id)

    assert controller.permission_owner.full_reference(
        session_key=f"websocket:{chat_id}",
        source_binding=binding,
    ) is None
    assert chats.binding(owner, chat_id) is None
