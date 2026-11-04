"""Cross-client invalidation of WebUI state projections.

Every WebUI mutation dispatches through one place
(:meth:`nanobot.webui.inbound_commands.WebUICommandHandler.execute_webui_request`).
Instead of each handler deciding whom to notify, the dispatcher maps the action
namespace onto the client-side caches that have to reload:

* ``sessions`` - topic rows and the session list (``session_updated``).
* ``settings`` - app-level settings, model presets, skills, automations.
* ``sidebar`` - sidebar layout; the handler broadcasts its full projection.

A namespace that is not classified here falls back to notifying every cache, so
a newly added action is correct before it is optimised. Add an entry when a new
namespace appears, not when a new action does.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nanobot.webui.session_identity import webui_chat_id

SESSIONS = "sessions"
SETTINGS = "settings"
SIDEBAR = "sidebar"

_NAMESPACE_DOMAINS: dict[str, frozenset[str]] = {
    # Session-scoped state lives in the topic row and the thread.
    "session": frozenset({SESSIONS}),
    "workspace": frozenset({SESSIONS}),
    "recovery": frozenset({SESSIONS}),
    # The sidebar handler broadcasts its own full projection.
    "sidebar": frozenset({SIDEBAR}),
    # App-level state is shared by every client's settings projection.
    "settings": frozenset({SETTINGS}),
    "skill": frozenset({SETTINGS}),
    "automation": frozenset({SETTINGS}),
    "repository": frozenset({SETTINGS}),
}

DEFAULT_DOMAINS: frozenset[str] = frozenset({SESSIONS, SETTINGS})


def mutation_domains(action: str) -> frozenset[str]:
    """Return the cached projections one mutation action invalidates."""
    return _NAMESPACE_DOMAINS.get(action.partition(".")[0], DEFAULT_DOMAINS)


def mutation_chat_id(payload: Mapping[str, Any]) -> str:
    """Return the session a mutation touched, when the payload names one."""
    chat_id = payload.get("chat_id")
    if isinstance(chat_id, str) and chat_id.strip():
        return chat_id.strip()
    key = payload.get("key")
    if isinstance(key, str):
        return webui_chat_id(key) or ""
    return ""


async def notify_domains(
    host: Any,
    domains: frozenset[str],
    chat_id: str = "",
    *,
    exclude: Any | None = None,
) -> None:
    """Broadcast the invalidation events for *domains* through a WebUI host.

    The host only has to offer ``broadcast_webui_event``; the wire events stay
    defined in one place for mutations, forks and any future state producer.
    ``exclude`` skips the client that already applied the change locally.
    """
    if SESSIONS in domains:
        await host.broadcast_webui_event(
            "session_updated",
            chat_id=chat_id,
            scope="metadata",
            exclude=exclude,
        )
    if SETTINGS in domains:
        await host.broadcast_webui_event("settings_updated", exclude=exclude)
