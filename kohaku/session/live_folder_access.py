"""Keep a folder access decision attached to its active turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio

    from kohaku.security.desktop_permissions import WorkspaceTaskAccessReference
    from kohaku.security.workspace_access import WorkspaceScope
    from kohaku.workspace_access_recovery import WorkspaceAccessRecord


@dataclass(frozen=True, slots=True)
class LiveFolderSelection:
    scope: WorkspaceScope
    task_reference: WorkspaceTaskAccessReference | None


@dataclass(frozen=True, slots=True)
class LiveFolderAccess:
    request_id: str
    run_id: str
    decision: asyncio.Future[LiveFolderSelection | None]

    def matches(self, record: WorkspaceAccessRecord) -> bool:
        return (
            not self.decision.done()
            and record["access_request_id"] == self.request_id
            and record["blocked_run_id"] == self.run_id
        )
