"""A01-F06: cron list/remove are bound to the requesting chat creator or admin.

Requirements case U2 (req-a38-cron/CASES.md).  The review finding named the
model-facing tool: ``list`` returned every enabled job and ``remove`` accepted
any non-system job from any chat when the instance and revision tokens matched.
"""

from __future__ import annotations

import pytest

from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.turn import TurnContext
from nanobot.cron.service import CronService
from tests.cron.stage3_fixtures import offline_denial  # noqa: F401

ADMIN_SESSION = "websocket:admin"


def _service(tmp_path) -> CronService:
    return CronService(
        tmp_path / "cron" / "jobs.json",
        admin_session_keys=(ADMIN_SESSION,),
    )


def _context(chat: str) -> RequestContext:
    return RequestContext(
        channel="websocket", chat_id=chat, session_key=f"websocket:{chat}"
    )


async def _add(tool: CronTool, chat: str) -> str:
    with request_context(_context(chat)):
        result = await tool.execute(
            action="add",
            name=f"job-{chat}",
            message=f"private message for {chat}",
            every_seconds=60,
        )
    assert result.startswith("Created job"), result
    return result.rsplit("id: ", 1)[1].rstrip(")")


async def _list(tool: CronTool, chat: str) -> str:
    with request_context(_context(chat)):
        return await tool.execute(action="list")


async def _remove(tool: CronTool, chat: str, job_id: str) -> str:
    with request_context(_context(chat)):
        return await tool.execute(action="remove", job_id=job_id)


@pytest.mark.asyncio
async def test_chat_cannot_see_or_remove_another_chats_job(tmp_path) -> None:
    service = _service(tmp_path)
    tool = CronTool(service)

    # 1. Three jobs created through the normal creation path.
    job_a = await _add(tool, "A")
    job_b = await _add(tool, "B")
    job_admin = await _add(tool, "admin")
    assert len({job_a, job_b, job_admin}) == 3

    # 1.2 Snapshot of B before any foreign request.
    snapshot_b = service.get_job(job_b)
    assert snapshot_b is not None

    # 2.1 Chat A lists only its own job; B and admin values never appear.
    listing_a = await _list(tool, "A")
    assert job_a in listing_a
    assert job_b not in listing_a
    assert job_admin not in listing_a
    assert "job-B" not in listing_a
    assert "private message for B" not in listing_a
    assert "Scheduled jobs:" in listing_a

    # 2.2 A known foreign id plus a wrong chat is not authority.
    refusal = await _remove(tool, "A", job_b)
    assert refusal == f"Job {job_b} not found"
    assert "job-B" not in refusal
    assert service.get_job(job_b) is not None

    # 2.3 Nothing on B was mutated, rescheduled, disabled or cancelled.
    assert service.get_job(job_b) == snapshot_b

    # 2.4 Chat B lists its own job and not A's.
    listing_b = await _list(tool, "B")
    assert job_b in listing_b
    assert job_a not in listing_b
    assert job_admin not in listing_b

    # 2.5 An anonymous request is refused and mutates nothing.
    anonymous = RequestContext(channel="", chat_id="", session_key=None)
    with request_context(anonymous):
        assert await tool.execute(action="list") == "No scheduled jobs."
        anonymous_refusal = await tool.execute(action="remove", job_id=job_b)
    assert anonymous_refusal == f"Job {job_b} not found"
    assert service.get_job(job_b) == snapshot_b

    # 2.6 Chat B cannot remove chat A's job.
    assert await _remove(tool, "B", job_a) == f"Job {job_a} not found"
    assert service.get_job(job_a) is not None

    # 2.7 The explicit admin session keeps global management.
    listing_admin = await _list(tool, "admin")
    assert {job_a, job_b, job_admin} <= set(_ids(listing_admin))

    # 2.8 Admin removal works once; a repeat has no second side effect.
    assert await _remove(tool, "admin", job_b) == f"Removed job {job_b}"
    assert job_b not in await _list(tool, "B")
    assert await _remove(tool, "admin", job_b) == f"Job {job_b} not found"

    # 2.9 No other model-facing mutation exists on this tool surface.
    assert tool.parameters["properties"]["action"]["enum"] == ["add", "list", "remove"]


def _ids(listing: str) -> list[str]:
    return [
        line.rsplit("id: ", 1)[1].split(",", 1)[0].rstrip(")")
        for line in listing.splitlines()
        if "(id: " in line
    ]


@pytest.mark.asyncio
async def test_turn_context_route_is_the_management_principal(tmp_path) -> None:
    service = _service(tmp_path)
    tool = CronTool(service)
    job_a = await _add(tool, "A")
    job_b = await _add(tool, "B")

    turn = TurnContext(run_id="scope-turn", session_key="websocket:A")
    listed = await tool.execute_with_context(turn, action="list")
    assert job_a in listed.content
    assert job_b not in listed.content

    removed = await tool.execute_with_context(turn, action="remove", job_id=job_b)
    assert removed.content == f"Job {job_b} not found"
    assert service.get_job(job_b) is not None


@pytest.mark.asyncio
async def test_single_owner_local_path_keeps_explicit_global_management(tmp_path) -> None:
    service = _service(tmp_path)
    tool = CronTool(service)
    job_a = await _add(tool, "A")
    job_b = await _add(tool, "B")

    # No admitted request means a local single-owner process, not a chat.
    listing = tool._list_jobs()
    assert job_a in listing and job_b in listing
