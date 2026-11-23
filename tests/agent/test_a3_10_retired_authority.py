"""A3-10 retirement regression tests.

These tests verify that the retired subagent authority remains importable
for compatibility, owns no tasks, and refuses to launch new work.
"""

import asyncio
import inspect

import pytest

import nanobot.agent
from nanobot.agent.subagent import RetiredSubagentAuthority, SubagentManager


def test_compatibility_import_remains_available() -> None:
    assert nanobot.agent.SubagentManager is SubagentManager


def test_retired_authority_owns_no_task_and_refuses_launch() -> None:
    authority = RetiredSubagentAuthority(max_iterations=7, disabled_skills={"x"})

    assert authority.max_iterations == 7
    assert authority.disabled_skills == {"x"}
    assert authority.runtime_statuses() == {}
    assert authority.get_running_count() == 0
    assert authority.get_running_count_by_session("s") == 0
    assert asyncio.run(authority.cancel_by_session("s")) == 0

    with pytest.raises(RuntimeError, match="retired"):
        asyncio.run(authority.spawn("x"))
    with pytest.raises(RuntimeError, match="retired"):
        asyncio.run(authority.run_inline("x"))

    assert inspect.iscoroutinefunction(authority.close)
