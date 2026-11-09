"""Regression tests for native admission reservations in ExecTool."""

from types import SimpleNamespace

import pytest

from nanobot.agent.tools.base import ToolResult
from nanobot.agent.tools.shell import ExecTool

REVALIDATION_ERROR = "input_files permission changed before command launch"


class _Reservation:
    def __init__(self) -> None:
        self.cancel_calls = 0

    def cancel_if_unlaunched(self) -> bool:
        self.cancel_calls += 1
        return True


def _result_text(result: ToolResult) -> str:
    parts: list[str] = []
    for name in ("output", "content", "error", "message", "text"):
        value = getattr(result, name, None)
        if value is not None:
            parts.append(str(value))
    parts.append(repr(result))
    return "\n".join(parts)


@pytest.mark.asyncio
async def test_revalidation_denial_cancels_reservation_without_spawning(
    tmp_path, monkeypatch
) -> None:
    tool = ExecTool(
        working_dir=str(tmp_path),
        restrict_to_workspace=False,
        timeout=5,
    )

    prepared = SimpleNamespace(
        command="echo hi",
        cwd=str(tmp_path),
        env={},
        timeout=5,
        shell_program=None,
        login=False,
        managed=False,
        logical_cwd=None,
        grant_snapshot=None,
    )
    reservation = _Reservation()
    spawn_calls: list[tuple[tuple, dict]] = []

    async def fake_spawn(*args, **kwargs):
        spawn_calls.append((args, kwargs))
        raise AssertionError("must not spawn")

    monkeypatch.setattr(tool, "_prepare_command", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(
        tool,
        "_reserve_native_admission",
        lambda *args, **kwargs: (prepared, reservation),
    )
    monkeypatch.setattr(
        tool,
        "_revalidate_input_files",
        lambda *args, **kwargs: ToolResult.error(
            "Error: input_files permission changed before command launch"
        ),
    )
    monkeypatch.setattr(tool, "_spawn", fake_spawn)

    result = await tool._execute_impl(command="echo hi", staged_inputs=[object()])

    assert REVALIDATION_ERROR in _result_text(result)
    assert reservation.cancel_calls == 1
    assert spawn_calls == []
