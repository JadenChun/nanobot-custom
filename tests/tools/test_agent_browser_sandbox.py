from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kohaku.agent.capabilities import REVIEWER_CAPABILITIES, role_capabilities
from kohaku.agent.tools import agent_browser as agent_browser_mod
from kohaku.agent.tools.agent_browser import AgentBrowserTool, classify_browser_action
from kohaku.agent.turn import RunStatus, ToolOutcome
from kohaku.context_repo import ResourceAccessPolicy
from kohaku.security.workspace_access import (
    bind_workspace_scope,
    build_workspace_scope,
    reset_workspace_scope,
)

_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
_JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


@pytest.mark.parametrize(
    "args, expected",
    [
        (["open", "https://example.com"], "read_navigation"),
        (["--headed", "open", "https://example.com"], "read_navigation"),
        (["screenshot"], "read_navigation"),
        (["--json", "screenshot"], "read_navigation"),
        (["--headed", "--json", "screenshot", "--full"], "read_navigation"),
        (["--json", "screenshot", "@e1", "--annotate"], "read_navigation"),
        (["screenshot", "out.png"], "state_changing"),
        (["--json", "screenshot", "out.png"], "state_changing"),
        (["screenshot", "--screenshot-dir", "/tmp"], "state_changing"),
        (["--screenshot-dir", "/tmp", "screenshot"], "state_changing"),
        (["--screenshot-dir=/tmp", "screenshot"], "state_changing"),
        (["click", "@e1"], "state_changing"),
        (["open", "file:///etc/passwd"], "state_changing"),
        (["open", "javascript:alert(1)"], "state_changing"),
        (["open", "data:text/html,<script>alert(1)</script>"], "state_changing"),
        (["open", "view-source:file:///etc/passwd"], "state_changing"),
        (["open", "/etc/passwd"], "state_changing"),
        (["open", "~/secret.txt"], "state_changing"),
        (["open", "./local.html"], "state_changing"),
        (["goto", "/etc/passwd"], "state_changing"),
    ],
)
def test_classify_browser_action_cases(args: list[str], expected: str) -> None:
    assert classify_browser_action(args) == expected


@pytest.mark.asyncio
async def test_screenshot_returns_inline_png_from_json_stdout(tmp_path: Path, monkeypatch) -> None:
    shot = tmp_path / "shot.png"
    shot.write_bytes(_PNG_BYTES)
    stdout = json.dumps({"success": True, "data": {"path": str(shot)}}).encode("utf-8")
    process = AsyncMock(
        return_value=SimpleNamespace(
            timed_out=False,
            stdout=stdout,
            stderr=b"",
            returncode=0,
        )
    )
    monkeypatch.setattr("kohaku.agent.tools.agent_browser._resolve_npx", lambda: "npx")
    monkeypatch.setattr("kohaku.agent.tools.agent_browser.run_owned_process", process)

    result = await AgentBrowserTool(working_dir=str(tmp_path)).execute(
        args=["--json", "screenshot"]
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["type"] == "text"
    payload = json.loads(result[0]["text"])
    assert payload["exitCode"] == 0
    assert payload["action_class"] == "read_navigation"
    assert result[1] == {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64.b64encode(_PNG_BYTES).decode('ascii')}"
        },
    }


@pytest.mark.asyncio
async def test_screenshot_returns_inline_jpeg_from_text_stdout(tmp_path: Path, monkeypatch) -> None:
    shot = tmp_path / "shot.jpg"
    shot.write_bytes(_JPEG_BYTES)
    stdout = f"✓ Screenshot saved to {shot}\n".encode("utf-8")
    process = AsyncMock(
        return_value=SimpleNamespace(
            timed_out=False,
            stdout=stdout,
            stderr=b"",
            returncode=0,
        )
    )
    monkeypatch.setattr("kohaku.agent.tools.agent_browser._resolve_npx", lambda: "npx")
    monkeypatch.setattr("kohaku.agent.tools.agent_browser.run_owned_process", process)

    result = await AgentBrowserTool(working_dir=str(tmp_path)).execute(args=["screenshot"])

    assert isinstance(result, list)
    assert result[1] == {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64.b64encode(_JPEG_BYTES).decode('ascii')}"
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setup_case",
    ["no_path", "missing_file", "oversized", "invalid_bytes"],
)
async def test_screenshot_falls_back_to_plain_json_when_unusable(
    tmp_path: Path, monkeypatch, setup_case: str
) -> None:
    if setup_case == "no_path":
        stdout = b"Screenshot completed\n"
    elif setup_case == "missing_file":
        stdout = json.dumps({"path": str(tmp_path / "missing.png")}).encode("utf-8")
    elif setup_case == "oversized":
        shot = tmp_path / "big.png"
        shot.write_bytes(_PNG_BYTES + b"0" * 128)
        monkeypatch.setattr(agent_browser_mod, "_MAX_INLINE_SCREENSHOT_BYTES", 32)
        stdout = json.dumps({"path": str(shot)}).encode("utf-8")
    else:
        shot = tmp_path / "not_an_image.png"
        shot.write_bytes(b"not actually a png")
        stdout = json.dumps({"path": str(shot)}).encode("utf-8")

    process = AsyncMock(
        return_value=SimpleNamespace(
            timed_out=False,
            stdout=stdout,
            stderr=b"",
            returncode=0,
        )
    )
    monkeypatch.setattr("kohaku.agent.tools.agent_browser._resolve_npx", lambda: "npx")
    monkeypatch.setattr("kohaku.agent.tools.agent_browser.run_owned_process", process)

    result = await AgentBrowserTool(working_dir=str(tmp_path)).execute(
        args=["--json", "screenshot"]
    )

    assert isinstance(result, str)
    payload = json.loads(result)
    assert payload["exitCode"] == 0
    assert payload["action_class"] == "read_navigation"


@pytest.mark.asyncio
async def test_reviewer_capabilities_retains_read_only_agent_browser_in_managed_turn(
    tmp_path: Path, monkeypatch
) -> None:
    assert "agent_browser" in REVIEWER_CAPABILITIES
    profile = role_capabilities("reviewer")
    assert profile.read_only is True
    assert "agent_browser" in profile.tools

    process = AsyncMock(
        return_value=SimpleNamespace(
            timed_out=False,
            stdout=b"ok\n",
            stderr=b"",
            returncode=0,
        )
    )
    monkeypatch.setattr("kohaku.agent.tools.agent_browser._resolve_npx", lambda: "npx")
    monkeypatch.setattr("kohaku.agent.tools.agent_browser.run_owned_process", process)

    policy = ResourceAccessPolicy(workspace=tmp_path, restrict_to_workspace=True)
    tool = AgentBrowserTool(
        working_dir=str(tmp_path),
        read_only=profile.read_only,
        resource_policy=policy,
    )

    token = bind_workspace_scope(build_workspace_scope(tmp_path, "restricted"))
    try:
        allowed_open = await tool.execute(args=["open", "https://example.com"])
        assert isinstance(allowed_open, str)
        assert json.loads(allowed_open)["exitCode"] == 0

        allowed_shot = await tool.execute(args=["--json", "screenshot"])
        assert isinstance(allowed_shot, str)
        assert json.loads(allowed_shot)["exitCode"] == 0

        for blocked_args in (
            ["click", "@e1"],
            ["screenshot", "out.png"],
            ["open", "file:///etc/passwd"],
            ["open", "data:text/html,hi"],
            ["open", "view-source:file:///etc/passwd"],
            ["open", "/etc/passwd"],
        ):
            blocked = await tool.execute(args=blocked_args)
            assert isinstance(blocked, ToolOutcome)
            assert blocked.stop_reason == RunStatus.POLICY_BLOCKED.value
    finally:
        reset_workspace_scope(token)
