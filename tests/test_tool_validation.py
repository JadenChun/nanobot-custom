import asyncio
import json
from typing import Any

from nanobot.agent.tools.agent_browser import AgentBrowserTool
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import Config


class SampleTool(Tool):
    @property
    def name(self) -> str:
        return "sample"

    @property
    def description(self) -> str:
        return "sample tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "mode": {"type": "string", "enum": ["fast", "full"]},
                "meta": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["tag"],
                },
            },
            "required": ["query", "count"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


def test_validate_params_missing_required() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi"})
    assert "missing required count" in "; ".join(errors)


def test_validate_params_type_and_range() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 0})
    assert any("count must be >= 1" in e for e in errors)

    errors = tool.validate_params({"query": "hi", "count": "2"})
    assert any("count should be integer" in e for e in errors)


def test_validate_params_enum_and_min_length() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "h", "count": 2, "mode": "slow"})
    assert any("query must be at least 2 chars" in e for e in errors)
    assert any("mode must be one of" in e for e in errors)


def test_validate_params_nested_object_and_array() -> None:
    tool = SampleTool()
    errors = tool.validate_params(
        {
            "query": "hi",
            "count": 2,
            "meta": {"flags": [1, "ok"]},
        }
    )
    assert any("missing required meta.tag" in e for e in errors)
    assert any("meta.flags[0] should be string" in e for e in errors)


def test_validate_params_ignores_unknown_fields() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 2, "extra": "x"})
    assert errors == []


async def test_registry_returns_validation_error() -> None:
    reg = ToolRegistry()
    reg.register(SampleTool())
    result = await reg.execute("sample", {"query": "hi"})
    assert "Invalid parameters" in result


class _FakeProcess:
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.killed = False

    async def communicate(self):
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> None:
        return None


async def test_agent_browser_runs_npx_with_expected_args(monkeypatch) -> None:
    captured: dict = {}

    async def _fake_create_subprocess_exec(*cmd, stdout=None, stderr=None, cwd=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return _FakeProcess(stdout=b"hello-world", stderr=b"warn")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    tool = AgentBrowserTool(
        package="@agentic/agent-browser@0.0.1",
        timeout=30,
        max_output_chars=5,
        working_dir="/tmp",
    )
    result = await tool.execute(args=["--help"])
    data = json.loads(result)

    assert captured["cmd"] == ("npx", "--yes", "@agentic/agent-browser@0.0.1", "--help")
    assert captured["cwd"] == "/tmp"
    assert data["exitCode"] == 0
    assert data["stdout"] == "hello"
    assert data["stderr"] == "warn"
    assert data["stdoutTruncated"] is True
    assert data["stderrTruncated"] is False


async def test_agent_browser_handles_timeout(monkeypatch) -> None:
    proc = _FakeProcess(stdout=b"", stderr=b"")

    async def _slow_communicate():
        await asyncio.sleep(0.05)
        return b"", b""

    proc.communicate = _slow_communicate  # type: ignore[assignment]

    async def _fake_create_subprocess_exec(*cmd, stdout=None, stderr=None, cwd=None):
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    tool = AgentBrowserTool(timeout=30)
    result = await tool.execute(args=["--help"], timeout=0)
    data = json.loads(result)

    assert "timed out" in data["error"]
    assert proc.killed is True


async def test_agent_browser_missing_npx_returns_error(monkeypatch) -> None:
    async def _fake_create_subprocess_exec(*cmd, stdout=None, stderr=None, cwd=None):
        raise FileNotFoundError

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    tool = AgentBrowserTool()
    result = await tool.execute(args=["--help"])

    assert "npx" in result
    assert "not found" in result.lower()


async def test_registry_validation_rejects_missing_agent_browser_args() -> None:
    reg = ToolRegistry()
    reg.register(AgentBrowserTool())
    result = await reg.execute("agent_browser", {})
    assert "Invalid parameters" in result


def test_config_accepts_agent_browser_camel_case_keys() -> None:
    cfg = Config.model_validate(
        {
            "tools": {
                "agentBrowser": {
                    "enabled": False,
                    "timeout": 90,
                    "maxOutputChars": 4321,
                }
            }
        }
    )
    assert cfg.tools.agent_browser.enabled is False
    assert cfg.tools.agent_browser.timeout == 90
    assert cfg.tools.agent_browser.max_output_chars == 4321
