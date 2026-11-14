"""Regression tests for A3-6 AC6.1 and AC6.2: MCP binding generation refusal."""

from types import SimpleNamespace

from mcp import types as mcp_types
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from nanobot.agent.tools.mcp import MCPToolWrapper


class DeadSession:
    def __init__(self):
        self.calls = 0

    async def call_tool(self, name, arguments):
        self.calls += 1
        raise McpError(ErrorData(code=-32000, message="Session terminated"))


class LiveSession:
    def __init__(self):
        self.calls = 0

    async def call_tool(self, name, arguments):
        self.calls += 1
        return SimpleNamespace(
            content=[mcp_types.TextContent(type="text", text="recovered")]
        )


def _build_scenario():
    tool_def = SimpleNamespace(
        name="quote",
        description="quote tool",
        inputSchema={"type": "object", "properties": {}},
    )
    dead = DeadSession()
    live = LiveSession()
    wrapper = MCPToolWrapper(dead, "remote", tool_def, tool_timeout=5)
    replacement_holder = {}

    async def reconnect(server_name, tool_name, stale_tool):
        replacement = MCPToolWrapper(live, server_name, tool_def, tool_timeout=5)
        replacement_holder["tool"] = replacement
        return replacement

    wrapper.set_reconnect_handler(reconnect)
    return dead, live, wrapper, replacement_holder


async def test_admitted_call_refuses_replacement_session():
    """AC6.1: an admitted call must not retry on a freshly reconnected binding."""
    dead, live, wrapper, replacement_holder = _build_scenario()

    result = await wrapper.execute(symbol="AAPL")

    assert result.is_error is True
    assert "refused" in result
    assert "bound connection generation ended" in result
    assert dead.calls == 1
    assert live.calls == 0
    assert wrapper._session is dead
    assert isinstance(replacement_holder["tool"], MCPToolWrapper)
    assert replacement_holder["tool"] is not wrapper


async def test_replacement_binding_serves_a_later_call():
    """AC6.2: a later call on the replacement binding succeeds exactly once."""
    dead, live, wrapper, replacement_holder = _build_scenario()

    # Drive the refused admitted call so the reconnect handler produces the
    # replacement binding under test.
    await wrapper.execute(symbol="AAPL")
    replacement = replacement_holder["tool"]

    output = await replacement.execute(symbol="AAPL")

    assert output == "recovered"
    assert live.calls == 1
