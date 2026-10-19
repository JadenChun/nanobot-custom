import pytest

from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.loader import ToolLoader
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import Config


@pytest.mark.parametrize("admitted", [False, True])
def test_retired_memory_scope_never_constructs_runtime_tools(tmp_path, monkeypatch, admitted):
    loader = ToolLoader()
    registry = ToolRegistry()
    ctx = ToolContext(config=Config().tools, workspace=str(tmp_path))
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append("construction")
        raise AssertionError("retired memory scope must not construct tools or plugins")

    monkeypatch.setattr("nanobot.agent.tools.fork_factories.construct_core_tools", forbidden)
    monkeypatch.setattr(ToolLoader, "_construct_plugins", forbidden)
    if admitted:
        # The foreground-only contract still refuses non-core scopes.
        monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    with pytest.raises(ValueError, match="Unsupported runtime tool scope"):
        loader.load(ctx, registry, scope="memory")

    assert attempts == []
    assert registry.tool_names == []
