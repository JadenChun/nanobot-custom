from __future__ import annotations

import hashlib
import inspect
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.image_generation import ImageGenerationTool
from nanobot.agent.tools.read_provenance import FilesystemReadReceipt
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.agent.turn import ToolOutcome, TurnContext
from nanobot.session.artifact import FilesystemArtifactOrigin


class _FakeTool(Tool):
    def __init__(self, name: str, schema: dict[str, Any] | None = None):
        self._name = name
        self._schema = schema

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return self._schema or {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> Any:
        return kwargs


async def test_contextual_call_preserves_schema_context_argument() -> None:
    class ContextTool(_FakeTool):
        async def execute(self, context: str, execution_context: Any = None) -> Any:
            return (context, execution_context)

    tool = ContextTool("context_tool", {
        "type": "object", "properties": {"context": {"type": "string"}},
    })
    turn = TurnContext(run_id="context-boundary")
    assert await tool.execute_with_context(turn, context="user evidence") == (
        "user evidence", turn,
    )


async def test_contextual_call_preserves_remote_kwargs() -> None:
    tool = _FakeTool("remote", {
        "type": "object", "properties": {"context": {"type": "string"}},
    })
    assert await tool.execute_with_context(
        TurnContext(run_id="remote"), context="remote value",
    ) == {"context": "remote value"}


async def test_contextual_call_still_injects_internal_context() -> None:
    class InternalContextTool(_FakeTool):
        async def execute(self, context: Any = None, **kwargs: Any) -> Any:
            return context

    turn = TurnContext(run_id="internal")
    assert await InternalContextTool("internal").execute_with_context(turn) is turn


async def test_contextual_registry_preserves_context_input() -> None:
    class ContextTool(_FakeTool):
        async def execute(self, context: str = "default", **kwargs: Any) -> str:
            return context

    registry = ToolRegistry()
    registry.register(ContextTool("context_tool", {
        "type": "object", "properties": {"context": {"type": "string"}},
    }))
    turn = TurnContext(run_id="registry-context")
    supplied = await registry.execute("context_tool", {"context": "evidence"}, context=turn)
    omitted = await registry.execute("context_tool", {}, context=turn)
    assert isinstance(supplied, ToolOutcome)
    assert isinstance(omitted, ToolOutcome)
    assert supplied.content == "evidence"
    assert omitted.content == "default"


async def test_contextual_call_preserves_schema_execution_context() -> None:
    class ContextTool(_FakeTool):
        async def execute(self, execution_context: str = "default", **kwargs: Any) -> str:
            return execution_context

    tool = ContextTool("context_tool", {
        "type": "object", "properties": {"execution_context": {"type": "string"}},
    })
    turn = TurnContext(run_id="execution-context")
    assert await tool.execute_with_context(turn, execution_context="evidence") == "evidence"
    assert await tool.execute_with_context(turn) == "default"

def _tool_names(definitions: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for definition in definitions:
        fn = definition.get("function", {})
        names.append(fn.get("name", ""))
    return names


def _registry_with_names(names: list[str]) -> ToolRegistry:
    registry = ToolRegistry()
    for name in names:
        registry.register(_FakeTool(name))
    return registry


def test_get_definitions_orders_builtins_then_mcp_tools() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("mcp_git_status"))
    registry.register(_FakeTool("write_file"))
    registry.register(_FakeTool("mcp_fs_list"))
    registry.register(_FakeTool("read_file"))

    assert _tool_names(registry.get_definitions()) == [
        "read_file",
        "write_file",
        "mcp_fs_list",
        "mcp_git_status",
    ]


def test_prepare_call_rejects_near_miss_tool_name_with_suggestion() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("read_file"))

    tool, params, error = registry.prepare_call("readFile", {"path": "foo.txt"})

    assert tool is None
    assert params == {"path": "foo.txt"}
    assert error is not None
    assert "Tool 'readFile' not found" in error
    assert "Did you mean 'read_file'?" in error
    assert "must match exactly" in error


def test_suggest_name_handles_canonical_tool_name_variants() -> None:
    registry = _registry_with_names(["read_file"])
    expected = {
        "readFile": "read_file",
        "read-file": "read_file",
        "READ_FILE": "read_file",
        "read file": "read_file",
        "readfile": "read_file",
    }

    assert {name: registry._suggest_name(name) for name in expected} == expected


def test_suggest_name_suppresses_low_confidence_and_non_unique_matches() -> None:
    registry = _registry_with_names(["read_file", "write_file"])

    for name in ["", "foo", "read", "file", "readfil", "read_file_tool"]:
        assert registry._suggest_name(name) is None

    ambiguous = _registry_with_names(["read_file", "readFile"])
    assert ambiguous._suggest_name("readfile") is None


def test_suggest_name_updates_after_register_and_unregister() -> None:
    registry = _registry_with_names(["read_file"])

    assert registry._suggest_name("readFile") == "read_file"

    registry.register(_FakeTool("readFile"))
    assert registry._suggest_name("read-file") is None

    registry.unregister("read_file")
    assert registry._suggest_name("read-file") == "readFile"


def test_prepare_call_read_file_rejects_non_object_params_with_actionable_hint() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("read_file"))

    tool, params, error = registry.prepare_call("read_file", ["foo.txt"])

    assert tool is not None
    assert params == ["foo.txt"]
    assert error is not None
    assert "must be a JSON object" in error
    assert 'tool_name(param1="value1", param2="value2")' in error
    assert "matching the tool schema" in error


def test_prepare_call_parses_json_string_arguments() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("read_file"))

    tool, params, error = registry.prepare_call("read_file", '{"path":"foo.txt"}')

    assert tool is not None
    assert params == {"path": "foo.txt"}
    assert error is None


def test_prepare_call_rejects_malformed_json_string_arguments() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("read_file"))

    tool, params, error = registry.prepare_call("read_file", '{path:"foo.txt"}')

    assert tool is not None
    assert params == '{path:"foo.txt"}'
    assert error is not None
    assert "parameters must be a JSON object" in error


def test_prepare_call_rejects_scalar_for_single_required_parameter() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool(
        "web_fetch",
        {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    ))

    tool, params, error = registry.prepare_call("web_fetch", "https://example.com")

    assert tool is not None
    assert params == "https://example.com"
    assert error is not None
    assert "parameters must be a JSON object" in error


def test_prepare_call_rejects_unquoted_scalar_strings_before_schema_cast() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool(
        "message",
        {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
    ))

    tool, params, error = registry.prepare_call("message", "true")

    assert tool is not None
    assert params == "true"
    assert error is not None
    assert "parameters must be a JSON object" in error


def test_prepare_call_unwraps_arguments_payload() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool(
        "read_file",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    ))

    tool, params, error = registry.prepare_call(
        "read_file",
        {"arguments": '{"path":"foo.txt"}'},
    )

    assert tool is not None
    assert params == {"path": "foo.txt"}
    assert error is None


def test_prepare_call_treats_none_arguments_as_empty_object() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("list_exec_sessions"))

    tool, params, error = registry.prepare_call("list_exec_sessions", None)

    assert tool is not None
    assert params == {}
    assert error is None

    tool, params, error = registry.prepare_call("list_exec_sessions", "null")

    assert tool is not None
    assert params == "null"
    assert error is not None
    assert "parameters must be a JSON object" in error


def test_prepare_call_other_tools_keep_generic_object_validation() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("grep"))

    tool, params, error = registry.prepare_call("grep", ["TODO"])

    assert tool is not None
    assert params == ["TODO"]
    assert error == (
        "Error: Tool 'grep' parameters must be a JSON object, got list. "
        'Use named parameters like tool_name(param1="value1", param2="value2") '
        "matching the tool schema."
    )


async def test_registry_rejects_unknown_builtin_tool_parameters(tmp_path) -> None:
    (tmp_path / "sample.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(
        ReadFileTool(
            workspace=tmp_path,
            allowed_dir=tmp_path,
            restrict_to_workspace=True,
        )
    )

    result = await registry.execute(
        "read_file",
        {"path": "sample.txt", "line_limit": 1},
    )

    assert "Invalid parameters" in result
    assert "unexpected parameter line_limit" in result
    assert "one" not in result


async def test_registry_preserves_successful_exec_output_that_starts_with_error() -> None:
    registry = ToolRegistry()
    output = "Error: generated report successfully\n\nExit code: 0"
    tool = _FakeTool("exec")
    tool.execute = AsyncMock(return_value=output)
    registry.register(tool)

    result = await registry.execute("exec", {})

    assert result == output


async def test_contextual_registry_treats_unstructured_error_prefix_as_success() -> None:
    registry = ToolRegistry()
    output = "Error: generated report successfully"
    tool = _FakeTool("report")
    tool.execute = AsyncMock(return_value=ToolOutcome(
        content=output,
        policy_metadata={"source": "fixture"},
    ))
    registry.register(tool)

    result = await registry.execute("report", {}, context=TurnContext(run_id="report-run"))

    assert isinstance(result, ToolOutcome)
    assert result.content == output
    assert result.stop_reason is None
    assert result.policy_metadata == {"source": "fixture"}


async def test_registry_keeps_valid_producer_receipt_for_unstructured_error_prefix(tmp_path) -> None:
    output = "Error: generated report successfully"
    encoded = output.encode("utf-8")
    receipt = FilesystemReadReceipt(
        FilesystemArtifactOrigin("filesystem", "workspace", None, "report.txt"),
        "a" * 64,
        len(encoded),
        hashlib.sha256(encoded).hexdigest(),
    )
    original = ToolOutcome(
        content=output,
        policy_metadata={"source": "fixture"},
        producer_receipt=receipt,
    )
    tool = ReadFileTool(workspace=tmp_path)
    tool.execute = AsyncMock(return_value=original)

    result = await ToolRegistry.execute_prepared(tool, {})

    assert result.content == output
    assert result.stop_reason is None
    assert result.policy_metadata == {"source": "fixture"}
    assert result.producer_receipt == receipt


async def test_registry_uses_structured_tool_result_for_errors() -> None:
    registry = ToolRegistry()
    output = "Error: plain tool output, not a structured failure"
    raw_tool = _FakeTool("raw_output")
    raw_tool.execute = AsyncMock(return_value=output)
    registry.register(raw_tool)

    raw_result = await registry.execute("raw_output", {})

    assert raw_result == output

    failing_tool = _FakeTool("failing_tool")
    failing_tool.execute = AsyncMock(return_value=ToolResult.error("Error: real failure"))
    registry.register(failing_tool)

    error_result = await registry.execute("failing_tool", {})

    assert isinstance(error_result, ToolResult)
    assert error_result.is_error
    assert error_result.startswith("Error: real failure")
    assert "[Analyze the error above" in error_result


def test_get_definitions_returns_cached_result() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("read_file"))
    first = registry.get_definitions()
    assert registry._cached_definitions is not None
    second = registry.get_definitions()
    assert first == second


def test_register_invalidates_cache() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("read_file"))
    first = registry.get_definitions()
    registry.register(_FakeTool("write_file"))
    second = registry.get_definitions()
    assert first is not second
    assert len(second) == 2


def test_unregister_invalidates_cache() -> None:
    registry = ToolRegistry()
    registry.register(_FakeTool("read_file"))
    registry.register(_FakeTool("write_file"))
    first = registry.get_definitions()
    registry.unregister("write_file")
    second = registry.get_definitions()
    assert first is not second
    assert len(second) == 1


def test_tool_interface_runtime_abstractness_and_signatures() -> None:
    assert Tool.__abstractmethods__ == {"name", "description", "parameters", "execute"}
    assert str(inspect.signature(Tool.execute)) == "(self, **kwargs: 'Any') -> 'Any'"
    assert str(inspect.signature(Tool.execute_with_context)) == (
        "(self, context: 'Any', /, **kwargs: 'Any') -> 'Any'"
    )
    for missing in ("name", "description", "parameters", "execute"):
        members = {
            "name": _FakeTool.name,
            "description": _FakeTool.description,
            "parameters": _FakeTool.parameters,
            "execute": _FakeTool.execute,
        }
        del members[missing]
        incomplete = type("IncompleteTool", (Tool,), members)
        with pytest.raises(TypeError, match=missing):
            incomplete()
        decorated = tool_parameters({"type": "object"})(incomplete)
        assert decorated is incomplete
        if missing != "parameters":
            with pytest.raises(TypeError, match=missing):
                decorated()
        else:
            assert decorated.__abstractmethods__ == frozenset()


def test_tool_interface_class_values_and_property_metadata() -> None:
    schema: dict[str, Any] = {"type": "object", "properties": {}}
    value_tool = type("ValueTool", (Tool,), {
        "name": "value", "description": "value metadata", "parameters": schema,
        "execute": _FakeTool.execute,
    })
    first = value_tool()
    second = value_tool()
    assert first.name == "value"
    assert first.description == "value metadata"
    assert first.parameters is schema
    schema["title"] = "class mutation"
    assert second.parameters["title"] == "class mutation"
    shadow: dict[str, Any] = {"type": "object", "title": "instance shadow"}
    setattr(first, "parameters", shadow)
    assert first.parameters is shadow
    assert second.parameters is schema
    property_tool = _FakeTool("property", schema)
    assert property_tool.name == "property"
    assert property_tool.description == "property tool"
    assert property_tool.parameters is schema


async def test_tool_interface_decorator_identity_binding_and_copies() -> None:
    schema: dict[str, Any] = {
        "type": "object", "properties": {"value": {"type": "string", "enum": ["seed"]}},
    }
    original = _FakeTool
    # A fresh subclass keeps the shared test helper's metadata untouched.
    owner = type("DecoratedTool", (original,), {})
    constructor = owner.__init__
    execute = owner.execute
    constructor_signature = inspect.signature(owner)
    execute_signature = inspect.signature(owner.execute)
    decorated = tool_parameters(schema)(owner)
    assert decorated is owner
    assert decorated.__init__ is constructor
    assert decorated.execute is execute
    assert inspect.signature(decorated) == constructor_signature
    assert inspect.signature(decorated.execute) == execute_signature
    instance = decorated("bound")
    assert instance.execute.__self__ is instance
    assert instance.execute.__func__ is execute
    assert await instance.execute(value="seed") == {"value": "seed"}
    schema["properties"]["value"]["enum"].append("input mutation")
    first = instance.parameters
    assert first["properties"]["value"]["enum"] == ["seed"]
    first["properties"]["value"]["enum"].append("access mutation")
    second = instance.parameters
    assert second is not first
    assert second["properties"]["value"]["enum"] == ["seed"]


@pytest.mark.parametrize("owner", [
    ReadFileTool, WriteFileTool, EditFileTool, ListDirTool,
    WebSearchTool, WebFetchTool, ImageGenerationTool,
])
def test_tool_interface_real_decorated_owner_schema_copies(owner: type[Tool]) -> None:
    descriptor = vars(owner)["parameters"]
    assert isinstance(descriptor, property)
    assert not getattr(descriptor, "__isabstractmethod__", False)
    assert "parameters" not in owner.__abstractmethods__
    # Access the installed property without activating constructors/providers.
    instance = object.__new__(owner)
    first = instance.parameters
    second = instance.parameters
    assert isinstance(first, dict)
    assert first == second
    assert first is not second
    assert first["type"] == "object"
    assert first["properties"] is not second["properties"]
    key = next(iter(first["properties"]))
    first["properties"][key]["title"] = "access mutation"
    assert "title" not in instance.parameters["properties"][key]
