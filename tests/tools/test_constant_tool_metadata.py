"""Constant metadata retains its class identity and normal attribute behavior."""
import inspect
from contextlib import asynccontextmanager

import pytest

from nanobot.agent.tools.artifact import ReadArtifactTool
from nanobot.agent.tools.artifact_handoff import ArtifactHandoffTool
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.image import ImageGenerationTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.social_crawl import SocialCrawlTool
from nanobot.agent.tools.trend_vpn import (
    TrendVpnBrowserFetchTool,
    TrendVpnFetchTool,
    TrendVpnSessionCloseTool,
    TrendVpnSessionStartTool,
)
from nanobot.session.artifact_grants import ArtifactHandoffCollector
from nanobot.session.artifact_reader import ArtifactReader
from tests.session.test_artifact_reader import setup_read

NAMES = (
    "read_artifact", "handoff_artifact", "generate_image", "social_crawl",
    "trend_vpn_session_start", "trend_vpn_fetch", "trend_vpn_browser_fetch",
    "trend_vpn_session_close",
)


@asynccontextmanager
async def constant_tools(tmp_path):
    """Use actual published artifacts and owner-issued root/child writers."""
    root_path, child_path = tmp_path / "root", tmp_path / "child"
    root_path.mkdir()
    child_path.mkdir()
    async with setup_read(root_path) as (writer, file_tool, store, _, _):
        reader = ArtifactReader(writer, "root_history", store, file_tool)
        async with setup_read(child_path, child=True) as (child, child_tool, child_store, _, _):
            child_reader = ArtifactReader(child, "isolated_stream", child_store, child_tool)
            collector = ArtifactHandoffCollector(child_reader)
            yield {
                tool.name: tool for tool in (
                    ReadArtifactTool(reader), ArtifactHandoffTool(collector),
                    ImageGenerationTool(workspace=tmp_path), SocialCrawlTool(),
                    TrendVpnSessionStartTool(), TrendVpnFetchTool(),
                    TrendVpnBrowserFetchTool(), TrendVpnSessionCloseTool(),
                )
            }


@pytest.mark.parametrize("name", NAMES)
async def test_constant_metadata_is_not_a_runtime_property(tmp_path, name):
    async with constant_tools(tmp_path) as tools:
        tool = tools[name]
        cls = type(tool)
        assert type(cls) is type(Tool)
        assert tool.name == name == cls.name
        assert isinstance(inspect.getattr_static(cls, "name"), str)
        assert isinstance(inspect.getattr_static(cls, "description"), str)
        assert isinstance(inspect.getattr_static(cls, "parameters"), dict)
        assert tool.description == cls.description
        assert tool.parameters is cls.parameters
        assert not cls.__abstractmethods__
        assert tool.read_only is (name in {"read_artifact", "handoff_artifact"})
        registry = ToolRegistry()
        registry.register(tool)
        definition, = registry.get_definitions()
        assert definition == tool.to_schema()
        assert definition["function"]["parameters"] is cls.parameters


@pytest.mark.parametrize("name", NAMES)
async def test_class_schema_mutation_and_inheritance_are_shared(tmp_path, name, monkeypatch):
    async with constant_tools(tmp_path) as tools:
        tool = tools[name]
        cls = type(tool)
        subclass = type("InheritedConstantTool", (cls,), {})
        assert subclass.parameters is cls.parameters is tool.parameters
        monkeypatch.setitem(cls.parameters, "metadata_identity_probe", True)
        assert tool.parameters["metadata_identity_probe"] is True
        assert subclass.parameters["metadata_identity_probe"] is True
        assert tool.to_schema()["function"]["parameters"] is cls.parameters
        assert subclass.name == cls.name == name
        replacement = {"type": "object", "properties": {}}
        monkeypatch.setattr(cls, "parameters", replacement)
        monkeypatch.setattr(cls, "name", "class_override")
        assert tool.parameters is subclass.parameters is cls.parameters is replacement
        assert tool.name == subclass.name == "class_override"


@pytest.mark.parametrize("name", NAMES)
async def test_instance_attributes_can_shadow_class_constants(tmp_path, name, monkeypatch):
    async with constant_tools(tmp_path) as tools:
        tool = tools[name]
        cls = type(tool)
        original = cls.parameters
        replacement = {"type": "object", "properties": {}}
        monkeypatch.setattr(tool, "parameters", replacement)
        monkeypatch.setattr(tool, "name", "instance_override")
        monkeypatch.setattr(tool, "description", "instance description")
        assert tool.parameters is replacement
        assert cls.parameters is original
        assert tool.name == "instance_override" and cls.name == name
        assert tool.description == "instance description" and cls.description != tool.description
        assert tool.to_schema()["function"]["parameters"] is replacement


def test_property_style_tools_and_base_abstract_contract_stay_properties():
    for member in ("name", "description", "parameters"):
        assert isinstance(inspect.getattr_static(Tool, member), property)
        assert isinstance(inspect.getattr_static(ReadFileTool, member), property)
    assert Tool.__abstractmethods__ == {"name", "description", "parameters", "execute"}
    with pytest.raises(TypeError):
        Tool()
