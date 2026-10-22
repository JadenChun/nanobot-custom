"""Regression tests for the cron tool's JSON-schema / runtime contract (#3113).

The schema advertised ``required=["action"]`` while ``_add_job`` rejected empty
``message``; LLMs rationally omitted ``message`` and looped on the runtime
error. The fix keeps ``required=["action"]`` (so ``list``/``remove`` stay
callable) but states the per-action requirement in each field's description
and tightens the runtime error for ``add`` without ``message``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.registry import ToolRegistry


class _SvcStub:
    """Minimal CronService stand-in; we only exercise schema/dispatch paths."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def list_jobs(self):
        return []

    def get_job(self, _job_id):
        return None

    def remove_job(self, _job_id):
        return "not-found"

    def add_job(self, **kwargs):
        self.calls.append(kwargs)
        class _J:
            pass

        j = _J()
        j.id = "id1"
        j.name = kwargs.get("name", "x")
        return j


@pytest.fixture
def registry() -> Iterator[ToolRegistry]:
    tool = CronTool(_SvcStub(), default_timezone="UTC")
    reg = ToolRegistry()
    reg.register(tool)
    with request_context(
        RequestContext(channel="channel", chat_id="chat-id", session_key="channel:chat-id")
    ):
        yield reg


class TestSchemaContract:
    def test_list_accepted_without_message(self, registry: ToolRegistry) -> None:
        # action='list' must pass schema validation with nothing but 'action'.
        _, _, err = registry.prepare_call("cron", {"action": "list"})
        assert err is None

    def test_remove_accepted_without_message(self, registry: ToolRegistry) -> None:
        # action='remove' must pass schema validation with just 'action' + 'job_id'.
        _, _, err = registry.prepare_call("cron", {"action": "remove", "job_id": "abc"})
        assert err is None

    def test_add_with_message_accepted(self, registry: ToolRegistry) -> None:
        _, _, err = registry.prepare_call(
            "cron", {"action": "add", "message": "ping", "at": "2030-01-01T00:00:00"}
        )
        assert err is None

    def test_add_without_message_surfaces_actionable_runtime_error(
        self, registry: ToolRegistry
    ) -> None:
        # Schema permits omitting message; the runtime must return a message
        # that tells the LLM exactly what's missing and how to retry, so it
        # doesn't loop like #3113 reports.
        import asyncio

        tool = registry._tools["cron"]  # type: ignore[attr-defined]
        out = asyncio.run(tool.execute(action="add", at="2030-01-01T00:00:00"))
        assert "message" in out
        assert "add" in out
        assert "Retry" in out or "retry" in out


class TestSchemaSelfDescribesRequirements:
    def test_message_description_flags_add_requirement(self) -> None:
        # LLMs rely on field descriptions to infer when something is actually
        # needed. Without this hint, #3113's loop returns.
        tool = CronTool(_SvcStub())
        desc = tool.parameters["properties"]["message"]["description"]
        assert "REQUIRED" in desc and "action='add'" in desc

    def test_job_id_description_flags_remove_requirement(self) -> None:
        tool = CronTool(_SvcStub())
        desc = tool.parameters["properties"]["job_id"]["description"]
        assert "REQUIRED" in desc and "action='remove'" in desc

    def test_top_level_required_stays_narrow(self) -> None:
        # If 'message' or 'job_id' ever creep back into top-level required,
        # list/remove start failing schema validation (the bug PR #3163 v1
        # accidentally introduced).
        tool = CronTool(_SvcStub())
        assert tool.parameters["required"] == ["action"]

    def test_additional_destination_schema_is_strict_and_bounded(self) -> None:
        tool = CronTool(_SvcStub())
        schema = tool.parameters["properties"]["additional_destinations"]
        item = schema["items"]
        assert schema["maxItems"] == 256
        assert item["required"] == ["channel", "to"]
        assert item["additionalProperties"] is False
        assert item["properties"]["channel"]["minLength"] == 1
        assert item["properties"]["to"]["minLength"] == 1

        invalid = [
            [{"channel": "telegram", "to": "chat", "extra": "reject"}],
            [{"channel": "telegram"}],
            [{"channel": "telegram", "to": "  "}],
            [{"channel": True, "to": "chat"}],
            [{"channel": "telegram", "to": 42}],
            [{"channel": "telegram", "to": "chat"}] * 257,
        ]
        for destinations in invalid:
            errors = tool.validate_params({
                "action": "add",
                "message": "ping",
                "every_seconds": 60,
                "additional_destinations": destinations,
            })
            assert errors

    def test_additional_destinations_direct_call_trims_preserves_order_and_primary(self) -> None:
        service = _SvcStub()
        tool = CronTool(service)
        metadata = {"nested": {"value": "before"}}
        with request_context(
            RequestContext(
                channel="telegram",
                chat_id="primary",
                session_key="telegram:primary",
                metadata=metadata,
            )
        ):
            result = asyncio.run(tool.execute(
                action="add",
                message="ping",
                every_seconds=60,
                additional_destinations=[
                    {"channel": " discord ", "to": " B "},
                    {"channel": "discord", "to": "B"},
                    {"channel": "telegram", "to": "primary"},
                ],
            ))
        metadata["nested"]["value"] = "after"
        assert "Created job" in result
        call = service.calls[0]
        assert call["deliver"] is True
        assert call["channel"] == "telegram"
        assert call["to"] == "primary"
        assert call["session_key"] == "telegram:primary"
        assert call["origin_channel"] == "telegram"
        assert call["origin_chat_id"] == "primary"
        assert [
            (destination.channel, destination.to)
            for destination in call["additional_destinations"]
        ] == [("discord", "B"), ("discord", "B"), ("telegram", "primary")]
        assert call["origin_metadata"] == {"nested": {"value": "before"}}

    def test_additional_destination_type_and_byte_limits_are_rejected_early(self) -> None:
        service = _SvcStub()
        tool = CronTool(service)
        with request_context(
            RequestContext(channel="telegram", chat_id="primary", session_key="telegram:primary")
        ):
            malformed = asyncio.run(tool.execute(
                action="add", message="ping", every_seconds=60,
                additional_destinations=["not an object"],
            ))
            too_long = asyncio.run(tool.execute(
                action="add", message="ping", every_seconds=60,
                additional_destinations=[{"channel": "telegram", "to": "é" * 257}],
            ))
        assert "must be an object" in malformed
        assert "512-byte" in too_long
        assert service.calls == []

    def test_direct_call_without_authenticated_route_is_rejected(self) -> None:
        tool = CronTool(_SvcStub())
        result = asyncio.run(tool.execute(action="add", message="ping", every_seconds=60))
        assert "chat session" in result

    def test_registry_keeps_destination_values_strict_before_validation(self) -> None:
        service = _SvcStub()
        tool = CronTool(service)
        registry = ToolRegistry()
        registry.register(tool)
        with request_context(
            RequestContext(channel="telegram", chat_id="primary", session_key="telegram:primary")
        ):
            _, _, err = registry.prepare_call(
                "cron",
                {
                    "action": "add",
                    "message": "ping",
                    "every_seconds": 60,
                    "additional_destinations": [{"channel": 42, "to": "chat"}],
                },
            )
        assert err is not None
