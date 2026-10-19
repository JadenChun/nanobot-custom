"""Invocation-local artifact grants and direct child handoff evidence."""

import asyncio
import hashlib
import json
import pickle
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.artifact_handoff import ArtifactHandoffTool
from nanobot.agent.tools.delegation import DelegateTaskTool
from nanobot.agent.tools.execution import KnownToolResult
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import ToolOutcome, TurnContext
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest
from nanobot.session.artifact import ArtifactValidationError, parse_artifact_descriptor
from nanobot.session.artifact_grants import (
    ArtifactGrantSet,
    ArtifactHandoffCollector,
    ArtifactHandoffReceipt,
)
from nanobot.session.artifact_reader import ArtifactReader
from nanobot.session.artifact_store import PrivateArtifactStore
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.source_store import SourceSessionStore
from nanobot.session.turn_writer import SourceTurnWriter
from nanobot.utils.llm_runtime import LLMRuntime
from tests.session.test_artifact import fixture


@asynccontextmanager
async def published_artifact(tmp_path, *, child: bool = False):
    """Publish one attested artifact and retain its direct parent stream."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "file").write_text("evidence 中文 🌟")
    backend = JsonlSessionStore(workspace)
    session = Session("test:artifact-grants")
    backend.save(session)
    backend.source.upgrade_source_session(
        session.key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    async with acquire_execution(backend, session.key, run_id="run") as owner:
        parent = SourceTurnWriter(backend.source, owner, run_id="run")
        writer = parent.child_stream() if child else parent
        tool = ReadFileTool(workspace=workspace, allowed_dir=workspace)
        outcome = await ToolRegistry.execute_prepared(
            tool, {"path": "file"}, SimpleNamespace(lock_owner=None)
        )
        writer.begin_tool_batch({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "read",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        })
        await writer.record_tool_result(KnownToolResult(
            0,
            "read",
            "read_file",
            outcome,
            outcome.content,
            {"status": "ok"},
            True,
            outcome.producer_receipt,
        ))
        snapshot = owner.read_snapshot()
        event = snapshot.events[-1]
        raw, _ = fixture()
        content = event["content"].encode()
        raw.update(
            workspace_id=backend.sessions_dir.name,
            session_key=session.key,
            incarnation=snapshot.state.incarnation,
            generation=snapshot.state.generation,
            event_id=event["_event_id"],
            execution_id=event["_execution_id"],
            stream_id=writer.stream_id,
            byte_length=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            source_access=outcome.producer_receipt.to_dict()["source_access"],
        )
        descriptor = parse_artifact_descriptor(raw)
        store = PrivateArtifactStore(
            tmp_path.resolve() / "artifacts",
            workspace_id=descriptor.workspace_id,
            session_key=session.key,
            incarnation=descriptor.incarnation,
            generation=descriptor.generation,
        )
        source = SourceSessionStore(backend, artifact_store=store)
        with store.transaction() as blobs:
            blobs.put(descriptor, content)
            source.commit_artifact_descriptor(
                session.key, snapshot.state, descriptor, owner=owner, blobs=blobs
            )
        yield parent, writer, tool, store, descriptor, content


async def test_direct_child_grant_reads_exact_parent_artifact_without_source_write(tmp_path):
    async with published_artifact(tmp_path) as (parent, writer, tool, store, descriptor, content):
        parent_reader = ArtifactReader(writer, "root_history", store, tool)
        before = writer.owner.read_snapshot()
        child = writer.child_stream()
        grants = ArtifactGrantSet.issue_downward(parent_reader, child, [descriptor.reference])
        assert writer.owner.read_snapshot() == before

        child_reader = ArtifactReader(child, "isolated_stream", store, tool, grants=grants)
        assert child_reader.read(descriptor.reference).text.encode() == content
        with pytest.raises(ArtifactValidationError):
            ArtifactReader(writer.child_stream(), "isolated_stream", store, tool).read(descriptor.reference)
        with pytest.raises(ArtifactValidationError):
            ArtifactGrantSet.issue_downward(
                child_reader, child.child_stream(), [descriptor.reference]
            )
        child.fail()
        with pytest.raises(ArtifactValidationError):
            child_reader.read(descriptor.reference)


async def test_grant_input_is_bounded_unique_and_nonserializable(tmp_path):
    async with published_artifact(tmp_path) as (_, writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, "root_history", store, tool)
        child = writer.child_stream()
        before = writer.owner.read_snapshot()
        for references in (
            [descriptor.reference, descriptor.reference],
            [descriptor.reference] * 17,
            (descriptor.reference,),
        ):
            with pytest.raises(ArtifactValidationError):
                ArtifactGrantSet.issue_downward(reader, child, references)
            assert writer.owner.read_snapshot() == before

        grants = ArtifactGrantSet.issue_downward(reader, child, [])
        with pytest.raises(TypeError):
            pickle.dumps(grants)


async def test_grants_reject_forged_or_cross_set_nonces(tmp_path):
    async with published_artifact(tmp_path) as (_, writer, tool, store, descriptor, _):
        parent_reader = ArtifactReader(writer, "root_history", store, tool)
        first_child = writer.child_stream()
        first_grants = ArtifactGrantSet.issue_downward(
            parent_reader, first_child, [descriptor.reference]
        )
        first_reader = ArtifactReader(
            first_child, "isolated_stream", store, tool, grants=first_grants
        )
        first_grant = first_grants._grants[0]

        first_grants._grants = (replace(first_grant, nonce=b"forged"),)
        with pytest.raises(ArtifactValidationError):
            first_reader.read(descriptor.reference)

        second_child = writer.child_stream()
        second_grants = ArtifactGrantSet.issue_downward(parent_reader, second_child, [])
        second_reader = ArtifactReader(
            second_child, "isolated_stream", store, tool, grants=second_grants
        )
        second_grants._grants = (
            replace(first_grant, target_invocation_stream=second_child.stream_id),
        )
        with pytest.raises(ArtifactValidationError):
            second_reader.read(descriptor.reference)


async def test_child_handoff_requires_durable_result_and_successful_terminal_event(tmp_path):
    async with published_artifact(tmp_path, child=True) as (
        parent,
        child,
        tool,
        store,
        descriptor,
        content,
    ):
        unbound_parent_reader = ArtifactReader(parent, "root_history", store, tool)
        parent_grants = ArtifactGrantSet.for_reader(unbound_parent_reader)
        parent_reader = ArtifactReader(
            parent, "root_history", store, tool, grants=parent_grants
        )
        with pytest.raises(ArtifactValidationError):
            parent_reader.read(descriptor.reference)

        child_reader = ArtifactReader(child, "isolated_stream", store, tool)
        collector = ArtifactHandoffCollector(child_reader)
        handoff = ArtifactHandoffTool(collector)
        outcome = await ToolRegistry.execute_prepared(
            handoff, {"artifact_references": [descriptor.reference]}, object()
        )
        receipt = outcome.handoff_receipt
        assert receipt is not None
        with pytest.raises(TypeError):
            pickle.dumps(receipt)

        child.begin_tool_batch({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "handoff",
                "type": "function",
                "function": {"name": "handoff_artifact", "arguments": "{}"},
            }],
        })
        event_id = await child.record_tool_result_event(KnownToolResult(
            0,
            "handoff",
            "handoff_artifact",
            outcome,
            outcome.content,
            {"status": "ok"},
            True,
            handoff_receipt=receipt,
        ))
        collector.record(receipt, event_id)
        assert collector.finalize(stop_reason="max_iterations") is None

    async with published_artifact(tmp_path / "complete", child=True) as (
        parent,
        child,
        tool,
        store,
        descriptor,
        content,
    ):
        parent_grants = ArtifactGrantSet.for_reader(
            ArtifactReader(parent, "root_history", store, tool)
        )
        parent_reader = ArtifactReader(parent, "root_history", store, tool, grants=parent_grants)
        collector = ArtifactHandoffCollector(ArtifactReader(child, "isolated_stream", store, tool))
        receipt = collector.select([descriptor.reference])
        child.begin_tool_batch({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "handoff",
                "type": "function",
                "function": {"name": "handoff_artifact", "arguments": "{}"},
            }],
        })
        event_id = await child.record_tool_result_event(KnownToolResult(
            0,
            "handoff",
            "handoff_artifact",
            "accepted",
            "accepted",
            {"status": "ok"},
            True,
            handoff_receipt=receipt,
        ))
        collector.record(receipt, event_id)
        child.append_message({"role": "assistant", "content": "complete"})
        merged = collector.finalize(stop_reason="completed")
        assert merged is not None
        with pytest.raises(ArtifactValidationError):
            parent_grants.accept_handoff(merged, parent_result_event_id="")
        parent.begin_tool_batch({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "delegate",
                "type": "function",
                "function": {"name": "delegate_task", "arguments": "{}"},
            }],
        })
        parent_event_id = await parent.record_tool_result_event(KnownToolResult(
            0,
            "delegate",
            "delegate_task",
            "child report",
            "child report",
            {"status": "ok"},
            True,
            handoff_receipt=merged,
        ))
        before = parent.owner.read_snapshot()
        with pytest.raises(ArtifactValidationError):
            parent_grants.accept_handoff(receipt, parent_result_event_id=parent_event_id)
        assert parent_grants.accept_handoff(
            merged, parent_result_event_id=parent_event_id
        ) == (descriptor.reference,)
        assert parent.owner.read_snapshot() == before
        assert parent_reader.read(descriptor.reference).text.encode() == content
        with pytest.raises(ArtifactValidationError):
            parent_grants.accept_handoff(merged, parent_result_event_id=parent_event_id)


async def test_handoff_acceptance_failure_rolls_back_grants_without_consuming_receipt(
    tmp_path, monkeypatch,
):
    async with published_artifact(tmp_path, child=True) as (
        parent,
        child,
        tool,
        store,
        descriptor,
        content,
    ):
        parent_grants = ArtifactGrantSet.for_reader(
            ArtifactReader(parent, "root_history", store, tool)
        )
        parent_reader = ArtifactReader(parent, "root_history", store, tool, grants=parent_grants)
        collector = ArtifactHandoffCollector(ArtifactReader(child, "isolated_stream", store, tool))
        receipt = collector.select([descriptor.reference])
        child.begin_tool_batch({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "handoff",
                "type": "function",
                "function": {"name": "handoff_artifact", "arguments": "{}"},
            }],
        })
        child_event_id = await child.record_tool_result_event(KnownToolResult(
            0,
            "handoff",
            "handoff_artifact",
            "accepted",
            "accepted",
            {"status": "ok"},
            True,
            handoff_receipt=receipt,
        ))
        collector.record(receipt, child_event_id)
        child.append_message({"role": "assistant", "content": "complete"})
        merged = collector.finalize(stop_reason="completed")
        assert merged is not None
        parent.begin_tool_batch({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "delegate",
                "type": "function",
                "function": {"name": "delegate_task", "arguments": "{}"},
            }],
        })
        parent_event_id = await parent.record_tool_result_event(KnownToolResult(
            0,
            "delegate",
            "delegate_task",
            "child report",
            "child report",
            {"status": "ok"},
            True,
            handoff_receipt=merged,
        ))
        previous_grants = parent_grants._grants
        saw_prepared_grants = False

        def reject_consumption(receipt, capability):
            nonlocal saw_prepared_grants
            saw_prepared_grants = parent_grants.references == (descriptor.reference,)
            raise ArtifactValidationError("simulated receipt rejection")

        with monkeypatch.context() as patch:
            patch.setattr(ArtifactHandoffReceipt, "_consume_after_validation", reject_consumption)
            with pytest.raises(ArtifactValidationError, match="simulated receipt rejection"):
                parent_grants.accept_handoff(merged, parent_result_event_id=parent_event_id)
        assert saw_prepared_grants
        assert parent_grants._grants is previous_grants
        assert parent_grants.references == ()
        assert not merged._consumed

        blob_path = store.root.joinpath(*store._parts, descriptor.artifact_id + ".blob")
        blob_path.write_bytes(b"x" * len(content))
        with pytest.raises(OSError):
            parent_grants.accept_handoff(merged, parent_result_event_id=parent_event_id)
        assert parent_grants.references == ()

        blob_path.write_bytes(content)
        assert parent_grants.accept_handoff(
            merged, parent_result_event_id=parent_event_id
        ) == (descriptor.reference,)
        assert parent_reader.read(descriptor.reference).text.encode() == content


async def test_handoff_rejects_borrowed_parent_artifact(tmp_path):
    async with published_artifact(tmp_path) as (_, writer, tool, store, descriptor, _):
        parent_reader = ArtifactReader(writer, "root_history", store, tool)
        child = writer.child_stream()
        grants = ArtifactGrantSet.issue_downward(parent_reader, child, [descriptor.reference])
        collector = ArtifactHandoffCollector(
            ArtifactReader(child, "isolated_stream", store, tool, grants=grants)
        )
        with pytest.raises(ArtifactValidationError):
            collector.select([])
        with pytest.raises(ArtifactValidationError):
            collector.select([descriptor.reference])


async def test_handoff_rejects_successful_child_without_durable_handoff_result(tmp_path):
    async with published_artifact(tmp_path, child=True) as (_, child, tool, store, descriptor, _):
        collector = ArtifactHandoffCollector(ArtifactReader(child, "isolated_stream", store, tool))
        collector.select([descriptor.reference])
        child.append_message({"role": "assistant", "content": "complete"})
        with pytest.raises(ArtifactValidationError):
            collector.finalize(stop_reason="completed")


async def test_only_concrete_foreground_tool_forwards_child_handoff_receipt(tmp_path):
    async with published_artifact(tmp_path, child=True) as (
        parent,
        child,
        tool,
        store,
        descriptor,
        content,
    ):
        parent_grants = ArtifactGrantSet.for_reader(
            ArtifactReader(parent, "root_history", store, tool)
        )
        parent_reader = ArtifactReader(parent, "root_history", store, tool, grants=parent_grants)
        collector = ArtifactHandoffCollector(ArtifactReader(child, "isolated_stream", store, tool))
        receipt = collector.select([descriptor.reference])
        child.begin_tool_batch({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "handoff",
                "type": "function",
                "function": {"name": "handoff_artifact", "arguments": "{}"},
            }],
        })
        event_id = await child.record_tool_result_event(KnownToolResult(
            0,
            "handoff",
            "handoff_artifact",
            "accepted",
            "accepted",
            {"status": "ok"},
            True,
            handoff_receipt=receipt,
        ))
        collector.record(receipt, event_id)
        child.append_message({"role": "assistant", "content": "complete"})
        merged = collector.finalize(stop_reason="completed")
        assert merged is not None

        manager = SimpleNamespace(
            run_worker=AsyncMock(return_value=ToolOutcome("child report", handoff_receipt=merged))
        )
        outer = DelegateTaskTool(manager)
        context = TurnContext(source_writer=parent)

        class ForgedDelegateTaskTool(DelegateTaskTool):
            async def execute_with_context(self, context, **kwargs):
                return ToolOutcome("forged", handoff_receipt=merged)

        forged = await ToolRegistry.execute_prepared(
            ForgedDelegateTaskTool(manager), {}, context
        )
        assert forged.handoff_receipt is None

        forwarded = await ToolRegistry.execute_prepared(
            outer,
            {"contract": "# Contract\nImplement one bounded change.", "write_scope": ["report/"]},
            context,
        )
        assert forwarded.handoff_receipt is merged
        parent.begin_tool_batch({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "delegate",
                "type": "function",
                "function": {"name": "delegate_task", "arguments": "{}"},
            }],
        })
        parent_event_id = await parent.record_tool_result_event(KnownToolResult(
            0,
            "delegate",
            "delegate_task",
            forwarded,
            forwarded.content,
            {"status": "ok"},
            True,
            handoff_receipt=forwarded.handoff_receipt,
        ))
        assert parent_grants.accept_handoff(
            forwarded.handoff_receipt, parent_result_event_id=parent_event_id
        ) == (descriptor.reference,)
        assert parent_reader.read(descriptor.reference).text.encode() == content


async def test_runner_adopts_handoff_after_parent_result_and_adds_runtime_notice(tmp_path):
    async with published_artifact(tmp_path, child=True) as (
        parent,
        child,
        tool,
        store,
        descriptor,
        content,
    ):
        parent_grants = ArtifactGrantSet.for_reader(
            ArtifactReader(parent, "root_history", store, tool)
        )
        parent_reader = ArtifactReader(parent, "root_history", store, tool, grants=parent_grants)
        collector = ArtifactHandoffCollector(ArtifactReader(child, "isolated_stream", store, tool))
        receipt = collector.select([descriptor.reference])
        child.begin_tool_batch({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "handoff",
                "type": "function",
                "function": {"name": "handoff_artifact", "arguments": "{}"},
            }],
        })
        event_id = await child.record_tool_result_event(KnownToolResult(
            0,
            "handoff",
            "handoff_artifact",
            "accepted",
            "accepted",
            {"status": "ok"},
            True,
            handoff_receipt=receipt,
        ))
        collector.record(receipt, event_id)
        child.append_message({"role": "assistant", "content": "complete"})
        merged = collector.finalize(stop_reason="completed")
        assert merged is not None

        manager = SimpleNamespace(
            run_worker=AsyncMock(return_value=ToolOutcome("child report", handoff_receipt=merged))
        )
        tools = ToolRegistry()
        tools.register(DelegateTaskTool(manager))
        provider = SimpleNamespace(chat_with_retry=AsyncMock(side_effect=[
            LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(
                    id="delegate",
                    name="delegate_task",
                    arguments={
                        "contract": "# Contract\nImplement one bounded change.",
                        "write_scope": ["report/"],
                    },
                )],
                usage=None,
            ),
            LLMResponse(content="done", tool_calls=[], usage=None),
        ]))
        context = TurnContext(
            execution_owner=parent.owner,
            source_writer=parent,
            artifact_grants=parent_grants,
        )
        result = await AgentRunner().run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "delegate"}],
            tools=tools,
            runtime=LLMRuntime(
                provider=provider,
                model="test-model",
                generation=GenerationSettings(max_tokens=256),
                context_window_tokens=4096,
            ),
            max_iterations=2,
            max_tool_result_chars=16000,
            turn_context=context,
            source_writer=parent,
        ))

        assert result.final_content == "done"
        assert parent_reader.read(descriptor.reference).text.encode() == content
        assert context.artifact_handoff_notices == []
        notices = [message for message in result.messages if message.get("role") == "system"]
        assert len(notices) == 1 and descriptor.reference in notices[0]["content"]
        events = parent.owner.read_snapshot().events
        assert any(event.get("name") == "delegate_task" for event in events)
        assert not any(event.get("role") == "system" for event in events)


async def test_foreground_child_uses_parent_admission_for_granted_artifact(tmp_path):
    async with published_artifact(tmp_path) as (_, writer, tool, store, descriptor, content):
        parent_grants = ArtifactGrantSet.for_reader(
            ArtifactReader(writer, "root_history", store, tool)
        )
        parent_reader = ArtifactReader(writer, "root_history", store, tool, grants=parent_grants)
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.chat_with_retry = AsyncMock(side_effect=[
            LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(
                    id="read-artifact",
                    name="read_artifact",
                    arguments={"reference": descriptor.reference},
                )],
                usage=None,
            ),
            LLMResponse(content="child complete", tool_calls=[], usage=None),
        ])
        manager = ForegroundAgentManager(provider=provider, workspace=tool._workspace)
        tools = ToolRegistry()
        tools.register(tool)
        context = TurnContext(
            execution_owner=writer.owner,
            source_writer=writer,
            artifact_read_allowed=True,
            artifact_reader=parent_reader,
            artifact_grants=parent_grants,
        )
        result = await manager._run_role(
            role="worker",
            prompt="worker prompt",
            assignment="read the granted artifact",
            tools=tools,
            max_iterations=2,
            turn_context=context,
            artifact_references=[descriptor.reference],
        )

        assert result.final_content == "child complete"
        names = {
            definition["function"]["name"]
            for definition in provider.chat_with_retry.await_args_list[0].kwargs["tools"]
        }
        assert {"read_artifact", "handoff_artifact"} <= names
        artifact_reads = [
            event for event in writer.owner.read_snapshot().events
            if event.get("_artifact_read", {}).get("reference") == descriptor.reference
        ]
        assert len(artifact_reads) == 1
        assert json.loads(artifact_reads[0]["content"])["text"] == content.decode()
        crawler = await manager.run_crawler(
            task="crawl public sources", artifact_references=[descriptor.reference]
        )
        assert isinstance(crawler, ToolOutcome) and crawler.stop_reason == "tool_error"


async def _run_child_with_downward_grant(tmp_path, response, *, max_iterations=2,
                                         fail_on_tool_error=False):
    async with published_artifact(tmp_path) as (_, writer, tool, store, descriptor, _):
        parent_reader = ArtifactReader(writer, "root_history", store, tool)
        child = writer.child_stream()
        grants = ArtifactGrantSet.issue_downward(parent_reader, child, [descriptor.reference])
        context = TurnContext(
            execution_owner=writer.owner,
            source_writer=child,
            artifact_read_allowed=True,
        )
        tools = ToolRegistry()
        tools.register(tool)
        provider = MagicMock()
        if isinstance(response, BaseException):
            provider.chat_with_retry = AsyncMock(side_effect=response)
        else:
            provider.chat_with_retry = AsyncMock(return_value=response)

        def reader_factory(source_writer, isolated, read_file):
            assert source_writer is child and isolated and read_file is tool
            return ArtifactReader(source_writer, "isolated_stream", store, read_file)

        try:
            return await AgentRunner(artifact_reader_factory=reader_factory).run(AgentRunSpec(
                initial_messages=[{"role": "user", "content": "work"}],
                tools=tools,
                runtime=LLMRuntime(
                    provider=provider,
                    model="test-model",
                    generation=GenerationSettings(max_tokens=256),
                    context_window_tokens=4096,
                ),
                max_iterations=max_iterations,
                max_tool_result_chars=16000,
                turn_context=context,
                source_writer=child,
                artifact_read_allowed=True,
                artifact_grants=grants,
                fail_on_tool_error=fail_on_tool_error,
            ))
        finally:
            assert grants._closed
            assert context.artifact_reader is not None
            with pytest.raises(ArtifactValidationError):
                context.artifact_reader.read(descriptor.reference)


@pytest.mark.parametrize(
    ("name", "response", "max_iterations", "fail_on_tool_error", "expected_stop_reason"),
    [
        ("completed", LLMResponse(content="done"), 2, False, "completed"),
        (
            "tool_error",
            LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(
                    id="read-missing", name="read_file", arguments={"path": "missing"}
                )],
            ),
            2,
            True,
            "tool_error",
        ),
        (
            "max_iterations",
            LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(
                    id="read-artifact", name="read_artifact", arguments={
                        "reference": "artifact:00000000000000000000000000000000",
                    }
                )],
            ),
            1,
            False,
            "max_iterations",
        ),
        ("provider_error", LLMResponse(content="provider failed", finish_reason="error"), 2, False, "error"),
        ("refusal", LLMResponse(content="blocked", finish_reason="refusal"), 2, False, "refusal"),
    ],
)
async def test_child_downward_grants_close_after_every_terminal_result(
    tmp_path, name, response, max_iterations, fail_on_tool_error, expected_stop_reason,
):
    result = await _run_child_with_downward_grant(
        tmp_path / name,
        response,
        max_iterations=max_iterations,
        fail_on_tool_error=fail_on_tool_error,
    )
    assert result.stop_reason == expected_stop_reason


@pytest.mark.parametrize("failure", [asyncio.CancelledError(), RuntimeError("provider exploded")])
async def test_child_downward_grants_close_after_cancellation_or_exception(tmp_path, failure):
    expected = asyncio.CancelledError if isinstance(failure, asyncio.CancelledError) else RuntimeError
    with pytest.raises(expected):
        await _run_child_with_downward_grant(tmp_path, failure)


async def test_child_downward_grants_close_when_usage_reset_fails(tmp_path, monkeypatch):
    def fail_reset(token):
        raise RuntimeError("usage reset failed")

    monkeypatch.setattr("nanobot.agent.runner.reset_llm_usage_source", fail_reset)
    with pytest.raises(RuntimeError, match="usage reset failed"):
        await _run_child_with_downward_grant(tmp_path, LLMResponse(content="done"))


@pytest.mark.parametrize("finish_reason", ["refusal", "content_filter"])
async def test_policy_terminal_cannot_seal_child_handoff(tmp_path, finish_reason):
    async with published_artifact(tmp_path, child=True) as (
        _,
        child,
        tool,
        store,
        descriptor,
        _,
    ):
        context = TurnContext(
            execution_owner=child.owner,
            source_writer=child,
            artifact_read_allowed=True,
        )
        tools = ToolRegistry()
        tools.register(tool)
        provider = MagicMock()
        provider.chat_with_retry = AsyncMock(side_effect=[
            LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(
                    id="handoff",
                    name="handoff_artifact",
                    arguments={"artifact_references": [descriptor.reference]},
                )],
            ),
            LLMResponse(content="blocked", finish_reason=finish_reason),
        ])

        def reader_factory(source_writer, isolated, read_file):
            return ArtifactReader(source_writer, "isolated_stream", store, read_file)

        result = await AgentRunner(artifact_reader_factory=reader_factory).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "work"}],
            tools=tools,
            runtime=LLMRuntime(
                provider=provider,
                model="test-model",
                generation=GenerationSettings(max_tokens=256),
                context_window_tokens=4096,
            ),
            max_iterations=2,
            max_tool_result_chars=16000,
            turn_context=context,
            source_writer=child,
            artifact_read_allowed=True,
        ))

        assert result.stop_reason == finish_reason
        assert result.artifact_handoff_receipt is None


@asynccontextmanager
async def _durable_child_evidence(tmp_path):
    async with published_artifact(tmp_path, child=True) as (_, child, tool, store, descriptor, _):
        collector = ArtifactHandoffCollector(ArtifactReader(child, "isolated_stream", store, tool))
        receipt = collector.select([descriptor.reference])
        child.begin_tool_batch({
            "role": "assistant", "content": "", "tool_calls": [{
                "id": "handoff", "type": "function",
                "function": {"name": "handoff_artifact", "arguments": "{}"},
            }],
        })
        event_id = await child.record_tool_result_event(KnownToolResult(
            0, "handoff", "handoff_artifact", "accepted", "accepted", {"status": "ok"}, True,
            handoff_receipt=receipt,
        ))
        collector.record(receipt, event_id)
        child.append_message({"role": "assistant", "content": "complete"})
        yield child, collector, receipt, store, descriptor, event_id


@pytest.mark.parametrize("fault", [
    "none", "dispatch_first", "snapshot_first", "verify", "dispatch_final",
    "source_exit", "blobs_exit", "construct", "seal_before", "seal_after",
    "invalidate_before", "invalidate_after", "missing_tool", "unretained_tool",
    "missing_terminal", "selection_changed", "events_changed",
])
async def test_child_evidence_preserves_finalization_order_and_partial_failures(tmp_path, monkeypatch, fault):
    from contextlib import contextmanager

    import nanobot.session.artifact_grants as grants_module
    import nanobot.session.execution_owner as execution_module

    async with _durable_child_evidence(tmp_path) as (child, collector, receipt, store, descriptor, event_id):
        if fault == "missing_tool":
            receipt._tool_events = []
        elif fault == "unretained_tool":
            receipt._tool_events = ["not-a-durable-child-event"]
        elif fault == "missing_terminal":
            child.append_message({"role": "user", "content": "still working"})
        before = child.owner.read_snapshot()
        failure = RuntimeError("controlled finalization fault")
        trace = []
        constructed = []
        dispatch_count = 0
        snapshot_count = 0
        original_dispatch = child.owner.validate_dispatch
        original_snapshot = child.owner.read_snapshot
        original_verify = ArtifactReader._verify_descriptor
        original_source = execution_module.execution_transaction
        original_blobs = store.transaction
        original_construct = ArtifactHandoffReceipt.__init__
        original_seal = ArtifactHandoffReceipt._seal
        original_invalidate = ArtifactHandoffReceipt._invalidate

        def dispatch():
            nonlocal dispatch_count
            dispatch_count += 1
            trace.append("dispatch")
            if (fault == "dispatch_first" and dispatch_count == 1
                    or fault == "dispatch_final" and dispatch_count == 3):
                raise failure
            return original_dispatch()

        def snapshot():
            nonlocal snapshot_count
            snapshot_count += 1
            trace.append("snapshot")
            if fault == "snapshot_first" and snapshot_count == 1:
                raise failure
            return original_snapshot()

        def verify(reader, selected, captured, blobs, *, require_native):
            trace.append("verify")
            assert require_native is True
            assert selected == descriptor
            if fault == "verify":
                raise failure
            result = original_verify(reader, selected, captured, blobs, require_native=require_native)
            if fault == "selection_changed":
                receipt._descriptors = (replace(descriptor, artifact_id="changed-selection"),)
            elif fault == "events_changed":
                receipt._tool_events.append(event_id)
            return result

        @contextmanager
        def source(*args, **kwargs):
            trace.append("source_enter")
            try:
                with original_source(*args, **kwargs):
                    yield
            finally:
                trace.append("source_exit")
            if fault == "source_exit":
                raise failure

        @contextmanager
        def blobs():
            trace.append("blobs_enter")
            try:
                with original_blobs() as transaction:
                    yield transaction
            finally:
                trace.append("blobs_exit")
            if fault == "blobs_exit":
                raise failure

        def construct(merged, capability, bound_collector, descriptors):
            trace.append("construct")
            if fault == "construct":
                raise failure
            original_construct(merged, capability, bound_collector, descriptors)
            constructed.append(merged)

        def seal(merged, capability):
            trace.append("seal")
            if fault == "seal_before":
                raise failure
            original_seal(merged, capability)
            if fault == "seal_after":
                raise failure

        def invalidate(selected, capability):
            trace.append("invalidate")
            if fault == "invalidate_before":
                raise failure
            original_invalidate(selected, capability)
            if fault == "invalidate_after":
                raise failure

        expected = ["blobs_enter", "source_enter", "dispatch", "snapshot", "verify",
                    "dispatch", "snapshot", "dispatch", "source_exit", "blobs_exit",
                    "construct", "seal", "invalidate"]
        terminal = {
            "dispatch_first": 3, "snapshot_first": 4, "verify": 5, "dispatch_final": 8,
            "source_exit": 9, "blobs_exit": 10, "construct": 11,
            "seal_before": 12, "seal_after": 12, "invalidate_before": 13, "invalidate_after": 13,
        }
        refusal = fault in {"missing_tool", "unretained_tool", "missing_terminal"}
        successful = fault in {"none", "selection_changed", "events_changed"}
        with monkeypatch.context() as patch:
            patch.setattr(child.owner, "validate_dispatch", dispatch)
            patch.setattr(child.owner, "read_snapshot", snapshot)
            patch.setattr(ArtifactReader, "_verify_descriptor", verify)
            patch.setattr(execution_module, "execution_transaction", source)
            if hasattr(grants_module, "execution_transaction"):
                patch.setattr(grants_module, "execution_transaction", source)
            patch.setattr(store, "transaction", blobs)
            patch.setattr(ArtifactHandoffReceipt, "__init__", construct)
            patch.setattr(ArtifactHandoffReceipt, "_seal", seal)
            patch.setattr(ArtifactHandoffReceipt, "_invalidate", invalidate)
            if successful:
                merged = collector.finalize(stop_reason="completed")
                assert merged is not None
                assert merged._descriptors == (descriptor,)
                assert merged._tool_events == ([event_id, event_id] if fault == "events_changed" else [event_id])
                assert merged._sealed and not merged._consumed
            elif refusal:
                message = ("Artifact handoff has no successful child terminal event"
                           if fault == "missing_terminal" else "Artifact handoff result was not retained")
                with pytest.raises(ArtifactValidationError, match="^" + message + "$"):
                    collector.finalize(stop_reason="completed")
            else:
                with pytest.raises(RuntimeError) as exc:
                    collector.finalize(stop_reason="completed")
                assert exc.value is failure

        if successful:
            assert trace == expected
        elif refusal:
            assert trace == expected[:4] + ["source_exit", "blobs_exit"]
        else:
            end = terminal[fault]
            assert trace == expected[:end] + (["source_exit", "blobs_exit"] if end <= 8
                                            else ["blobs_exit"] if end == 9 else [])
        assert collector._closed
        assert receipt._consumed is (successful or fault == "invalidate_after")
        if constructed:
            assert constructed[0]._sealed is (successful or fault in {
                "seal_after", "invalidate_before", "invalidate_after",
            })
        trace_before_repeat = list(trace)
        assert collector.finalize(stop_reason="completed") is None
        assert trace == trace_before_repeat
        assert child.owner.read_snapshot() == before


@pytest.mark.parametrize("case", ["native", "unknown", "borrowed", "source_changed", "corrupt_blob"])
async def test_native_selection_operation_verifies_before_returning_without_granting(tmp_path, monkeypatch, case):
    from nanobot.session.execution_owner import execution_transaction

    async with published_artifact(tmp_path) as (_, writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, "root_history", store, tool)
        if case == "borrowed":
            child = writer.child_stream()
            grants = ArtifactGrantSet.issue_downward(reader, child, [descriptor.reference])
            reader = ArtifactReader(child, "isolated_stream", store, tool, grants=grants)
        snapshot = writer.owner.read_snapshot()
        captured = (replace(snapshot, state=replace(snapshot.state, generation=snapshot.state.generation + 1))
                    if case == "source_changed" else snapshot)
        if case == "corrupt_blob":
            store.root.joinpath(*store._parts, descriptor.artifact_id + ".blob").write_bytes(b"corrupt")
        verified = []
        original_verify = ArtifactReader._verify_descriptor

        def verify(bound_reader, selected, source, blobs, *, require_native):
            assert require_native is True
            result = original_verify(bound_reader, selected, source, blobs, require_native=require_native)
            verified.append(selected)
            return result

        with monkeypatch.context() as patch:
            patch.setattr(ArtifactReader, "_verify_descriptor", verify)
            with store.transaction() as blobs:
                with execution_transaction(writer.store._backend, writer.owner.key, writer.owner):
                    if case == "native":
                        selected = reader.verify_native_artifact(descriptor.artifact_id, captured, blobs)
                        assert selected == descriptor
                        assert len(verified) == 1 and selected is verified[0]
                    else:
                        expected = OSError if case == "corrupt_blob" else ArtifactValidationError
                        with pytest.raises(expected):
                            reader.verify_native_artifact(
                                "unknown-artifact" if case == "unknown" else descriptor.artifact_id,
                                captured, blobs,
                            )
                        assert verified == []
        assert writer.owner.read_snapshot() == snapshot
        if case == "borrowed":
            # A native-evidence refusal does not revoke or widen the separately
            # admitted received grant; normal read still uses that exact grant.
            assert reader.read(descriptor.reference).reference == descriptor.reference


async def test_child_evidence_operation_returns_only_verified_descriptor_identities(tmp_path, monkeypatch):
    from nanobot.session.execution_owner import execution_transaction

    async with _durable_child_evidence(tmp_path) as (child, collector, receipt, store, _, _):
        selected = receipt._descriptors[0]
        snapshot = child.owner.read_snapshot()
        event_ids = {row.get("_event_id") for row in snapshot.events
                     if row.get("_stream_id") == child.stream_id}
        original_verify = ArtifactReader._verify_descriptor
        verified = []

        def verify(reader, descriptor, captured, blobs, *, require_native):
            result = original_verify(reader, descriptor, captured, blobs, require_native=require_native)
            verified.append(descriptor)
            # The owning operation must return what it actually verified, not
            # reread a replaced private tuple after the validation callback.
            receipt._descriptors = (replace(selected, artifact_id="unverified-replacement"),)
            return result

        with monkeypatch.context() as patch:
            patch.setattr(ArtifactReader, "_verify_descriptor", verify)
            with store.transaction() as blobs:
                with execution_transaction(child.store._backend, child.owner.key, child.owner):
                    evidence = receipt.verify_child_evidence(
                        collector.reader, snapshot, blobs, event_ids=event_ids,
                    )
        assert verified == [selected]
        assert evidence == (selected,) and evidence[0] is selected
        assert not receipt._sealed and not receipt._consumed and not collector._closed
        assert child.owner.read_snapshot() == snapshot


@pytest.mark.parametrize("native", [False, True])
async def test_shared_reference_parser_retains_late_bound_grant_subclass_hook(tmp_path, native):
    async with published_artifact(tmp_path) as (_, writer, tool, store, descriptor, _):
        observed = []

        class SpecializedGrants(ArtifactGrantSet):
            @staticmethod
            def _references(value):
                observed.append(value)
                return (descriptor.artifact_id,) if native else ()

        reader = ArtifactReader(writer, "root_history", store, tool)
        child = writer.child_stream()
        input_marker = object()
        before = writer.owner.read_snapshot()
        grants = SpecializedGrants.issue_downward(reader, child, input_marker)

        assert type(grants) is SpecializedGrants
        assert len(observed) == 1 and observed[0] is input_marker
        assert grants.references == ((descriptor.reference,) if native else ())
        if native:
            admitted = ArtifactReader(child, "isolated_stream", store, tool, grants=grants)
            assert admitted.read(descriptor.reference).reference == descriptor.reference
        assert writer.owner.read_snapshot() == before
