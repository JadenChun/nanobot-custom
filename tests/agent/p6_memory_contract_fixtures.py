"""Synthetic inputs at real Q1 summary, source and receipt ownership boundaries.

No productive guard or recovery proof is mocked. Provider fixtures replace only
transport; all durable files live under the caller's isolated test directory.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from nanobot.agent.context_governance import (
    ContextGovernanceConfig,
    ContextGovernor,
    ModelRequestState,
    SourceCompactionState,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse, ProviderCallContext
from nanobot.providers.conversation_state import ProviderConversationStateController
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.request_budget import ProviderRequestLimits
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.source_projection import RuntimeNode, source_nodes
from nanobot.session.source_request import SourceRequestLedger, SourceTranscript
from nanobot.session.turn_writer import SourceTurnWriter
from nanobot.utils.llm_runtime import LLMRuntime


class SummaryProvider(LLMProvider):
    def __init__(self, responses):
        super().__init__(provider_name="summary-fixture")
        self.generation = GenerationSettings(max_tokens=256, temperature=0.25, reasoning_effort="high")
        self.responses = deque(responses)
        self.calls = 0
        self.prompts = []
        self.kwargs = []
        self.contexts = []

    def get_default_model(self):
        return "test-model"

    async def chat(self, messages, **kwargs):
        self.calls += 1
        self.prompts.append(deepcopy(messages))
        self.kwargs.append(deepcopy(kwargs))
        response = self.responses.popleft()
        if isinstance(response, BaseException):
            raise response
        return response

    async def chat_with_context(self, *, provider_context, **kwargs):
        self.contexts.append(provider_context)
        return await self.chat(**kwargs)


def summary_runtime(provider, *, window=8192):
    return LLMRuntime.capture(provider, "test-model", context_window_tokens=window)


def file_snapshot(root: Path):
    """Include directory entries, bytes and mode, not just absent MEMORY/HISTORY."""
    return {
        str(path.relative_to(root)): (path.is_dir(), None if path.is_dir() else path.read_bytes(),
                                      path.stat().st_mode)
        for path in sorted(root.rglob("*"))
    }


def source_store(root: Path):
    backend = JsonlSessionStore(root)
    session = Session("test:compact")
    backend.save(session)
    backend.source.upgrade_source_session(
        session.key, expected_revision=session.revision,
        expected_generation=session.generation, expected_incarnation=session.incarnation,
    )
    return backend


def source_state(writer, provider, *, window=12000, eligible_count=None):
    nodes = source_nodes(writer.owner.read_snapshot().events)
    active = SourceTranscript((RuntimeNode("policy", {"role": "system", "content": "policy"}), *nodes))
    if eligible_count is None:
        eligible_count = len(nodes) - 1
    raw = active.messages()
    ledger = SourceRequestLedger(active, initial_eligible_ids=frozenset(
        node.event_id for node in nodes[:eligible_count]
    ))
    return ModelRequestState(
        ContextGovernanceConfig(provider, "test-model", ToolRegistry(), None, "test:compact", 16000,
                                context_window_tokens=window, max_tokens=256, preserve_source=True),
        ProviderConversationStateController(provider=provider, model="test-model", messages=raw,
                                            session_id="test:compact"),
        source=ledger,
        source_compaction=SourceCompactionState(writer, summary_runtime(provider, window=window), raw, None),
    )


async def prepare_source(state):
    return await ContextGovernor().prepare_request(
        state, state.source.transcript.messages(), source_view=state.source.transcript,
        transcript=state.source_compaction.raw_messages, tool_definitions=None,
    )


def receipt_layout(root: Path, *, cursor=0, start=0, end=2, base=b"base memory\n",
                   output="# Recovered\n", marker="receipt-marker"):
    """Real authenticated layout; chunk digest is the exact legacy receipt digest."""
    root.mkdir(parents=True, exist_ok=True)
    workspace, sessions = root / "workspace", root / "sessions"
    workspace.mkdir(mode=0o700)
    sessions.mkdir(mode=0o700)
    state = workspace / ".nanobot"
    state.mkdir(mode=0o700)
    workspace_id = "a" * 32
    identity = state / "workspace-id"
    identity.write_text(workspace_id + "\n")
    identity.chmod(0o600)
    namespace = sessions / workspace_id
    namespace.mkdir(mode=0o700)
    binding = namespace / ".workspace"
    binding.write_text(str(workspace.resolve()) + "\n")
    binding.chmod(0o600)
    key = "cli:receipt"
    events = [
        {"role": "user", "content": "one", "_event_id": "event-one"},
        {"role": "assistant", "content": "two", "_event_id": "event-two"},
        {"role": "user", "content": "three", "_event_id": "event-three"},
    ]
    header = {
        "_type": "metadata", "key": key, "created_at": "2026-09-25T00:00:00",
        "updated_at": "2026-09-25T00:01:00", "revision": 7, "generation": 3,
        "incarnation": "receipt-incarnation", "last_archived": cursor,
        "last_consolidated": cursor, "metadata": {"source_schema_version": 1},
    }
    source = namespace / f"{JsonlSessionStore.storage_key(key)}.jsonl"
    source.write_bytes(b"".join((json.dumps(row) + "\n").encode() for row in [header, *events]))
    source.chmod(0o600)
    memory = workspace / "memory"
    memory.mkdir(mode=0o700)
    (memory / "HISTORY.md").write_bytes(b"[2026-09-24 12:00] Prior note.\n\n")
    (memory / "MEMORY.md").write_bytes(base)
    chunk = json.dumps(events[start:end], ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), default=str).encode()
    receipt = {
        "version": 1, "chunk_digest": hashlib.sha256(chunk).hexdigest(),
        "history_marker_id": marker, "history_entry": "[2026-01-01] recovered summary",
        "memory_output": output, "base_memory_digest": hashlib.sha256(base).hexdigest(),
        "output_memory_digest": hashlib.sha256(output.encode()).hexdigest(),
        "reference_only": False, "created_at": "2026-09-25T12:00:00Z",
        "session_key": key, "start_offset": start, "end_offset": end, "session_generation": 3,
        "metadata_sentinel": {"unknown": ["preserve", {"exact": True}]},
    }
    pending = memory / ".consolidation.pending.json"
    pending.write_bytes(json.dumps(receipt, sort_keys=True).encode())
    pending.chmod(0o600)
    return workspace, sessions, source, pending


def provider_limits(total):
    return ProviderCallContext(request_limits=ProviderRequestLimits(
        total_tokens=total, completion_reserve_tokens=1, safety_buffer_tokens=0,
    ))


async def fitting_prefix(root, rows, boundary):
    """Construct an exact synthetic auxiliary boundary, never a production cap."""
    from nanobot.agent.memory import Consolidator

    backend = source_store(root)
    provider = SummaryProvider([LLMResponse(content="retained facts")])

    def count(messages, tools, model):
        if messages[0]["content"].startswith("Summarize historical evidence"):
            payload = json.loads(messages[1]["content"])
            return len(payload["events"]) * 40 + len(payload["previous_summary"] or ""), "exact fixture"
        return 100 + len(messages) * 60, "exact fixture"

    provider.estimate_prompt_tokens = count
    async with acquire_execution(backend, "test:compact", run_id="run") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="run")
        for row in rows:
            writer.append_message(row)
        before = owner.read_snapshot()
        state = source_state(writer, provider, eligible_count=len(rows))
        # Allowance is 2048 for this main-request fixture; safety remains 1024.
        state.source_compaction.runtime = summary_runtime(provider, window=boundary * 40 + 2048 + 1024)
        attempt = ContextGovernor().capture_source_compaction(state, state.source.transcript,
                                                             tool_definitions=None)
        selected = attempt.candidate.summary_evidence(boundary)
        assert [node.event_id for node in selected.nodes] == [row["_event_id"] for row in before.events[:boundary]]
        messages, context = await prepare_source(state)
        after = owner.read_snapshot()
        payload = json.loads(provider.prompts[0][1]["content"])
        assert provider.calls == 1
        assert payload == {"previous_summary": None,
                           "events": [node.message for node in source_nodes(before.events[:boundary])]}
        assert after.checkpoint["boundary"] == boundary
        assert after.events == before.events
        assert len(after.events) == len(rows)
        assert messages[-len(rows[boundary:]):] == [node.message for node in source_nodes(before.events[boundary:])]
        assert any(message.get("content") == rows[next(i for i in range(len(rows) - 1, -1, -1)
                   if rows[i]["role"] == "user")]["content"] for message in messages)
        assert provider.kwargs[0]["tools"] is None
        assert provider.contexts[0].conversation_state is None
        assert count(provider.prompts[0], None, "test-model")[0] + 2048 + 1024 == state.source_compaction.runtime.context_window_tokens
        larger = Consolidator.source_summary_messages(
            [node.message for node in source_nodes(before.events[:boundary + 1])], None,
        )
        assert count(larger, None, "test-model")[0] + 2048 + 1024 > state.source_compaction.runtime.context_window_tokens
        assert count(messages, None, "test-model")[0] <= ContextGovernor.input_budget(state.config) * 0.6
        assert context is not None and not state.source.has_pending_requests
        assert state.source_compaction.raw_messages == state.source.transcript.messages()
        return payload["events"], after


def concrete_provider(*, content="fitted"):
    """Only SDK create/client are synthetic; finalized body validation stays real."""
    calls = []

    async def chunks():
        yield SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=content, tool_calls=None, reasoning_content=None),
            finish_reason=None,
        )], usage=None)
        yield SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=None, tool_calls=None, reasoning_content=None),
            finish_reason="stop",
        )], usage=None)

    async def create(**kwargs):
        calls.append("stream" if kwargs.get("stream") else "chat")
        if kwargs.get("stream"):
            return chunks()
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=None), finish_reason="stop",
        )], usage=None)

    provider = OpenAICompatProvider(api_key="fixture", default_model="test-model",
                                    provider_name="q1-finalized-fixture")
    provider._ensure_client = AsyncMock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=create))),
    ))
    return provider, calls
