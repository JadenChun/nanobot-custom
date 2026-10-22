"""Artifact text access rechecks source scope, policy and private blob integrity."""

import hashlib
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from nanobot.agent.tools.artifact import ReadArtifactTool
from nanobot.agent.tools.execution import KnownToolResult
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.session.artifact import ArtifactValidationError, parse_artifact_descriptor
from nanobot.session.artifact_reader import ArtifactReader
from nanobot.session.artifact_store import ArtifactStorageError, PrivateArtifactStore
from nanobot.session.execution_owner import ExecutionOwnershipError, acquire_execution
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.source import SourceValidationError
from nanobot.session.source_store import SourceSessionStore
from nanobot.session.turn_writer import SourceTurnWriter
from tests.session.test_artifact import fixture


@asynccontextmanager
async def setup_read(tmp_path, text='evidence 中文 🌟', *, child=False):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'file').write_text(text)
    backend = JsonlSessionStore(workspace)
    session = Session('test:read')
    backend.save(session)
    backend.source.upgrade_source_session(session.key, expected_revision=session.revision,
        expected_generation=session.generation, expected_incarnation=session.incarnation)
    async with acquire_execution(backend, session.key, run_id='run') as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id='run')
        if child:
            writer = writer.child_stream()
        tool = ReadFileTool(workspace=workspace, allowed_dir=workspace)
        outcome = await ToolRegistry.execute_prepared(tool, {'path': 'file'}, SimpleNamespace(lock_owner=None))
        assert outcome.producer_receipt is not None, outcome.content
        writer.begin_tool_batch({'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'read', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}]})
        await writer.record_tool_result(KnownToolResult(0, 'read', 'read_file', outcome, outcome.content,
            {'status': 'ok'}, True, outcome.producer_receipt))
        snapshot = owner.read_snapshot()
        event = snapshot.events[-1]
        raw, _ = fixture()
        content = event['content'].encode()
        raw.update(workspace_id=backend.sessions_dir.name, session_key=session.key,
            incarnation=snapshot.state.incarnation, generation=snapshot.state.generation,
            event_id=event['_event_id'], execution_id=event['_execution_id'], stream_id=writer.stream_id,
            byte_length=len(content), sha256=hashlib.sha256(content).hexdigest(),
            source_access=outcome.producer_receipt.to_dict()['source_access'])
        descriptor = parse_artifact_descriptor(raw)
        store = PrivateArtifactStore(tmp_path.resolve() / 'artifacts', workspace_id=descriptor.workspace_id,
            session_key=session.key, incarnation=descriptor.incarnation, generation=descriptor.generation)
        source = SourceSessionStore(backend, artifact_store=store)
        with store.transaction() as blobs:
            blobs.put(descriptor, content)
            source.commit_artifact_descriptor(session.key, snapshot.state, descriptor, owner=owner, blobs=blobs)
        yield writer, tool, store, descriptor, content


async def test_utf8_ranges_and_historical_text_survive_original_file_change(tmp_path):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, content):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        (tool._workspace / 'file').write_text('different current content')
        for offset in range(len(content) + 3):
            page = reader.read(descriptor.reference, offset=offset, max_bytes=4)
            assert page.text.encode() == content[page.start_offset:page.next_offset]
            assert len(page.text.encode()) <= 4
            assert page.next_offset > offset or page.eof
        position, pieces = 0, []
        while True:
            page = reader.read(descriptor.reference, offset=position, max_bytes=4)
            pieces.append(page.text)
            position = page.next_offset
            if page.eof:
                break
        assert ''.join(pieces).encode() == content


async def test_new_root_stream_can_read_history_but_fresh_and_child_cannot(tmp_path):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, content):
        new = SourceTurnWriter(writer.store, writer.owner, run_id='next')
        assert ArtifactReader(new, 'root_history', store, tool).read(descriptor.reference).text.encode() == content
        for isolated in (new, writer.child_stream()):
            with pytest.raises(ArtifactValidationError):
                ArtifactReader(isolated, 'isolated_stream', store, tool).read(descriptor.reference)


async def test_child_can_read_own_artifact_but_parent_and_sibling_cannot(tmp_path):
    async with setup_read(tmp_path, child=True) as (writer, tool, store, descriptor, content):
        assert ArtifactReader(writer, 'isolated_stream', store, tool).read(descriptor.reference).text.encode() == content
        parent = SourceTurnWriter(writer.store, writer.owner, run_id='run')
        for caller, mode in ((parent, 'root_history'), (parent.child_stream(), 'isolated_stream')):
            with pytest.raises(ArtifactValidationError):
                ArtifactReader(caller, mode, store, tool).read(descriptor.reference)


@pytest.mark.parametrize('failure', ['missing', 'corrupt', 'closed', 'reset', 'workspace', 'unpublished'])
async def test_current_authority_and_integrity_are_required(tmp_path, failure):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        if failure == 'missing':
            next(store.root.rglob('*.blob')).unlink()
        elif failure == 'corrupt':
            next(store.root.rglob('*.blob')).write_bytes(b'corrupt')
        elif failure in {'closed', 'reset'}:
            await writer.owner.quiesce()
            if failure == 'reset':
                writer.store.reset_source_session(writer.owner.key, writer.owner.read_snapshot().state,
                    [], owner=writer.owner)
        elif failure == 'workspace':
            tool._workspace = tmp_path / 'other'
        else:
            descriptor = replace(descriptor, artifact_id='c' * 32)
        with pytest.raises((SourceValidationError, ArtifactStorageError, ExecutionOwnershipError, OSError)):
            reader.read(descriptor.reference)


@pytest.mark.parametrize('offset,max_bytes', [(True, 4), (-1, 4), (0, True), (0, 3), (0, 16385)])
async def test_invalid_ranges_fail_closed(tmp_path, offset, max_bytes):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        with pytest.raises(ArtifactValidationError):
            ArtifactReader(writer, 'root_history', store, tool).read(descriptor.reference,
                                                                  offset=offset, max_bytes=max_bytes)


async def test_tool_returns_trusted_inline_retrieval_marker(tmp_path):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        artifact_tool = ReadArtifactTool(reader)
        outcome = await ToolRegistry.execute_prepared(artifact_tool, {'reference': descriptor.reference}, object())
        assert outcome.retrieval_receipt.matches(outcome.content)
        assert outcome.producer_receipt is None
        assert str(store.root) not in outcome.content
        writer.begin_tool_batch({'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'artifact', 'type': 'function', 'function': {'name': 'read_artifact', 'arguments': '{}'}}]})
        await writer.record_tool_result(KnownToolResult(0, 'artifact', 'read_artifact', outcome,
            outcome.content, {'status': 'ok'}, True, retrieval_receipt=outcome.retrieval_receipt))
        event = writer.owner.read_snapshot().events[-1]
        assert event['_artifact_read']['reference'] == descriptor.reference
        assert event['content'] == outcome.content and '_source_origin' not in event
        class Forged(ReadArtifactTool):
            async def execute_with_context(self, *args, **kwargs):
                return outcome
        forged = await ToolRegistry.execute_prepared(Forged(reader), {}, object())
        assert forged.retrieval_receipt is None


async def test_unregistered_task_cannot_use_reader(tmp_path):
    import asyncio

    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        async def child():
            return reader.read(descriptor.reference)
        with pytest.raises(ExecutionOwnershipError):
            await asyncio.create_task(child())


async def test_namespace_replacement_before_return_rejects_bytes(tmp_path, monkeypatch):
    from nanobot.session.artifact_store import ArtifactBlobTransaction

    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        original = ArtifactBlobTransaction.read_range
        def replace_namespace(transaction, *args, **kwargs):
            result = original(transaction, *args, **kwargs)
            namespace = store.root.joinpath(*store._parts)
            namespace.rename(namespace.with_name(namespace.name + '-old'))
            namespace.mkdir(mode=0o700)
            return result
        monkeypatch.setattr(ArtifactBlobTransaction, 'read_range', replace_namespace)
        with pytest.raises((OSError, ArtifactStorageError)):
            reader.read(descriptor.reference)


@pytest.mark.parametrize('reference', ['artifact:../file', '/private/file', 'artifact:' + 'c' * 32])
async def test_opaque_reference_cannot_select_other_authority(tmp_path, reference):
    async with setup_read(tmp_path) as (writer, tool, store, _, _):
        with pytest.raises(ArtifactValidationError):
            ArtifactReader(writer, 'root_history', store, tool).read(reference)


async def test_reader_registration_requires_invocation_file_capability(tmp_path):
    from unittest.mock import AsyncMock, Mock

    from nanobot.agent.runner import AgentRunner, AgentRunSpec

    async with setup_read(tmp_path) as (writer, tool, store, _, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        registry = ToolRegistry()
        registry.register(ReadArtifactTool(reader))
        factory = Mock(side_effect=AssertionError('Missing file capability'))
        runner = AgentRunner(artifact_reader_factory=factory)
        # Stop immediately after admission: inspect the effective registry passed
        # into transcript construction, before transport and budget preparation.
        def inspect(spec):
            assert spec.tools.tool_names == []
            raise RuntimeError('admission checked')
        runner._initial_transcript_and_compaction = inspect
        spec = AgentRunSpec(initial_messages=[], tools=registry, runtime=AsyncMock(),
            max_iterations=1, max_tool_result_chars=100)
        with pytest.raises(RuntimeError, match='admission checked'):
            await runner.run(spec)
        assert factory.call_count == 0
        assert registry.tool_names == ['read_artifact']


async def test_dynamic_project_and_new_read_roots_cannot_expose_private_store(tmp_path):
    from nanobot.security.workspace_access import (
        bind_workspace_scope,
        build_workspace_scope,
        reset_workspace_scope,
    )

    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        token = bind_workspace_scope(build_workspace_scope(tmp_path, 'restricted'))
        try:
            with pytest.raises(ArtifactValidationError):
                ArtifactReader(writer, 'root_history', store, tool)
            with pytest.raises(ArtifactValidationError):
                reader.read(descriptor.reference)
        finally:
            reset_workspace_scope(token)
        tool._extra_read_allowed_dirs.append(store.root)
        with pytest.raises(ArtifactValidationError):
            reader.read(descriptor.reference)


def test_complete_json_response_is_bounded_without_losing_next_page():
    import json

    from nanobot.agent.tools.artifact import bounded_artifact_response
    from nanobot.session.artifact_reader import ArtifactReadResult

    text = '\x00' * 16000 + '🌟'
    original = text.encode()
    position, collected = 0, []
    while position < len(original):
        result = ArtifactReadResult('artifact:' + 'b' * 32, original[position:].decode(),
                                    position, len(original), True)
        payload, bounded = bounded_artifact_response(result)
        assert len(payload.encode()) <= 16384
        assert json.loads(payload)['text'] == bounded.text
        assert bounded.next_offset > position
        assert bounded.text.encode() == original[position:bounded.next_offset]
        collected.append(bounded.text)
        position = bounded.next_offset
    assert ''.join(collected) == text


@pytest.mark.parametrize('field,value', [('reference', 'artifact:../escape'), ('start_offset', True),
    ('start_offset', -1), ('next_offset', -1), ('next_offset', False)])
def test_retrieval_receipts_reject_invalid_authority_and_ranges(field, value):
    from nanobot.agent.tools.artifact import ArtifactRetrievalReceipt, parse_retrieval_receipt

    raw = ArtifactRetrievalReceipt('artifact:' + 'b' * 32, 0, 4, 'a' * 64).to_dict()
    raw[field] = value
    with pytest.raises(ArtifactValidationError):
        parse_retrieval_receipt(raw)


async def test_invalid_retrieval_receipt_does_not_mark_or_discard_inline_evidence(tmp_path):
    from nanobot.agent.tools.artifact import ArtifactRetrievalReceipt

    async with setup_read(tmp_path) as (writer, _, _, _, _):
        receipt = ArtifactRetrievalReceipt('artifact:../forged', True, -1,
                                           hashlib.sha256(b'inline').hexdigest())
        writer.begin_tool_batch({'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'result', 'type': 'function', 'function': {'name': 'read_artifact', 'arguments': '{}'}}]})
        await writer.record_tool_result(KnownToolResult(0, 'result', 'read_artifact', 'inline',
            'inline', {'status': 'ok'}, True, retrieval_receipt=receipt))
        event = writer.owner.read_snapshot().events[-1]
        assert event['content'] == 'inline' and '_artifact_read' not in event


@pytest.mark.parametrize('operation', ['source', 'published'])
@pytest.mark.parametrize('fault', ['none', 'resolver_refusal', 'private_root'])
async def test_artifact_file_policy_preserves_existing_guard_and_resolver_order(
    tmp_path, monkeypatch, operation, fault,
):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        trace = []
        reader = ArtifactReader(writer, 'root_history', store, tool,
                                location_validator=lambda: trace.append('root'))
        snapshot = writer.owner.read_snapshot()
        dispatch = writer.owner.validate_dispatch
        read_snapshot = writer.owner.read_snapshot
        resolve_read = tool._resolve_read

        def traced_dispatch():
            trace.append('dispatch')
            return dispatch()

        def traced_snapshot():
            trace.append('snapshot')
            return read_snapshot()

        def traced_resolve(path):
            trace.append('resolve')
            assert path == str(tool._workspace / 'file')
            if fault == 'resolver_refusal':
                raise OSError('denied by current resolver')
            return resolve_read(path)

        if fault == 'private_root':
            tool._extra_read_allowed_dirs.append(store.root)
        trace.clear()

        def validate():
            if operation == 'source':
                receipt = reader.authorize_source(descriptor, snapshot)
                assert receipt.source_access == descriptor.source_access
                assert receipt.sha256 == descriptor.sha256
            else:
                with store.transaction() as blobs:
                    assert reader.verify_published(descriptor, snapshot, blobs) is None

        with monkeypatch.context() as patch:
            patch.setattr(writer.owner, 'validate_dispatch', traced_dispatch)
            patch.setattr(writer.owner, 'read_snapshot', traced_snapshot)
            patch.setattr(tool, '_resolve_read', traced_resolve)
            if fault == 'none':
                validate()
            else:
                with pytest.raises(ArtifactValidationError):
                    validate()

        assert trace == (['root'] if fault == 'private_root' else
                         ['root', 'dispatch', 'resolve'] if operation == 'source' else
                         ['root', 'dispatch', 'snapshot', 'resolve'])
        assert writer.owner.read_snapshot() == snapshot


@pytest.mark.parametrize('operation', ['source', 'published'])
@pytest.mark.parametrize('policy_changed', [False, True])
async def test_file_tool_artifact_validation_uses_current_policy_without_returning_text(
    tmp_path, operation, policy_changed,
):
    from nanobot.session.artifact import attested_artifact_origin

    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        snapshot = writer.owner.read_snapshot()
        expected = attested_artifact_origin(descriptor, snapshot, workspace_id=store.workspace_id)
        if policy_changed:
            tool._workspace = tmp_path / 'other-workspace'

        def validate():
            if operation == 'source':
                assert tool.revalidate_artifact_source(expected) is None
            else:
                with store.transaction() as blobs:
                    receipt = tool.verify_artifact_evidence(descriptor, snapshot, blobs, store=store)
                    assert receipt == expected
                    assert receipt.sha256 == descriptor.sha256

        if policy_changed:
            with pytest.raises(ArtifactValidationError):
                validate()
        else:
            validate()
        assert writer.owner.read_snapshot() == snapshot
