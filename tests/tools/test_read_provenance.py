"""Producer receipts qualify exact returned text without changing legacy reads."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.read_provenance import (
    locate_read,
    parse_read_receipt,
    qualified_text_bytes,
    root_binding,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import ToolOutcome
from nanobot.context_repo.manager import ContextRepoManager, ManagedContextRepo
from nanobot.context_repo.policy import ResourceAccessPolicy


async def read(tool, **params):
    return await ToolRegistry.execute_prepared(tool, params, SimpleNamespace(lock_owner=None))


@pytest.mark.parametrize('raw', [b'first\r\nsecond\r\nthird', b'first\n\xe9\nthird'])
async def test_exact_paginated_decoded_representation_and_legacy_parity(tmp_path, raw):
    path = tmp_path / 'file.txt'
    path.write_bytes(raw)
    tool = ReadFileTool(workspace=tmp_path, allowed_dir=tmp_path)
    outcome = await read(tool, path='file.txt', offset=2, limit=1, force=True)
    assert outcome.producer_receipt is not None
    assert outcome.producer_receipt.matches(outcome.content)
    assert outcome.producer_receipt.source_access.relative_path == 'file.txt'
    assert outcome.producer_receipt.root_binding == root_binding(tmp_path)
    assert outcome.content == await tool.execute(path='file.txt', offset=2, limit=1, force=True)
    assert type(outcome.content) is str


async def test_concurrent_same_instance_results_do_not_exchange_receipts(tmp_path):
    for name in ('one', 'two'):
        (tmp_path / name).write_text(name)
    tool = ReadFileTool(workspace=tmp_path, allowed_dir=tmp_path)
    results = await asyncio.gather(*(read(tool, path=name, force=True) for name in ('one', 'two')))
    for name, result in zip(('one', 'two'), results):
        assert result.producer_receipt.source_access.relative_path == name
        assert result.producer_receipt.matches(result.content)
    assert not results[0].producer_receipt.matches(results[1].content)


@pytest.mark.parametrize('kind', ['empty', 'error', 'binary', 'office', 'pdf', 'external'])
async def test_unsupported_results_have_no_producer_receipt(tmp_path, monkeypatch, kind):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    path = workspace / {'office': 'file.docx', 'pdf': 'file.pdf'}.get(kind, 'file.bin')
    path.write_bytes(b'' if kind == 'empty' else b'\x00\xff')
    tool = ReadFileTool(workspace=workspace)
    if kind == 'error':
        path.unlink()
    if kind == 'external':
        path = tmp_path / 'external.txt'
        path.write_text('external')
    monkeypatch.setattr(tool, '_read_pdf', lambda *a: 'pdf text')
    monkeypatch.setattr(tool, '_read_office_doc', lambda *a: 'office text')
    result = await read(tool, path=str(path))
    assert result.producer_receipt is None


async def test_unqualified_platform_preserves_inline_text(tmp_path, monkeypatch):
    (tmp_path / 'file').write_text('evidence')
    from nanobot.agent.tools import read_provenance
    monkeypatch.setattr(read_provenance.os, 'supports_dir_fd', set())
    result = await read(ReadFileTool(workspace=tmp_path), path='file')
    assert 'evidence' in result.content and result.producer_receipt is None


async def test_forged_registered_implementation_cannot_supply_receipt(tmp_path):
    (tmp_path / 'file').write_text('evidence')
    original = await read(ReadFileTool(workspace=tmp_path), path='file')
    class Impostor(ReadFileTool):
        async def execute_with_context(self, context, **kwargs):
            return original
    forged = await read(Impostor(workspace=tmp_path), path='file')
    assert forged.content == original.content and forged.producer_receipt is None


async def test_mutated_returned_text_drops_receipt(tmp_path, monkeypatch):
    (tmp_path / 'file').write_text('evidence')
    tool = ReadFileTool(workspace=tmp_path)
    original = await read(tool, path='file')
    async def changed(*args, **kwargs):
        return ToolOutcome('changed', producer_receipt=original.producer_receipt)
    monkeypatch.setattr(tool, 'execute_with_context', changed)
    result = await read(tool, path='file')
    assert result.content == 'changed' and result.producer_receipt is None


@pytest.mark.parametrize('kind', ['file', 'parent', 'symlink'])
def test_replacement_cannot_attest_another_object(tmp_path, kind):
    directory = tmp_path / 'directory'
    directory.mkdir()
    path = directory / 'file'
    path.write_text('evidence')
    with qualified_text_bytes(path, max_bytes=100) as qualified:
        assert qualified[0] == b'evidence'
        if kind == 'parent':
            directory.rename(tmp_path / 'old')
            directory.mkdir()
            path.write_text('evidence')
        else:
            path.rename(directory / 'old')
            if kind == 'symlink':
                path.symlink_to(directory / 'old')
            else:
                path.write_text('evidence')
        with pytest.raises(OSError):
            qualified[1]()


def test_policy_precedence_and_root_binding_prevent_relabeling(tmp_path):
    context = tmp_path / 'context'
    target = context / 'target'
    target.mkdir(parents=True)
    path = target / 'file'
    path.write_text('evidence')
    repo = ManagedContextRepo(context, 'context-id', True, manifest={
        'targetRepos': {'target-id': {'path': str(target), 'read': ['**']}}})
    policy = ResourceAccessPolicy(workspace=tmp_path, context_manager=ContextRepoManager([repo]))
    origin, binding = locate_read(path, tmp_path, policy)
    assert origin.access_class == 'target_repository'
    assert origin.repository == '["context-id","target-id"]'
    assert origin.relative_path == 'file' and binding == root_binding(target)
    assert locate_read(context / 'notes', tmp_path, policy)[0].access_class == 'context_repository'
    assert locate_read(tmp_path / 'notes', tmp_path, policy)[0].access_class == 'workspace'
    assert root_binding(tmp_path / 'different') != binding


async def test_protected_managed_file_is_not_attested(tmp_path):
    path = tmp_path / 'secret.txt'
    path.write_text('secret')
    repo = ManagedContextRepo(tmp_path, 'protected', True, manifest={'protected': ['secret.txt']})
    policy = ResourceAccessPolicy(workspace=tmp_path, context_manager=ContextRepoManager([repo]))
    result = await read(ReadFileTool(workspace=tmp_path, resource_policy=policy), path='secret.txt')
    assert result.stop_reason == 'tool_error' and result.producer_receipt is None


async def test_receipt_parser_rejects_extra_keys_and_digest_mismatch(tmp_path):
    (tmp_path / 'file').write_text('evidence')
    result = await read(ReadFileTool(workspace=tmp_path), path='file')
    raw = result.producer_receipt.to_dict()
    assert parse_read_receipt(raw) == result.producer_receipt
    assert not replace(result.producer_receipt, byte_length=1).matches(result.content)
    raw['execution_id'] = 'forged'
    with pytest.raises(ValueError):
        parse_read_receipt(raw)


async def test_legacy_upgrade_cannot_retroactively_authenticate_read(tmp_path):
    from nanobot.session.manager import JsonlSessionStore, Session

    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'file').write_text('evidence')
    result = await read(ReadFileTool(workspace=workspace), path='file')
    backend = JsonlSessionStore(workspace)
    session = Session('test:legacy-origin')
    session.messages = [
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'call', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'call', 'content': result.content,
         '_source_origin': result.producer_receipt.to_dict(), '_dispatched': True, '_outcome_status': 'ok'},
    ]
    backend.save(session)
    snapshot = backend.source.upgrade_source_session(session.key,
        expected_revision=session.revision, expected_generation=session.generation,
        expected_incarnation=session.incarnation)
    assert snapshot.events[1]['content'] == result.content
    assert '_source_origin' not in snapshot.events[1]
