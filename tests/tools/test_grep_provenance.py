"""Single-file grep producer provenance tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.search import GrepTool
from nanobot.agent.turn import ToolOutcome
from nanobot.context_repo.manager import ContextRepoManager, ManagedContextRepo
from nanobot.context_repo.policy import ResourceAccessPolicy


async def _attested_grep(tool: GrepTool, **params: object) -> ToolOutcome:
    outcome = await ToolRegistry.execute_prepared(tool, params, SimpleNamespace(lock_owner=None))
    assert isinstance(outcome, ToolOutcome)
    return outcome


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {
            "output_mode": "content",
            "context_before": 1,
            "context_after": 1,
            "head_limit": 1,
            "offset": 1,
        },
        {"output_mode": "files_with_matches", "head_limit": 1, "offset": 0},
        {"output_mode": "count", "head_limit": 1, "offset": 0},
    ],
)
async def test_single_file_grep_attests_the_exact_legacy_output(tmp_path, params):
    (tmp_path / "evidence.txt").write_bytes(
        "before\r\nneedle caf\u00e9\r\nafter\r\nneedle second\n".encode("utf-8")
    )
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)

    ordinary = await tool.execute(pattern="needle", path="evidence.txt", fixed_strings=True, **params)
    attested = await _attested_grep(
        tool,
        pattern="needle",
        path="evidence.txt",
        fixed_strings=True,
        **params,
    )

    assert attested.content == ordinary
    assert attested.producer_receipt is not None
    assert attested.producer_receipt.producer == "builtin_grep_file_text"
    assert attested.producer_receipt.matches(ordinary)
    assert attested.producer_receipt.source_access.relative_path == "evidence.txt"


@pytest.mark.asyncio
async def test_single_file_grep_no_match_is_attested(tmp_path):
    (tmp_path / "evidence.txt").write_text("ordinary text\n", encoding="utf-8")
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)

    ordinary = await tool.execute(pattern="needle", path="evidence.txt", fixed_strings=True)
    attested = await _attested_grep(tool, pattern="needle", path="evidence.txt", fixed_strings=True)

    assert attested.content == ordinary == "No matches found for pattern 'needle' in evidence.txt"
    assert attested.producer_receipt is not None
    assert attested.producer_receipt.matches(ordinary)


@pytest.mark.asyncio
async def test_oversized_explicit_file_keeps_legacy_output_without_a_receipt(tmp_path, monkeypatch):
    path = tmp_path / "oversized.txt"
    path.write_text("needle\n" + "x" * 20, encoding="utf-8")
    monkeypatch.setattr(GrepTool, "_MAX_EXPLICIT_FILE_BYTES", 10)
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)

    ordinary = await tool.execute(pattern="needle", path="oversized.txt", fixed_strings=True)
    contextual = await _attested_grep(tool, pattern="needle", path="oversized.txt", fixed_strings=True)

    assert ordinary == (
        "No matches found for pattern 'needle' in oversized.txt\n\n"
        "(skipped 1 large files)"
    )
    assert contextual.content == ordinary
    assert contextual.producer_receipt is None


@pytest.mark.asyncio
async def test_grep_subclasses_cannot_claim_the_builtin_producer(tmp_path):
    class ImpostorGrepTool(GrepTool):
        pass

    (tmp_path / "evidence.txt").write_text("needle\n", encoding="utf-8")
    outcome = await _attested_grep(
        ImpostorGrepTool(workspace=tmp_path, allowed_dir=tmp_path),
        pattern="needle",
        path="evidence.txt",
        fixed_strings=True,
    )

    assert outcome.content == "evidence.txt:1\n> 1| needle"
    assert outcome.producer_receipt is None


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["file", "symlink"])
async def test_replaced_or_symlinked_source_cannot_issue_grep_receipt(tmp_path, monkeypatch, replacement):
    path = tmp_path / "evidence.txt"
    path.write_text("needle\n", encoding="utf-8")
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path)
    original_format = GrepTool._format_block
    changed = False

    def replace_during_render(*args, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            old = tmp_path / "old-evidence.txt"
            path.rename(old)
            if replacement == "symlink":
                path.symlink_to(old)
            else:
                path.write_text("replacement\n", encoding="utf-8")
        return original_format(*args, **kwargs)

    monkeypatch.setattr(GrepTool, "_format_block", staticmethod(replace_during_render))
    outcome = await _attested_grep(tool, pattern="needle", path="evidence.txt", fixed_strings=True)

    assert outcome.stop_reason == "tool_error"
    assert outcome.producer_receipt is None
    assert "Error searching files:" in str(outcome.content)


@pytest.mark.asyncio
async def test_grep_policy_denial_never_issues_a_receipt(tmp_path):
    path = tmp_path / "protected.txt"
    path.write_text("needle\n", encoding="utf-8")
    repo = ManagedContextRepo(tmp_path, "protected", True, manifest={"protected": ["protected.txt"]})
    policy = ResourceAccessPolicy(workspace=tmp_path, context_manager=ContextRepoManager([repo]))
    tool = GrepTool(workspace=tmp_path, allowed_dir=tmp_path, resource_policy=policy)

    outcome = await _attested_grep(tool, pattern="needle", path="protected.txt", fixed_strings=True)

    assert outcome.stop_reason == "tool_error"
    assert outcome.producer_receipt is None


@pytest.mark.asyncio
@pytest.mark.parametrize("namespace", ["context_repository", "target_repository"])
async def test_grep_attests_the_current_managed_namespace(tmp_path, namespace):
    workspace = tmp_path / "workspace"
    context = tmp_path / "context"
    target = context / "target"
    workspace.mkdir()
    target.mkdir(parents=True)
    if namespace == "context_repository":
        path = context / "evidence.txt"
        manifest = {}
    else:
        path = target / "evidence.txt"
        manifest = {"targetRepos": {"target-id": {"path": str(target), "read": ["**"]}}}
    path.write_text("needle\n", encoding="utf-8")
    repo = ManagedContextRepo(context, "context-id", True, manifest=manifest)
    policy = ResourceAccessPolicy(workspace=workspace, context_manager=ContextRepoManager([repo]))
    tool = GrepTool(
        workspace=workspace,
        allowed_dir=workspace,
        extra_read_allowed_dirs=policy.extra_read_dirs(),
        resource_policy=policy,
    )

    outcome = await _attested_grep(tool, pattern="needle", path=str(path), fixed_strings=True)

    assert outcome.producer_receipt is not None
    assert outcome.producer_receipt.source_access.access_class == namespace


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["directory", "document", "binary", "unsupported_namespace"])
async def test_ineligible_grep_sources_remain_inline_without_a_receipt(tmp_path, kind):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if kind == "directory":
        (workspace / "evidence.txt").write_text("needle\n", encoding="utf-8")
        path = "."
        tool = GrepTool(workspace=workspace, allowed_dir=workspace)
    elif kind == "document":
        path = "evidence.pdf"
        (workspace / path).write_bytes(b"not a document")
        tool = GrepTool(workspace=workspace, allowed_dir=workspace)
    elif kind == "binary":
        path = "evidence.bin"
        (workspace / path).write_bytes(b"\x00needle")
        tool = GrepTool(workspace=workspace, allowed_dir=workspace)
    else:
        external = tmp_path / "external.txt"
        external.write_text("needle\n", encoding="utf-8")
        path = str(external)
        tool = GrepTool(workspace=workspace)

    outcome = await _attested_grep(tool, pattern="needle", path=path, fixed_strings=True)

    assert outcome.producer_receipt is None
