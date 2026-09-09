"""Fork command policy remains active under upstream execution modes."""
import json
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.tools.shell import ExecTool
from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy


@pytest.mark.parametrize('command', ['rm -rf data', 'curl http://127.0.0.1/private', 'osascript -e beep'])
async def test_full_access_still_checks_fork_guards_before_spawn(tmp_path, command):
    tool = ExecTool(working_dir=str(tmp_path), restrict_to_workspace=False, desktop_use_active=True)
    tool._spawn = AsyncMock(side_effect=AssertionError('blocked command spawned'))
    result = await tool.execute(command)
    assert result.is_error
    tool._spawn.assert_not_awaited()


def test_allowlist_never_overrides_deny_and_checks_every_segment(tmp_path):
    tool = ExecTool(working_dir=str(tmp_path), allow_patterns=['.*'])
    assert 'deny' in tool._guard_command('rm -rf data', str(tmp_path))
    tool.allow_patterns = ['echo safe']
    assert 'allowlist' in tool._guard_command('echo safe; cat secret', str(tmp_path))


def test_managed_cwd_and_protected_paths_keep_separate_sandbox_bounds(tmp_path):
    workspace, repo, outside = (tmp_path / name for name in ('workspace', 'repo', 'outside'))
    for root in (workspace, repo, outside):
        root.mkdir()
    (repo / 'nanobot.context.json').write_text(json.dumps({'name': 'fixture', 'protected': ['private/**']}))
    manager = ContextRepoManager.from_config(context_paths=[repo])
    policy = ResourceAccessPolicy(workspace=workspace, context_manager=manager, restrict_to_workspace=True)
    tool = ExecTool(working_dir=str(workspace), restrict_to_workspace=True, resource_policy=policy)
    assert not isinstance(tool._prepare_command('echo safe', str(repo)), str)
    assert 'outside' in tool._prepare_command('echo safe', str(outside))
    assert 'protected' in tool._prepare_command('cat private/secret', str(repo))
    (repo / 'escape').symlink_to(outside, target_is_directory=True)
    assert 'outside' in tool._prepare_command('echo safe', str(repo / 'escape'))
    tool.sandbox = 'bwrap'
    before = list(tool.sandbox_rw_binds)
    assert 'not mounted' in tool._prepare_command('echo safe', str(repo))
    assert tool.sandbox_rw_binds == before
