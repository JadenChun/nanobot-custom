"""Preserved context assets must reach the upstream prompt loader."""
from nanobot.agent.context import ContextBuilder
from nanobot.agent.skills import SkillsLoader


def test_custom_profile_files_are_loaded_from_agent_workspace(tmp_path):
    workspace = tmp_path / 'agent'
    project = tmp_path / 'project'
    workspace.mkdir()
    project.mkdir()
    (workspace / 'TOOLS.md').write_text('custom tool guidance')
    (workspace / 'IDENTITY.md').write_text('custom identity')
    (project / 'AGENTS.md').write_text('project instructions')
    (project / 'IDENTITY.md').write_text('other identity')
    text = ContextBuilder(workspace)._load_bootstrap_files(project)
    assert 'custom tool guidance' in text
    assert 'custom identity' in text
    assert 'project instructions' in text
    assert 'other identity' not in text


def test_ordered_external_skills_preserve_workspace_and_disabled_precedence(tmp_path):
    workspace = tmp_path / 'workspace'
    first, second, builtin = [tmp_path / name for name in ('first', 'second', 'builtin')]
    for root, names in ((workspace / 'skills', ['local']), (first, ['local', 'shared']),
                        (second, ['shared', 'external']), (builtin, ['shared', 'external'])):
        for name in names:
            folder = root / name
            folder.mkdir(parents=True)
            (folder / 'SKILL.md').write_text(str(root))
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin, extra_paths=[first, second],
                          disabled_skills={'external'})
    assert loader.load_skill('local') == str(workspace / 'skills')
    assert loader.load_skill('shared') == str(first)
    assert loader.load_skill('external') is None
    assert {s['name'] for s in loader.list_skills(False)} == {'local', 'shared'}


def test_managed_repository_context_and_memory_reach_prompt(tmp_path):
    from nanobot.context_repo import ContextRepoManager

    workspace, repo = tmp_path / 'agent', tmp_path / 'context'
    workspace.mkdir()
    (repo / 'memory').mkdir(parents=True)
    (repo / 'runs').mkdir()
    (repo / 'CONTEXT.md').write_text('repository entrypoint')
    (repo / 'memory' / 'MEMORY.md').write_text('repository memory')
    (repo / 'runs' / 'summaries.md').write_text('repository run summary')
    manager = ContextRepoManager.from_config(context_paths=[repo])
    builder = ContextBuilder(workspace, context_manager=manager)
    text = builder.build_system_prompt(include_memory=False)
    assert 'repository entrypoint' in text
    assert 'repository memory' in text
    assert 'repository run summary' in text
