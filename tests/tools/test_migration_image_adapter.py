"""Legacy image requests reach the canonical artifact-producing tool."""
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.image_configuration import effective_image_configuration
from nanobot.agent.tools.image_generation import ImageGenerationTool, ImageGenerationToolConfig
from nanobot.config.schema import Config, ImageConfig, ToolsConfig
from nanobot.providers.codex_cli_image import CodexCLIImageGenerationClient
from nanobot.providers.image_generation import (
    GeneratedImageResponse,
    ImageGenerationError,
    OpenRouterImageGenerationClient,
)

DATA = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII='
RAW = base64.b64decode(DATA.split(',', 1)[1])


def test_legacy_default_and_explicit_upstream_selection_survive_roundtrip():
    config = Config()
    assert config.tools.image_generation.configuration_source == 'legacy'
    assert effective_image_configuration(config.tools).config.enabled
    restored = Config.model_validate_json(config.model_dump_json(by_alias=True))
    assert effective_image_configuration(restored.tools).config.provider == 'codex_cli'
    explicit = Config.model_validate({'tools': {'imageGeneration': {'enabled': False}}})
    assert not effective_image_configuration(explicit.tools).config.enabled
    assert explicit.tools.image_generation.configuration_source == 'upstream'


@pytest.mark.parametrize('alias', ['codex', 'codex-cli', 'codex_cli'])
def test_legacy_cli_aliases_do_not_select_oauth(alias):
    config = ToolsConfig(image={'provider': alias})
    assert effective_image_configuration(config).config.provider == 'codex_cli'


def test_legacy_openrouter_settings_win_over_upstream_credentials(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'environment')
    tools = ToolsConfig(image={'provider': 'openrouter', 'apiKey': 'configured', 'model': 'old-model'})
    selected = effective_image_configuration(tools)
    assert selected.providers['openrouter'].api_key == 'configured'
    assert selected.config.model == 'old-model'
    assert selected.config.default_aspect_ratio == '16:9'


@pytest.mark.parametrize('count,path', [(0, None), (-1, None), (True, None), (9, None), (2, 'out.png')])
async def test_invalid_calls_never_construct_backend(tmp_path, count, path, monkeypatch):
    tool = ImageGenerationTool(workspace=tmp_path, config=ImageGenerationToolConfig())
    monkeypatch.setattr(tool, '_provider_client', lambda: pytest.fail('backend constructed'))
    result = await tool.execute('test', count=count, output_path=path)
    assert result.is_error


async def test_output_copy_and_artifact_and_empty_response(tmp_path, monkeypatch):
    monkeypatch.setattr('nanobot.config.loader._current_config_path', tmp_path / 'config.json')
    tool = ImageGenerationTool(workspace=tmp_path, config=ImageGenerationToolConfig())
    client = SimpleNamespace(generate=AsyncMock(return_value=GeneratedImageResponse([DATA], '', {})))
    monkeypatch.setattr(tool, '_provider_client', lambda: client)
    result = json.loads(await tool.execute('visible prompt', output_path='nested/picture.png'))
    target = tmp_path / 'generated_images' / 'picture.png'
    assert result['path'] == str(target)
    assert target.read_bytes() == RAW
    assert tool._resolve_reference_image(result['artifacts'][0]['path']) == result['artifacts'][0]['path']
    client.generate.return_value = GeneratedImageResponse([], '', {})
    client.generate.reset_mock()
    result = await tool.execute('test')
    assert result.is_error
    assert client.generate.await_count == 1


async def test_denied_destination_and_publication_failure_do_not_regenerate(tmp_path, monkeypatch):
    monkeypatch.setattr('nanobot.config.loader._current_config_path', tmp_path / 'config.json')
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    tool = ImageGenerationTool(workspace=workspace, config=ImageGenerationToolConfig())
    client = SimpleNamespace(generate=AsyncMock(return_value=GeneratedImageResponse([DATA], '', {})))
    monkeypatch.setattr(tool, '_provider_client', lambda: client)
    assert (await tool.execute('test', output_path=str(tmp_path / 'outside.png'))).is_error
    client.generate.assert_not_awaited()
    destination = workspace / 'generated_images' / 'out.png'
    destination.parent.mkdir()
    destination.write_bytes(b'previous')
    monkeypatch.setattr('nanobot.agent.tools.image_generation.os.replace', lambda *_: (_ for _ in ()).throw(OSError('publication failed')))
    result = await tool.execute('test', output_path='out.png')
    assert result.is_error and 'retained artifacts' in result
    assert destination.read_bytes() == b'previous'
    assert client.generate.await_count == 1


@pytest.mark.parametrize('shape', ['images', 'blocks', 'direct'])
async def test_openrouter_accepts_legacy_and_upstream_image_shapes(shape):
    message = {'images': [{'image_url': {'url': DATA}}]} if shape == 'images' else {
        'content': DATA if shape == 'direct' else [{'type': 'image_url', 'image_url': {'url': DATA}}]}
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': message}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        client = OpenRouterImageGenerationClient(api_key='fake', client=http, legacy_request=True)
        response = await client.generate(prompt='a tree', model='legacy', aspect_ratio='16:9')
    assert response.images == [DATA]
    assert requests[0]['messages'][0]['content'] == 'a tree (aspect ratio: 16:9)'
    assert requests[0]['modalities'] == ['image']


@pytest.mark.parametrize('mode', ['fresh', 'basename', 'relative', 'stale', 'touched', 'ambiguous', 'outside', 'unattributed', 'timeout'])
async def test_codex_cli_uses_owned_argv_and_rejects_unrelated_files(tmp_path, monkeypatch, mode):
    root = tmp_path / 'generated_images'
    root.mkdir()
    stale = root / 'stale.png'
    stale.write_bytes(RAW)
    monkeypatch.setattr('nanobot.providers.codex_cli_image.resolve_command', lambda _: '/fake/codex')
    observed = []
    async def invoke(args, **kwargs):
        observed.append((args, kwargs))
        if mode in {'fresh', 'basename', 'relative', 'ambiguous', 'unattributed'}:
            (root / 'new.png').write_bytes(RAW)
        if mode == 'ambiguous':
            (root / 'sub').mkdir()
            (root / 'sub' / 'new.png').write_bytes(RAW)
        if mode == 'touched':
            import os
            stamp = stale.stat().st_mtime_ns + 1000000000
            os.utime(stale, ns=(stamp, stamp))
        outside = tmp_path / 'unrelated.png'
        outside.write_bytes(RAW)
        output = {'fresh': str(root / 'new.png'), 'basename': 'new.png', 'relative': 'generated_images/new.png', 'ambiguous': 'new.png',
                  'stale': str(stale), 'touched': str(stale), 'unattributed': 'done'}.get(mode, str(outside))
        return SimpleNamespace(returncode=0, timed_out=mode == 'timeout', stdout=output.encode(), stderr=b'')
    monkeypatch.setattr('nanobot.providers.codex_cli_image.run_owned_process', invoke)
    client = CodexCLIImageGenerationClient(ImageConfig(codex_home=str(tmp_path), codex_timeout=12))
    if mode in {'fresh', 'basename', 'relative'}:
        response = await client.generate(prompt='  a tree  ', model='chosen')
        assert response.images == [DATA]
        assert response.raw['source_path'] == str(root / 'new.png')
    else:
        with pytest.raises(ImageGenerationError):
            await client.generate(prompt='  a tree  ', model='chosen')
    assert observed[0][0][-3:] == ['-m', 'chosen', 'a tree']
    assert observed[0][1]['timeout'] == 12
    assert observed[0][1]['env']['CODEX_HOME'] == str(tmp_path)


async def test_default_canonical_tool_reports_missing_cli(tmp_path, monkeypatch):
    monkeypatch.setattr('nanobot.providers.codex_cli_image.resolve_command', lambda _: None)
    context = ToolContext(config=ToolsConfig(), workspace=str(tmp_path))
    assert ImageGenerationTool.enabled(context)
    result = await ImageGenerationTool.create(context).execute('test')
    assert result.is_error and 'command not found' in result


async def test_protected_reference_is_denied_before_backend(tmp_path, monkeypatch):
    from nanobot.agent.tools.filesystem import WriteFileTool
    from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy

    (tmp_path / 'nanobot.context.json').write_text(json.dumps({'protected': ['secret.png']}))
    (tmp_path / 'secret.png').write_bytes(RAW)
    policy = ResourceAccessPolicy(workspace=tmp_path, context_manager=ContextRepoManager.from_config(context_paths=[tmp_path]))
    writer = WriteFileTool(workspace=tmp_path, allowed_dir=tmp_path, resource_policy=policy)
    tool = ImageGenerationTool(workspace=tmp_path, config=ImageGenerationToolConfig(), write_tool=writer)
    monkeypatch.setattr(tool, '_provider_client', lambda: pytest.fail('unauthorized reference reached backend'))
    assert (await tool.execute('test', reference_images=['secret.png'])).is_error


async def test_reload_without_previous_tool_keeps_sandbox_path_restriction(tmp_path, monkeypatch):
    from nanobot.agent.tools.image_generation import reload_image_generation_tool
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.config.loader import save_config

    monkeypatch.setattr('nanobot.config.loader._current_config_path', tmp_path / 'config.json')
    config = Config()
    config.tools.exec.sandbox = 'bwrap'
    config.tools.restrict_to_workspace = False
    save_config(config)
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    state = SimpleNamespace(workspace=workspace, tools_config=config.tools)
    registry = ToolRegistry()
    assert (await reload_image_generation_tool(state, registry))['ok']
    tool = registry.get('generate_image')
    monkeypatch.setattr(tool, '_provider_client', lambda: pytest.fail('outside output reached backend'))
    assert (await tool.execute('test', output_path=str(tmp_path / 'outside.png'))).is_error


async def test_authorized_external_output_and_symlink_escape(tmp_path, monkeypatch):
    from nanobot.agent.tools.filesystem import WriteFileTool
    from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy

    monkeypatch.setattr('nanobot.config.loader._current_config_path', tmp_path / 'config.json')
    workspace, repo, outside = (tmp_path / name for name in ('workspace', 'repo', 'outside'))
    for root in (workspace, repo, outside):
        root.mkdir()
    (repo / 'outputs').mkdir()
    (repo / 'nanobot.context.json').write_text(json.dumps({'writable': ['outputs/**']}))
    policy = ResourceAccessPolicy(workspace=workspace, context_manager=ContextRepoManager.from_config(context_paths=[repo]),
                                  restrict_to_workspace=True)
    writer = WriteFileTool(workspace=workspace, allowed_dir=workspace, resource_policy=policy)
    tool = ImageGenerationTool(workspace=workspace, config=ImageGenerationToolConfig(), write_tool=writer)
    client = SimpleNamespace(generate=AsyncMock(return_value=GeneratedImageResponse([DATA], '', {})))
    monkeypatch.setattr(tool, '_provider_client', lambda: client)
    target = repo / 'outputs' / 'allowed.png'
    assert json.loads(await tool.execute('test', output_path=str(target)))['path'] == str(target)
    assert target.read_bytes() == RAW
    client.generate.reset_mock()
    (repo / 'outputs' / 'escape').symlink_to(outside, target_is_directory=True)
    assert (await tool.execute('test', output_path=str(repo / 'outputs' / 'escape' / 'out.png'))).is_error
    client.generate.assert_not_awaited()


async def test_cli_cancellation_propagates_through_canonical_tool(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setattr('nanobot.providers.codex_cli_image.resolve_command', lambda _: '/fake/codex')
    monkeypatch.setattr('nanobot.providers.codex_cli_image.run_owned_process', AsyncMock(side_effect=asyncio.CancelledError))
    tools = ToolsConfig(image={'codexHome': str(tmp_path)})
    tool = ImageGenerationTool.create(ToolContext(config=tools, workspace=str(tmp_path)))
    with pytest.raises(asyncio.CancelledError):
        await tool.execute('test')


def test_explicit_settings_disable_switches_out_of_legacy_mode():
    from nanobot.webui.settings_capabilities import update_image_generation_settings

    config = Config()
    assert effective_image_configuration(config.tools).config.enabled
    assert update_image_generation_settings(config, {'enabled': ['false']}, oauth_status=lambda _: {})
    assert config.tools.image_generation.configuration_source == 'upstream'
    assert not effective_image_configuration(config.tools).config.enabled


async def test_cli_rejects_replacement_during_capture(tmp_path, monkeypatch):
    from pathlib import Path

    root = tmp_path / 'generated_images'
    root.mkdir()
    target = root / 'new.png'
    monkeypatch.setattr('nanobot.providers.codex_cli_image.resolve_command', lambda _: '/fake/codex')
    async def invoke(*args, **kwargs):
        target.write_bytes(RAW)
        return SimpleNamespace(returncode=0, timed_out=False, stdout=str(target).encode(), stderr=b'')
    monkeypatch.setattr('nanobot.providers.codex_cli_image.run_owned_process', invoke)
    original = Path.read_bytes
    reads = 0
    def read(path):
        nonlocal reads
        raw = original(path)
        if path == target:
            reads += 1
            if reads == 2:
                replacement = root / 'replacement.png'
                replacement.write_bytes(RAW)
                replacement.replace(target)
        return raw
    monkeypatch.setattr(Path, 'read_bytes', read)
    client = CodexCLIImageGenerationClient(ImageConfig(codex_home=str(tmp_path)))
    with pytest.raises(ImageGenerationError, match='changed during capture'):
        await client.generate(prompt='test', model='test')
