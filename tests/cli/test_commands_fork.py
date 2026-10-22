import asyncio
import json
import re
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from nanobot.bus.events import OutboundMessage
from nanobot.cli.commands import app
from nanobot.config.schema import Config, FallbackEntry
from nanobot.providers.factory import make_provider as _make_provider
from nanobot.providers.legacy_fallback_provider import LegacyFallbackProvider
from nanobot.providers.openai_codex_provider import _strip_model_prefix
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.pooled_provider import PooledProvider
from nanobot.providers.registry import find_by_name
from tests.agent import p6_admitted_fixtures
from tests.cli import p6_classic_cli_fixtures

runner = CliRunner()
classic_cli_runtime = p6_classic_cli_fixtures.classic_cli_runtime
admitted_async = p6_admitted_fixtures.admitted_async


@pytest.fixture(autouse=True)
def preserve_loguru_activation_state():
    """Keep CLI logging flags from changing Loguru state for later test modules."""
    from loguru import logger

    core = logger._core
    with core.lock:
        activation_list = list(core.activation_list)
        enabled = dict(core.enabled)
        activation_none = core.activation_none

    yield

    with core.lock:
        core.activation_list = activation_list
        core.enabled = enabled
        core.activation_none = activation_none


class _StopGatewayError(RuntimeError):
    pass


@pytest.fixture
def isolated_cli_effects(tmp_path, monkeypatch):
    """Positive-only disposable paths and effect fences, not startup admission."""
    from nanobot.config import loader

    def denied(*_args, **_kwargs):
        raise AssertionError("CLI option test attempted an external effect")

    monkeypatch.setattr(loader, "_current_config_path", loader._current_config_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))
    monkeypatch.setattr(subprocess.Popen, "__init__", denied)
    for name in ("connect", "connect_ex", "bind", "listen", "sendto"):
        monkeypatch.setattr(socket.socket, name, denied)
    for name in ("getaddrinfo", "create_connection"):
        monkeypatch.setattr(socket, name, denied)


@pytest.fixture
def gateway_graphs(tmp_path, monkeypatch, isolated_cli_effects):
    """Observe real failed acquisition and verify the durable lease is released."""
    from nanobot.runtime_composition import RuntimeComposition, RuntimeState
    from nanobot.session.runtime_authority import DurableSessionAuthority

    original_define = RuntimeComposition.define
    graphs = []

    def define(config, **kwargs):
        workspace = config.workspace_path
        assert workspace.is_absolute() and workspace.is_relative_to(tmp_path)
        assert not workspace.is_symlink()
        workspace.mkdir(parents=True, exist_ok=True)
        graph = original_define(config, **kwargs)
        graphs.append(graph)
        return graph

    monkeypatch.setattr(RuntimeComposition, "define", define)
    yield graphs
    for graph in graphs:
        assert graph.state is RuntimeState.FAILED
        assert graph._owner_loop.is_closed()

        async def verify_release():
            lease = await DurableSessionAuthority(
                graph.preflight.workspace, sessions_root=graph.preflight.sessions_root,
            ).acquire()
            assert lease.manager.has_managed_authority
            assert lease.close is not None
            await lease.close()

        asyncio.run(verify_release())
        assert not (graph.preflight.workspace / "AGENTS.md").exists()


@pytest.fixture
def mock_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep onboarding's config, workspace and writes inside pytest's temp root."""
    base_dir = tmp_path / "test_onboard_data"
    base_dir.mkdir()
    config_file = base_dir / "config.json"
    workspace_dir = base_dir / "workspace"

    def configured_default() -> Config:
        config = Config()
        config.agents.defaults.workspace = str(workspace_dir)
        return config

    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)
    with patch("nanobot.config.loader.get_config_path") as mock_cp, \
         patch("nanobot.config.loader.save_config") as mock_sc, \
         patch("nanobot.config.loader.load_config") as mock_lc, \
         patch("nanobot.config.schema.Config", side_effect=configured_default):
        mock_cp.return_value = config_file
        mock_lc.side_effect = lambda _config_path=None: configured_default()

        def _save_config(config: Config, config_path: Path | None = None):
            target = config_path or config_file
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(config.model_dump(by_alias=True)), encoding="utf-8")

        mock_sc.side_effect = _save_config

        yield config_file, workspace_dir


def test_onboard_fresh_install(mock_paths):
    """No existing config — should create from scratch."""
    config_file, workspace_dir = mock_paths

    result = runner.invoke(app, ["onboard"])

    assert result.exit_code == 0
    assert "Created config" in result.stdout
    assert "Created workspace" in result.stdout
    assert "nanobot is ready" in result.stdout
    assert config_file.exists()
    assert (workspace_dir / "AGENTS.md").exists()
    assert (workspace_dir / "memory" / "MEMORY.md").exists()
    assert json.loads(config_file.read_text())["agents"]["defaults"]["workspace"] == str(
        workspace_dir
    )


def test_onboard_existing_config_refresh(mock_paths):
    """Config exists, user declines overwrite — should refresh (load-merge-save)."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "existing values preserved" in "".join(result.stdout.splitlines())
    assert workspace_dir.exists()
    assert (workspace_dir / "AGENTS.md").exists()


def test_onboard_existing_config_overwrite(mock_paths):
    """Config exists, user confirms overwrite — should reset to defaults."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="y\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "Config reset to defaults" in result.stdout
    assert workspace_dir.exists()


def test_onboard_existing_workspace_safe_create(mock_paths):
    """Workspace exists — should not recreate, but still add missing templates."""
    config_file, workspace_dir = mock_paths
    workspace_dir.mkdir(parents=True)
    config_file.write_text("{}")

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Created workspace" not in result.stdout
    assert "Created AGENTS.md" in result.stdout
    assert (workspace_dir / "AGENTS.md").exists()


def _strip_ansi(text):
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)


def test_onboard_help_shows_workspace_and_config_options(monkeypatch):
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)
    result = runner.invoke(app, ["onboard", "--help"])

    assert result.exit_code == 0
    stripped_output = _strip_ansi(result.stdout)
    assert "--workspace" in stripped_output
    assert "-w" in stripped_output
    assert "--config" in stripped_output
    assert "-c" in stripped_output
    assert "--wizard" in stripped_output
    assert "--dir" not in stripped_output


def test_onboard_interactive_discard_does_not_save_or_create_workspace(mock_paths, monkeypatch):
    config_file, workspace_dir = mock_paths

    from nanobot.cli.onboard import OnboardResult

    monkeypatch.setattr(
        "nanobot.cli.onboard.run_onboard",
        lambda initial_config: OnboardResult(config=initial_config, should_save=False),
    )

    result = runner.invoke(app, ["onboard", "--wizard"])

    assert result.exit_code == 0
    assert "No changes were saved" in result.stdout
    assert not config_file.exists()
    assert not workspace_dir.exists()


def test_onboard_uses_explicit_config_and_workspace_paths(tmp_path, monkeypatch):
    config_path = tmp_path / "instance" / "config.json"
    workspace_path = tmp_path / "workspace"

    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)
    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("nanobot.channels.registry.discover_plugins", lambda: {})

    result = runner.invoke(
        app,
        ["onboard", "--config", str(config_path), "--workspace", str(workspace_path)],
    )

    assert result.exit_code == 0
    saved = Config.model_validate(json.loads(config_path.read_text(encoding="utf-8")))
    assert saved.workspace_path == workspace_path
    assert (workspace_path / "AGENTS.md").exists()
    stripped_output = _strip_ansi(result.stdout)
    compact_output = stripped_output.replace("\n", "")
    resolved_config = str(config_path.resolve())
    assert resolved_config in compact_output
    assert f'nanobot webui -c "{resolved_config}"' in result.stdout


def test_onboard_wizard_preserves_explicit_config_in_next_steps(tmp_path, monkeypatch):
    config_path = tmp_path / "instance" / "config.json"
    workspace_path = tmp_path / "workspace"

    from nanobot.cli.onboard import OnboardResult

    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)
    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr(
        "nanobot.cli.onboard.run_onboard",
        lambda initial_config: OnboardResult(config=initial_config, should_save=True),
    )
    monkeypatch.setattr("nanobot.channels.registry.discover_plugins", lambda: {})

    result = runner.invoke(
        app,
        ["onboard", "--wizard", "--config", str(config_path), "--workspace", str(workspace_path)],
    )

    assert result.exit_code == 0
    resolved_config = str(config_path.resolve())
    assert f'nanobot webui -c "{resolved_config}"' in result.stdout


def test_artifacts_path_reports_workspace_scoped_browser_test_dir(tmp_path, monkeypatch):
    runtime_config = Config()
    runtime_config.agents.defaults.workspace = str(tmp_path / "workspace")

    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)

    monkeypatch.setattr(
        "nanobot.cli.commands._load_runtime_config",
        lambda config=None, workspace=None: runtime_config,
    )

    result = runner.invoke(app, ["artifacts", "path"])

    assert result.exit_code == 0
    expected = tmp_path / "workspace" / "artifacts" / "browser-tests"
    assert str(expected) in _strip_ansi(result.stdout).replace("\n", "")
    assert expected.exists()


def test_artifacts_clean_removes_saved_browser_test_artifacts(tmp_path, monkeypatch):
    runtime_config = Config()
    runtime_config.agents.defaults.workspace = str(tmp_path / "workspace")

    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)

    monkeypatch.setattr(
        "nanobot.cli.commands._load_runtime_config",
        lambda config=None, workspace=None: runtime_config,
    )

    artifacts_root = Path(runtime_config.workspace_path) / "artifacts" / "browser-tests"
    run_dir = artifacts_root / "20260408-login-flow"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "desktop-annotated.png").write_bytes(b"png")
    (run_dir / "notes.md").write_text("notes", encoding="utf-8")

    result = runner.invoke(app, ["artifacts", "clean", "--yes"])

    assert result.exit_code == 0
    assert artifacts_root.exists()
    assert list(artifacts_root.iterdir()) == []


def test_artifacts_clean_decline_keeps_workspace_files(tmp_path, monkeypatch):
    runtime_config = Config()
    runtime_config.agents.defaults.workspace = str(tmp_path / "workspace")
    artifacts_root = runtime_config.workspace_path / "artifacts" / "browser-tests"
    artifacts_root.mkdir(parents=True)
    saved = artifacts_root / "notes.md"
    saved.write_text("keep", encoding="utf-8")
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)
    monkeypatch.setattr("nanobot.cli.commands._load_runtime_config", lambda *_: runtime_config)

    result = runner.invoke(app, ["artifacts", "clean"], input="n\n")

    assert result.exit_code == 1
    assert saved.read_text(encoding="utf-8") == "keep"


def test_artifacts_clean_rejects_symlinked_root_outside_workspace(tmp_path, monkeypatch):
    runtime_config = Config()
    runtime_config.agents.defaults.workspace = str(tmp_path / "workspace")
    external = tmp_path / "external"
    external.mkdir()
    saved = external / "notes.md"
    saved.write_text("keep", encoding="utf-8")
    parent = runtime_config.workspace_path / "artifacts"
    parent.mkdir(parents=True)
    (parent / "browser-tests").symlink_to(external, target_is_directory=True)
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)
    monkeypatch.setattr("nanobot.cli.commands._load_runtime_config", lambda *_: runtime_config)

    result = runner.invoke(app, ["artifacts", "clean", "--yes"])

    assert result.exit_code == 1
    assert "must remain inside the workspace" in result.stdout
    assert saved.read_text(encoding="utf-8") == "keep"


def test_artifacts_path_stops_before_config_when_migration_is_incomplete(tmp_path, monkeypatch):
    from nanobot.migration_guard import MigrationIncompleteError

    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)
    monkeypatch.setattr(
        "nanobot.cli.commands._load_runtime_config",
        lambda *_: pytest.fail("migration guard must precede config and artifacts writes"),
    )

    result = runner.invoke(app, ["artifacts", "path", "-w", str(tmp_path / "workspace")])

    assert isinstance(result.exception, MigrationIncompleteError)
    assert not (tmp_path / "workspace").exists()


def test_artifacts_clean_receipt_denial_keeps_workspace_files(tmp_path, monkeypatch):
    from nanobot.agent.memory_recovery_startup_admission import StartupReceiptAdmissionError

    runtime_config = Config()
    runtime_config.agents.defaults.workspace = str(tmp_path / "workspace")
    artifacts_root = runtime_config.workspace_path / "artifacts" / "browser-tests"
    artifacts_root.mkdir(parents=True)
    saved = artifacts_root / "notes.md"
    saved.write_text("keep", encoding="utf-8")

    def deny(_workspace, *, sessions_root):
        raise StartupReceiptAdmissionError()

    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)
    monkeypatch.setattr("nanobot.cli.commands._load_runtime_config", lambda *_: runtime_config)
    monkeypatch.setattr(
        "nanobot.agent.memory_recovery_startup_admission.require_startup_receipt_admission",
        deny,
    )

    result = runner.invoke(app, ["artifacts", "clean", "--yes"])

    assert result.exit_code == 1
    assert "memory recovery required" in result.stdout
    assert saved.read_text(encoding="utf-8") == "keep"


def test_make_provider_wraps_cli_fallback_chain_for_same_provider_models():
    config = Config()
    config.agents.defaults.model = "gemini-3-flash-preview"
    config.agents.defaults.fallback = [
        FallbackEntry(provider="gemini", model="gemma-4-31b-it"),
    ]
    config.providers.gemini.api_key = "test-gemini-key"

    provider = _make_provider(config)

    assert isinstance(provider, LegacyFallbackProvider)
    assert len(provider._candidates) == 2
    assert provider._candidates[0].get_default_model() == "gemini-3-flash-preview"
    assert provider._candidates[1].get_default_model() == "gemma-4-31b-it"

    fallback_provider = provider._candidates[1]
    assert isinstance(fallback_provider, OpenAICompatProvider)
    assert fallback_provider._spec is not None
    assert fallback_provider._spec.name == "gemini"
    assert fallback_provider._effective_base == "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_config_matches_github_copilot_codex_with_hyphen_prefix():
    config = Config()
    config.agents.defaults.model = "github-copilot/gpt-5.3-codex"

    assert config.get_provider_name() == "github_copilot"


def test_config_matches_openai_codex_with_hyphen_prefix():
    config = Config()
    config.agents.defaults.model = "openai-codex/gpt-5.1-codex"

    assert config.get_provider_name() == "openai_codex"


def test_config_matches_opencode_go_with_hyphen_prefix():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"model": "opencode-go/kimi-k2.6"}},
            "providers": {"opencodeGo": {"apiKey": "test-key"}},
        }
    )

    assert config.get_provider_name() == "opencode_go"

    provider = _make_provider(config)
    assert isinstance(provider, OpenAICompatProvider)
    assert provider._effective_base == "https://opencode.ai/zen/go/v1"


def test_config_dump_excludes_oauth_provider_blocks():
    config = Config()

    providers = config.model_dump(by_alias=True)["providers"]

    assert "openaiCodex" not in providers
    assert "githubCopilot" not in providers


def test_config_matches_explicit_ollama_prefix_without_api_key():
    config = Config()
    config.agents.defaults.model = "ollama/llama3.2"

    assert config.get_provider_name() == "ollama"
    assert config.get_api_base() == "http://localhost:11434/v1"


def test_config_explicit_ollama_provider_uses_default_localhost_api_base():
    config = Config()
    config.agents.defaults.provider = "ollama"
    config.agents.defaults.model = "llama3.2"

    assert config.get_provider_name() == "ollama"
    assert config.get_api_base() == "http://localhost:11434/v1"


def test_config_accepts_camel_case_explicit_provider_name_for_coding_plan():
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "provider": "volcengineCodingPlan",
                    "model": "doubao-1-5-pro",
                }
            },
            "providers": {
                "volcengineCodingPlan": {
                    "apiKey": "test-key",
                }
            },
        }
    )

    assert config.get_provider_name() == "volcengine_coding_plan"
    assert config.get_api_base() == "https://ark.cn-beijing.volces.com/api/coding/v3"


def test_find_by_name_accepts_camel_case_and_hyphen_aliases():
    assert find_by_name("volcengineCodingPlan") is not None
    assert find_by_name("volcengineCodingPlan").name == "volcengine_coding_plan"
    assert find_by_name("github-copilot") is not None
    assert find_by_name("github-copilot").name == "github_copilot"


def test_config_auto_detects_ollama_from_local_api_base():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "auto", "model": "llama3.2"}},
            "providers": {"ollama": {"apiBase": "http://localhost:11434/v1"}},
        }
    )

    assert config.get_provider_name() == "ollama"
    assert config.get_api_base() == "http://localhost:11434/v1"


def test_config_prefers_api_keys_list_for_provider_key_resolution():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "auto", "model": "google/gemini-2.5-pro"}},
            "providers": {
                "gemini": {
                    "apiKey": "legacy-single",
                    "apiKeys": ["multi-1", "multi-2"],
                }
            },
        }
    )

    assert config.get_provider_name() == "gemini"
    assert config.get_api_key() == "multi-1"


def test_config_prefers_ollama_over_vllm_when_both_local_providers_configured():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "auto", "model": "llama3.2"}},
            "providers": {
                "vllm": {"apiBase": "http://localhost:8000"},
                "ollama": {"apiBase": "http://localhost:11434/v1"},
            },
        }
    )

    assert config.get_provider_name() == "ollama"
    assert config.get_api_base() == "http://localhost:11434/v1"


def test_config_falls_back_to_vllm_when_ollama_not_configured():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "auto", "model": "llama3.2"}},
            "providers": {
                "vllm": {"apiBase": "http://localhost:8000"},
            },
        }
    )

    assert config.get_provider_name() == "vllm"
    assert config.get_api_base() == "http://localhost:8000"


def test_openai_compat_provider_passes_model_through():
    from nanobot.providers.openai_compat_provider import OpenAICompatProvider

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):
        provider = OpenAICompatProvider(default_model="github-copilot/gpt-5.3-codex")

    assert provider.get_default_model() == "github-copilot/gpt-5.3-codex"


def test_openai_codex_strip_prefix_supports_hyphen_and_underscore():
    assert _strip_model_prefix("openai-codex/gpt-5.1-codex") == "gpt-5.1-codex"
    assert _strip_model_prefix("openai_codex/gpt-5.1-codex") == "gpt-5.1-codex"


def test_make_provider_passes_extra_headers_to_custom_provider():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "custom", "model": "gpt-4o-mini"}},
            "providers": {
                "custom": {
                    "apiKey": "test-key",
                    "apiBase": "https://example.com/v1",
                    "extraHeaders": {
                        "APP-Code": "demo-app",
                        "x-session-affinity": "sticky-session",
                    },
                }
            },
        }
    )

    provider = _make_provider(config)

    assert provider.api_key == "test-key"
    assert provider._effective_base == "https://example.com/v1"
    assert provider._default_headers["APP-Code"] == "demo-app"
    assert provider._default_headers["x-session-affinity"] == "sticky-session"


def test_make_provider_passes_api_keys_pool_to_openai_compat():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "gemini", "model": "google/gemini-2.5-pro"}},
            "providers": {
                "gemini": {
                    "apiKeys": ["gem-key-1", "gem-key-2"],
                    "apiBase": "https://generativelanguage.googleapis.com/v1beta/openai/",
                }
            },
        }
    )

    provider = _make_provider(config)

    assert isinstance(provider, PooledProvider)
    assert [leaf.api_key for leaf in provider._leaves] == ["gem-key-1", "gem-key-2"]
    assert all(isinstance(leaf, OpenAICompatProvider) for leaf in provider._leaves)
    assert all(leaf.provider_name == "gemini" for leaf in provider._leaves)


def test_make_provider_passes_rate_limit_to_openai_compat():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "gemini", "model": "google/gemini-2.5-pro"}},
            "providers": {
                "gemini": {
                    "apiKey": "gem-key-1",
                    "rateLimit": 5,
                }
            },
        }
    )

    provider = _make_provider(config)

    assert isinstance(provider, PooledProvider)
    assert [leaf.api_key for leaf in provider._leaves] == ["gem-key-1"]
    assert provider._pool._rate_limit == 5


@pytest.fixture
def mock_agent_runtime(tmp_path, monkeypatch, classic_cli_runtime):
    """Mock agent command dependencies for focused CLI tests."""
    config = Config()
    config.providers.openrouter.api_key = "test-only-key"
    config.agents.defaults.workspace = str(tmp_path / "default-workspace")
    provider = MagicMock()
    provider.generation.max_tokens = 4096

    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)

    with patch("nanobot.config.loader.load_config", return_value=config) as mock_load_config, \
         patch("nanobot.config.loader.resolve_config_env_vars", side_effect=lambda c: c), \
         patch("nanobot.cli.agent.sync_workspace_templates") as mock_sync_templates, \
         patch("nanobot.providers.factory.make_provider", return_value=provider), \
         patch("nanobot.cli.terminal._print_agent_response") as mock_print_response, \
         patch("nanobot.cron.service.CronService") as mock_cron, \
         patch("nanobot.cli.agent.AgentLoop.from_config") as mock_from_config:

        agent_loop = MagicMock()
        agent_loop.channels_config = None
        agent_loop.process_direct = AsyncMock(
            return_value=OutboundMessage(channel="cli", chat_id="direct", content="mock-response"),
        )
        agent_loop.aclose = AsyncMock(return_value=None)
        mock_from_config.return_value = agent_loop

        yield {
            "config": config,
            "load_config": mock_load_config,
            "sync_templates": mock_sync_templates,
            "from_config": mock_from_config,
            "agent_loop": agent_loop,
            "cron": mock_cron,
            "print_response": mock_print_response,
        }


def test_agent_help_shows_workspace_and_config_options(monkeypatch):
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)
    result = runner.invoke(app, ["agent", "--help"])

    assert result.exit_code == 0
    stripped_output = _strip_ansi(result.stdout)
    assert "--workspace" in stripped_output
    assert "-w" in stripped_output
    assert "--config" in stripped_output
    assert "-c" in stripped_output


def test_agent_uses_default_config_when_no_workspace_or_config_flags(mock_agent_runtime):
    result = runner.invoke(app, ["agent", "-m", "hello"])

    assert result.exit_code == 0
    assert mock_agent_runtime["load_config"].call_args.args == (None,)
    assert mock_agent_runtime["sync_templates"].call_args.args == (
        mock_agent_runtime["config"].workspace_path,
    )
    passed_config = mock_agent_runtime["from_config"].call_args.args[0]
    assert passed_config.workspace_path == mock_agent_runtime["config"].workspace_path
    mock_agent_runtime["agent_loop"].process_direct.assert_awaited_once()
    mock_agent_runtime["print_response"].assert_called_once_with(
        "mock-response", render_markdown=True, metadata={},
    )


def test_agent_logs_flag_prints_explicit_runtime_logs_banner(mock_agent_runtime):
    with patch("loguru.logger.remove"), \
         patch("loguru.logger.add"), \
         patch("loguru.logger.enable"):
        result = runner.invoke(app, ["agent", "-m", "hello", "--logs"])

    assert result.exit_code == 0
    stripped_output = _strip_ansi(result.stdout)
    assert "Runtime logs enabled (--logs)." in stripped_output
    assert "Use --no-logs to hide internal logs." in stripped_output


def test_agent_log_file_adds_rotating_file_sink(mock_agent_runtime, tmp_path: Path):
    log_path = tmp_path / "agent.log"
    with patch("loguru.logger.remove"), \
         patch("loguru.logger.add") as add_log, \
         patch("loguru.logger.enable"):
        result = runner.invoke(
            app,
            ["agent", "-m", "hello", "--log-file", str(log_path)],
        )

    assert result.exit_code == 0
    add_log.assert_called_once_with(
        str(log_path),
        rotation="20 MB",
        retention="7 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        encoding="utf-8",
        enqueue=True,
    )
    output = _strip_ansi(result.stdout)
    assert "Logging to" in output
    assert "agent.log" in output


def test_agent_uses_explicit_config_path(mock_agent_runtime, tmp_path: Path):
    config_path = tmp_path / "agent-config.json"
    config_path.write_text("{}")

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_path)])

    assert result.exit_code == 0
    assert mock_agent_runtime["load_config"].call_args.args == (config_path.resolve(),)


def test_agent_config_sets_active_path(mock_agent_runtime, monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    seen: dict[str, Path] = {}

    monkeypatch.setattr(
        "nanobot.config.loader.set_config_path",
        lambda path: seen.__setitem__("config_path", path),
    )
    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_file)])

    assert result.exit_code == 0
    assert seen["config_path"] == config_file.resolve()
    assert mock_agent_runtime["load_config"].call_args.args == (config_file.resolve(),)


def test_agent_uses_workspace_directory_for_cron_store(mock_agent_runtime, tmp_path: Path) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = mock_agent_runtime["config"]
    config.agents.defaults.workspace = str(tmp_path / "agent-workspace")

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_file)])

    assert result.exit_code == 0
    mock_agent_runtime["cron"].assert_called_once_with(
        config.workspace_path / "cron" / "jobs.json"
    )


def test_agent_workspace_override_does_not_migrate_legacy_cron(
    mock_agent_runtime, monkeypatch, tmp_path: Path
) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    legacy_dir = tmp_path / "global" / "cron"
    legacy_dir.mkdir(parents=True)
    legacy_file = legacy_dir / "jobs.json"
    legacy_file.write_text('{"jobs": []}')

    override = tmp_path / "override-workspace"
    monkeypatch.setattr("nanobot.config.paths.get_cron_dir", lambda: legacy_dir)

    result = runner.invoke(
        app,
        ["agent", "-m", "hello", "-c", str(config_file), "-w", str(override)],
    )

    assert result.exit_code == 0
    mock_agent_runtime["cron"].assert_called_once_with(override / "cron" / "jobs.json")
    assert legacy_file.exists()
    assert not (override / "cron" / "jobs.json").exists()


def test_agent_custom_config_workspace_does_not_migrate_legacy_cron(
    mock_agent_runtime, monkeypatch, tmp_path: Path
) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    legacy_dir = tmp_path / "global" / "cron"
    legacy_dir.mkdir(parents=True)
    legacy_file = legacy_dir / "jobs.json"
    legacy_file.write_text('{"jobs": []}')

    custom_workspace = tmp_path / "custom-workspace"
    config = mock_agent_runtime["config"]
    config.agents.defaults.workspace = str(custom_workspace)
    monkeypatch.setattr("nanobot.config.paths.get_cron_dir", lambda: legacy_dir)

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_file)])

    assert result.exit_code == 0
    mock_agent_runtime["cron"].assert_called_once_with(
        custom_workspace / "cron" / "jobs.json"
    )
    assert legacy_file.exists()
    assert not (custom_workspace / "cron" / "jobs.json").exists()


def test_agent_overrides_workspace_path(mock_agent_runtime, tmp_path: Path):
    workspace_path = tmp_path / "agent-workspace"

    result = runner.invoke(app, ["agent", "-m", "hello", "-w", str(workspace_path)])

    assert result.exit_code == 0
    assert mock_agent_runtime["config"].agents.defaults.workspace == str(workspace_path)
    assert mock_agent_runtime["sync_templates"].call_args.args == (workspace_path,)
    assert mock_agent_runtime["from_config"].call_args.args[0].workspace_path == workspace_path


def test_agent_workspace_override_wins_over_config_workspace(mock_agent_runtime, tmp_path: Path):
    config_path = tmp_path / "agent-config.json"
    config_path.write_text("{}")
    workspace_path = tmp_path / "agent-workspace"

    result = runner.invoke(
        app,
        ["agent", "-m", "hello", "-c", str(config_path), "-w", str(workspace_path)],
    )

    assert result.exit_code == 0
    assert mock_agent_runtime["load_config"].call_args.args == (config_path.resolve(),)
    assert mock_agent_runtime["config"].agents.defaults.workspace == str(workspace_path)
    assert mock_agent_runtime["sync_templates"].call_args.args == (workspace_path,)
    assert mock_agent_runtime["from_config"].call_args.args[0].workspace_path == workspace_path


def test_agent_hints_about_deprecated_memory_window(mock_agent_runtime, tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"agents": {"defaults": {"memoryWindow": 42}}}))

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_file)])

    assert result.exit_code == 0
    assert "memoryWindow" in result.stdout
    assert "no longer used" in result.stdout


@pytest.mark.parametrize("legacy_key", ["contextWindowTokens", "maxInputTokens"])
def test_agent_hints_about_legacy_input_limit_keys(mock_agent_runtime, tmp_path, legacy_key):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"agents": {"defaults": {legacy_key: 42}}}))

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_file)])

    assert result.exit_code == 0
    assert legacy_key in result.stdout
    assert "maxTokens.input" in result.stdout


def test_heartbeat_retains_recent_messages_by_default():
    config = Config()

    assert config.gateway.heartbeat.keep_recent_messages == 8


@pytest.mark.asyncio
async def test_cron_execution_uses_fresh_detail_and_links_exact_visible_pair(
    tmp_path: Path, admitted_async, monkeypatch,
) -> None:
    """Current contract: fresh detail plus an exact metadata link, not mirrored turns."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.turn import HistoryMode, TurnRequest, TurnSource
    from nanobot.cron.canonical_execution import execute_claimed_fresh_text
    from nanobot.cron.occurrence_store import OccurrenceExecutionStartedError
    from nanobot.providers.base import LLMResponse
    from tests.cron.test_canonical_execution import ScriptedProvider, _service_claim

    service, claim = _service_claim(tmp_path)
    detail_key = f"cron:{claim.identity.job_id}"
    provider = ScriptedProvider([LLMResponse(content="finished report")])

    def build(inputs):
        return AgentLoop.from_config(
            inputs.config, inputs.bus, provider=provider, model="synthetic-model",
            context_window_tokens=200_000, session_manager=inputs.sessions,
            tool_registry=inputs.tools, cron_service=inputs.cron,
            turn_delivery_factory=inputs.delivery, enable_legacy_auto_compact=False,
        )

    agent = await admitted_async(tmp_path, build, keys=(detail_key, "telegram:primary"))
    sessions = agent.sessions
    detail = sessions.get_or_create(detail_key)
    detail.add_run_message("old-run", "user", "older instruction")
    detail.add_run_message("old-run", "assistant", "older result")
    sessions.save(detail)
    execute = AsyncMock(wraps=agent.execute_silent_cron_turn)
    monkeypatch.setattr(agent, "execute_silent_cron_turn", execute)
    ordinary_send = AsyncMock(side_effect=AssertionError("cron must use durable delivery"))
    monkeypatch.setattr(agent.bus, "publish_outbound", ordinary_send)
    monkeypatch.setattr(agent.bus, "publish_event", ordinary_send)

    result, projection, outbox = await execute_claimed_fresh_text(
        service, agent, claim, visible_sessions=sessions,
    )

    request = execute.await_args.args[0]
    assert isinstance(request, TurnRequest)
    assert request.source is TurnSource.CRON
    assert request.session_key == detail_key
    assert request.history_mode is HistoryMode.FRESH
    assert request.scheduled_link is not None
    assert request.scheduled_link.instruction == "compute report"
    assert request.scheduled_link.visible_session_key == "telegram:primary"
    assert request.scheduled_link.occurrence_id == claim.occurrence_id
    assert request.callbacks.on_progress is None
    assert result.run_id == projection.run_id == outbox.run_id == claim.run_id
    assert "older instruction" not in str(provider.calls[0][0])
    assert "older result" not in str(provider.calls[0][0])
    detail = sessions.get_or_create(detail_key)
    assert [message["content"] for message in detail.get_run_messages("old-run")] == [
        "older instruction",
        "older result",
    ]
    assert [(message["role"], message["content"])
            for message in detail.get_run_messages(claim.run_id)] == [
        ("user", "compute report"), ("assistant", "finished report"),
    ]
    visible = sessions.get_or_create("telegram:primary")
    assert visible.messages == []
    linked = visible.recent_scheduled_runs()
    assert len(linked) == 1
    assert linked[0]["run_id"] == claim.run_id
    assert linked[0]["occurrence_id"] == claim.occurrence_id
    assert linked[0]["detail_ref"] == {"session_key": detail_key, "run_id": claim.run_id}
    assert linked[0]["instruction"] == "compute report"
    assert linked[0]["result"] == "finished report"
    assert claim.run_id in visible.get_model_history()[0]["content"]
    assert service.project_occurrence_completion(claim.occurrence_id) == projection
    assert outbox.messages == ({"content": "finished report", "media": []},)
    ordinary_send.assert_not_awaited()
    with pytest.raises(OccurrenceExecutionStartedError):
        await execute_claimed_fresh_text(service, agent, claim, visible_sessions=sessions)
    assert len(provider.calls) == 1
    execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_cron_empty_canonical_ledger_ignores_stale_message_tool_delivery(
    tmp_path: Path, isolated_cli_effects,
) -> None:
    from nanobot.agent.tools.message import MessageTool, capture_message_deliveries
    from nanobot.agent.turn import TurnResult
    from nanobot.channels.delivery_contract import ChannelTextPartReceipt, ChannelTextReceipt
    from nanobot.cron.canonical_execution import execute_claimed_fresh_text
    from nanobot.cron.channel_sender import CronChannelSender
    from nanobot.cron.delivery_runtime import ConfirmedDelivered
    from tests.cron.test_canonical_execution import _service_claim

    service, claim = _service_claim(tmp_path)
    stale_send = AsyncMock()
    message_tool = MessageTool(
        send_callback=stale_send,
        default_channel="telegram",
        default_chat_id="primary",
        workspace=tmp_path / "workspace",
    )
    previous_records = []
    with capture_message_deliveries(previous_records):
        await message_tool.execute("finished report")
    assert len(previous_records) == 1
    assert previous_records[0].content == "finished report"

    agent = SimpleNamespace(
        tools=SimpleNamespace(get=MagicMock(return_value=message_tool)),
        execute_silent_cron_turn=AsyncMock(return_value=TurnResult(
            run_id=claim.run_id, content="finished report",
            final_content="finished report", sent_messages=(),
        )),
    )
    # Even an ambient older capture must not override this result's exact ledger.
    with capture_message_deliveries(previous_records):
        result, projection, outbox = await execute_claimed_fresh_text(service, agent, claim)

    assert result.sent_messages == ()
    assert projection.run_id == claim.run_id
    assert outbox.messages == ({"content": "finished report", "media": []},)
    assert [(target.channel, target.to) for target in outbox.destinations] == [
        ("telegram", "primary"), ("discord", "secondary"),
    ]
    agent.tools.get.assert_not_called()
    stale_send.assert_awaited_once()
    published = []

    class ReceiptChannel:
        def __init__(self, name):
            self.name = name

        async def send_text_with_receipt(self, target, messages, *, action_id, request_digest):
            published.append((self.name, target, messages))
            return ChannelTextReceipt(
                channel=self.name, target=target, action_id=action_id,
                request_digest=request_digest,
                parts=(ChannelTextPartReceipt(0, 0, 0, len(published), 1),),
            )

    sender = CronChannelSender(lambda name: ReceiptChannel(name))
    runtime = service.delivery_runtime
    assert runtime is not None
    for target in outbox.destinations:
        current = runtime.get(outbox.outbox_id)
        delivery_claim = runtime.claim(
            current.outbox_id, target.destination_id, current.record_revision,
            owner="test-sender",
        )
        outcome = await runtime.dispatch(delivery_claim, sender=sender)
        assert isinstance(outcome, ConfirmedDelivered)

    assert published == [
        ("telegram", "primary", ("finished report",)),
        ("discord", "secondary", ("finished report",)),
    ]
    agent.execute_silent_cron_turn.assert_awaited_once()
    stale_send.assert_awaited_once()


def _write_instance_config(tmp_path: Path) -> Path:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")
    return config_file


def _stop_gateway_provider(_config) -> object:
    raise _StopGatewayError("stop")


def _patch_cli_command_runtime(
    monkeypatch,
    config: Config,
    *,
    set_config_path=None,
    sync_templates=None,
    make_provider=None,
    message_bus=None,
    session_manager=None,
    cron_service=None,
    get_cron_dir=None,
) -> None:
    from nanobot.providers.factory import ProviderSnapshot

    provider_factory = make_provider or (lambda _config: MagicMock())

    def load_config(path=None):
        assert path is not None
        config.bind_source_path(Path(path))
        if config.agents.defaults.workspace == Config().agents.defaults.workspace:
            config.agents.defaults.workspace = str(Path(path).parent / "workspace")
        return config

    # This module checks CLI behavior behind the production migration guard.
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)

    def provider_snapshot(target_config: Config) -> ProviderSnapshot:
        return ProviderSnapshot(
            provider=provider_factory(target_config),
            model=target_config.agents.defaults.model,
            context_window_tokens=target_config.agents.defaults.context_window_tokens,
            signature=("fork-cli-test",),
        )

    monkeypatch.setattr(
        "nanobot.config.loader.set_config_path",
        set_config_path or (lambda _path: None),
    )
    monkeypatch.setattr("nanobot.config.loader.load_config", load_config)
    monkeypatch.setattr("nanobot.config.loader.resolve_config_env_vars", lambda loaded: loaded)
    monkeypatch.setattr(
        "nanobot.cli.gateway_runtime.sync_workspace_templates",
        sync_templates or (lambda _path: None),
    )
    monkeypatch.setattr("nanobot.providers.factory.make_provider", provider_factory)
    monkeypatch.setattr(
        "nanobot.providers.factory.build_provider_snapshot",
        provider_snapshot,
    )
    monkeypatch.setattr(
        "nanobot.providers.factory.load_provider_snapshot",
        lambda _config_path=None: provider_snapshot(config),
    )
    monkeypatch.setattr("nanobot.cli.webui_support._provider_setup_error", lambda _config: None)
    monkeypatch.setattr("nanobot.cli.gateway_runtime._tcp_endpoint_reachable", lambda *_a: False)
    monkeypatch.setattr("nanobot.cli.gateway_runtime._webui_endpoint_reachable", lambda *_a: False)

    if message_bus is not None:
        monkeypatch.setattr("nanobot.bus.queue.MessageBus", message_bus)
    if session_manager is not None:
        monkeypatch.setattr("nanobot.session.manager.SessionManager", session_manager)
    if cron_service is not None:
        monkeypatch.setattr("nanobot.cron.service.CronService", cron_service)
    if get_cron_dir is not None:
        monkeypatch.setattr("nanobot.config.paths.get_cron_dir", get_cron_dir)


def _patch_serve_runtime(monkeypatch, config: Config, seen: dict[str, object]) -> None:
    from nanobot.api.server import create_app
    from nanobot.runtime_composition import RuntimeComposition, RuntimeState

    def forbidden(*_args, **_kwargs):
        raise AssertionError("API option parsing must not acquire runtime resources")

    def _record_create_app(
        *, model_name: str, request_timeout: float, api_key: str,
        composition: RuntimeComposition,
    ):
        assert composition.state is RuntimeState.NEW
        seen["composition"] = composition
        seen["workspace"] = composition.preflight.workspace
        seen["model_name"] = model_name
        seen["request_timeout"] = request_timeout
        seen["api_key"] = api_key
        return create_app(
            model_name=model_name, request_timeout=request_timeout,
            api_key=api_key, composition=composition,
        )

    def _fake_run_app(api_app, host: str, port: int, print):
        assert api_app.cleanup_ctx
        assert seen["composition"].state is RuntimeState.NEW
        seen["api_app"] = api_app
        seen["host"] = host
        seen["port"] = port

    _patch_cli_command_runtime(monkeypatch, config, make_provider=forbidden)
    monkeypatch.setattr(RuntimeComposition, "acquire", forbidden)
    monkeypatch.setattr("nanobot.cli.commands.sync_workspace_templates", forbidden)
    monkeypatch.setattr("nanobot.api.server.create_app", _record_create_app)
    monkeypatch.setattr("aiohttp.web.run_app", _fake_run_app)


def test_gateway_uses_workspace_from_config_by_default(monkeypatch, tmp_path: Path, gateway_graphs) -> None:
    config_file = _write_instance_config(tmp_path)
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    seen: dict[str, Path] = {}

    def stop_provider(runtime_config):
        seen["workspace"] = runtime_config.workspace_path
        raise _StopGatewayError("stop")

    _patch_cli_command_runtime(
        monkeypatch,
        config,
        set_config_path=lambda path: seen.__setitem__("config_path", path),
        sync_templates=lambda _path: pytest.fail("failed acquisition must not write templates"),
        make_provider=stop_provider,
    )

    result = runner.invoke(app, ["gateway", "--config", str(config_file)])

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["config_path"] == config_file.resolve()
    assert seen["workspace"] == Path(config.agents.defaults.workspace)


def test_gateway_workspace_option_overrides_config(monkeypatch, tmp_path: Path, gateway_graphs) -> None:
    config_file = _write_instance_config(tmp_path)
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    override = tmp_path / "override-workspace"
    seen: dict[str, Path] = {}

    def stop_provider(runtime_config):
        seen["workspace"] = runtime_config.workspace_path
        raise _StopGatewayError("stop")

    _patch_cli_command_runtime(
        monkeypatch,
        config,
        sync_templates=lambda _path: pytest.fail("failed acquisition must not write templates"),
        make_provider=stop_provider,
    )

    result = runner.invoke(
        app,
        ["gateway", "--config", str(config_file), "--workspace", str(override)],
    )

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["workspace"] == override
    assert config.workspace_path == override


def test_gateway_uses_workspace_directory_for_cron_store(monkeypatch, tmp_path: Path, gateway_graphs) -> None:
    config_file = _write_instance_config(tmp_path)
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    seen: dict[str, Path] = {}

    class _StopCron:
        def __init__(self, store_path: Path) -> None:
            seen["cron_store"] = store_path
            raise _StopGatewayError("stop")

    _patch_cli_command_runtime(
        monkeypatch,
        config,
        cron_service=_StopCron,
    )

    result = runner.invoke(app, ["gateway", "--config", str(config_file)])

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["cron_store"] == config.workspace_path / "cron" / "jobs.json"


def test_gateway_workspace_override_does_not_migrate_legacy_cron(
    monkeypatch, tmp_path: Path, gateway_graphs,
) -> None:
    config_file = _write_instance_config(tmp_path)
    legacy_dir = tmp_path / "global" / "cron"
    legacy_dir.mkdir(parents=True)
    legacy_file = legacy_dir / "jobs.json"
    legacy_file.write_text('{"jobs": []}')

    override = tmp_path / "override-workspace"
    config = Config()
    seen: dict[str, Path] = {}

    class _StopCron:
        def __init__(self, store_path: Path) -> None:
            seen["cron_store"] = store_path
            raise _StopGatewayError("stop")

    _patch_cli_command_runtime(
        monkeypatch,
        config,
        cron_service=_StopCron,
        get_cron_dir=lambda: legacy_dir,
    )

    result = runner.invoke(
        app,
        ["gateway", "--config", str(config_file), "--workspace", str(override)],
    )

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["cron_store"] == override / "cron" / "jobs.json"
    assert legacy_file.exists()
    assert not (override / "cron" / "jobs.json").exists()


def test_gateway_custom_config_workspace_does_not_migrate_legacy_cron(
    monkeypatch, tmp_path: Path, gateway_graphs,
) -> None:
    config_file = _write_instance_config(tmp_path)
    legacy_dir = tmp_path / "global" / "cron"
    legacy_dir.mkdir(parents=True)
    legacy_file = legacy_dir / "jobs.json"
    legacy_file.write_text('{"jobs": []}')

    custom_workspace = tmp_path / "custom-workspace"
    config = Config()
    config.agents.defaults.workspace = str(custom_workspace)
    seen: dict[str, Path] = {}

    class _StopCron:
        def __init__(self, store_path: Path) -> None:
            seen["cron_store"] = store_path
            raise _StopGatewayError("stop")

    _patch_cli_command_runtime(
        monkeypatch,
        config,
        cron_service=_StopCron,
        get_cron_dir=lambda: legacy_dir,
    )

    result = runner.invoke(app, ["gateway", "--config", str(config_file)])

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["cron_store"] == custom_workspace / "cron" / "jobs.json"
    assert legacy_file.exists()
    assert not (custom_workspace / "cron" / "jobs.json").exists()


def test_migrate_cron_store_moves_legacy_file(tmp_path: Path) -> None:
    """Legacy global jobs.json is moved into the workspace on first run."""
    from nanobot.cli.runtime_config import _migrate_cron_store

    legacy_dir = tmp_path / "global" / "cron"
    legacy_dir.mkdir(parents=True)
    legacy_file = legacy_dir / "jobs.json"
    legacy_file.write_text('{"jobs": []}')

    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "workspace")
    workspace_cron = config.workspace_path / "cron" / "jobs.json"

    with patch("nanobot.config.paths.get_cron_dir", return_value=legacy_dir):
        _migrate_cron_store(config)

    assert workspace_cron.exists()
    assert workspace_cron.read_text() == '{"jobs": []}'
    assert not legacy_file.exists()


def test_migrate_cron_store_skips_when_workspace_file_exists(tmp_path: Path) -> None:
    """Migration does not overwrite an existing workspace cron store."""
    from nanobot.cli.runtime_config import _migrate_cron_store

    legacy_dir = tmp_path / "global" / "cron"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "jobs.json").write_text('{"old": true}')

    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "workspace")
    workspace_cron = config.workspace_path / "cron" / "jobs.json"
    workspace_cron.parent.mkdir(parents=True)
    workspace_cron.write_text('{"new": true}')

    with patch("nanobot.config.paths.get_cron_dir", return_value=legacy_dir):
        _migrate_cron_store(config)

    assert workspace_cron.read_text() == '{"new": true}'


def _probe_gateway_health_port(monkeypatch, seen):
    """The selected port reaches the health check before any listener is bound."""
    class Composition:
        async def acquire(self):
            return SimpleNamespace(
                agent=SimpleNamespace(stop=lambda: seen.__setitem__("stopped", True)),
                channels=SimpleNamespace(), bus=SimpleNamespace(drain=AsyncMock()),
            )

        async def aclose(self):
            seen["closed"] = True

    def probe(host, port):
        seen["port"] = port
        raise _StopGatewayError("stop before socket bind")

    monkeypatch.setattr(
        "nanobot.runtime_composition.RuntimeComposition.define",
        lambda *_args, **_kwargs: Composition(),
    )
    monkeypatch.setattr("nanobot.cli.gateway_runtime._tcp_endpoint_reachable", probe)
    monkeypatch.setattr(
        "nanobot.cli.gateway_runtime._install_gateway_shutdown_handlers",
        lambda *_args: lambda: seen.__setitem__("restored", True),
    )


def test_gateway_uses_configured_port_when_cli_flag_is_missing(monkeypatch, tmp_path: Path, isolated_cli_effects) -> None:
    config_file = _write_instance_config(tmp_path)
    config = Config()
    config.gateway.port = 18791
    seen = {}

    _patch_cli_command_runtime(
        monkeypatch,
        config,
        make_provider=_stop_gateway_provider,
    )
    _probe_gateway_health_port(monkeypatch, seen)

    result = runner.invoke(app, ["gateway", "--config", str(config_file)])

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["port"] == 18791
    assert seen["stopped"] and seen["closed"] and seen["restored"]


def test_gateway_cli_port_overrides_configured_port(monkeypatch, tmp_path: Path, isolated_cli_effects) -> None:
    config_file = _write_instance_config(tmp_path)
    config = Config()
    config.gateway.port = 18791
    seen = {}

    _patch_cli_command_runtime(
        monkeypatch,
        config,
        make_provider=_stop_gateway_provider,
    )
    _probe_gateway_health_port(monkeypatch, seen)

    result = runner.invoke(app, ["gateway", "--config", str(config_file), "--port", "18792"])

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["port"] == 18792
    assert seen["stopped"] and seen["closed"] and seen["restored"]


def test_serve_uses_api_config_defaults_and_workspace_override(
    monkeypatch, tmp_path: Path, isolated_cli_effects,
) -> None:
    config_file = _write_instance_config(tmp_path)
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    config.api.host = "127.0.0.2"
    config.api.port = 18900
    config.api.timeout = 45.0
    override_workspace = tmp_path / "override-workspace"
    seen: dict[str, object] = {}

    _patch_serve_runtime(monkeypatch, config, seen)

    result = runner.invoke(
        app,
        ["serve", "--config", str(config_file), "--workspace", str(override_workspace)],
    )

    assert result.exit_code == 0
    assert seen["workspace"] == override_workspace
    assert seen["host"] == "127.0.0.2"
    assert seen["port"] == 18900
    assert seen["request_timeout"] == 45.0


def test_serve_cli_options_override_api_config(monkeypatch, tmp_path: Path, isolated_cli_effects) -> None:
    config_file = _write_instance_config(tmp_path)
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    config.api.host = "127.0.0.2"
    config.api.port = 18900
    config.api.timeout = 45.0
    seen: dict[str, object] = {}

    _patch_serve_runtime(monkeypatch, config, seen)

    result = runner.invoke(
        app,
        [
            "serve",
            "--config",
            str(config_file),
            "--host",
            "127.0.0.1",
            "--port",
            "18901",
            "--timeout",
            "46",
        ],
    )

    assert result.exit_code == 0
    assert seen["host"] == "127.0.0.1"
    assert seen["port"] == 18901
    assert seen["request_timeout"] == 46.0


def test_channels_login_requires_channel_name(monkeypatch) -> None:
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)
    result = runner.invoke(app, ["channels", "login"])

    assert result.exit_code == 2
