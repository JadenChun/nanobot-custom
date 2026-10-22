import asyncio
import json
import re
import signal
import urllib.error
from contextlib import contextmanager, suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from typer.testing import CliRunner

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.cli import gateway_runtime as cli_gateway_runtime
from nanobot.cli import provider as provider_commands
from nanobot.cli import terminal as cli_terminal
from nanobot.cli import webui as cli_webui
from nanobot.cli import webui_support as cli_webui_support
from nanobot.cli.commands import app
from nanobot.config.schema import Config
from nanobot.cron.webui_metadata import cron_proactive_delivery_metadata
from nanobot.providers.factory import ProviderSnapshot, make_provider, provider_signature
from nanobot.providers.openai_codex_provider import _strip_model_prefix
from nanobot.providers.registry import find_by_name
from nanobot.webui.dev import WebUIDevError
from nanobot.webui.metadata import (
    WEBUI_MESSAGE_SOURCE_METADATA_KEY,
    WEBUI_TURN_METADATA_KEY,
)
from tests.cli import p6_classic_cli_fixtures

runner = CliRunner()
classic_cli_runtime = p6_classic_cli_fixtures.classic_cli_runtime


@pytest.fixture(autouse=True)
def _admit_candidate_for_cli_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise CLI compatibility without enabling the guarded candidate.

    Startup-denial tests own the real guard; this module uses isolated fakes to
    reach the command behavior behind it.
    """
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", lambda _title: None)


def _normalized_rendered_text(output: str) -> str:
    return " ".join(output.split())


@pytest.mark.parametrize("output", [
    "existing values preserved",
    "existing values\npreserved",
    "existing\n  values\n preserved\n",
])
def test_normalized_rendered_text_preserves_word_boundaries(output):
    assert _normalized_rendered_text(output) == "existing values preserved"


def test_proactive_websocket_delivery_gets_fresh_turn_id() -> None:
    metadata = {
        "webui": True,
        WEBUI_TURN_METADATA_KEY: "turn-that-created-the-reminder",
        "workspace_scope": {"mode": "default"},
    }

    out = cron_proactive_delivery_metadata(
        "websocket",
        metadata,
        turn_seed="cron:drink-water",
        source_label="drink water",
    )

    assert out["webui"] is True
    assert out["workspace_scope"] == {"mode": "default"}
    assert out[WEBUI_TURN_METADATA_KEY].startswith("cron:drink-water:")
    assert out[WEBUI_TURN_METADATA_KEY] != metadata[WEBUI_TURN_METADATA_KEY]
    assert out[WEBUI_MESSAGE_SOURCE_METADATA_KEY] == {"kind": "cron", "label": "drink water"}


def _fake_provider():
    """Return a minimal fake provider that satisfies AgentLoop.__init__."""
    p = MagicMock()
    p.generation.max_tokens = 4096
    return p


class _StopGatewayError(RuntimeError):
    pass


class _GatewayAgentContractStub:
    """Minimal stable AgentLoop surface required by gateway assembly tests."""

    tools = ToolRegistry()

    @staticmethod
    def mcp_runtime_status() -> dict[str, str]:
        return {}

    @staticmethod
    def pending_cron_job_ids_for_session(_session_key: str) -> set[str]:
        return set()

    @staticmethod
    def pending_local_trigger_ids_for_session(_session_key: str) -> set[str]:
        return set()

    async def submit_local_trigger_turn(
        self,
        _msg: InboundMessage,
    ) -> OutboundMessage | None:
        return None

    def preserve_inflight_turns_on_shutdown(self) -> None:
        return None


class _EmptyGatewaySessionManager:
    """Minimal session-manager contract for gateway assembly tests."""

    def list_sessions(self) -> list[dict[str, object]]:
        return []

    def flush_all(self) -> int:
        return 0


def test_gateway_signal_handler_first_signal_stops_and_second_forces() -> None:
    class _FakeLoop:
        def __init__(self) -> None:
            self.handlers: dict[int, tuple[object, tuple[object, ...]]] = {}
            self.removed: list[int] = []

        def add_signal_handler(self, signum, callback, *args) -> None:
            self.handlers[int(signum)] = (callback, args)

        def remove_signal_handler(self, signum) -> bool:
            self.removed.append(int(signum))
            self.handlers.pop(int(signum), None)
            return True

    async def _run() -> None:
        loop = _FakeLoop()
        shutdown_event = asyncio.Event()
        never = asyncio.Event()
        task = asyncio.create_task(never.wait())
        output: list[str] = []

        restore = cli_gateway_runtime._install_gateway_shutdown_handlers(
            loop, shutdown_event, [task], output.append,
        )
        try:
            callback, args = loop.handlers[int(signal.SIGINT)]
            assert callable(callback)

            callback(*args)
            assert shutdown_event.is_set()
            assert output == ["\nShutting down... Press Ctrl+C again to force."]
            assert not task.done()

            callback(*args)
            await asyncio.sleep(0)
            assert task.cancelled()
        finally:
            restore()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        assert int(signal.SIGINT) in loop.removed
        assert int(signal.SIGTERM) in loop.removed

    asyncio.run(_run())


def test_interactive_tty_mode_restores_line_input(monkeypatch) -> None:
    try:
        import os
        import pty
        import termios
    except ImportError:  # pragma: no cover - platform without POSIX termios
        pytest.skip("termios unavailable")

    master_fd, slave_fd = pty.openpty()

    class _Stdin:
        def fileno(self) -> int:
            return slave_fd

    try:
        attrs = termios.tcgetattr(slave_fd)
        attrs[0] &= ~termios.ICRNL
        attrs[0] |= termios.IGNCR
        attrs[3] &= ~(termios.ISIG | termios.ICANON | termios.ECHO)
        termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)

        monkeypatch.setattr(cli_terminal.sys, "stdin", _Stdin())
        cli_terminal._ensure_interactive_tty_mode()

        restored = termios.tcgetattr(slave_fd)
        assert restored[0] & termios.ICRNL
        assert not restored[0] & termios.IGNCR
        assert restored[3] & termios.ISIG
        assert restored[3] & termios.ICANON
        assert restored[3] & termios.ECHO
    finally:
        os.close(master_fd)
        os.close(slave_fd)


def test_webui_restores_tty_before_loading_config(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text("{}", encoding="utf-8")
    calls: list[str] = []
    original_resolve = cli_webui._resolve_webui_config_path

    monkeypatch.setattr(
        cli_terminal,
        "_ensure_interactive_tty_mode",
        lambda: calls.append("tty"),
    )
    monkeypatch.setattr(
        cli_webui,
        "_resolve_webui_config_path",
        lambda path: calls.append("config") or original_resolve(path),
    )
    _patch_webui_provider_ready(monkeypatch)
    monkeypatch.setattr(cli_webui, "sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr(cli_webui, "_gateway_health_ready", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(cli_webui, "_webui_endpoint_reachable", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(cli_webui, "_tcp_endpoint_reachable", lambda *_args, **_kwargs: False)
    _patch_webui_managed_gateway(monkeypatch)

    result = runner.invoke(app, ["webui", "--config", str(config_file), "--yes", "--no-open"])

    assert result.exit_code == 0
    assert calls[:2] == ["tty", "config"]








@pytest.fixture
def mock_paths(tmp_path: Path):
    """Mock config/workspace paths for test isolation."""
    def configured_default() -> Config:
        config = Config()
        config.agents.defaults.workspace = str(workspace_dir)
        return config

    with patch("nanobot.config.loader.get_config_path") as mock_cp, \
         patch("nanobot.config.loader.save_config") as mock_sc, \
         patch("nanobot.config.loader.load_config") as mock_lc, \
         patch("nanobot.config.schema.Config", side_effect=configured_default):
        base_dir = tmp_path / "test_onboard_data"
        base_dir.mkdir()

        config_file = base_dir / "config.json"
        workspace_dir = base_dir / "workspace"

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
    assert json.loads(config_file.read_text())["agents"]["defaults"]["workspace"] == str(workspace_dir)


def test_onboard_recommends_webui(mock_paths):
    """Default onboarding should recommend the guided WebUI launcher."""
    result = runner.invoke(app, ["onboard"])

    assert result.exit_code == 0
    assert "✓ nanobot is ready. Run: nanobot webui" in result.stdout


def test_onboard_existing_config_refresh(mock_paths):
    """Config exists, user declines overwrite — should refresh (load-merge-save)."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "existing values preserved" in _normalized_rendered_text(result.stdout)
    assert workspace_dir.exists()
    assert (workspace_dir / "AGENTS.md").exists()


def test_onboard_existing_config_refresh_non_interactive(mock_paths):
    """Config exists, user specifies --refresh — should refresh non-interactively (no prompt)."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard", "--refresh"])

    assert result.exit_code == 0
    assert "Config already exists" not in result.stdout
    assert "existing values preserved" in _normalized_rendered_text(result.stdout)
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


def test_onboard_help_shows_workspace_and_config_options():
    result = runner.invoke(app, ["onboard", "--help"])

    assert result.exit_code == 0
    stripped_output = _strip_ansi(result.stdout)
    assert "--workspace" in stripped_output
    assert "-w" in stripped_output
    assert "--config" in stripped_output
    assert "-c" in stripped_output
    assert "--wizard" in stripped_output
    assert "--refresh" in stripped_output
    assert "--dir" not in stripped_output


def test_status_help_shows_workspace_and_config_options():
    result = runner.invoke(app, ["status", "--help"])

    assert result.exit_code == 0
    stripped_output = _strip_ansi(result.stdout)
    assert "--workspace" in stripped_output
    assert "-w" in stripped_output
    assert "--config" in stripped_output
    assert "-c" in stripped_output


def test_status_uses_explicit_config_and_workspace(tmp_path: Path):
    config_path = tmp_path / "instance" / "config.json"
    config_workspace = tmp_path / "config-workspace"
    override_workspace = tmp_path / "override-workspace"
    config = Config()
    config.agents.defaults.workspace = str(config_workspace)
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config.model_dump(mode="json", by_alias=True)))

    result = runner.invoke(
        app,
        ["status", "--config", str(config_path), "--workspace", str(override_workspace)],
    )

    assert result.exit_code == 0
    stripped_output = _strip_ansi(result.stdout)
    compact_output = stripped_output.replace("\n", "")
    assert str(config_path.resolve(strict=False)) in compact_output
    assert str(override_workspace) in compact_output
    assert str(config_workspace) not in compact_output


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

    monkeypatch.setattr("nanobot.channels.registry.discover_all", lambda: {})

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

    monkeypatch.setattr(
        "nanobot.cli.onboard.run_onboard",
        lambda initial_config: OnboardResult(config=initial_config, should_save=True),
    )
    monkeypatch.setattr("nanobot.channels.registry.discover_all", lambda: {})

    result = runner.invoke(
        app,
        ["onboard", "--wizard", "--config", str(config_path), "--workspace", str(workspace_path)],
    )

    assert result.exit_code == 0
    resolved_config = str(config_path.resolve())
    assert f'nanobot webui -c "{resolved_config}"' in result.stdout


def test_config_matches_github_copilot_codex_with_hyphen_prefix():
    config = Config()
    config.agents.defaults.model = "github-copilot/gpt-5.3-codex"

    assert config.get_provider_name() == "github_copilot"


def test_config_matches_openai_codex_with_hyphen_prefix():
    config = Config()
    config.agents.defaults.model = "openai-codex/gpt-5.6-sol"

    assert config.get_provider_name() == "openai_codex"


def test_openai_codex_oauth_default_matches_curated_flagship():
    spec = find_by_name("openai_codex")

    assert spec is not None
    assert spec.builtin_models
    assert provider_commands._OAUTH_PROVIDER_DEFAULT_MODELS["openai_codex"] == (
        spec.builtin_models[0].id
    )


def test_config_dump_excludes_oauth_provider_blocks():
    config = Config()

    providers = config.model_dump(by_alias=True)["providers"]

    assert "openaiCodex" not in providers
    assert "xaiGrok" not in providers
    assert "githubCopilot" not in providers


def test_plugins_list_uses_explicit_config(monkeypatch, tmp_path: Path):
    from nanobot.channels.plugin import ChannelPlugin

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"channels": {"example": {"enabled": True}}}),
        encoding="utf-8",
    )
    plugin = ChannelPlugin(
        name="example",
        display_name="Example",
        runtime="example.runtime:ExampleChannel",
    )
    monkeypatch.setattr(
        "nanobot.channels.registry.discover_plugins",
        lambda enabled_names=None: (
            {"example": plugin}
            if enabled_names is None or "example" in enabled_names
            else {}
        ),
    )
    monkeypatch.setattr(
        "nanobot.optional_features.optional_dependency_groups",
        lambda: {},
    )

    result = runner.invoke(app, ["plugins", "list", "--config", str(config_path)])

    assert result.exit_code == 0
    stripped_output = _strip_ansi(result.stdout)
    assert "example" in stripped_output
    assert "yes" in stripped_output


def test_provider_logout_openai_codex_removes_local_oauth_files(tmp_path, monkeypatch):
    token_path = tmp_path / "auth" / "codex.json"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("NANOBOT_CODEX_AUTH_PATH", str(token_path))

    result = runner.invoke(app, ["provider", "logout", "openai-codex"])

    assert result.exit_code == 0
    assert not token_path.exists()
    state = json.loads((token_path.parent / f"{token_path.name}.state.json").read_text())
    assert state["logged_out"] is True
    assert "Logged out from OpenAI Codex" in result.stdout


def test_provider_logout_openai_codex_succeeds_when_no_local_oauth_file(monkeypatch, tmp_path):
    token_path = tmp_path / "auth" / "codex.json"
    monkeypatch.setenv("NANOBOT_CODEX_AUTH_PATH", str(token_path))

    result = runner.invoke(app, ["provider", "logout", "openai-codex"])

    assert result.exit_code == 0
    assert "Logged out from OpenAI Codex" in result.stdout


def test_provider_logout_xai_grok_removes_instance_credentials(tmp_path, monkeypatch):
    token_path = tmp_path / "auth" / "xai.json"
    lock_path = token_path.with_suffix(".lock")
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text("{}", encoding="utf-8")
    lock_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "nanobot.providers.xai_oauth.get_xai_oauth_storage_path",
        lambda: token_path,
    )

    result = runner.invoke(app, ["provider", "logout", "xai-grok"])

    assert result.exit_code == 0
    assert not token_path.exists()
    assert "Logged out from xAI Grok" in result.stdout


def test_provider_logout_xai_grok_uses_explicit_config_path(tmp_path, monkeypatch):
    from nanobot.config import loader

    default_config = tmp_path / "default" / "config.json"
    selected_config = tmp_path / "selected" / "config.json"
    default_token = default_config.parent / "auth" / "xai.json"
    selected_token = selected_config.parent / "auth" / "xai.json"
    for token_path in (default_token, selected_token):
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(loader, "_current_config_path", default_config)

    result = runner.invoke(
        app,
        ["provider", "logout", "xai-grok", "--config", str(selected_config)],
    )

    assert result.exit_code == 0
    assert default_token.exists()
    assert not selected_token.exists()
    assert "Using config:" in result.stdout


def test_provider_logout_github_copilot_removes_local_oauth_files(tmp_path, monkeypatch):
    token_path = tmp_path / "auth" / "github-copilot.json"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("NANOBOT_COPILOT_AUTH_PATH", str(token_path))

    result = runner.invoke(app, ["provider", "logout", "github-copilot"])

    assert result.exit_code == 0
    assert not token_path.exists()
    state = json.loads((token_path.parent / f"{token_path.name}.state.json").read_text())
    assert state["logged_out"] is True
    assert "Logged out from GitHub Copilot" in result.stdout


def test_provider_logout_github_copilot_succeeds_when_no_local_oauth_file(monkeypatch, tmp_path):
    token_path = tmp_path / "auth" / "github-copilot.json"
    monkeypatch.setenv("NANOBOT_COPILOT_AUTH_PATH", str(token_path))

    result = runner.invoke(app, ["provider", "logout", "github-copilot"])

    assert result.exit_code == 0
    assert "Logged out from GitHub Copilot" in result.stdout


def test_provider_logout_rejects_unknown_provider():
    result = runner.invoke(app, ["provider", "logout", "not-a-real-provider"])

    assert result.exit_code == 1
    assert "Unknown OAuth provider" in result.stdout


def test_provider_logout_paths_resolve_to_expected_files():
    from oauth_cli_kit.providers import OPENAI_CODEX_PROVIDER
    from oauth_cli_kit.storage import FileTokenStorage

    from nanobot.providers.github_copilot_provider import get_storage
    from nanobot.providers.xai_oauth import get_xai_oauth_storage_path

    codex_storage = FileTokenStorage(token_filename=OPENAI_CODEX_PROVIDER.token_filename)
    codex_path = codex_storage.get_token_path()
    assert codex_path.name == "codex.json"
    assert codex_path.parent.name == "auth"

    xai_path = get_xai_oauth_storage_path()
    assert xai_path.name == "xai.json"
    assert xai_path.parent.name == "auth"

    gh_storage = get_storage()
    gh_path = gh_storage.get_token_path()
    assert gh_path.name == "github-copilot.json"
    assert gh_path.parent.name == "auth"


def test_provider_login_rejects_unknown_provider():
    result = runner.invoke(app, ["provider", "login", "not-a-real-provider"])

    assert result.exit_code == 1
    assert "Unknown OAuth provider" in result.stdout


def test_provider_login_openai_codex_handles_missing_oauth_symbol(monkeypatch):
    import oauth_cli_kit

    monkeypatch.delattr(oauth_cli_kit, "login_oauth_interactive")

    result = runner.invoke(app, ["provider", "login", "openai-codex"])

    assert result.exit_code == 1
    assert (
        "This nanobot installation is missing the required oauth-cli-kit package. "
        "Reinstall or upgrade nanobot-ai using the same installation method."
    ) in re.sub(r"\s+", " ", result.stdout)
    assert result.exception is not None


def test_provider_login_can_set_openai_codex_as_main_provider(tmp_path):
    config_path = tmp_path / "config.json"
    called = False
    original = provider_commands._LOGIN_HANDLERS["openai_codex"]

    def fake_login() -> None:
        nonlocal called
        called = True

    provider_commands._LOGIN_HANDLERS["openai_codex"] = fake_login
    try:
        result = runner.invoke(
            app,
            [
                "provider",
                "login",
                "openai-codex",
                "--set-main",
                "--config",
                str(config_path),
            ],
        )
    finally:
        provider_commands._LOGIN_HANDLERS["openai_codex"] = original

    assert result.exit_code == 0
    assert called is True
    assert "Set openai-codex as the main provider" in result.stdout

    saved = Config.model_validate(json.loads(config_path.read_text(encoding="utf-8")))
    assert saved.agents.defaults.provider == "openai_codex"
    assert saved.agents.defaults.model == "openai-codex/gpt-5.6-sol"
    assert saved.agents.defaults.model_preset is None
    assert make_provider(saved).__class__.__name__ == "OpenAICodexProvider"


def test_provider_login_can_set_github_copilot_as_main_provider(tmp_path):
    config_path = tmp_path / "config.json"
    original = provider_commands._LOGIN_HANDLERS["github_copilot"]
    provider_commands._LOGIN_HANDLERS["github_copilot"] = lambda: None
    try:
        result = runner.invoke(
            app,
            [
                "provider",
                "login",
                "github-copilot",
                "--set-main",
                "--config",
                str(config_path),
            ],
        )
    finally:
        provider_commands._LOGIN_HANDLERS["github_copilot"] = original

    assert result.exit_code == 0
    assert "Set github-copilot as the main provider" in result.stdout

    saved = Config.model_validate(json.loads(config_path.read_text(encoding="utf-8")))
    assert saved.agents.defaults.provider == "github_copilot"
    assert saved.agents.defaults.model == "github-copilot/gpt-5.4-mini"
    assert saved.agents.defaults.model_preset is None
    assert make_provider(saved).__class__.__name__ == "GitHubCopilotProvider"


def test_provider_login_can_set_xai_grok_as_main_provider(tmp_path):
    config_path = tmp_path / "config.json"
    original = provider_commands._LOGIN_HANDLERS["xai_grok"]
    provider_commands._LOGIN_HANDLERS["xai_grok"] = lambda: None
    try:
        result = runner.invoke(
            app,
            [
                "provider",
                "login",
                "xai-grok",
                "--set-main",
                "--config",
                str(config_path),
            ],
        )
    finally:
        provider_commands._LOGIN_HANDLERS["xai_grok"] = original

    assert result.exit_code == 0
    assert "Set xai-grok as the main provider" in result.stdout

    saved = Config.model_validate(json.loads(config_path.read_text(encoding="utf-8")))
    assert saved.agents.defaults.provider == "xai_grok"
    assert saved.agents.defaults.model == "xai-grok/grok-4.6"
    assert saved.agents.defaults.context_window_tokens == 500_000
    assert saved.agents.defaults.model_preset is None
    assert make_provider(saved).__class__.__name__ == "XAIGrokProvider"


def test_provider_login_model_implies_set_main_provider(tmp_path):
    config_path = tmp_path / "config.json"
    original = provider_commands._LOGIN_HANDLERS["github_copilot"]
    provider_commands._LOGIN_HANDLERS["github_copilot"] = lambda: None
    try:
        result = runner.invoke(
            app,
            [
                "provider",
                "login",
                "github-copilot",
                "--model",
                "github-copilot/gpt-5.4-mini",
                "--config",
                str(config_path),
            ],
        )
    finally:
        provider_commands._LOGIN_HANDLERS["github_copilot"] = original

    assert result.exit_code == 0
    assert "Set github-copilot as the main provider" in result.stdout

    saved = Config.model_validate(json.loads(config_path.read_text(encoding="utf-8")))
    assert saved.agents.defaults.provider == "github_copilot"
    assert saved.agents.defaults.model == "github-copilot/gpt-5.4-mini"
    assert make_provider(saved).__class__.__name__ == "GitHubCopilotProvider"


def test_provider_login_openai_codex_passes_configured_proxy(monkeypatch, tmp_path):
    monkeypatch.setenv("NANOBOT_CODEX_AUTH_PATH", str(tmp_path / "codex.json"))
    proxy = "http://127.0.0.1:23458"
    monkeypatch.setattr(
        "nanobot.config.loader.load_config",
        lambda: Config.model_validate({"providers": {"openaiCodex": {"proxy": proxy}}}),
    )

    import oauth_cli_kit

    def fake_get_token(**_kwargs):
        raise RuntimeError("no-token")

    monkeypatch.setattr(oauth_cli_kit, "get_token", fake_get_token)

    captured: dict[str, str | None] = {}

    def fake_login(*, print_fn, prompt_fn, proxy=None, storage=None):
        captured["proxy"] = proxy
        return SimpleNamespace(
            access="access-token", refresh="refresh-token", expires=9_999_999_999_999,
            account_id="acct-test"
        )

    monkeypatch.setattr(oauth_cli_kit, "login_oauth_interactive", fake_login)

    result = runner.invoke(app, ["provider", "login", "openai-codex"])

    assert result.exit_code == 0
    assert captured["proxy"] == proxy


def test_provider_login_openai_codex_uses_explicit_config_proxy(tmp_path, monkeypatch):
    from nanobot.config import loader

    proxy = "http://127.0.0.1:23458"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"providers": {"openaiCodex": {"proxy": proxy}}}),
        encoding="utf-8",
    )
    active_path: dict[str, Path] = {}
    real_load_config = loader.load_config

    def fake_set_config_path(path: Path) -> None:
        active_path["path"] = path

    def fake_load_config(config_path: Path | None = None) -> Config:
        path = config_path or active_path.get("path")
        if path is None:
            return Config.model_validate(
                {"providers": {"openaiCodex": {"proxy": "http://default-proxy:8080"}}}
            )
        return real_load_config(path)

    monkeypatch.setattr(loader, "set_config_path", fake_set_config_path)
    monkeypatch.setattr(loader, "load_config", fake_load_config)

    import oauth_cli_kit

    captured: dict[str, str | None] = {}

    def fake_login(*, proxy=None, **_kwargs):
        captured["proxy"] = proxy
        return SimpleNamespace(
            access="access-token", refresh="refresh-token", expires=9_999_999_999_999,
            account_id="acct-test"
        )

    monkeypatch.setattr(oauth_cli_kit, "login_oauth_interactive", fake_login)
    monkeypatch.setenv("NANOBOT_CODEX_AUTH_PATH", str(tmp_path / "codex.json"))

    result = runner.invoke(
        app,
        ["provider", "login", "openai-codex", "--config", str(config_path)],
    )

    assert result.exit_code == 0
    assert active_path["path"] == config_path.resolve()
    assert captured["proxy"] == proxy


def test_provider_login_openai_codex_resolves_proxy_env_ref(monkeypatch, tmp_path):
    proxy = "http://127.0.0.1:23458"
    monkeypatch.setenv("CODEX_PROXY_FOR_TEST", proxy)
    monkeypatch.setattr(
        "nanobot.config.loader.load_config",
        lambda: Config.model_validate(
            {"providers": {"openaiCodex": {"proxy": "${CODEX_PROXY_FOR_TEST}"}}}
        ),
    )

    import oauth_cli_kit

    captured: dict[str, str | None] = {}

    def fake_login(*, proxy=None, **_kwargs):
        captured["proxy"] = proxy
        return SimpleNamespace(
            access="access-token", refresh="refresh-token", expires=9_999_999_999_999,
            account_id="acct-test"
        )

    monkeypatch.setattr(oauth_cli_kit, "login_oauth_interactive", fake_login)
    monkeypatch.setenv("NANOBOT_CODEX_AUTH_PATH", str(tmp_path / "codex.json"))

    result = runner.invoke(app, ["provider", "login", "openai-codex"])

    assert result.exit_code == 0
    assert captured["proxy"] == proxy


def test_provider_login_xai_grok_runs_browser_flow_with_configured_proxy(monkeypatch):
    proxy = "http://127.0.0.1:23458"
    monkeypatch.setattr(
        "nanobot.config.loader.load_config",
        lambda: Config.model_validate({"providers": {"xaiGrok": {"proxy": proxy}}}),
    )
    monkeypatch.setattr(
        "nanobot.providers.xai_oauth.get_xai_oauth_token",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("not signed in")),
    )
    captured: dict[str, object] = {}

    def fake_login(*, print_fn, prompt_fn, proxy=None):
        captured.update(print_fn=print_fn, prompt_fn=prompt_fn, proxy=proxy)
        return SimpleNamespace(access="access-token", account_id="user@example.com")

    monkeypatch.setattr("nanobot.providers.xai_oauth.login_xai_oauth", fake_login)

    result = runner.invoke(app, ["provider", "login", "xai-grok"])

    assert result.exit_code == 0
    assert captured["proxy"] == proxy
    assert callable(captured["print_fn"])
    assert callable(captured["prompt_fn"])
    assert "Hosted X Search is enabled automatically when the selected model supports it" in result.stdout


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


def test_config_accepts_lm_studio_without_api_key_and_uses_default_localhost_api_base():
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "provider": "lm_studio",
                    "model": "local-model",
                }
            },
            "providers": {
                "lmStudio": {
                    "apiKey": None,
                }
            },
        }
    )

    assert config.get_provider_name() == "lm_studio"
    assert config.get_api_key() is None
    assert config.get_api_base() == "http://localhost:1234/v1"


def test_config_accepts_atomic_chat_without_api_key_and_uses_default_localhost_api_base():
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "provider": "atomic_chat",
                    "model": "local-model",
                }
            },
            "providers": {
                "atomicChat": {
                    "apiKey": None,
                }
            },
        }
    )

    assert config.get_provider_name() == "atomic_chat"
    assert config.get_api_key() is None
    assert config.get_api_base() == "http://localhost:1337/v1"


def test_find_by_name_accepts_camel_case_and_hyphen_aliases():
    assert find_by_name("volcengineCodingPlan") is not None
    assert find_by_name("volcengineCodingPlan").name == "volcengine_coding_plan"
    assert find_by_name("github-copilot") is not None
    assert find_by_name("github-copilot").name == "github_copilot"
    assert find_by_name("longcat") is not None
    assert find_by_name("longcat").name == "longcat"
    assert find_by_name("atomic-chat") is not None
    assert find_by_name("atomic-chat").name == "atomic_chat"


def test_config_explicit_longcat_provider_resolves_provider_name():
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "provider": "longcat",
                    "model": "LongCat-Flash-Chat",
                }
            },
            "providers": {
                "longcat": {
                    "apiKey": "test-key",
                }
            },
        }
    )

    assert config.get_provider_name() == "longcat"
    assert config.get_api_base() == "https://api.longcat.chat/openai/v1"


def test_config_auto_detects_longcat_from_model_keyword():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "auto", "model": "longcat/LongCat-Flash-Chat"}},
            "providers": {"longcat": {"apiKey": "test-key"}},
        }
    )

    assert config.get_provider_name() == "longcat"


def test_config_explicit_xiaomi_mimo_provider_uses_default_api_base():
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "provider": "xiaomi_mimo",
                    "model": "MiniMax-M1-80k",
                }
            },
            "providers": {
                "xiaomiMimo": {
                    "apiKey": "test-key",
                }
            },
        }
    )

    assert config.get_provider_name() == "xiaomi_mimo"
    assert config.get_api_base() == "https://api.xiaomimimo.com/v1"


def test_config_auto_detects_xiaomi_mimo_from_model_keyword():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "auto", "model": "mimo/MiniMax-M1-80k"}},
            "providers": {"xiaomiMimo": {"apiKey": "test-key"}},
        }
    )

    assert config.get_provider_name() == "xiaomi_mimo"
    assert config.get_api_base() == "https://api.xiaomimimo.com/v1"


def test_config_explicit_minimax_anthropic_provider_uses_default_api_base():
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "provider": "minimax_anthropic",
                    "model": "MiniMax-M2.7-highspeed",
                }
            },
            "providers": {
                "minimaxAnthropic": {
                    "apiKey": "test-key",
                }
            },
        }
    )

    assert config.get_provider_name() == "minimax_anthropic"
    assert config.get_api_key() == "test-key"
    assert config.get_api_base() == "https://api.minimax.io/anthropic"


def test_config_auto_detects_ollama_from_local_api_base():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "auto", "model": "llama3.2"}},
            "providers": {"ollama": {"apiBase": "http://localhost:11434/v1"}},
        }
    )

    assert config.get_provider_name() == "ollama"
    assert config.get_api_base() == "http://localhost:11434/v1"


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


def test_config_cloud_nemotron_is_not_hijacked_by_unconfigured_ollama():
    """`nvidia/nemotron-*` via a gateway must not route to Ollama when no
    Ollama endpoint is configured. Ollama keeps "nemotron" in its keywords
    for bare-model auto-routing (PR #1863), which previously hijacked
    cloud-hosted nemotron variants and silently sent traffic to
    http://localhost:11434/v1."""
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "provider": "auto",
                    "model": "nvidia/nemotron-3-super-120b-a12b",
                }
            },
            "providers": {"openrouter": {"apiKey": "sk-or-test"}},
        }
    )

    assert config.get_provider_name() == "openrouter"
    assert config.get_api_base() == "https://openrouter.ai/api/v1"


def test_config_bare_nemotron_still_auto_routes_to_configured_ollama():
    """Preserves PR #1863 intent: when the user has actually configured an
    Ollama endpoint, a bare nemotron model still auto-routes there."""
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "auto", "model": "nemotron-3-nano"}},
            "providers": {"ollama": {"apiBase": "http://localhost:11434/v1"}},
        }
    )

    assert config.get_provider_name() == "ollama"
    assert config.get_api_base() == "http://localhost:11434/v1"


def test_config_cloud_nemotron_is_not_hijacked_by_configured_ollama():
    """An explicit cloud namespace takes precedence over local keywords."""
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "provider": "auto",
                    "model": "nvidia/nemotron-3-super-120b-a12b",
                }
            },
            "providers": {
                "ollama": {"apiBase": "http://localhost:11434/v1"},
                "openrouter": {"apiKey": "sk-or-test"},
            },
        }
    )

    assert config.get_provider_name() == "openrouter"
    assert config.get_api_base() == "https://openrouter.ai/api/v1"


def test_openai_compat_provider_passes_model_through():
    from nanobot.providers.openai_compat_provider import OpenAICompatProvider

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):
        provider = OpenAICompatProvider(default_model="github-copilot/gpt-5.3-codex")

    assert provider.get_default_model() == "github-copilot/gpt-5.3-codex"


def test_openai_codex_proxy_config_affects_provider_and_signature():
    def config_with_proxy(proxy: str) -> Config:
        return Config.model_validate(
            {
                "agents": {
                    "defaults": {
                        "provider": "openai-codex",
                        "model": "openai-codex/gpt-5.5",
                    }
                },
                "providers": {"openaiCodex": {"proxy": proxy}},
            }
        )

    proxy = "http://127.0.0.1:23458"
    config = config_with_proxy(proxy)

    provider = make_provider(config)

    assert provider.__class__.__name__ == "OpenAICodexProvider"
    assert provider.proxy == proxy
    assert provider_signature(config) != provider_signature(
        config_with_proxy("http://127.0.0.1:23459")
    )


def test_provider_proxy_rejects_unsupported_backend():
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "provider": "anthropic",
                    "model": "anthropic/claude-opus-4-5",
                }
            },
            "providers": {
                "anthropic": {
                    "apiKey": "sk-test",
                    "proxy": "http://127.0.0.1:23458",
                }
            },
        }
    )

    with pytest.raises(ValueError, match=r"providers\.anthropic\.proxy"):
        make_provider(config)


def test_github_copilot_provider_strips_prefixed_model_name():
    from nanobot.providers.github_copilot_provider import GitHubCopilotProvider

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):
        provider = GitHubCopilotProvider(default_model="github-copilot/gpt-5.1")

    kwargs = provider._build_kwargs(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model="github-copilot/gpt-5.1",
        max_tokens=16,
        temperature=0.1,
        reasoning_effort=None,
        tool_choice=None,
    )

    assert kwargs["model"] == "gpt-5.1"


@pytest.mark.asyncio
async def test_github_copilot_provider_refreshes_client_api_key_before_chat():
    from nanobot.providers.github_copilot_provider import GitHubCopilotProvider

    mock_client = MagicMock()
    mock_client.api_key = "no-key"
    mock_client.chat.completions.create = AsyncMock(return_value={
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", return_value=mock_client):
        provider = GitHubCopilotProvider(default_model="github-copilot/gpt-4")
        await provider._ensure_client()

    provider._get_copilot_access_token = AsyncMock(return_value="copilot-access-token")

    response = await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="github-copilot/gpt-4",
        max_tokens=16,
        temperature=0.1,
    )

    assert response.content == "ok"
    assert provider._client.api_key == "copilot-access-token"
    provider._get_copilot_access_token.assert_awaited_once()
    mock_client.chat.completions.create.assert_awaited_once()


def test_openai_codex_strip_prefix_supports_hyphen_and_underscore():
    assert _strip_model_prefix("openai-codex/gpt-5.6-sol") == "gpt-5.6-sol"
    assert _strip_model_prefix("openai_codex/gpt-5.6-sol") == "gpt-5.6-sol"


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
            }
        }
    )

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as mock_async_openai:
        provider = make_provider(config)
        asyncio.run(provider._ensure_client())

    kwargs = mock_async_openai.call_args.kwargs
    assert kwargs["api_key"] == "test-key"
    assert kwargs["base_url"] == "https://example.com/v1"
    assert kwargs["default_headers"]["APP-Code"] == "demo-app"
    assert kwargs["default_headers"]["x-session-affinity"] == "sticky-session"


def test_make_provider_treats_dynamic_custom_provider_as_direct():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "my-company-api", "model": "gpt-4o-mini"}},
            "providers": {
                "my-company-api": {
                    "apiBase": "https://example.com/v1",
                }
            },
        }
    )

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as mock_async_openai:
        provider = make_provider(config)
        asyncio.run(provider._ensure_client())

    assert provider.get_default_model() == "gpt-4o-mini"
    assert provider._spec.name == "my_company_api"
    assert provider._spec.is_direct is True
    kwargs = mock_async_openai.call_args.kwargs
    assert kwargs["api_key"] == "no-key"
    assert kwargs["base_url"] == "https://example.com/v1"


def test_make_provider_strips_dynamic_custom_route_prefix_from_request_model():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "auto", "model": "my-company-api/gpt-4o-mini"}},
            "providers": {
                "my-company-api": {
                    "apiBase": "https://example.com/v1",
                }
            },
        }
    )

    provider = make_provider(config)

    kwargs = provider._build_kwargs(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model=None,
        max_tokens=16,
        temperature=0.1,
        reasoning_effort=None,
        tool_choice=None,
    )
    body = provider._build_responses_body(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model=None,
        max_tokens=16,
        temperature=0.1,
        reasoning_effort=None,
        tool_choice=None,
    )

    assert config.get_provider_name() == "my-company-api"
    assert kwargs["model"] == "gpt-4o-mini"
    assert body["model"] == "gpt-4o-mini"


def test_make_provider_preserves_namespaced_model_for_forced_dynamic_provider():
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "provider": "my-company-api",
                    "model": "openai/gpt-4o-mini",
                }
            },
            "providers": {
                "my-company-api": {
                    "apiBase": "https://example.com/v1",
                }
            },
        }
    )

    provider = make_provider(config)
    kwargs = provider._build_kwargs(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model=None,
        max_tokens=16,
        temperature=0.1,
        reasoning_effort=None,
        tool_choice=None,
    )

    assert kwargs["model"] == "openai/gpt-4o-mini"


def test_make_provider_strips_dynamic_custom_route_prefix_once():
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "provider": "auto",
                    "model": "my-company-api/openai/gpt-4o-mini",
                }
            },
            "providers": {
                "my-company-api": {
                    "apiBase": "https://example.com/v1",
                }
            },
        }
    )

    provider = make_provider(config)
    kwargs = provider._build_kwargs(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model=None,
        max_tokens=16,
        temperature=0.1,
        reasoning_effort=None,
        tool_choice=None,
    )

    assert kwargs["model"] == "openai/gpt-4o-mini"


def test_make_provider_rejects_dynamic_custom_provider_without_api_base():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "my-company-api", "model": "gpt-4o-mini"}},
            "providers": {
                "my-company-api": {
                    "apiKey": "sk-test",
                }
            },
        }
    )

    with pytest.raises(ValueError, match="Provider 'my-company-api' requires api_base"):
        make_provider(config)


def test_make_provider_rejects_auto_dynamic_custom_prefix_without_api_base():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "auto", "model": "companyProxy/gpt-4o"}},
            "providers": {
                "otherProxy": {
                    "apiBase": "https://other.example.test/v1",
                },
                "companyProxy": {
                    "apiKey": "sk-company",
                },
            },
        }
    )

    with pytest.raises(ValueError, match="Provider 'companyProxy' requires api_base"):
        make_provider(config)


@pytest.fixture
def mock_agent_runtime(tmp_path, classic_cli_runtime):
    """Mock agent command dependencies for focused CLI tests."""
    config = Config()
    config.providers.openrouter.api_key = "test-only-key"
    config.agents.defaults.workspace = str(tmp_path / "default-workspace")

    with patch("nanobot.config.loader.load_config", return_value=config) as mock_load_config, \
         patch("nanobot.config.loader.resolve_config_env_vars", side_effect=lambda c: c), \
         patch("nanobot.cli.agent.sync_workspace_templates") as mock_sync_templates, \
         patch("nanobot.providers.factory.make_provider", return_value=_fake_provider()), \
         patch("nanobot.cli.terminal._print_agent_response") as mock_print_response, \
         patch("nanobot.cron.service.CronService"), \
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
            "print_response": mock_print_response,
        }


def test_agent_help_shows_workspace_and_config_options():
    result = runner.invoke(app, ["agent", "--help"])

    assert result.exit_code == 0
    stripped_output = _strip_ansi(result.stdout)
    assert "--workspace" in stripped_output
    assert "-w" in stripped_output
    assert "--config" in stripped_output
    assert "-c" in stripped_output
    assert "--theme" in stripped_output


def test_agent_rejects_unknown_tui_theme(mock_agent_runtime):
    result = runner.invoke(app, ["agent", "-m", "hello", "--theme", "sepia"])

    assert result.exit_code != 0
    assert "must be auto, dark, or light" in result.output
    mock_agent_runtime["from_config"].assert_not_called()


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


def test_agent_uses_explicit_config_path(mock_agent_runtime, tmp_path: Path):
    config_path = tmp_path / "agent-config.json"
    config_path.write_text("{}")

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_path)])

    assert result.exit_code == 0
    assert mock_agent_runtime["load_config"].call_args.args == (config_path.resolve(),)


def test_agent_config_sets_active_path(monkeypatch, tmp_path: Path, classic_cli_runtime) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.providers.openrouter.api_key = "test-only-key"
    config.agents.defaults.workspace = str(tmp_path / "default-workspace")
    seen: dict[str, Path] = {}

    monkeypatch.setattr(
        "nanobot.config.loader.set_config_path",
        lambda path: seen.__setitem__("config_path", path),
    )
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("nanobot.cli.agent.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: _fake_provider())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", lambda: object())
    monkeypatch.setattr("nanobot.cron.service.CronService", lambda _store: object())

    class _FakeAgentLoop:
        @classmethod
        def from_config(cls, config, bus=None, **extra):
            return cls(**extra)
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def process_direct(self, *_args, **_kwargs):
            return OutboundMessage(channel="cli", chat_id="direct", content="ok")

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("nanobot.cli.agent.AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr("nanobot.cli.terminal._print_agent_response", lambda *_args, **_kwargs: None)

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_file)])

    assert result.exit_code == 0
    assert seen["config_path"] == config_file.resolve()


def test_agent_uses_workspace_directory_for_cron_store(
    monkeypatch, tmp_path: Path, classic_cli_runtime,
) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    config = Config()
    config.providers.openrouter.api_key = "test-only-key"
    config.agents.defaults.workspace = str(tmp_path / "agent-workspace")
    seen: dict[str, Path] = {}

    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("nanobot.cli.agent.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: _fake_provider())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", lambda: object())

    class _FakeCron:
        def __init__(self, store_path: Path) -> None:
            seen["cron_store"] = store_path

    class _FakeAgentLoop:
        @classmethod
        def from_config(cls, config, bus=None, **extra):
            return cls(**extra)
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def process_direct(self, *_args, **_kwargs):
            return OutboundMessage(channel="cli", chat_id="direct", content="ok")

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("nanobot.cron.service.CronService", _FakeCron)
    monkeypatch.setattr("nanobot.cli.agent.AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr("nanobot.cli.terminal._print_agent_response", lambda *_args, **_kwargs: None)

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_file)])

    assert result.exit_code == 0
    assert seen["cron_store"] == config.workspace_path / "cron" / "jobs.json"


def test_agent_workspace_override_does_not_migrate_legacy_cron(
    monkeypatch, tmp_path: Path, classic_cli_runtime,
) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    legacy_dir = tmp_path / "global" / "cron"
    legacy_dir.mkdir(parents=True)
    legacy_file = legacy_dir / "jobs.json"
    legacy_file.write_text('{"jobs": []}')

    override = tmp_path / "override-workspace"
    config = Config()
    config.providers.openrouter.api_key = "test-only-key"
    seen: dict[str, Path] = {}

    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("nanobot.cli.agent.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: _fake_provider())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", lambda: object())
    monkeypatch.setattr("nanobot.config.paths.get_cron_dir", lambda: legacy_dir)

    class _FakeCron:
        def __init__(self, store_path: Path) -> None:
            seen["cron_store"] = store_path

    class _FakeAgentLoop:
        @classmethod
        def from_config(cls, config, bus=None, **extra):
            return cls(**extra)
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def process_direct(self, *_args, **_kwargs):
            return OutboundMessage(channel="cli", chat_id="direct", content="ok")

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("nanobot.cron.service.CronService", _FakeCron)
    monkeypatch.setattr("nanobot.cli.agent.AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr("nanobot.cli.terminal._print_agent_response", lambda *_args, **_kwargs: None)

    result = runner.invoke(
        app,
        ["agent", "-m", "hello", "-c", str(config_file), "-w", str(override)],
    )

    assert result.exit_code == 0
    assert seen["cron_store"] == override / "cron" / "jobs.json"
    assert legacy_file.exists()
    assert not (override / "cron" / "jobs.json").exists()


def test_agent_custom_config_workspace_does_not_migrate_legacy_cron(
    monkeypatch, tmp_path: Path, classic_cli_runtime,
) -> None:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")

    legacy_dir = tmp_path / "global" / "cron"
    legacy_dir.mkdir(parents=True)
    legacy_file = legacy_dir / "jobs.json"
    legacy_file.write_text('{"jobs": []}')

    custom_workspace = tmp_path / "custom-workspace"
    config = Config()
    config.providers.openrouter.api_key = "test-only-key"
    config.agents.defaults.workspace = str(custom_workspace)
    seen: dict[str, Path] = {}

    monkeypatch.setattr("nanobot.config.loader.set_config_path", lambda _path: None)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path=None: config)
    monkeypatch.setattr("nanobot.cli.agent.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda _config: _fake_provider())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", lambda: object())
    monkeypatch.setattr("nanobot.config.paths.get_cron_dir", lambda: legacy_dir)

    class _FakeCron:
        def __init__(self, store_path: Path) -> None:
            seen["cron_store"] = store_path

    class _FakeAgentLoop:
        @classmethod
        def from_config(cls, config, bus=None, **extra):
            return cls(**extra)
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def process_direct(self, *_args, **_kwargs):
            return OutboundMessage(channel="cli", chat_id="direct", content="ok")

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("nanobot.cron.service.CronService", _FakeCron)
    monkeypatch.setattr("nanobot.cli.agent.AgentLoop", _FakeAgentLoop)
    monkeypatch.setattr(
        "nanobot.cli.terminal._print_agent_response", lambda *_args, **_kwargs: None
    )

    result = runner.invoke(app, ["agent", "-m", "hello", "-c", str(config_file)])

    assert result.exit_code == 0
    assert seen["cron_store"] == custom_workspace / "cron" / "jobs.json"
    assert legacy_file.exists()
    assert not (custom_workspace / "cron" / "jobs.json").exists()


def test_agent_overrides_workspace_path(mock_agent_runtime, tmp_path: Path):
    workspace_path = tmp_path / "agent-workspace"

    result = runner.invoke(app, ["agent", "-m", "hello", "-w", str(workspace_path)])

    assert result.exit_code == 0
    assert mock_agent_runtime["config"].agents.defaults.workspace == str(workspace_path)
    assert mock_agent_runtime["sync_templates"].call_args.args == (workspace_path,)
    passed_config = mock_agent_runtime["from_config"].call_args.args[0]
    assert passed_config.workspace_path == workspace_path


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
    passed_config = mock_agent_runtime["from_config"].call_args.args[0]
    assert passed_config.workspace_path == workspace_path












def _write_instance_config(tmp_path: Path) -> Path:
    config_file = tmp_path / "instance" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}")
    return config_file


def _stop_gateway_provider(_config) -> object:
    raise _StopGatewayError("stop")


def _test_provider_snapshot(provider: object, config: Config) -> ProviderSnapshot:
    return ProviderSnapshot(
        provider=provider,
        model=config.agents.defaults.model,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        signature=("test",),
    )


def _patch_webui_provider_ready(monkeypatch) -> None:
    monkeypatch.setattr(
        "nanobot.providers.factory.validate_provider_setup",
        lambda _config: None,
    )


def _patch_gateway_ports_free(monkeypatch) -> None:
    monkeypatch.setattr("nanobot.cli.webui._gateway_health_ready", lambda *_a, **_kw: False)
    monkeypatch.setattr("nanobot.cli.webui._tcp_endpoint_reachable", lambda *_a, **_kw: False)
    monkeypatch.setattr("nanobot.cli.webui._webui_endpoint_reachable", lambda *_a, **_kw: False)
    monkeypatch.setattr(
        "nanobot.cli.gateway_runtime._tcp_endpoint_reachable",
        lambda *_a, **_kw: False,
    )
    monkeypatch.setattr(
        "nanobot.cli.gateway_runtime._webui_endpoint_reachable",
        lambda *_a, **_kw: False,
    )


def _record_gateway_lease_release(monkeypatch, captured: dict[str, object]) -> None:
    from nanobot.gateway import GatewayClientLease

    original_release = GatewayClientLease.release

    def record_release(
        lease: GatewayClientLease,
        *,
        timeout_s: int = 20,
        wait_for_stop: bool = True,
    ) -> bool:
        captured["lease_release_wait_for_stop"] = wait_for_stop
        captured["dev_running_at_release"] = captured.get("dev_running")
        return original_release(
            lease,
            timeout_s=timeout_s,
            wait_for_stop=wait_for_stop,
        )

    monkeypatch.setattr(GatewayClientLease, "release", record_release)


def _patch_webui_managed_gateway(
    monkeypatch,
    seen: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run WebUI CLI tests against a managed gateway without spawning a process."""
    from nanobot.gateway import GatewayStatus, RuntimeResult

    captured = seen if seen is not None else {}
    _record_gateway_lease_release(monkeypatch, captured)

    class _FakeRuntime:
        def __init__(self, **kwargs) -> None:
            captured["runtime_kwargs"] = kwargs
            self.paths = kwargs["paths"]
            self.running = False

        def start_background(self, options):
            captured["start_options"] = options
            self.running = True
            status = GatewayStatus(
                running=True,
                pid=123,
                state_path=Path("gateway.json"),
                log_path=Path("gateway.log"),
                port=options.port,
                reason="running",
            )
            return RuntimeResult(True, "gateway_started_background", status)

        def start_on_demand(self, options):
            from nanobot.gateway import GatewayClientLease

            GatewayClientLease(self, kind="test-webui").mark_ephemeral()
            return self.start_background(options)

        def status(self):
            return SimpleNamespace(running=self.running)

        def stop(self, *, timeout_s: int):
            captured["stop_timeout"] = timeout_s
            self.running = False
            return RuntimeResult(True, "gateway_stopped", self.status())

        _stop = stop

    monkeypatch.setattr("nanobot.gateway.GatewayRuntime", _FakeRuntime)
    monkeypatch.setattr(
        "nanobot.cli.webui._prepare_webui_bundle_for_gateway",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "nanobot.cli.webui._attach_to_background_gateway",
        lambda runtime, **_kwargs: captured.__setitem__("attached_runtime", runtime),
    )
    return captured


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
    provider_factory = make_provider or (lambda _config: _fake_provider())

    def load_config(path=None):
        if path is not None:
            config.bind_source_path(Path(path))
        return config

    monkeypatch.setattr(
        "nanobot.config.loader.set_config_path",
        set_config_path or (lambda _path: None),
    )
    monkeypatch.setattr("nanobot.config.loader.load_config", load_config)
    monkeypatch.setattr("nanobot.config.loader.resolve_config_env_vars", lambda c: c)
    monkeypatch.setattr(
        "nanobot.cli.commands.sync_workspace_templates",
        sync_templates or (lambda _path: None),
    )
    monkeypatch.setattr(
        "nanobot.cli.webui.sync_workspace_templates",
        sync_templates or (lambda _path: None),
    )
    monkeypatch.setattr(
        "nanobot.cli.gateway_runtime.sync_workspace_templates",
        sync_templates or (lambda _path: None),
    )
    monkeypatch.setattr(
        "nanobot.providers.factory.make_provider",
        provider_factory,
    )
    monkeypatch.setattr(
        "nanobot.providers.factory.build_provider_snapshot",
        lambda _config: _test_provider_snapshot(provider_factory(_config), _config),
    )
    monkeypatch.setattr(
        "nanobot.providers.factory.load_provider_snapshot",
        lambda _config_path=None: _test_provider_snapshot(provider_factory(config), config),
    )
    monkeypatch.setattr(
        "nanobot.cli.webui_support._provider_setup_error",
        lambda _config: None,
    )
    _patch_gateway_ports_free(monkeypatch)

    if message_bus is not None:
        monkeypatch.setattr("nanobot.bus.queue.MessageBus", message_bus)
    if session_manager is not None:
        monkeypatch.setattr("nanobot.session.manager.SessionManager", session_manager)
    if cron_service is not None:
        monkeypatch.setattr("nanobot.cron.service.CronService", cron_service)
    if get_cron_dir is not None:
        monkeypatch.setattr("nanobot.config.paths.get_cron_dir", get_cron_dir)






def test_webui_yes_creates_config_and_enables_local_websocket(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "instance" / "config.json"
    workspace = tmp_path / "workspace"
    seen: dict[str, object] = {}
    _patch_webui_provider_ready(monkeypatch)
    monkeypatch.setattr(
        "nanobot.cli.webui.sync_workspace_templates",
        lambda path: seen.__setitem__("templates", path),
    )

    _patch_webui_managed_gateway(monkeypatch, seen)

    result = runner.invoke(
        app,
        [
            "webui",
            "--config",
            str(config_file),
            "--workspace",
            str(workspace),
            "--port",
            "8899",
            "--gateway-port",
            "18888",
            "--yes",
            "--no-open",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(config_file.read_text(encoding="utf-8"))
    websocket = data["channels"]["websocket"]
    assert websocket["enabled"] is True
    assert websocket["host"] == "127.0.0.1"
    assert websocket["port"] == 8899
    assert websocket["websocketRequiresToken"] is True
    assert isinstance(websocket["tokenIssueSecret"], str)
    assert len(websocket["tokenIssueSecret"]) >= 32
    assert data["agents"]["defaults"]["workspace"] == str(workspace)
    assert seen["templates"] == workspace
    options = seen["start_options"]
    assert options.port == 18888
    assert options.config_path == str(config_file.resolve(strict=False))
    assert options.workspace == str(workspace.resolve(strict=False))
    compact_output = re.sub(r"\s+", " ", _strip_ansi(result.stdout))
    assert "bootstrap secret was generated" in compact_output
    assert "channels.websocket.tokenIssueSecret" in compact_output
    assert "rerun without --no-open" in compact_output
    assert seen["lease_release_wait_for_stop"] is False
    assert "stop_timeout" not in seen


def test_webui_background_points_to_the_single_persistent_gateway_command(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "config.json"
    workspace = tmp_path / "workspace"

    result = runner.invoke(
        app,
        [
            "webui",
            "--background",
            "--config",
            str(config_file),
            "--workspace",
            str(workspace),
        ],
    )

    assert result.exit_code == 1
    compact_output = _strip_ansi(result.stdout).replace("\n", " ")
    assert "webui --background` no longer owns gateway lifecycle" in compact_output
    assert "nanobot gateway --background --config" in compact_output
    assert "--workspace" in compact_output
    assert not config_file.exists()


def test_webui_dev_starts_vite_sidecar_and_gateway(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text("{}", encoding="utf-8")
    seen: dict[str, object] = {}
    _patch_webui_provider_ready(monkeypatch)
    _patch_gateway_ports_free(monkeypatch)
    monkeypatch.setattr("nanobot.cli.webui.sync_workspace_templates", lambda _path: None)

    @contextmanager
    def fake_dev_server(**kwargs):
        seen["dev_kwargs"] = kwargs
        seen["dev_running"] = True
        dev_server = SimpleNamespace(
            url=kwargs["browser_url"],
            ensure_running=lambda: None,
        )
        seen["dev_server"] = dev_server
        try:
            yield dev_server
        finally:
            seen["dev_running"] = False

    monkeypatch.setattr("nanobot.cli.webui.run_webui_dev_server", fake_dev_server)
    _patch_webui_managed_gateway(monkeypatch, seen)
    monkeypatch.setattr(
        "nanobot.cli.webui._prepare_webui_bundle_for_gateway",
        lambda *_args, **_kwargs: pytest.fail("dev mode must not inspect the bundled WebUI"),
    )
    monkeypatch.setattr(
        "nanobot.cli.webui._attach_to_background_gateway",
        lambda runtime, **kwargs: seen.update(
            attached_runtime=runtime,
            attach_kwargs=kwargs,
            attached_while_dev_running=seen["dev_running"],
        ),
    )
    monkeypatch.setattr(
        "nanobot.cli.webui._open_webui_browser",
        lambda url: seen.__setitem__("opened_url", url),
    )

    result = runner.invoke(
        app,
        [
            "webui",
            "--dev",
            "--config",
            str(config_file),
            "--port",
            "8899",
            "--gateway-port",
            "18888",
            "--yes",
        ],
    )

    assert result.exit_code == 0
    dev_kwargs = seen["dev_kwargs"]
    assert isinstance(dev_kwargs, dict)
    assert dev_kwargs["target_url"] == "http://127.0.0.1:8899"
    browser_url = dev_kwargs["browser_url"]
    assert isinstance(browser_url, str)
    assert browser_url.startswith("http://127.0.0.1:5173/#/?bootstrapSecret=")
    assert seen["start_options"].port == 18888
    assert seen["attached_while_dev_running"] is True
    assert seen["attach_kwargs"] == {"poll_hook": seen["dev_server"].ensure_running}
    assert seen["opened_url"] == browser_url
    assert seen["dev_running"] is False
    assert seen["dev_running_at_release"] is False
    assert seen["lease_release_wait_for_stop"] is False
    assert "WebUI dev: http://127.0.0.1:5173/#/?bootstrapSecret=<redacted>" in re.sub(
        r"\s+", " ", _strip_ansi(result.stdout)
    )


def test_webui_dev_waits_for_external_gateway_via_health_endpoint(monkeypatch) -> None:
    health_results = iter((True, False))
    health_calls: list[tuple[str, int]] = []
    sidecar_checks = 0

    def fake_health(host: str, port: int) -> bool:
        health_calls.append((host, port))
        return next(health_results)

    monkeypatch.setattr("nanobot.cli.webui._gateway_health_ready", fake_health)
    monkeypatch.setattr(
        "nanobot.cli.webui._webui_endpoint_reachable",
        lambda _url: pytest.fail("must not probe the WebSocket endpoint while waiting"),
    )
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    def ensure_sidecar_running() -> None:
        nonlocal sidecar_checks
        sidecar_checks += 1

    dev_server = MagicMock()
    dev_server.ensure_running.side_effect = ensure_sidecar_running
    cli_webui._wait_with_existing_foreground_gateway("127.0.0.1", 18888, dev_server)

    assert health_calls == [("127.0.0.1", 18888), ("127.0.0.1", 18888)]
    assert sidecar_checks == 2


async def test_webui_dev_monitor_fails_when_sidecar_exits() -> None:
    dev_server = MagicMock()
    dev_server.ensure_running.side_effect = WebUIDevError(
        "WebUI development server exited unexpectedly (code 23)"
    )

    with pytest.raises(WebUIDevError, match=r"exited unexpectedly \(code 23\)"):
        await cli_gateway_runtime._watch_webui_dev_server(
            dev_server,
            asyncio.Event(),
            poll_interval_s=0,
        )


async def test_webui_dev_monitor_ignores_an_expected_gateway_shutdown() -> None:
    dev_server = MagicMock()
    shutdown_event = asyncio.Event()
    shutdown_event.set()

    await cli_gateway_runtime._watch_webui_dev_server(
        dev_server,
        shutdown_event,
        poll_interval_s=0,
    )

    dev_server.ensure_running.assert_not_called()


def test_browser_readiness_accepts_http_auth_response(monkeypatch) -> None:
    def auth_required(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "http://127.0.0.1:8765/webui/bootstrap",
            401,
            "authentication required",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlopen", auth_required)

    assert cli_gateway_runtime._http_endpoint_responding(
        "http://127.0.0.1:8765/webui/bootstrap"
    ) is True


def test_browser_readiness_rejects_connection_error(monkeypatch) -> None:
    def unavailable(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", unavailable)

    assert cli_gateway_runtime._http_endpoint_responding(
        "http://127.0.0.1:8765/webui/bootstrap"
    ) is False


@pytest.mark.parametrize("resume_args", [[], ["--yes"]])
def test_webui_resumes_first_run_without_provider_setup(
    monkeypatch, tmp_path: Path, resume_args: list[str],
) -> None:
    config_file = tmp_path / "config.json"
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        "nanobot.cli.webui_support._provider_setup_error",
        lambda _config: "No API key configured for provider 'custom'.",
    )
    monkeypatch.setattr(
        "nanobot.cli.webui._provider_setup_error",
        lambda _config: "No API key configured for provider 'custom'.",
    )
    _patch_gateway_ports_free(monkeypatch)
    monkeypatch.setattr("nanobot.cli.webui.sync_workspace_templates", lambda _path: None)
    _patch_webui_managed_gateway(monkeypatch, seen)

    args = ["webui", "--config", str(config_file), "--workspace", str(tmp_path / "workspace")]
    result = runner.invoke(app, [*args, "--yes", "--no-open"])

    assert result.exit_code == 0
    assert config_file.exists()
    assert seen["start_options"].config_path == str(config_file.resolve(strict=False))
    assert "Configure a provider and model in WebUI Settings → Models." in result.stdout

    saved_config = json.loads(config_file.read_text(encoding="utf-8"))
    seen.clear()
    resumed = runner.invoke(app, [*args, *resume_args, "--no-open"])

    assert resumed.exit_code == 0, resumed.stdout
    assert seen["start_options"].config_path == str(config_file.resolve(strict=False))
    assert "Configure a provider and model in WebUI Settings → Models." in resumed.stdout
    assert "Quick Start" not in resumed.stdout
    assert json.loads(config_file.read_text(encoding="utf-8")) == saved_config
    assert saved_config["channels"]["websocket"]["host"] == "127.0.0.1"
    assert saved_config["channels"]["websocket"]["tokenIssueSecret"]


def test_webui_missing_runtime_env_fails_before_starting_gateway(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "config.json"
    missing_env = "NANOBOT_TEST_MISSING_WEBUI_SECRET"
    monkeypatch.delenv(missing_env, raising=False)
    config_file.write_text(
        json.dumps({
            "agents": {
                "defaults": {
                    "provider": "ollama",
                    "model": "ollama/qwen3",
                }
            },
            "channels": {
                "websocket": {
                    "enabled": True,
                    "host": "0.0.0.0",
                    "tokenIssueSecret": f"${{{missing_env}}}",
                }
            },
        }),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["webui", "--config", str(config_file), "--yes", "--no-open"])

    assert result.exit_code == 1
    assert missing_env in result.stdout
    assert "nanobot status --config" in result.stdout
    assert config_file.name in result.stdout
    assert "Traceback" not in result.stdout
    assert f"${{{missing_env}}}" in config_file.read_text(encoding="utf-8")


def test_webui_yes_opens_settings_for_incomplete_custom_model_setup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({
            "agents": {
                "defaults": {
                    "provider": "custom",
                    "model": "custom/test-model",
                }
            },
            "providers": {
                "custom": {
                    "displayName": "Custom",
                }
            },
        }),
        encoding="utf-8",
    )

    _patch_gateway_ports_free(monkeypatch)
    seen = _patch_webui_managed_gateway(monkeypatch)
    result = runner.invoke(app, [
        "webui", "--config", str(config_file), "--workspace", str(tmp_path / "workspace"),
        "--yes", "--no-open",
    ])

    assert result.exit_code == 0, result.stdout
    assert seen["start_options"].config_path == str(config_file.resolve(strict=False))
    assert "Model setup is incomplete" in result.stdout
    assert "Settings → Models" in _normalized_rendered_text(result.stdout)
    saved = json.loads(config_file.read_text(encoding="utf-8"))
    assert saved["agents"]["defaults"]["model"] == "custom/test-model"
    assert saved["providers"]["custom"]["displayName"] == "Custom"


def test_open_webui_browser_redacts_bootstrap_secret(monkeypatch, capsys) -> None:
    opened: list[str] = []
    url = "http://127.0.0.1:8765/#/?bootstrapSecret=super-secret"
    monkeypatch.setattr(
        cli_webui_support,
        "_launch_browser",
        lambda value: opened.append(value) or True,
    )

    cli_webui_support._open_webui_browser(url, wait=False)

    assert opened == [url]
    output = _strip_ansi(capsys.readouterr().out)
    assert "bootstrapSecret=<redacted>" in output
    assert "super-secret" not in output


def test_open_webui_browser_reports_launch_failure(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli_webui_support, "_launch_browser", lambda _value: False)

    cli_webui_support._open_webui_browser("http://127.0.0.1:8765/", wait=False)

    assert "Could not open browser; visit http://127.0.0.1:8765/" in _strip_ansi(
        capsys.readouterr().out
    )


def test_launch_browser_uses_macos_url_services(monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(cli_webui_support.sys, "platform", "darwin")
    monkeypatch.setattr(
        cli_webui_support,
        "_launch_macos_browser",
        lambda url: seen.append(url) or True,
    )

    assert cli_webui_support._launch_browser("http://127.0.0.1:8765/") is True
    assert seen == ["http://127.0.0.1:8765/"]


def test_launch_browser_uses_default_browser_off_macos(monkeypatch) -> None:
    opened: list[tuple[str, int, bool]] = []
    monkeypatch.setattr(cli_webui_support.sys, "platform", "linux")
    monkeypatch.setattr(
        cli_webui_support.webbrowser,
        "open",
        lambda url, *, new, autoraise: opened.append((url, new, autoraise)) or True,
    )

    assert cli_webui_support._launch_browser("http://127.0.0.1:8765/") is True
    assert opened == [("http://127.0.0.1:8765/", 2, True)]


def test_webui_foreground_attaches_to_existing_managed_gateway(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")
    seen: dict[str, object] = {}
    _record_gateway_lease_release(monkeypatch, seen)
    _patch_webui_provider_ready(monkeypatch)
    monkeypatch.setattr("nanobot.cli.webui.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.cli.webui._gateway_health_ready", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "nanobot.cli.webui_support._gateway_health_ready",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "nanobot.cli.webui._prepare_webui_bundle_for_gateway",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("nanobot.cli.webui._webui_endpoint_reachable", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "nanobot.cli.webui._open_webui_browser",
        lambda url, **kwargs: seen.update({"opened_url": url, "open_kwargs": kwargs}),
    )
    class _FakeRuntime:
        def __init__(self, **kwargs) -> None:
            seen["runtime_kwargs"] = kwargs
            self.paths = kwargs["paths"]
            self.status_calls = 0

        def status(self):
            self.status_calls += 1
            # The command checks during config refresh and once more before
            # delegating to the attach helper.
            # If that helper is not patched as intended, make its polling loop
            # terminate instead of hanging the Windows test worker forever.
            return SimpleNamespace(running=self.status_calls <= 2)

        def start_background(self, options):
            return SimpleNamespace(
                ok=False,
                message="gateway_already_running",
                status=SimpleNamespace(log_path=self.paths.log_path),
            )

        start_on_demand = start_background

        def restart(self, options, *, timeout_s: int):
            seen["restart_options"] = options
            seen["restart_timeout"] = timeout_s
            return SimpleNamespace(
                ok=True,
                message="gateway_started_background",
                status=SimpleNamespace(log_path=self.paths.log_path),
            )

    monkeypatch.setattr("nanobot.gateway.GatewayRuntime", _FakeRuntime)
    monkeypatch.setattr(
        "nanobot.cli.webui._attach_to_background_gateway",
        lambda runtime: seen.__setitem__("attached_runtime", runtime),
    )

    result = runner.invoke(app, ["webui", "--config", str(config_file), "--yes"])

    assert result.exit_code == 0
    assert "Gateway is already running; attaching to the existing WebUI" in result.stdout
    assert isinstance(seen["attached_runtime"], _FakeRuntime)
    opened_url = seen["opened_url"]
    assert isinstance(opened_url, str)
    parsed = urlparse(opened_url)
    assert f"{parsed.scheme}://{parsed.netloc}" == "http://127.0.0.1:8765"
    fragment = parsed.fragment.removeprefix("/?")
    assert parse_qs(fragment).get("bootstrapSecret")
    assert seen["open_kwargs"] == {"wait": False}
    assert seen["lease_release_wait_for_stop"] is False


def test_attach_to_background_gateway_detaches_on_ctrl_c(capsys, tmp_path: Path) -> None:
    stopped = False
    log_path = tmp_path / "gateway.log"
    log_path.touch()

    class _FakeRuntime:
        def status(self):
            return SimpleNamespace(running=True, log_path=log_path)

        def stop(self):
            nonlocal stopped
            stopped = True
            return SimpleNamespace(ok=True, message="gateway_stopped")

    def _interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    cli_webui_support._attach_to_background_gateway(_FakeRuntime(), sleep=_interrupt)

    assert stopped is False
    output = capsys.readouterr().out
    rendered = " ".join(output.split())
    assert "Closing the browser does not stop channels or automations" in rendered
    assert "gateway stops only when the last local client exits" in rendered
    assert "WebUI launcher detached" in rendered


def test_attach_to_background_gateway_follows_only_new_logs(capsys, tmp_path: Path) -> None:
    log_path = tmp_path / "gateway.log"
    log_path.write_text("historical log\n", encoding="utf-8")
    polls = 0

    class _FakeRuntime:
        def status(self):
            return SimpleNamespace(running=True, log_path=log_path)

    def _append_then_interrupt(_seconds: float) -> None:
        nonlocal polls
        if polls == 0:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("[websocket] live log\n")
            polls += 1
            return
        raise KeyboardInterrupt

    cli_webui_support._attach_to_background_gateway(
        _FakeRuntime(),
        sleep=_append_then_interrupt,
    )

    output = capsys.readouterr().out
    assert "[websocket] live log" in output
    assert "historical log" not in output


def test_read_new_gateway_logs_recovers_after_truncation(tmp_path: Path) -> None:
    log_path = tmp_path / "gateway.log"
    log_path.write_text("a much longer historical log line\n", encoding="utf-8")
    cursor = cli_webui_support._start_gateway_log_cursor(log_path)
    log_path.write_text("fresh log\n", encoding="utf-8")

    lines = cli_webui_support._read_new_gateway_logs(log_path, cursor)

    assert lines == ["fresh log"]
    assert cursor.offset == log_path.stat().st_size


def test_read_new_gateway_logs_detects_fast_rewrite_past_offset(tmp_path: Path) -> None:
    log_path = tmp_path / "gateway.log"
    log_path.write_text("historical log\n", encoding="utf-8")
    cursor = cli_webui_support._start_gateway_log_cursor(log_path)
    log_path.write_text("first fresh log\nsecond fresh log\n", encoding="utf-8")

    lines = cli_webui_support._read_new_gateway_logs(log_path, cursor)

    assert lines == ["first fresh log", "second fresh log"]


def test_read_new_gateway_logs_waits_for_complete_utf8_line(tmp_path: Path) -> None:
    log_path = tmp_path / "gateway.log"
    log_path.touch()
    cursor = cli_webui_support._start_gateway_log_cursor(log_path)
    encoded = "模型 ready\n".encode()
    log_path.write_bytes(encoded[:2])

    assert cli_webui_support._read_new_gateway_logs(log_path, cursor) == []

    with log_path.open("ab") as handle:
        handle.write(encoded[2:])

    assert cli_webui_support._read_new_gateway_logs(log_path, cursor) == ["模型 ready"]


def test_read_new_gateway_logs_tolerates_missing_file(tmp_path: Path) -> None:
    log_path = tmp_path / "missing.log"
    cursor = cli_webui_support._start_gateway_log_cursor(log_path)
    lines = cli_webui_support._read_new_gateway_logs(log_path, cursor)

    assert lines == []
    assert cursor.offset == 0


def test_attach_to_background_gateway_checks_owned_sidecar(tmp_path: Path) -> None:
    log_path = tmp_path / "gateway.log"
    log_path.touch()

    class _FakeRuntime:
        def status(self):
            return SimpleNamespace(running=True, log_path=log_path)

    def sidecar_exited() -> None:
        raise WebUIDevError("WebUI development server exited unexpectedly (code 23)")

    with pytest.raises(WebUIDevError, match=r"exited unexpectedly \(code 23\)"):
        cli_webui_support._attach_to_background_gateway(
            _FakeRuntime(),
            poll_hook=sidecar_exited,
        )


def test_webui_foreground_does_not_claim_unmanaged_gateway(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({
        "agents": {"defaults": {"workspace": str(tmp_path / "workspace")}},
    }))
    _patch_webui_provider_ready(monkeypatch)
    monkeypatch.setattr("nanobot.cli.webui._prepare_webui_bundle_for_gateway", lambda *_a, **_kw: None)
    monkeypatch.setattr("nanobot.cli.webui.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.cli.webui._gateway_health_ready", lambda *_args: True)
    monkeypatch.setattr("nanobot.cli.webui._webui_endpoint_reachable", lambda *_args: True)
    monkeypatch.setattr("nanobot.cli.webui._open_webui_browser", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "nanobot.cli.webui._attach_to_background_gateway",
        lambda _runtime: pytest.fail("unmanaged gateway must not be attached"),
    )

    class _FakeRuntime:
        def __init__(self, **kwargs) -> None:
            self.paths = kwargs["paths"]

        def status(self):
            return SimpleNamespace(running=False)

    monkeypatch.setattr("nanobot.gateway.GatewayRuntime", _FakeRuntime)

    result = runner.invoke(app, ["webui", "--config", str(config_file), "--yes"])

    assert result.exit_code == 0
    assert "controlled by another foreground command" in result.stdout


def test_webui_foreground_refuses_occupied_webui_port(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({
        "agents": {"defaults": {"workspace": str(tmp_path / "workspace")}},
    }))
    _patch_webui_provider_ready(monkeypatch)
    monkeypatch.setattr("nanobot.cli.webui._prepare_webui_bundle_for_gateway", lambda *_a, **_kw: None)
    monkeypatch.setattr("nanobot.cli.webui.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.cli.webui._gateway_health_ready", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        "nanobot.cli.webui_support._gateway_health_ready",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr("nanobot.cli.webui._webui_endpoint_reachable", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("nanobot.cli.webui._tcp_endpoint_reachable", lambda *_args, **_kwargs: False)
    result = runner.invoke(app, ["webui", "--config", str(config_file), "--yes"])

    assert result.exit_code == 1
    assert "nanobot cannot start because one of its local ports is already in use" in result.stdout
    assert "--port" in result.stdout
    assert "--gateway-port" in result.stdout


def test_webui_foreground_reports_an_existing_gateway_without_leaking_secret(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({
            "agents": {"defaults": {"workspace": str(tmp_path / "workspace")}},
            "channels": {"websocket": {"tokenIssueSecret": "do-not-leak"}},
        }),
        encoding="utf-8",
    )
    _patch_webui_provider_ready(monkeypatch)
    monkeypatch.setattr("nanobot.cli.webui._prepare_webui_bundle_for_gateway", lambda *_a, **_kw: None)
    monkeypatch.setattr("nanobot.cli.webui.sync_workspace_templates", lambda _path: None)
    monkeypatch.setattr("nanobot.cli.webui._gateway_health_ready", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "nanobot.cli.webui_support._gateway_health_ready",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr("nanobot.cli.webui._webui_endpoint_reachable", lambda *_args, **_kwargs: False)

    result = runner.invoke(app, ["webui", "--config", str(config_file), "--yes"])

    assert result.exit_code == 1
    assert "gateway is already running for this local instance" in result.stdout
    assert "bootstrapSecret=<redacted>" in result.stdout
    assert "do-not-leak" not in result.stdout


def _patch_serve_runtime(
    monkeypatch, config: Config, seen: dict[str, object], config_file: Path,
) -> None:
    from nanobot.api.server import create_app
    from nanobot.runtime_composition import RuntimeComposition, RuntimeState

    # Command-option tests define the real graph/app without starting a server.
    # Runtime acquisition is separately exercised by API composition tests.
    if config.agents.defaults.workspace == Config().agents.defaults.workspace:
        config.agents.defaults.workspace = str(config_file.parent / "workspace")
    config.bind_source_path(config_file)
    config.agents.defaults.session_ttl_minutes = 0
    config.agents.defaults.idle_compaction.enabled = False

    def forbidden(*_args, **_kwargs):
        raise AssertionError("API command configuration must not acquire runtime resources")

    async def forbidden_acquire(*_args, **_kwargs):
        forbidden()

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
    monkeypatch.setattr(RuntimeComposition, "acquire", forbidden_acquire)
    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("socket.socket.bind", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr("nanobot.api.server.create_app", _record_create_app)
    monkeypatch.setattr("aiohttp.web.run_app", _fake_run_app)


def test_gateway_uses_workspace_from_config_by_default(monkeypatch, tmp_path: Path) -> None:
    import traceback

    config_file = _write_instance_config(tmp_path)
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    config.workspace_path.mkdir()
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

    assert isinstance(result.exception, _StopGatewayError), "".join(traceback.format_exception(result.exception))
    assert seen["config_path"] == config_file.resolve()
    assert seen["workspace"] == Path(config.agents.defaults.workspace)


def test_gateway_workspace_option_overrides_config(monkeypatch, tmp_path: Path) -> None:
    config_file = _write_instance_config(tmp_path)
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    override = tmp_path / "override-workspace"
    override.mkdir()
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


def test_gateway_uses_workspace_directory_for_cron_store(monkeypatch, tmp_path: Path) -> None:
    config_file = _write_instance_config(tmp_path)
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    config.workspace_path.mkdir()
    seen: dict[str, Path] = {}

    class _StopCron:
        def __init__(self, store_path: Path) -> None:
            seen["cron_store"] = store_path
            raise _StopGatewayError("stop")

    _patch_cli_command_runtime(
        monkeypatch,
        config,
        message_bus=MessageBus,
        cron_service=_StopCron,
    )

    result = runner.invoke(app, ["gateway", "--config", str(config_file)])

    assert isinstance(result.exception, _StopGatewayError)
    assert seen["cron_store"] == config.workspace_path / "cron" / "jobs.json"








def test_gateway_workspace_override_does_not_migrate_legacy_cron(
    monkeypatch, tmp_path: Path
) -> None:
    config_file = _write_instance_config(tmp_path)
    legacy_dir = tmp_path / "global" / "cron"
    legacy_dir.mkdir(parents=True)
    legacy_file = legacy_dir / "jobs.json"
    legacy_file.write_text('{"jobs": []}')

    override = tmp_path / "override-workspace"
    override.mkdir()
    config = Config()
    seen: dict[str, Path] = {}

    class _StopCron:
        def __init__(self, store_path: Path) -> None:
            seen["cron_store"] = store_path
            raise _StopGatewayError("stop")

    _patch_cli_command_runtime(
        monkeypatch,
        config,
        message_bus=MessageBus,
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
    monkeypatch, tmp_path: Path
) -> None:
    config_file = _write_instance_config(tmp_path)
    legacy_dir = tmp_path / "global" / "cron"
    legacy_dir.mkdir(parents=True)
    legacy_file = legacy_dir / "jobs.json"
    legacy_file.write_text('{"jobs": []}')

    custom_workspace = tmp_path / "custom-workspace"
    custom_workspace.mkdir()
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
        message_bus=MessageBus,
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
    """Observe CLI port handoff without constructing providers or binding sockets."""
    class Composition:
        async def acquire(self):
            return SimpleNamespace(
                agent=SimpleNamespace(stop=lambda: seen.__setitem__("stopped", True)),
                channels=SimpleNamespace(),
                bus=SimpleNamespace(drain=AsyncMock()),
            )

        async def aclose(self):
            seen["closed"] = True

    def probe(host, port):
        seen["host"] = host
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


def test_gateway_uses_configured_port_when_cli_flag_is_missing(monkeypatch, tmp_path: Path) -> None:
    config_file = _write_instance_config(tmp_path)
    config = Config()
    config.gateway.port = 18791
    config.agents.defaults.workspace = str(tmp_path / "workspace")
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


def test_gateway_cli_port_overrides_configured_port(monkeypatch, tmp_path: Path) -> None:
    config_file = _write_instance_config(tmp_path)
    config = Config()
    config.gateway.port = 18791
    config.agents.defaults.workspace = str(tmp_path / "workspace")
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


@pytest.mark.parametrize(
    ("host", "display_url", "warns_about_public_bind"),
    [
        ("127.0.0.1", "http://127.0.0.1:18791/health", False),
        ("0.0.0.0", "http://127.0.0.1:18791/health", True),
    ],
)
def test_gateway_health_endpoint_binds_and_serves_expected_responses(
    monkeypatch,
    capsys,
    host: str,
    display_url: str,
    warns_about_public_bind: bool,
) -> None:
    captured: dict[str, object] = {}

    class _FakeServer:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            captured["server_closed"] = True
            return False

        async def serve_forever(self) -> None:
            return None

    async def _fake_start_server(handler, host: str, port: int):
        captured["handler"] = handler
        captured["host"] = host
        captured["port"] = port
        return _FakeServer()

    class _FakeReader:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        async def read(self, _size: int) -> bytes:
            return self.payload

    class _FakeWriter:
        def __init__(self) -> None:
            self.output = b""
            self.closed = False

        def write(self, data: bytes) -> None:
            self.output += data

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("asyncio.start_server", _fake_start_server)
    asyncio.run(cli_gateway_runtime._health_server(
        host, 18791, SimpleNamespace(enabled_channels=["telegram", "discord"]),
    ))
    output = _normalized_rendered_text(capsys.readouterr().out)

    assert captured["server_closed"] is True
    assert captured["host"] == host
    assert captured["port"] == 18791
    assert f"Health endpoint: {display_url}" in output
    assert ("unauthenticated health endpoint" in output) is warns_about_public_bind
    assert ("listening beyond loopback" in output) is warns_about_public_bind
    assert ("Keep port 18791 private or protect it" in output) is warns_about_public_bind
    assert ("listening on 0.0.0.0" in output) is warns_about_public_bind

    health_handler = captured["handler"]
    assert callable(health_handler)

    def _call_handler(path: str) -> tuple[str, _FakeWriter]:
        request = f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode()
        writer = _FakeWriter()
        asyncio.run(health_handler(_FakeReader(request), writer))
        return writer.output.decode(), writer

    root_response, root_writer = _call_handler("/")
    assert root_writer.closed is True
    assert "HTTP/1.0 404 Not Found" in root_response
    assert "Connection: close" in root_response
    assert root_response.endswith("\r\n\r\nNot Found")

    health_response, health_writer = _call_handler("/health")
    assert health_writer.closed is True
    assert "HTTP/1.0 200 OK" in health_response
    health_body = json.loads(health_response.split("\r\n\r\n", 1)[1])
    assert health_body == {
        "status": "ok",
        "process": "alive",
        "ready": True,
        "websocket": "disabled",
    }

    missing_response, missing_writer = _call_handler("/missing")
    assert missing_writer.closed is True
    assert "HTTP/1.0 404 Not Found" in missing_response
    assert missing_response.endswith("\r\n\r\nNot Found")

    if host == "127.0.0.1":
        async def _exercise_connection_limit() -> None:
            release = asyncio.Event()
            all_started = asyncio.Event()
            started = 0

            class _BlockingReader:
                async def read(self, _size: int) -> bytes:
                    nonlocal started
                    started += 1
                    if started == cli_gateway_runtime._GATEWAY_HEALTH_MAX_CONNECTIONS:
                        all_started.set()
                    await release.wait()
                    return b"GET /health HTTP/1.1\r\n\r\n"

            active_writers = [
                _FakeWriter()
                for _ in range(cli_gateway_runtime._GATEWAY_HEALTH_MAX_CONNECTIONS)
            ]
            active_tasks = [
                asyncio.create_task(health_handler(_BlockingReader(), writer))
                for writer in active_writers
            ]
            await asyncio.wait_for(all_started.wait(), timeout=1)

            overflow_writer = _FakeWriter()
            await health_handler(
                _FakeReader(b"GET /health HTTP/1.1\r\n\r\n"),
                overflow_writer,
            )
            assert overflow_writer.closed is True
            assert overflow_writer.output == b""

            release.set()
            await asyncio.gather(*active_tasks)

        asyncio.run(_exercise_connection_limit())

        class _NeverRespondingReader:
            async def read(self, _size: int) -> bytes:
                await asyncio.Event().wait()

        monkeypatch.setattr(
            cli_gateway_runtime,
            "_GATEWAY_HEALTH_READ_TIMEOUT_SECONDS",
            0.01,
        )
        timed_out_writer = _FakeWriter()
        asyncio.run(health_handler(_NeverRespondingReader(), timed_out_writer))
        assert timed_out_writer.closed is True
        assert timed_out_writer.output == b""






def test_serve_uses_api_config_defaults_and_workspace_override(
    monkeypatch, tmp_path: Path
) -> None:
    config_file = _write_instance_config(tmp_path)
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "config-workspace")
    config.api.host = "127.0.0.2"
    config.api.port = 18900
    config.api.timeout = 45.0
    config.api.api_key = "secret"
    override_workspace = tmp_path / "override-workspace"
    seen: dict[str, object] = {}

    _patch_serve_runtime(monkeypatch, config, seen, config_file)

    result = runner.invoke(
        app,
        ["serve", "--config", str(config_file), "--workspace", str(override_workspace)],
    )

    assert result.exit_code == 0
    assert seen["workspace"] == override_workspace
    assert seen["host"] == "127.0.0.2"
    assert seen["port"] == 18900
    assert seen["request_timeout"] == 45.0
    assert seen["api_key"] == "secret"


def test_trigger_cli_queues_message_in_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from nanobot.triggers.local_store import LocalTriggerStore

    config_file = _write_instance_config(tmp_path)
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "workspace")
    _patch_cli_command_runtime(monkeypatch, config)

    store = LocalTriggerStore(config.workspace_path)
    trigger = store.create(
        name="Review hook",
        channel="websocket",
        chat_id="chat-1",
        session_key="websocket:chat-1",
    )

    result = runner.invoke(
        app,
        ["trigger", "--config", str(config_file), trigger.id, "Review PR #4502"],
    )

    assert result.exit_code == 0
    assert f"Queued {trigger.id}" in result.stdout
    deliveries = store.claim_deliveries()
    assert len(deliveries) == 1
    assert deliveries[0].trigger_id == trigger.id
    assert deliveries[0].content == "Review PR #4502"


def test_serve_cli_options_override_api_config(monkeypatch, tmp_path: Path) -> None:
    config_file = _write_instance_config(tmp_path)
    config = Config()
    config.api.host = "127.0.0.2"
    config.api.port = 18900
    config.api.timeout = 45.0
    config.api.api_key = "secret"
    seen: dict[str, object] = {}

    _patch_serve_runtime(monkeypatch, config, seen, config_file)

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
    assert seen["api_key"] == "secret"


def test_serve_allows_loopback_without_api_key(monkeypatch, tmp_path: Path) -> None:
    config_file = _write_instance_config(tmp_path)
    config = Config()
    seen: dict[str, object] = {}

    _patch_serve_runtime(monkeypatch, config, seen, config_file)

    result = runner.invoke(app, ["serve", "--config", str(config_file)])

    assert result.exit_code == 0
    assert seen["host"] == "127.0.0.1"
    assert seen["api_key"] == ""


def test_serve_passes_configured_api_key(monkeypatch, tmp_path: Path) -> None:
    config_file = _write_instance_config(tmp_path)
    config = Config()
    config.api.api_key = " secret "
    seen: dict[str, object] = {}

    _patch_serve_runtime(monkeypatch, config, seen, config_file)

    result = runner.invoke(app, ["serve", "--config", str(config_file)])

    assert result.exit_code == 0
    assert seen["api_key"] == "secret"


def test_serve_rejects_wildcard_host_without_api_key(monkeypatch, tmp_path: Path) -> None:
    config_file = _write_instance_config(tmp_path)
    config = Config()
    seen: dict[str, object] = {}

    _patch_serve_runtime(monkeypatch, config, seen, config_file)

    result = runner.invoke(app, ["serve", "--config", str(config_file), "--host", "0.0.0.0"])

    assert result.exit_code == 1
    assert "api_key is not set" in result.stdout
    assert "workspace" not in seen
    assert "api_app" not in seen


def test_serve_rejects_specific_network_interface_without_api_key(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_file = _write_instance_config(tmp_path)
    config = Config()
    seen: dict[str, object] = {}

    _patch_serve_runtime(monkeypatch, config, seen, config_file)

    result = runner.invoke(
        app,
        ["serve", "--config", str(config_file), "--host", "192.168.1.10"],
    )

    assert result.exit_code == 1
    assert "api_key" in result.stdout
    assert "prevent unauthenticated access" in result.stdout
    assert "api_app" not in seen


def test_channels_login_requires_channel_name() -> None:
    result = runner.invoke(app, ["channels", "login"])

    assert result.exit_code == 2
