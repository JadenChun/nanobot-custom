"""CLI commands for nanobot."""

# pyright: reportConstantRedefinition=false, reportMissingTypeStubs=false, reportPrivateUsage=false, reportUnusedFunction=false, reportUnusedImport=false

import asyncio
import os
import sys
from collections.abc import AsyncGenerator, Callable, Coroutine, Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    if sys.stdout.encoding != "utf-8":
        os.environ["PYTHONIOENCODING"] = "utf-8"
        # Re-open stdout/stderr with UTF-8 encoding
        with suppress(Exception):
            for stream in (sys.stdout, sys.stderr):
                reconfigure = getattr(stream, "reconfigure", None)
                if callable(reconfigure):
                    reconfigure(encoding="utf-8", errors="replace")

# Keep console encoding setup before importing CLI UI/logging libraries.
import typer  # noqa: E402
from loguru import logger  # noqa: E402
from typer.core import TyperGroup  # noqa: E402

from nanobot.cli.log_control import register_cli_console_sink  # noqa: E402

# Remove default handler and re-add with unified nanobot format
logger.remove()
_log_handler_id = logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <5}</level> | "
        "<cyan>{extra[channel]}</cyan> | "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=None,
    filter=lambda record: record["extra"].setdefault("channel", "-") or True,
)
register_cli_console_sink(_log_handler_id)


from rich.console import Console  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from nanobot import __logo__, __version__  # noqa: E402
from nanobot import optional_features as feature_support  # noqa: E402
from nanobot.cli import terminal as cli_terminal  # noqa: E402
from nanobot.cli.agent import agent  # noqa: E402
from nanobot.cli.cron_delivery import cron_delivery_app  # noqa: E402
from nanobot.cli.cron_runtime import cron_runtime_app  # noqa: E402
from nanobot.cli.gateway import create_gateway_app  # noqa: E402
from nanobot.cli.gateway_runtime import _run_gateway  # noqa: E402
from nanobot.cli.log_control import _set_nanobot_logs  # noqa: E402
from nanobot.cli.process_identity import set_cli_process_identity  # noqa: E402
from nanobot.cli.provider import provider_app  # noqa: E402
from nanobot.cli.runtime_config import (  # noqa: E402
    _load_inspection_config,
    _load_runtime_config,
    _model_display,
    _print_config_error,
    _print_model_setup_steps,
    _provider_setup_error,
)
from nanobot.cli.webui import webui  # noqa: E402
from nanobot.cli.webui_support import (  # noqa: E402
    _prepare_webui_bundle_for_gateway,
    _validate_gateway_startup,
)
from nanobot.config.schema import Config  # noqa: E402
from nanobot.security.network import is_loopback_host  # noqa: E402
from nanobot.utils.helpers import sanitize_surrogates as _sanitize_surrogates  # noqa: E402,F401
from nanobot.utils.helpers import (  # noqa: E402
    sync_workspace_templates,
)

# Backward-compatible import for callers that used the former module location.
SafeFileHistory = cli_terminal.SafeFileHistory

if TYPE_CHECKING:
    from nanobot.session.artifact_operator import CopiedArtifactOperator
    from nanobot.session.manager import SessionManager

_T = TypeVar("_T")


class _DesktopAwareGroup(TyperGroup):
    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        # Keep exact arguments: explicitly passing even a default-valued option
        # must bypass the picker. Also covers older commands:app launchers.
        ctx.meta["desktop_target_args"] = list(args)
        return super().parse_args(ctx, args)


app = typer.Typer(
    cls=_DesktopAwareGroup,
    name="nanobot",
    context_settings={"help_option_names": ["-h", "--help"]},
    help=f"{__logo__} nanobot - Personal AI Assistant",
    epilog=(
        "Run `nanobot` without a subcommand to start the terminal agent. "
        "Use `nanobot agent --help` for agent options."
    ),
    invoke_without_command=True,
    no_args_is_help=False,
)
app.add_typer(cron_delivery_app, name="cron-delivery")
app.add_typer(cron_runtime_app, name="cron-runtime")

console = Console()


artifacts_app = typer.Typer(help="Manage browser test artifacts")
app.add_typer(artifacts_app, name="artifacts")


@contextmanager
def _admitted_browser_artifacts_dir(
    config: str | None, workspace: str | None,
) -> Generator[Path, None, None]:
    """Select only this workspace's browser-test directory under startup admission."""
    from nanobot.agent.memory_recovery_startup_admission import (
        StartupReceiptAdmissionError,
        admitted_startup_mutation,
        require_startup_receipt_admission,
    )
    from nanobot.config.paths import (
        get_browser_test_artifacts_dir,
        get_default_sessions_root,
    )
    from nanobot.migration_guard import require_runtime_ready

    require_runtime_ready()
    runtime_config = _load_runtime_config(config, workspace)
    workspace_path = runtime_config.workspace_path.resolve(strict=False)
    if runtime_config.runtime_data_dir is not None:
        sessions_root = runtime_config.runtime_data_dir / "sessions"
    elif config:
        sessions_root = Path(config).expanduser().resolve(strict=False).parent / "sessions"
    else:
        sessions_root = get_default_sessions_root()
    candidate = runtime_config.workspace_path / "artifacts" / "browser-tests"
    if (
        candidate.parent.is_symlink()
        or candidate.is_symlink()
        or not candidate.resolve(strict=False).is_relative_to(workspace_path)
    ):
        console.print("[red]Error: browser test artifacts must remain inside the workspace.[/red]")
        raise typer.Exit(1)
    try:
        require_startup_receipt_admission(workspace_path, sessions_root=sessions_root)
        with admitted_startup_mutation(workspace_path, sessions_root=sessions_root):
            yield get_browser_test_artifacts_dir(runtime_config.workspace_path)
    except StartupReceiptAdmissionError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from None


@artifacts_app.command("path")
def artifacts_path(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
) -> None:
    """Show the workspace path used for browser test artifacts."""
    with _admitted_browser_artifacts_dir(config, workspace) as artifacts_dir:
        entries = [entry for entry in artifacts_dir.iterdir() if entry.name != ".gitkeep"]
        console.print(f"Browser test artifacts: {artifacts_dir}")
        console.print(f"Entries: {len(entries)}")


@artifacts_app.command("clean")
def artifacts_clean(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Remove saved browser test artifacts while preserving their root directory."""
    import shutil

    with _admitted_browser_artifacts_dir(config, workspace) as artifacts_dir:
        targets = [entry for entry in artifacts_dir.iterdir() if entry.name != ".gitkeep"]
        if not targets:
            console.print(f"[green]✓[/green] Browser test artifacts already empty: {artifacts_dir}")
            return
        if not yes and not typer.confirm(
            f"Delete {len(targets)} item(s) from {artifacts_dir}?", default=False,
        ):
            console.print("Cancelled.")
            raise typer.Exit(1)
        for target in targets:
            if target.is_symlink() or not target.is_dir():
                target.unlink()
            else:
                shutil.rmtree(target)
        console.print(f"[green]✓[/green] Removed {len(targets)} item(s) from {artifacts_dir}")

def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} nanobot v{__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    """nanobot - Personal AI Assistant."""
    # Editable/source installs can retain an older generated console script that
    # imports this Typer app directly instead of ``nanobot.cli.entry``. Keep the
    # role identity correct until that launcher is regenerated.
    command = ctx.invoked_subcommand
    set_cli_process_identity([command] if command else ["agent"])
    if command is None:
        console.print(ctx.get_help())


# ============================================================================
# Onboard / Setup
# ============================================================================


@app.command()
def onboard(
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    wizard: bool = typer.Option(False, "--wizard", help="Use interactive wizard"),
    non_interactive_refresh: bool = typer.Option(False, "--refresh", help="Refresh config, preserving existing settings without prompting"),
):
    """Initialize nanobot configuration and workspace."""
    from nanobot.migration_guard import require_runtime_ready
    require_runtime_ready()
    from nanobot.agent.memory_recovery_startup_admission import (
        StartupReceiptAdmissionError,
        admitted_startup_mutation,
        require_startup_receipt_admission,
    )
    from nanobot.config.loader import get_config_path, load_config, save_config, set_config_path
    from nanobot.config.schema import Config

    explicit_config = config is not None
    if config:
        config_path = Path(config).expanduser().resolve()
        set_config_path(config_path)
        console.print(f"[dim]Using config: {config_path}[/dim]")
    else:
        config_path = get_config_path()

    def _apply_workspace_override(loaded: Config) -> Config:
        if workspace:
            loaded.agents.defaults.workspace = workspace
        return loaded

    loaded_config: Config | None = None
    save_note = ""
    # Create or update config
    if config_path.exists():
        if wizard:
            loaded_config = _apply_workspace_override(load_config(config_path))
        else:
            should_refresh = non_interactive_refresh
            if not non_interactive_refresh:
                console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
                console.print(
                    "  [bold]y[/bold] = overwrite with defaults (existing values will be lost)"
                )
                console.print(
                    "  [bold]N[/bold] = refresh config, keeping existing values and adding new fields"
                )
                if typer.confirm("Overwrite?"):
                    loaded_config = _apply_workspace_override(Config())
                    save_note = f"Config reset to defaults at {config_path}"
                else:
                    should_refresh = True

            if should_refresh:
                loaded_config = _apply_workspace_override(load_config(config_path))
                save_note = f"Config refreshed at {config_path} (existing values preserved)"
    else:
        loaded_config = _apply_workspace_override(Config())
        # In wizard mode, don't save yet - the wizard will handle saving if should_save=True
        if not wizard:
            save_note = f"Created config at {config_path}"

    assert loaded_config is not None

    # Run interactive wizard if enabled
    if wizard:
        from nanobot.cli.onboard import run_onboard

        try:
            result = run_onboard(initial_config=loaded_config)
            if not result.should_save:
                console.print("[yellow]Configuration discarded. No changes were saved.[/yellow]")
                return

            loaded_config = result.config
            save_note = f"Config saved at {config_path}"
        except Exception as e:
            console.print(f"[red]✗[/red] Error during configuration: {e}")
            console.print("[yellow]Please run 'nanobot onboard' again to complete setup.[/yellow]")
            raise typer.Exit(1)
    workspace_path = loaded_config.workspace_path.resolve(strict=False)
    workspace_was_missing = not workspace_path.exists()
    sessions_root = (loaded_config.runtime_data_dir or config_path.parent) / "sessions"
    try:
        require_startup_receipt_admission(workspace_path, sessions_root=sessions_root)
        with admitted_startup_mutation(workspace_path, sessions_root=sessions_root):
            save_config(loaded_config, config_path)
            if save_note:
                console.print(f"[green]✓[/green] {save_note}")
            _onboard_plugins(config_path)
            if not workspace_path.exists():
                workspace_path.mkdir(parents=True, exist_ok=True)
            if workspace_was_missing:
                console.print(f"[green]✓[/green] Created workspace at {workspace_path}")
            sync_workspace_templates(workspace_path)
    except StartupReceiptAdmissionError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from None

    webui_cmd = "nanobot webui"
    if explicit_config:
        webui_cmd += f' -c "{config_path}"'

    typer.echo(f"\n✓ nanobot is ready. Run: {webui_cmd}")


def _onboard_plugins(config_path: Path) -> None:
    """Inject default config for all discovered channels (built-in + plugins)."""
    import json

    from nanobot.channels.contracts import channel_default_config
    from nanobot.channels.registry import discover_plugins
    from nanobot.config.loader import merge_missing_defaults

    plugins = discover_plugins()
    if not plugins:
        return

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    channels = data.setdefault("channels", {})
    for name, plugin in plugins.items():
        defaults = channel_default_config(plugin)
        if name not in channels:
            channels[name] = defaults
        else:
            channels[name] = merge_missing_defaults(channels[name], defaults)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _print_enable_options(
    extras: dict[str, list[str] | None],
    channel_plugins: dict[str, Any],
    config: Config,
) -> None:
    table = Table(title="Available Features")
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Enabled")

    for item in sorted(set(channel_plugins) | set(extras)):
        plugin = channel_plugins.get(item)
        is_channel = plugin is not None
        enabled = (
            feature_support.channel_enabled(
                config,
                item,
                plugin,
                default_enabled=plugin.default_enabled,
            )
            if is_channel
            else feature_support.extra_installed(item, extras[item])
        )
        table.add_row(
            item,
            "channel" if is_channel else "feature",
            "[green]yes[/green]" if enabled else "[dim]no[/dim]",
        )

    console.print(table)


def _read_trigger_cli_message(message: str | None) -> str:
    """Read a trigger message from an argument or stdin."""
    if message and message.strip():
        return message
    try:
        if not sys.stdin.isatty():
            content = sys.stdin.read()
            if content.strip():
                return content
    except Exception:
        pass
    console.print("[red]Error: trigger message is required[/red]")
    raise typer.Exit(1)


@app.command()
def trigger(
    trigger_id: str = typer.Argument(..., help="Trigger ID returned by /trigger"),
    message: str | None = typer.Argument(None, help="Message to deliver; stdin is used when omitted"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Deliver a local trigger message to its bound chat session."""
    from nanobot.triggers.local_store import (
        LocalTriggerStore,
        TriggerDisabledError,
        TriggerNotFoundError,
        TriggerStoreError,
    )

    runtime_config = _load_runtime_config(config, workspace)
    content = _read_trigger_cli_message(message)
    store = LocalTriggerStore(runtime_config.workspace_path)
    try:
        delivery = store.enqueue(trigger_id, content)
    except (TriggerNotFoundError, TriggerDisabledError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    except (TriggerStoreError, ValueError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Queued[/green] {delivery.trigger_id} ({delivery.id})")


# ============================================================================
# OpenAI-Compatible API Server
# ============================================================================


@app.command()
def serve(
    port: int | None = typer.Option(None, "--port", "-p", help="API server port"),
    host: str | None = typer.Option(None, "--host", "-H", help="Bind address"),
    timeout: float | None = typer.Option(None, "--timeout", "-t", help="Per-request timeout (seconds)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show nanobot runtime logs"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Start the OpenAI-compatible API server (/v1/chat/completions)."""
    try:
        from aiohttp import web  # noqa: F401
    except ImportError:
        console.print("[red]aiohttp is required. Install with: nanobot plugins enable api[/red]")
        raise typer.Exit(1)

    from nanobot.agent.memory_recovery_startup_admission import (
        StartupReceiptAdmissionError,
        admitted_startup_mutation,
        require_startup_receipt_admission,
    )
    from nanobot.api.server import create_app
    from nanobot.config.paths import get_default_sessions_root
    from nanobot.runtime_composition import RuntimeComposition, RuntimePreflightError

    _set_nanobot_logs(verbose)

    runtime_config = _load_runtime_config(config, workspace)
    api_cfg = runtime_config.api
    host = host if host is not None else api_cfg.host
    port = port if port is not None else api_cfg.port
    timeout = timeout if timeout is not None else api_cfg.timeout
    api_key = api_cfg.api_key.strip() if api_cfg.api_key else ""
    if not is_loopback_host(host) and not api_key:
        console.print(
            f"[red]Error: host {host} is available beyond this device but api_key is not set. "
            "Set api.api_key in config to prevent unauthenticated access.[/red]"
        )
        raise typer.Exit(1)
    workspace_path = runtime_config.workspace_path.resolve(strict=False)
    if runtime_config.runtime_data_dir is not None:
        sessions_root = runtime_config.runtime_data_dir / "sessions"
    elif config:
        sessions_root = Path(config).expanduser().resolve(strict=False).parent / "sessions"
    else:
        sessions_root = get_default_sessions_root()
    try:
        require_startup_receipt_admission(workspace_path, sessions_root=sessions_root)
    except StartupReceiptAdmissionError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from None
    try:
        composition = RuntimeComposition.define(
            runtime_config,
            sessions_root=sessions_root,
            startup_receipt_admitted=True,
        )
    except (StartupReceiptAdmissionError, RuntimePreflightError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from None
    model_name, preset_tag = _model_display(runtime_config)
    console.print(f"{__logo__} Starting OpenAI-compatible API server")
    console.print(f"  [cyan]Endpoint[/cyan] : http://{host}:{port}/v1/chat/completions")
    console.print(f"  [cyan]Model[/cyan]    : {model_name}{preset_tag}")
    console.print("  [cyan]Session[/cyan]  : api:default")
    console.print(f"  [cyan]Timeout[/cyan]  : {timeout}s")
    if not is_loopback_host(host):
        console.print(
            "[yellow]API is available beyond this device "
            "(authentication required).[/yellow]"
        )
    console.print()

    api_app = create_app(
        model_name=model_name, request_timeout=timeout,
        api_key=api_key, composition=composition,
    )

    async def prepare_runtime_context(_app: web.Application) -> AsyncGenerator[None, None]:
        with admitted_startup_mutation(workspace_path, sessions_root=sessions_root):
            sync_workspace_templates(workspace_path)
        yield

    api_app.cleanup_ctx.append(prepare_runtime_context)

    def _log_aiohttp(message: object) -> None:
        logger.info("{}", message)

    web.run_app(api_app, host=host, port=port, print=_log_aiohttp)


# ============================================================================
# WebUI Launcher
# ============================================================================

app.command(name="webui")(webui)


# ============================================================================
# Gateway / Server
# ============================================================================


app.add_typer(
    create_gateway_app(
        console=console,
        log_handler_id=_log_handler_id,
        load_runtime_config=_load_runtime_config,
        run_gateway=_run_gateway,
        validate_startup_config=_validate_gateway_startup,
        prepare_webui_bundle=lambda config, mode: _prepare_webui_bundle_for_gateway(
            config,
            mode=mode,
        ),
    ),
    name="gateway",
)


# ============================================================================
# Agent Commands
# ============================================================================


app.command(name="agent")(agent)


# ============================================================================
# Session Commands
# ============================================================================


sessions_app = typer.Typer(help="Manage persisted session history")
app.add_typer(sessions_app, name="sessions")

session_artifacts_app = typer.Typer(help="Inspect copied canonical-session artifacts")
sessions_app.add_typer(session_artifacts_app, name="artifacts")

_OFFLINE_WARNING = (
    "Offline copy confirmed. Stop every writer first: older nanobot binaries do not honor "
    "the Stage 2 writer lease."
)
_OFFLINE_ERROR = "Copied-state session operation is unavailable."


def _require_offline_copy(offline_copy: bool) -> None:
    if not offline_copy:
        console.print("[red]Error: --offline-copy is required.[/red]")
        raise typer.Exit(1)
    console.print(f"[yellow]Warning:[/yellow] {_OFFLINE_WARNING}")


def _parse_copied_generation(value: str) -> int:
    if (
        not value
        or len(value) > 20
        or (value != "0" and (not value.isascii() or not value.isdigit() or value[0] == "0"))
    ):
        console.print(f"[red]Error: {_OFFLINE_ERROR}[/red]")
        raise typer.Exit(1)
    return int(value)


def _copied_state_config(config: str | None) -> tuple[Config, Path]:
    from nanobot.config.loader import get_config_path, load_config
    from nanobot.config.paths import get_default_sessions_root

    config_path = Path(config).expanduser().resolve(strict=False) if config else get_config_path()
    try:
        if config is not None and not config_path.is_file():
            raise ValueError
        loaded = load_config(config_path)
    except (OSError, RuntimeError, UnicodeError, ValueError):
        console.print(f"[red]Error: {_OFFLINE_ERROR}[/red]")
        raise typer.Exit(1) from None
    live_sessions_root = (
        loaded.runtime_data_dir / "sessions"
        if loaded.runtime_data_dir is not None
        else get_default_sessions_root()
    )
    return loaded, live_sessions_root.expanduser().resolve(strict=False)


async def _with_offline_manager(
    loaded: Config,
    live_sessions_root: Path,
    copied_workspace: Path,
    copied_sessions_root: Path,
    operation: Callable[["SessionManager"], _T],
) -> _T:
    from nanobot.session.runtime_authority import OfflineSessionOperator

    lease = await OfflineSessionOperator(
        copied_workspace,
        copied_sessions_root=copied_sessions_root,
        live_workspace=loaded.workspace_path,
        live_sessions_root=live_sessions_root,
    ).acquire()
    try:
        return operation(lease.manager)
    finally:
        if lease.close is not None:
            close_result = lease.close()
            if close_result is not None:
                await close_result


def _run_copied_operation(operation: Coroutine[object, object, _T]) -> _T:
    try:
        return asyncio.run(operation)
    except (OSError, RuntimeError, UnicodeError, ValueError):
        console.print(f"[red]Error: {_OFFLINE_ERROR}[/red]")
        raise typer.Exit(1) from None


def _configured_artifact_operator(
    loaded: Config,
    live_sessions_root: Path,
    copied_workspace: Path,
    copied_sessions_root: Path,
    copied_artifact_root: Path,
) -> "CopiedArtifactOperator":
    from nanobot.agent.tools.filesystem import ReadFileTool
    from nanobot.config.paths import get_media_dir
    from nanobot.context_repo import ContextRepoManager
    from nanobot.context_repo.policy import ResourceAccessPolicy
    from nanobot.session.artifact_factory import ConfiguredArtifactReaderFactory
    from nanobot.session.artifact_operator import CopiedArtifactOperator

    workspace = loaded.workspace_path.expanduser().resolve(strict=False)
    context_manager = ContextRepoManager.from_config(
        context_paths=loaded.agents.defaults.context_paths,
        context_repos=loaded.agents.defaults.context_repos,
    )
    policy = ResourceAccessPolicy(
        workspace=workspace,
        context_manager=context_manager,
        restrict_to_workspace=loaded.tools.restrict_to_workspace,
    )
    allowed_dir = workspace if loaded.tools.restrict_to_workspace else None
    read_file = ReadFileTool(
        workspace=workspace,
        allowed_dir=allowed_dir,
        extra_read_allowed_dirs=policy.extra_read_dirs(),
        resource_policy=policy,
        restrict_to_workspace=loaded.tools.restrict_to_workspace,
    )
    factory = None
    if loaded.artifacts.enabled:
        configured_root = loaded.artifacts.root
        if configured_root is None:
            if loaded.runtime_data_dir is None:
                raise ValueError
            configured_root = str(loaded.runtime_data_dir / "context-artifacts")
        factory = ConfiguredArtifactReaderFactory(
            root=configured_root,
            max_blob_bytes=loaded.artifacts.max_blob_bytes,
            max_namespace_bytes=loaded.artifacts.max_namespace_bytes,
            static_roots=(
                workspace,
                live_sessions_root,
                live_sessions_root.parent,
                *context_manager.read_roots(),
                *context_manager.write_roots(),
                *(path for path in (target.path for target in context_manager.target_repos())
                  if path is not None),
            ),
            media_root=(
                loaded.runtime_data_dir / "media"
                if loaded.runtime_data_dir is not None
                else get_media_dir()
            ),
        )
    return CopiedArtifactOperator(
        copied_workspace,
        copied_sessions_root=copied_sessions_root,
        copied_artifact_root=copied_artifact_root,
        artifact_factory=factory,
        read_file=read_file,
        live_workspace=workspace,
        live_sessions_root=live_sessions_root,
    )


@sessions_app.command("bootstrap")
def sessions_bootstrap(
    copied_workspace: Path = typer.Option(..., "--copied-workspace"),
    copied_sessions_root: Path = typer.Option(..., "--copied-sessions-root"),
    offline_copy: bool = typer.Option(False, "--offline-copy"),
    config: str | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Adopt valid sessions in an explicitly stopped copied namespace."""
    _require_offline_copy(offline_copy)
    loaded, live_root = _copied_state_config(config)
    count = _run_copied_operation(
        _with_offline_manager(
            loaded,
            live_root,
            copied_workspace,
            copied_sessions_root,
            lambda manager: manager.bootstrap_existing_sessions(offline_copy=True),
        )
    )
    console.print(f"Bootstrap complete: {count} session(s) adopted.")


@sessions_app.command("convert")
def sessions_convert(
    key: str = typer.Argument(...),
    copied_workspace: Path = typer.Option(..., "--copied-workspace"),
    copied_sessions_root: Path = typer.Option(..., "--copied-sessions-root"),
    offline_copy: bool = typer.Option(False, "--offline-copy"),
    config: str | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Convert one bootstrapped legacy session in copied state."""
    _require_offline_copy(offline_copy)
    loaded, live_root = _copied_state_config(config)
    _run_copied_operation(
        _with_offline_manager(
            loaded,
            live_root,
            copied_workspace,
            copied_sessions_root,
            lambda manager: manager.convert_legacy_session(key, offline_copy=True),
        )
    )
    console.print("Conversion complete.")


@sessions_app.command("recover")
def sessions_recover(
    key: str = typer.Argument(...),
    copied_workspace: Path = typer.Option(..., "--copied-workspace"),
    copied_sessions_root: Path = typer.Option(..., "--copied-sessions-root"),
    offline_copy: bool = typer.Option(False, "--offline-copy"),
    config: str | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Recover one exact interrupted copied-state transition."""
    _require_offline_copy(offline_copy)
    loaded, live_root = _copied_state_config(config)
    _run_copied_operation(
        _with_offline_manager(
            loaded,
            live_root,
            copied_workspace,
            copied_sessions_root,
            lambda manager: manager.recover_session_mode(key),
        )
    )
    console.print("Recovery complete.")


@sessions_app.command("export")
def sessions_export(
    key: str = typer.Argument(...),
    copied_workspace: Path = typer.Option(..., "--copied-workspace"),
    copied_sessions_root: Path = typer.Option(..., "--copied-sessions-root"),
    destination: Path = typer.Option(..., "--destination"),
    include_compatibility: bool = typer.Option(False, "--include-compatibility"),
    offline_copy: bool = typer.Option(False, "--offline-copy"),
    config: str | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Export one canonical session from copied state to a new absolute destination."""
    from nanobot.session.session_export import export_canonical_session

    _require_offline_copy(offline_copy)
    loaded, live_root = _copied_state_config(config)
    _run_copied_operation(
        _with_offline_manager(
            loaded,
            live_root,
            copied_workspace,
            copied_sessions_root,
            lambda manager: export_canonical_session(
                manager,
                key,
                destination,
                include_compatibility=include_compatibility,
            ),
        )
    )
    console.print("Export complete.")


@session_artifacts_app.command("inspect")
def sessions_artifacts_inspect(
    key: str = typer.Argument(...),
    copied_workspace: Path = typer.Option(..., "--copied-workspace"),
    copied_sessions_root: Path = typer.Option(..., "--copied-sessions-root"),
    copied_artifact_root: Path = typer.Option(..., "--copied-artifact-root"),
    offline_copy: bool = typer.Option(False, "--offline-copy"),
    config: str | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Report aggregate artifact reachability for one copied canonical session."""
    _require_offline_copy(offline_copy)
    loaded, live_root = _copied_state_config(config)
    try:
        operator = _configured_artifact_operator(
            loaded,
            live_root,
            copied_workspace,
            copied_sessions_root,
            copied_artifact_root,
        )
    except (OSError, RuntimeError, ValueError):
        console.print(f"[red]Error: {_OFFLINE_ERROR}[/red]")
        raise typer.Exit(1) from None
    report = _run_copied_operation(operator.inspect(key))
    console.print(
        f"status={report.status} reachable_blobs={report.reachable_blobs} "
        f"reachable_bytes={report.reachable_bytes} unreachable_blobs={report.unreachable_blobs} "
        f"unreachable_bytes={report.unreachable_bytes} missing_blobs={report.missing_blobs} "
        f"missing_bytes={report.missing_bytes}"
    )


@session_artifacts_app.command("rematerialize")
def sessions_artifacts_rematerialize(
    key: str = typer.Argument(...),
    incarnation: str = typer.Option(..., "--incarnation"),
    generation: str = typer.Option(..., "--generation"),
    reference: str = typer.Option(..., "--reference"),
    copied_workspace: Path = typer.Option(..., "--copied-workspace"),
    copied_sessions_root: Path = typer.Option(..., "--copied-sessions-root"),
    copied_artifact_root: Path = typer.Option(..., "--copied-artifact-root"),
    offline_copy: bool = typer.Option(False, "--offline-copy"),
    config: str | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Rematerialize one exact source-bound artifact into copied storage."""
    _require_offline_copy(offline_copy)
    parsed_generation = _parse_copied_generation(generation)
    loaded, live_root = _copied_state_config(config)
    try:
        operator = _configured_artifact_operator(
            loaded,
            live_root,
            copied_workspace,
            copied_sessions_root,
            copied_artifact_root,
        )
    except (OSError, RuntimeError, ValueError):
        console.print(f"[red]Error: {_OFFLINE_ERROR}[/red]")
        raise typer.Exit(1) from None
    result = _run_copied_operation(
        operator.rematerialize(
            key,
            incarnation=incarnation,
            generation=parsed_generation,
            reference=reference,
        )
    )
    console.print(f"status={result.status}")


@sessions_app.command("restore-workspace")
def sessions_restore_workspace(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
) -> None:
    """Copy sessions back into the workspace before downgrading nanobot."""
    from nanobot.config.loader import get_config_path, load_config
    from nanobot.config.paths import get_default_sessions_root
    from nanobot.session.manager import SessionManager
    from nanobot.session.mode_storage import namespace_claimed, workspace_id_if_present
    from nanobot.session.session_mode import SessionModeError

    # Avoid rendering a config/state path before the fixed managed-namespace guidance.
    # Unmanaged calls retain the existing loader and output below.
    quiet_path = Path(config).expanduser().resolve(strict=False) if config else get_config_path()
    try:
        quiet_config = load_config(quiet_path)
    except (OSError, RuntimeError, UnicodeError, ValueError):
        quiet_config = None
    if quiet_config is not None:
        try:
            if workspace:
                quiet_config.agents.defaults.workspace = workspace
            quiet_workspace = quiet_config.workspace_path.expanduser().resolve(strict=False)
            quiet_root = (
                quiet_config.runtime_data_dir / "sessions"
                if quiet_config.runtime_data_dir is not None
                else get_default_sessions_root()
            ).expanduser().resolve(strict=False)
            managed = namespace_claimed(
                quiet_workspace,
                quiet_root,
                workspace_id_if_present(quiet_workspace),
            )
        except (OSError, RuntimeError, UnicodeError, ValueError):
            console.print(
                "[red]Managed session restore is unavailable. Use sessions export against a "
                "verified offline copy.[/red]"
            )
            raise typer.Exit(1) from None
        if managed:
            console.print(
                "[red]Managed session restore is unavailable. Use sessions export against a "
                "verified offline copy.[/red]"
            )
            raise typer.Exit(1)

    runtime_config = _load_runtime_config(config, workspace)
    data_dir = runtime_config.runtime_data_dir
    try:
        manager = SessionManager(
            runtime_config.workspace_path,
            sessions_root=data_dir / "sessions" if data_dir is not None else None,
        )
        result = manager.restore_sessions_to_workspace()
    except RuntimeError as exc:
        if not isinstance(exc, SessionModeError) and str(exc) != "managed sessions require verified export":
            raise
        console.print(
            "[red]Managed session restore is unavailable. Use sessions export against a "
            "verified offline copy.[/red]"
        )
        raise typer.Exit(1) from None
    console.print(
        f"Restored {result.restored} session file(s) to "
        f"{escape(str(runtime_config.workspace_path / 'sessions'))}; "
        f"{result.unchanged} already matched."
    )
    if result.conflicts:
        console.print(
            "[red]Rollback is incomplete: existing or invalid files require manual review.[/red]"
        )
        for path in result.conflicts:
            console.print(Text(f"- {path}", style="red"))
        raise typer.Exit(1)


# ============================================================================
# Channel Commands
# ============================================================================


channels_app = typer.Typer(help="Manage channels")
app.add_typer(channels_app, name="channels")


@channels_app.command("status")
def channels_status(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Show channel status."""
    from nanobot.channels.registry import discover_all

    _, loaded = _load_inspection_config(config=config)

    table = Table(title="Channel Status")
    table.add_column("Channel", style="cyan")
    table.add_column("Enabled")

    for name, cls in sorted(discover_all().items()):
        section = getattr(loaded.channels, name, None)
        if section is None:
            enabled = False
        elif isinstance(section, dict):
            enabled = cast(dict[str, Any], section).get("enabled", False)
        else:
            enabled = getattr(section, "enabled", False)
        table.add_row(
            cls.display_name,
            "[green]\u2713[/green]" if enabled else "[dim]\u2717[/dim]",
        )

    console.print(table)


@channels_app.command("login")
def channels_login(
    channel_name: str = typer.Argument(..., help="Channel name (e.g. weixin, whatsapp)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-authentication even if already logged in"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Authenticate with a channel via QR code or other interactive login."""
    from nanobot.bus.queue import MessageBus
    from nanobot.channels.registry import discover_all

    _, loaded = _load_inspection_config(config=config)
    channel_cfg: Any = getattr(loaded.channels, channel_name, None) or {}

    # Validate channel exists
    all_channels = discover_all()
    if channel_name not in all_channels:
        available = ", ".join(all_channels.keys())
        console.print(f"[red]Unknown channel: {channel_name}[/red]  Available: {available}")
        raise typer.Exit(1)

    console.print(f"{__logo__} {all_channels[channel_name].display_name} Login\n")

    channel_factory = all_channels[channel_name]
    channel = channel_factory(channel_cfg, bus=MessageBus())

    success = asyncio.run(channel.login(force=force))

    if not success:
        raise typer.Exit(1)


# ============================================================================
# Plugin Commands
# ============================================================================

plugins_app = typer.Typer(help="Manage optional nanobot features")
app.add_typer(plugins_app, name="plugins")


@plugins_app.command("list")
def plugins_list(
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """List optional nanobot features."""
    from nanobot.channels.registry import discover_plugins
    from nanobot.config.loader import load_config, set_config_path

    resolved_config_path = Path(config_path).expanduser().resolve() if config_path else None
    if resolved_config_path is not None:
        set_config_path(resolved_config_path)

    _print_enable_options(
        feature_support.optional_dependency_groups(),
        discover_plugins(),
        load_config(resolved_config_path),
    )


@plugins_app.command("enable")
def plugins_enable(
    name: str = typer.Argument(..., help="Feature name (e.g. weixin, matrix, bedrock)"),
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    logs: bool = typer.Option(False, "--logs/--no-logs", help="Show optional package install logs"),
):
    """Enable a nanobot feature."""
    from nanobot.config.loader import get_config_path, set_config_path

    resolved_config_path = Path(config_path).expanduser().resolve() if config_path else None
    if resolved_config_path is not None:
        set_config_path(resolved_config_path)
    resolved_config_path = resolved_config_path or get_config_path()
    _set_nanobot_logs(logs)

    try:
        payload = feature_support.enable_optional_feature(
            name,
            config_path=resolved_config_path,
            runner=feature_support.run_install_command,
        )
    except feature_support.OptionalFeatureError as exc:
        console.print(f"[red]{escape(exc.message)}[/red]")
        raise typer.Exit(1) from exc

    message = payload.get("last_action", {}).get("message") or f"Enabled feature '{name}'"
    console.print(f"[green]{escape(message)}[/green]")


@plugins_app.command("disable")
def plugins_disable(
    name: str = typer.Argument(..., help="Channel name (e.g. telegram, matrix, slack)"),
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Disable a nanobot channel feature."""
    from nanobot.config.loader import get_config_path, set_config_path

    resolved_config_path = Path(config_path).expanduser().resolve() if config_path else None
    if resolved_config_path is not None:
        set_config_path(resolved_config_path)
    resolved_config_path = resolved_config_path or get_config_path()

    try:
        payload = feature_support.disable_optional_feature(name, config_path=resolved_config_path)
    except feature_support.OptionalFeatureError as exc:
        console.print(f"[red]{escape(exc.message)}[/red]")
        raise typer.Exit(1) from exc

    message = payload.get("last_action", {}).get("message") or f"Disabled channel '{name}'"
    console.print(f"[green]{escape(message)}[/green] in {resolved_config_path}")


# ============================================================================
# Status Commands
# ============================================================================


@app.command()
def status(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
):
    """Show nanobot status."""
    config_path, loaded = _load_inspection_config(config=config, workspace=workspace)
    workspace_path = loaded.workspace_path

    console.print(f"{__logo__} nanobot Status\n")

    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(
        f"Workspace: {workspace_path} "
        f"{'[green]✓[/green]' if workspace_path.exists() else '[red]✗[/red]'}"
    )

    if config_path.exists():
        from nanobot.config.errors import ConfigLoadError
        from nanobot.config.loader import resolve_config_env_vars, resolve_env_refs
        from nanobot.providers.registry import PROVIDERS

        _model, _preset_tag = _model_display(loaded)
        console.print(f"Model: {_model}{_preset_tag}")

        provider_ready = False
        try:
            resolved = resolve_config_env_vars(
                loaded.model_copy(deep=True),
                config_path=config_path,
            )
        except ConfigLoadError as exc:
            console.print("Agent: [red]✗ configuration is not ready[/red]")
            _print_config_error(exc)
        else:
            provider_error = _provider_setup_error(resolved)
            if provider_error:
                console.print(Text(f"Agent: ✗ {provider_error}", style="red"))
                console.print("Complete provider/model setup:")
                _print_model_setup_steps(config_path)
            else:
                provider_ready = True
                console.print("Agent: [green]✓ provider/model configuration is ready[/green]")

        # Check API keys from registry
        for spec in PROVIDERS:
            p = getattr(loaded.providers, spec.name, None)
            if p is None:
                continue
            if spec.is_oauth:
                console.print(f"{spec.label}: [green]✓ (OAuth)[/green]")
            elif spec.is_local:
                # Local deployments show api_base instead of api_key
                if resolve_env_refs(p.api_base or ""):
                    console.print(f"{spec.label}: [green]✓ {p.api_base}[/green]")
                else:
                    console.print(f"{spec.label}: [dim]not set[/dim]")
            else:
                has_key = bool(resolve_env_refs(p.api_key or ""))
                console.print(f"{spec.label}: {'[green]✓[/green]' if has_key else '[dim]not set[/dim]'}")

        if provider_ready:
            console.print()
            console.print('Next: [cyan]nanobot agent -m "Hello!"[/cyan]')
            console.print(
                "[dim]Status does not call the model or verify network access and credentials.[/dim]"
            )
    else:
        console.print("Agent: [red]✗ configuration file not found[/red]")
        console.print("Create the provider/model configuration:")
        _print_model_setup_steps(config_path)


# ============================================================================
# OAuth Login
# ============================================================================

app.add_typer(provider_app, name="provider")


if __name__ == "__main__":
    app()
