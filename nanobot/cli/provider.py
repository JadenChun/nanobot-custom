"""Typer commands for OAuth provider authentication."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import typer
from rich.console import Console

from nanobot import __logo__
from nanobot.providers.oauth_guidance import OAUTH_CLI_KIT_MISSING_MESSAGE

if TYPE_CHECKING:
    from nanobot.providers.registry import ProviderSpec


console = Console()
provider_app = typer.Typer(help="Manage providers")

_PROVIDER_DISPLAY: dict[str, str] = {
    "openai_codex": "OpenAI Codex",
    "xai_grok": "xAI Grok",
    "github_copilot": "GitHub Copilot",
}

_OAUTH_PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "openai_codex": "openai-codex/gpt-5.6-sol",
    "xai_grok": "xai-grok/grok-4.6",
    "github_copilot": "github-copilot/gpt-5.4-mini",
}


class _OAuthToken(Protocol):
    access: str | None
    refresh: str
    expires: int
    account_id: str | None


class _LoginOAuthInteractive(Protocol):
    def __call__(
        self,
        *,
        print_fn: Callable[[str], None],
        prompt_fn: Callable[[str], str],
        proxy: str | None = None,
        storage: object | None = None,
    ) -> _OAuthToken | None: ...


class AcquisitionOnlyStorage:
    """Satisfy oauth-cli-kit without granting it persistent storage authority."""

    def save(self, token: object) -> None:
        del token

    def load(self) -> None:
        return None

    def get_token_path(self) -> Path:
        raise RuntimeError("acquisition-only storage has no path")


def _required_module_attribute(module_name: str, attribute: str) -> object:
    """Load an optional dependency attribute with import-compatible errors."""
    module = import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ImportError(f"{module_name}.{attribute} is unavailable") from exc


def _load_openai_oauth_client() -> _LoginOAuthInteractive:
    """Load the untyped OAuth client behind a typed boundary."""
    return cast(
        _LoginOAuthInteractive,
        _required_module_attribute("oauth_cli_kit", "login_oauth_interactive"),
    )


def _resolve_oauth_provider(provider: str) -> ProviderSpec:
    """Resolve and validate an OAuth provider configuration."""
    from nanobot.providers.registry import PROVIDERS

    key = provider.replace("-", "_")
    spec = next((s for s in PROVIDERS if s.name == key and s.is_oauth), None)
    if not spec:
        names = ", ".join(s.name.replace("_", "-") for s in PROVIDERS if s.is_oauth)
        console.print(f"[red]Unknown OAuth provider: {provider}[/red]  Supported: {names}")
        raise typer.Exit(1)
    return spec


def _set_oauth_provider_as_main(
    provider_name: str,
    *,
    model: str | None = None,
    config_path: str | None = None,
) -> None:
    """Persist an OAuth provider as the active agent provider."""
    from nanobot.config.loader import get_config_path, load_config, save_config, set_config_path

    resolved_config_path = Path(config_path).expanduser().resolve() if config_path else None
    if resolved_config_path is not None and get_config_path() != resolved_config_path:
        set_config_path(resolved_config_path)
        console.print(f"[dim]Using config: {resolved_config_path}[/dim]")

    config = load_config(resolved_config_path)
    selected_model = (model or "").strip() or _OAUTH_PROVIDER_DEFAULT_MODELS[provider_name]
    config.agents.defaults.model_preset = None
    config.agents.defaults.provider = provider_name
    config.agents.defaults.model = selected_model
    if provider_name == "xai_grok" and selected_model in {
        "xai-grok/grok-4.5",
        "xai-grok/grok-4.6",
    }:
        config.agents.defaults.context_window_tokens = 500_000
    save_config(config, resolved_config_path)

    saved_path = resolved_config_path or get_config_path()
    console.print(
        f"[green]✓ Set {provider_name.replace('_', '-')} as the main provider[/green]  "
        f"[dim]{selected_model}[/dim]"
    )
    console.print(f"[dim]Saved: {saved_path}[/dim]")


@provider_app.command("login")
def provider_login(
    provider: str = typer.Argument(
        ...,
        help="OAuth provider (e.g. 'openai-codex', 'xai-grok', 'github-copilot')",
    ),
    set_main: bool = typer.Option(
        False,
        "--set-main",
        "--main",
        help="Set this OAuth provider as the active agent provider after login",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use when setting this provider as the active provider",
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Authenticate with an OAuth provider."""
    spec = _resolve_oauth_provider(provider)

    handler = _LOGIN_HANDLERS.get(spec.name)
    if not handler:
        console.print(f"[red]Login not implemented for {spec.label}[/red]")
        raise typer.Exit(1)

    if config:
        from nanobot.config.loader import set_config_path

        resolved_config_path = Path(config).expanduser().resolve()
        set_config_path(resolved_config_path)
        console.print(f"[dim]Using config: {resolved_config_path}[/dim]")

    console.print(f"{__logo__} OAuth Login - {spec.label}\n")
    handler()
    if set_main or model:
        _set_oauth_provider_as_main(spec.name, model=model, config_path=config)


@provider_app.command("logout")
def provider_logout(
    provider: str = typer.Argument(
        ...,
        help="OAuth provider (e.g. 'openai-codex', 'xai-grok', 'github-copilot')",
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Log out from an OAuth provider."""
    spec = _resolve_oauth_provider(provider)

    handler = _LOGOUT_HANDLERS.get(spec.name)
    if not handler:
        console.print(f"[red]Logout not implemented for {spec.label}[/red]")
        raise typer.Exit(1)

    if config:
        from nanobot.config.loader import set_config_path

        resolved_config_path = Path(config).expanduser().resolve()
        set_config_path(resolved_config_path)
        console.print(f"[dim]Using config: {resolved_config_path}[/dim]")

    console.print(f"{__logo__} OAuth Logout - {spec.label}\n")
    handler()


@provider_app.command("reimport")
def provider_reimport(
    provider: str = typer.Argument(
        ...,
        help="Credential source to reimport (openai-codex or github-copilot)",
    ),
) -> None:
    """Explicitly replace the canonical credential from a read-only legacy source."""
    from nanobot.providers.oauth_model_catalog import invalidate_oauth_model_catalog

    spec = _resolve_oauth_provider(provider)
    if spec.name == "openai_codex":
        from nanobot.providers.codex_auth import reimport_codex

        token = reimport_codex()
    elif spec.name == "github_copilot":
        from nanobot.providers.copilot_auth import reimport_copilot

        token = reimport_copilot()
    else:
        console.print(f"[red]Reimport is not supported for {spec.label}[/red]")
        raise typer.Exit(1)
    if token is None:
        console.print("[yellow]! No importable credential was found[/yellow]")
        raise typer.Exit(1)
    invalidate_oauth_model_catalog(spec.name)
    console.print(f"[green]✓ Reimported {spec.label} credential[/green]")


def _login_openai_codex() -> None:
    try:
        from nanobot.config.loader import load_config, resolve_config_env_vars
        from nanobot.providers.codex_auth import get_codex_authority
        from nanobot.providers.oauth_credentials import OAuthToken as CanonicalToken

        login_oauth_interactive = _load_openai_oauth_client()
        proxy = None
        try:
            proxy = resolve_config_env_vars(load_config()).providers.openai_codex.proxy or None
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e
        authority = get_codex_authority()
        generation = authority.begin_login()
        console.print("[cyan]Starting interactive OAuth login...[/cyan]\n")
        capture = AcquisitionOnlyStorage()
        token = login_oauth_interactive(
            print_fn=lambda s: console.print(s),
            prompt_fn=lambda s: typer.prompt(s),
            proxy=proxy,
            storage=capture,
        )
        if not (token and token.access):
            console.print("[red]✗ Authentication failed[/red]")
            raise typer.Exit(1)
        authority.commit_login(
            CanonicalToken(token.access, token.refresh, token.expires, token.account_id),
            generation,
        )
        from nanobot.providers.oauth_model_catalog import invalidate_oauth_model_catalog

        invalidate_oauth_model_catalog("openai_codex")
        console.print("[green]✓ Authenticated with OpenAI Codex[/green]")
    except ImportError:
        console.print(f"[red]{OAUTH_CLI_KIT_MISSING_MESSAGE}[/red]")
        raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Authentication error ({type(exc).__name__})[/red]")
        raise typer.Exit(1) from exc


def _logout_openai_codex() -> None:
    """Clear local OAuth credentials for OpenAI Codex."""
    from nanobot.providers.codex_auth import logout_codex
    from nanobot.providers.oauth_model_catalog import invalidate_oauth_model_catalog

    logout_codex()
    invalidate_oauth_model_catalog("openai_codex")
    console.print("[green]✓ Logged out from OpenAI Codex[/green]")


def _login_xai_grok() -> None:
    """Authenticate with xAI using the Grok subscription OAuth contract."""
    from nanobot.config.loader import load_config, resolve_config_env_vars
    from nanobot.providers.xai_oauth import get_xai_oauth_token, login_xai_oauth

    try:
        proxy = resolve_config_env_vars(load_config()).providers.xai_grok.proxy or None
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    token = None
    with suppress(Exception):
        token = get_xai_oauth_token(proxy=proxy)
    if not (token and token.access):
        console.print(
            "[cyan]Starting xAI browser sign-in for your X Premium / Grok subscription...[/cyan]\n"
        )
        try:
            token = login_xai_oauth(
                print_fn=lambda message: console.print(message),
                prompt_fn=lambda prompt: typer.prompt(prompt),
                proxy=proxy,
            )
        except Exception as exc:
            console.print(f"[red]Authentication error: {exc}[/red]")
            raise typer.Exit(1) from exc
    account = token.account_id or "xAI account"
    console.print(f"[green]✓ Authenticated with xAI[/green]  [dim]{account}[/dim]")
    console.print(
        "[dim]Hosted X Search is enabled automatically when the selected model supports it.[/dim]"
    )


def _logout_xai_grok() -> None:
    """Clear local xAI OAuth credentials for this nanobot instance."""
    from nanobot.providers.xai_oauth import get_xai_oauth_storage_path, logout_xai_oauth

    token_path = get_xai_oauth_storage_path()
    provider_label = _PROVIDER_DISPLAY["xai_grok"]
    if logout_xai_oauth():
        console.print(f"[green]✓ Logged out from {provider_label}[/green]")
        console.print(f"[dim]Removed: {token_path}[/dim]")
    else:
        console.print(f"[yellow]! No local OAuth credentials found for {provider_label}[/yellow]")


def _logout_github_copilot() -> None:
    """Clear local OAuth credentials for GitHub Copilot."""
    from nanobot.providers.copilot_auth import logout_copilot
    from nanobot.providers.oauth_model_catalog import invalidate_oauth_model_catalog

    logout_copilot()
    invalidate_oauth_model_catalog("github_copilot")
    console.print("[green]✓ Logged out from GitHub Copilot[/green]")


def _login_github_copilot() -> None:
    try:
        from nanobot.providers.github_copilot_provider import login_github_copilot

        console.print("[cyan]Starting GitHub Copilot device flow...[/cyan]\n")
        login_github_copilot(
            print_fn=lambda s: console.print(s),
            prompt_fn=lambda s: typer.prompt(s),
        )
        from nanobot.providers.oauth_model_catalog import invalidate_oauth_model_catalog

        invalidate_oauth_model_catalog("github_copilot")
        console.print("[green]✓ Authenticated with GitHub Copilot[/green]")
    except Exception as e:
        console.print(f"[red]Authentication error ({type(e).__name__})[/red]")
        raise typer.Exit(1)


_LOGIN_HANDLERS: dict[str, Callable[[], None]] = {
    "openai_codex": _login_openai_codex,
    "xai_grok": _login_xai_grok,
    "github_copilot": _login_github_copilot,
}
_LOGOUT_HANDLERS: dict[str, Callable[[], None]] = {
    "openai_codex": _logout_openai_codex,
    "xai_grok": _logout_xai_grok,
    "github_copilot": _logout_github_copilot,
}
