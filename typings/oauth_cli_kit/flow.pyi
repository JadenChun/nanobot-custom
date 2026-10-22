from collections.abc import Callable
from typing import Protocol

from .models import OAuthProviderConfig, OAuthToken
from .storage import TokenStorage

class _InteractiveLoginSink(Protocol):
    """The installed interactive path consumes save, not load or file locking."""

    def save(self, token: OAuthToken) -> None: ...

def get_token(
    provider: OAuthProviderConfig = ...,
    storage: TokenStorage | None = None,
    min_ttl_seconds: int = 60,
    proxy: str | None = None,
) -> OAuthToken: ...

def login_oauth_interactive(
    print_fn: Callable[[str], None],
    prompt_fn: Callable[[str], str],
    provider: OAuthProviderConfig = ...,
    originator: str | None = None,
    storage: _InteractiveLoginSink | None = None,
    proxy: str | None = None,
    open_browser: bool | None = None,
) -> OAuthToken: ...

login_codex_oauth_interactive = login_oauth_interactive
get_codex_token = get_token
