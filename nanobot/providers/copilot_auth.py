"""GitHub Copilot OAuth (device flow) and bearer-token management.

Two-stage credential model used by GitHub Copilot (and matched by every
open-source Copilot integration — copilot.vim, aider, opencode, etc.):

1. **GitHub OAuth token** (long-lived; obtained via GitHub device-code flow
   against the public Copilot client ID). Stored in
   ``~/.nanobot/auth/github-copilot.json``; older locations are imported.
2. **Copilot internal bearer** (short-lived, ~30 minutes; minted by exchanging
   the GitHub token at ``https://api.github.com/copilot_internal/v2/token``).
   Cached in memory and on disk; refreshed automatically a few minutes before
   expiry.

Public surface:

* :func:`device_login` — run the interactive device-code flow and persist the
  GitHub token.
* :func:`get_copilot_bearer` — return a currently-valid Copilot bearer,
  refreshing on demand. Used by the OpenAI-compatible provider as its
  ``token_provider`` callback.
* :func:`copilot_request_headers` — extra HTTP headers Copilot requires.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Callable, cast

import httpx

from nanobot.providers.oauth_credentials import (
    CredentialError,
    CredentialStatus,
    DerivedBearer,
    OAuthCredentialCoordinator,
    OAuthToken,
)

# Public GitHub Copilot OAuth client ID. Identical value is published in
# copilot.vim, opencode, aider, neovim CopilotChat, etc. Not a secret.
_COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"
_DEVICE_CODE_URL = "https://github.com/login/device/code"
_OAUTH_TOKEN_URL = "https://github.com/login/oauth/access_token"
_COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
_DEVICE_SCOPE = "read:user"

# Identify ourselves as a VS Code-style Copilot client. The Copilot API
# rejects requests that don't look like a known editor integration.
_EDITOR_VERSION = "vscode/1.95.0"
_EDITOR_PLUGIN_VERSION = "copilot-chat/0.22.0"
_COPILOT_INTEGRATION_ID = "vscode-chat"
_USER_AGENT = "GitHubCopilotChat/0.22.0"

# Refresh the short-lived Copilot bearer a few minutes before it actually
# expires so in-flight requests don't race the expiry boundary.
_REFRESH_EARLY_SECONDS = 300

def _auth_path() -> Path:
    override = os.environ.get("NANOBOT_COPILOT_AUTH_PATH")
    if override:
        return Path(os.path.abspath(os.fspath(Path(override).expanduser())))
    home = Path(os.environ.get("HOME", str(Path.home())))
    return Path(os.path.abspath(os.fspath(home / ".nanobot" / "auth" / "github-copilot.json")))


def _legacy_auth_path() -> Path:
    override = os.environ.get("OAUTH_CLI_KIT_TOKEN_PATH")
    if override:
        return Path(override).expanduser()
    home = Path(os.environ.get("HOME", str(Path.home())))
    return home / ".local" / "share" / "nanobot" / "auth" / "github-copilot.json"


def _load_auth() -> dict[str, Any]:
    paths = [_legacy_auth_path()]
    if not os.environ.get("OAUTH_CLI_KIT_TOKEN_PATH"):
        home = Path(os.environ.get("HOME", str(Path.home())))
        paths.append(home / ".nanobot" / "copilot_auth.json")
    for path in paths:
        try:
            raw: object = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                continue
            # JSON object keys are strings; token contents are checked below.
            data = cast(dict[str, Any], raw)
            if isinstance(data.get("github_token") or data.get("access"), str):
                return data
        except (OSError, UnicodeError, ValueError):
            continue
    return {}


def _import_legacy_token() -> str | None:
    data = _load_auth()
    token = data.get("github_token") or data.get("access")
    return token if isinstance(token, str) and token else None


def get_copilot_authority() -> OAuthCredentialCoordinator:
    path = _auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.parent.resolve(strict=True) / path.name
    try:
        info = path.lstat()
    except FileNotFoundError:
        info = None
    if info is not None:
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise CredentialError("invalid credential store")
    if os.name != "nt":
        path.parent.chmod(0o700)
    return OAuthCredentialCoordinator(path, "copilot", importer=_import_legacy_token)


def get_copilot_login_status() -> CredentialStatus:
    return get_copilot_authority().peek_status()


def get_copilot_identity() -> OAuthToken | None:
    return get_copilot_authority().get_cached_or_import()


def logout_copilot() -> None:
    get_copilot_authority().logout()
    from nanobot.providers.oauth_model_catalog import invalidate_oauth_model_catalog

    invalidate_oauth_model_catalog("github_copilot")


def reimport_copilot() -> OAuthToken | None:
    authority = get_copilot_authority()
    generation = authority.begin_login()
    token = authority.explicit_reimport(generation)
    if token is not None:
        from nanobot.providers.oauth_model_catalog import invalidate_oauth_model_catalog

        invalidate_oauth_model_catalog("github_copilot")
    return token


# ---------------------------------------------------------------------------
# Device-code flow
# ---------------------------------------------------------------------------


def device_login(
    *,
    on_user_code: Callable[[str, str], None] | None = None,
    poll_timeout: float = 600.0,
) -> dict[str, Any]:
    """Run the GitHub device-code flow for the Copilot client and persist.

    Args:
        on_user_code: Optional callback invoked with ``(user_code, verification_uri)``
            so the caller can display the prompt. If omitted, prints to stdout.
        poll_timeout: Max seconds to wait for the user to approve.

    Returns:
        The full saved auth dict.
    """
    authority = get_copilot_authority()
    generation = authority.begin_login()
    try:
        client = httpx.Client(timeout=30.0)
        with client:
            resp = client.post(
                _DEVICE_CODE_URL,
                data={"client_id": _COPILOT_CLIENT_ID, "scope": _DEVICE_SCOPE},
                headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
            )
            if resp.status_code != 200:
                raise RuntimeError("GitHub device authorization request was rejected.")
            try:
                device = resp.json()
            except Exception:
                raise RuntimeError("GitHub device authorization returned invalid data.") from None
            if not isinstance(device, dict):
                raise RuntimeError("GitHub device authorization returned invalid data.")
            device_code = cast(dict[object, object], device).get("device_code")
            user_code = cast(dict[object, object], device).get("user_code")
            verification_uri = cast(dict[object, object], device).get("verification_uri") or "https://github.com/login/device"
            try:
                # Dynamic JSON stays at this edge; int retains its exact coercion/error contract.
                interval = max(1, int(cast(dict[object, Any], device).get("interval") or 5))
            except (TypeError, ValueError):
                raise RuntimeError("GitHub device authorization returned invalid data.") from None
            if not isinstance(device_code, str) or not device_code or not isinstance(user_code, str) or not user_code:
                raise RuntimeError("GitHub device authorization returned invalid data.")

            if on_user_code is not None:
                on_user_code(user_code, str(verification_uri))
            else:
                print(f"\nGo to {verification_uri} and enter code: {user_code}\n")

            deadline = time.time() + poll_timeout
            access_token: str | None = None
            while time.time() < deadline:
                time.sleep(interval)
                poll = client.post(
                    _OAUTH_TOKEN_URL,
                    data={
                        "client_id": _COPILOT_CLIENT_ID,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
                )
                if poll.status_code != 200:
                    raise RuntimeError("GitHub device authorization polling failed.")
                try:
                    data = poll.json()
                except Exception:
                    raise RuntimeError(
                        "GitHub device authorization returned invalid data."
                    ) from None
                if not isinstance(data, dict):
                    raise RuntimeError("GitHub device authorization returned invalid data.")
                err = cast(dict[object, object], data).get("error")
                if err == "authorization_pending":
                    continue
                if err == "slow_down":
                    interval += 5
                    continue
                if err in {"expired_token", "access_denied"}:
                    raise RuntimeError("GitHub device authorization did not complete.")
                if err:
                    raise RuntimeError("GitHub device authorization failed.")
                candidate = cast(dict[object, object], data).get("access_token")
                access_token = candidate if isinstance(candidate, str) and candidate else None
                if access_token:
                    break

        if not access_token:
            raise RuntimeError("GitHub device authorization timed out.")
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError("GitHub device authorization network failure.") from None

    authority.commit_login(access_token, generation)
    saved = {"github_token": access_token}
    # Eagerly mint a Copilot bearer so we surface entitlement errors at login
    # time (e.g. the GitHub account has no Copilot subscription).
    get_copilot_bearer()
    return saved


# ---------------------------------------------------------------------------
# Copilot bearer
# ---------------------------------------------------------------------------


def _refresh_copilot_bearer(auth: dict[str, Any]) -> dict[str, Any]:
    """Exchange the GitHub OAuth token for a fresh in-memory bearer payload."""
    github_token = auth.get("github_token")
    if not github_token:
        raise RuntimeError(
            "GitHub Copilot is not authenticated. Run `nanobot provider login github-copilot`."
        )

    token_url = os.environ.get("NANOBOT_COPILOT_TOKEN_URL", "").strip() or _COPILOT_TOKEN_URL
    resp = httpx.get(
        token_url,
        headers={
            "Authorization": f"token {github_token}",
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
            "Editor-Version": _EDITOR_VERSION,
            "Editor-Plugin-Version": _EDITOR_PLUGIN_VERSION,
        },
        timeout=30.0,
    )
    if resp.status_code == 401:
        raise RuntimeError(
            "GitHub token is no longer valid. Re-run `nanobot provider login github-copilot`."
        )
    if resp.status_code == 403:
        raise RuntimeError(
            "GitHub account is not entitled to Copilot. "
            "Ensure the account has an active Copilot subscription."
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Copilot token exchange failed with HTTP {resp.status_code}.")

    payload = resp.json()
    bearer = payload.get("token")
    expires_at = payload.get("expires_at")
    if not isinstance(expires_at, int):
        refresh_in = payload.get("refresh_in")
        expires_at = int(time.time()) + (
            int(refresh_in) if isinstance(refresh_in, (int, float)) else 1500
        )
    if not bearer:
        raise RuntimeError("Copilot token exchange returned an invalid response.")

    auth["copilot_token"] = bearer
    auth["copilot_expires_at"] = expires_at
    auth["copilot_endpoints"] = payload.get("endpoints") or {}
    return auth


def get_copilot_bearer_view(
    min_ttl_seconds: int = _REFRESH_EARLY_SECONDS,
    *,
    expected_generation: str | None = None,
) -> DerivedBearer:
    """Return a valid Copilot bearer token, refreshing if near expiry.

    Safe to call from multiple threads / coroutines — refresh is serialized.
    """
    def exchange(identity: OAuthToken) -> dict[str, object]:
        auth = _refresh_copilot_bearer({"github_token": identity.access})
        endpoint: object = None
        endpoints = auth.get("copilot_endpoints")
        if isinstance(endpoints, dict) and isinstance(cast(dict[object, object], endpoints).get("api"), str):
            endpoint = cast(dict[object, object], endpoints)["api"]
        return {
            "bearer": auth["copilot_token"],
            "expires_at_ms": int(auth["copilot_expires_at"]) * 1000,
            "endpoint": endpoint,
        }

    return get_copilot_authority().get_copilot_bearer(
        min_ttl_seconds,
        exchange,
        expected_generation=expected_generation,
    )


def get_copilot_bearer(min_ttl_seconds: int = _REFRESH_EARLY_SECONDS) -> str:
    return get_copilot_bearer_view(min_ttl_seconds).bearer


def copilot_request_headers() -> dict[str, str]:
    """HTTP headers Copilot expects on every request."""
    return {
        "Editor-Version": _EDITOR_VERSION,
        "Editor-Plugin-Version": _EDITOR_PLUGIN_VERSION,
        "Copilot-Integration-Id": _COPILOT_INTEGRATION_ID,
        "User-Agent": _USER_AGENT,
    }


def is_authenticated() -> bool:
    return get_copilot_identity() is not None
