"""Compatibility helpers for OpenAI Codex OAuth tokens."""

from __future__ import annotations

import base64
import json
import os
import stat
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import httpx
from oauth_cli_kit.models import OAuthToken
from oauth_cli_kit.providers import OPENAI_CODEX_PROVIDER

from nanobot.providers.oauth_credentials import (
    CredentialError,
    CredentialStatus,
    CredentialView,
    OAuthCredentialCoordinator,
)
from nanobot.providers.oauth_credentials import (
    OAuthToken as CanonicalToken,
)

_REFRESH_EARLY_SECONDS = 300
DEFAULT_CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
MALAYSIA_TIMEZONE = ZoneInfo("Asia/Kuala_Lumpur")


class CodexUsageError(RuntimeError):
    """Raised when the Codex OAuth usage endpoint cannot be read."""


def get_token(min_ttl_seconds: int = 60, proxy: str | None = None) -> OAuthToken:
    """Load or refresh the canonical Codex credential under its process lock."""
    authority = get_codex_authority()

    def refresh(current: CanonicalToken) -> CanonicalToken:
        token = _refresh_token(current.refresh, current.account_id, proxy=proxy)
        return _to_canonical(token)

    try:
        return _to_oauth(authority.refresh_codex(min_ttl_seconds, refresh).token)
    except Exception as exc:
        raise RuntimeError("OpenAI Codex OAuth credential is unavailable.") from exc


def get_codex_credential_view(
    min_ttl_seconds: int = 60, proxy: str | None = None
) -> CredentialView:
    authority = get_codex_authority()

    def refresh(current: CanonicalToken) -> CanonicalToken:
        return _to_canonical(_refresh_token(current.refresh, current.account_id, proxy=proxy))

    return authority.refresh_codex(min_ttl_seconds, refresh)


def get_codex_authority() -> OAuthCredentialCoordinator:
    path = _canonical_codex_path()
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
    return OAuthCredentialCoordinator(path, "codex", importer=_import_codex_token)


def get_codex_login_status() -> CredentialStatus:
    return get_codex_authority().peek_status()


def logout_codex() -> None:
    get_codex_authority().logout()
    from nanobot.providers.oauth_model_catalog import invalidate_oauth_model_catalog

    invalidate_oauth_model_catalog("openai_codex")


def reimport_codex() -> OAuthToken | None:
    authority = get_codex_authority()
    generation = authority.begin_login()
    token = authority.explicit_reimport(generation)
    if token is not None:
        from nanobot.providers.oauth_model_catalog import invalidate_oauth_model_catalog

        invalidate_oauth_model_catalog("openai_codex")
    return None if token is None else _to_oauth(token)


def _to_canonical(token: OAuthToken) -> CanonicalToken:
    return CanonicalToken(token.access, token.refresh, token.expires, token.account_id)


def _to_oauth(token: CanonicalToken) -> OAuthToken:
    return OAuthToken(token.access, token.refresh, token.expires_at_ms, token.account_id)


def _import_codex_token() -> CanonicalToken | None:
    # Both official Codex and oauth-cli-kit remain read-only migration sources.
    token = _load_official_codex_token(_official_codex_auth_path())
    if token is None:
        token = _load_nanobot_token(_legacy_oauth_token_path())
    return None if token is None else _to_canonical(token)


def get_codex_usage() -> dict[str, Any]:
    """Fetch the current ChatGPT/Codex usage windows for the OAuth account."""
    token = get_token()
    usage_url = os.environ.get("NANOBOT_CODEX_USAGE_URL", DEFAULT_CODEX_USAGE_URL)
    try:
        response = httpx.get(
            usage_url,
            headers={
                "Authorization": f"Bearer {token.access}",
                # The Codex coordinator rejects tokens without an account ID.
                "ChatGPT-Account-ID": cast(str, token.account_id),
                "originator": "codex_cli_rs",
                "Accept": "application/json",
            },
            timeout=30.0,
        )
    except httpx.HTTPError as exc:
        raise CodexUsageError("Codex quota request failed due to a network error.") from exc

    if response.status_code == 401:
        raise CodexUsageError("Codex OAuth credentials were rejected. Run the OpenAI login again.")
    if response.status_code == 429:
        raise CodexUsageError("Codex quota endpoint is rate-limited. Please try again shortly.")
    if response.status_code != 200:
        raise CodexUsageError(f"Codex quota request failed (HTTP {response.status_code}).")

    try:
        payload = response.json()
    except ValueError as exc:
        raise CodexUsageError("Codex returned an invalid quota response.") from exc
    if not isinstance(payload, dict):
        raise CodexUsageError("Codex returned an unexpected quota response.")
    # httpx's default JSON decoder produces string-keyed objects.
    return cast(dict[str, object], payload)


def format_codex_usage(payload: dict[str, Any]) -> str:
    """Render the Codex usage response without exposing token data."""
    plan = _first_value(payload, "plan_type", "planType")
    rate_limit = _first_value(payload, "rate_limit", "rateLimits")
    lines = ["Codex quota" + (f" ({plan})" if plan else "") + ":"]

    if not isinstance(rate_limit, dict):
        lines.append("No rate-limit data was returned.")
        return "\n".join(lines)

    windows = (
        ("Primary", _first_value(cast(dict[object, object], rate_limit), "primary_window", "primary")),
        ("Secondary", _first_value(cast(dict[object, object], rate_limit), "secondary_window", "secondary")),
    )
    rendered = 0
    for label, window in windows:
        if not isinstance(window, dict):
            continue
        details: list[str] = []
        reset_detail: str | None = None
        used = _first_value(cast(dict[object, object], window), "used_percent", "usedPercent")
        used_number = _as_number(used)
        if used_number is not None:
            remaining = max(0.0, 100.0 - used_number)
            details.append(f"{remaining:g}% remaining")

        reset_at = _as_timestamp(_first_value(cast(dict[object, object], window), "reset_at", "resetsAt"))
        if reset_at is not None:
            reset_time = datetime.fromtimestamp(reset_at, tz=MALAYSIA_TIMEZONE).strftime(
                "%Y-%m-%d %H:%M MYT"
            )
            reset_detail = f"Reset at {reset_time}"

        if details:
            lines.append(f"{label}: " + "; ".join(details))
            rendered += 1
        if reset_detail:
            lines.append(reset_detail)

    if rendered == 0:
        lines.append("No rate-limit windows were returned.")
    return "\n".join(lines)


def _first_value(
    payload: Mapping[str, object] | Mapping[object, object], *keys: str
) -> object:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _as_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _as_timestamp(value: object) -> float | None:
    timestamp = _as_number(value)
    if timestamp is None:
        return None
    return timestamp / 1000 if timestamp > 10_000_000_000 else timestamp


def _canonical_codex_path() -> Path:
    override = os.environ.get("NANOBOT_CODEX_AUTH_PATH")
    if override:
        return Path(os.path.abspath(os.fspath(Path(override).expanduser())))
    home = Path(os.environ.get("HOME", str(Path.home())))
    return Path(os.path.abspath(os.fspath(home / ".nanobot" / "auth" / "openai-codex.json")))


# Compatibility name for callers that only need the active Nanobot path.
_nanobot_token_path = _canonical_codex_path


def _legacy_oauth_token_path() -> Path:
    override = os.environ.get("OAUTH_CLI_KIT_TOKEN_PATH")
    if override:
        return Path(override)
    xdg_home = os.environ.get("XDG_DATA_HOME")
    if xdg_home:
        base = Path(xdg_home)
    else:
        home = Path(os.environ.get("HOME", str(Path.home())))
        base = home / ".local" / "share"
    return base / "oauth-cli-kit" / "auth" / OPENAI_CODEX_PROVIDER.token_filename


def _official_codex_auth_path() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home) / "auth.json"
    home = Path(os.environ.get("HOME", str(Path.home())))
    return home / ".codex" / "auth.json"


def _load_nanobot_token(path: Path) -> OAuthToken | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    access = _clean_str(payload.get("access"))
    refresh = _clean_str(payload.get("refresh"))
    account_id = _clean_str(payload.get("account_id")) or _account_id_from_access_token(access)
    expires = _coerce_expires_ms(payload.get("expires"), access)
    if not access or not account_id or expires is None:
        return None
    return OAuthToken(access=access, refresh=refresh or "", expires=expires, account_id=account_id)


def _load_official_codex_token(path: Path) -> OAuthToken | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    token_block = payload.get("tokens")
    data = cast(dict[object, object], token_block) if isinstance(token_block, dict) else payload
    access = _clean_str(data.get("access_token"))
    refresh = _clean_str(data.get("refresh_token"))
    account_id = _clean_str(data.get("account_id")) or _account_id_from_access_token(access)
    expires = _coerce_expires_ms(
        data.get("expires") or data.get("expires_at") or payload.get("expires") or payload.get("expires_at"),
        access,
        last_refresh=payload.get("last_refresh"),
    )
    if not access or not account_id or expires is None:
        return None
    return OAuthToken(access=access, refresh=refresh or "", expires=expires, account_id=account_id)


def _refresh_token(
    refresh_token: str, account_id: str | None, *, proxy: str | None = None
) -> OAuthToken:
    client_kwargs: dict[str, Any] = {"timeout": 30.0}
    if proxy:
        client_kwargs.update(proxy=proxy, trust_env=False)
    response = httpx.post(
        OPENAI_CODEX_PROVIDER.token_url,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": OPENAI_CODEX_PROVIDER.client_id,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        **client_kwargs,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Token refresh failed: {response.status_code} {response.text}")

    payload = response.json()
    access = _clean_str(payload.get("access_token"))
    if not access:
        raise RuntimeError("Token refresh response missing access_token")

    refreshed_token = _clean_str(payload.get("refresh_token")) or refresh_token
    refreshed_account = _clean_str(payload.get("account_id")) or account_id or _account_id_from_access_token(access)
    expires = _coerce_expires_ms(
        payload.get("expires_in") if payload.get("expires_in") is not None else payload.get("expires_at"),
        access, duration=payload.get("expires_in") is not None,
    )
    if not refreshed_account or expires is None:
        raise RuntimeError("Token refresh response missing account metadata")

    return OAuthToken(
        access=access,
        refresh=refreshed_token,
        expires=expires,
        account_id=refreshed_account,
    )


def _coerce_expires_ms(raw: object, access_token: str | None, last_refresh: object = None, *, duration: bool = False) -> int | None:
    if isinstance(raw, int):
        return raw if raw > 10_000_000_000 else (_now_ms() if duration else 0) + raw * 1000
    if isinstance(raw, float):
        value = int(raw)
        return value if value > 10_000_000_000 else (_now_ms() if duration else 0) + value * 1000
    if isinstance(raw, str) and raw.strip():
        text = raw.strip()
        if text.isdigit():
            value = int(text)
            return value if value > 10_000_000_000 else (_now_ms() if duration else 0) + value * 1000
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            dt = None
        if dt is not None:
            return int(dt.timestamp() * 1000)

    exp = _jwt_expiry_ms(access_token)
    if exp is not None:
        return exp

    if isinstance(last_refresh, str):
        try:
            dt = datetime.fromisoformat(last_refresh.replace("Z", "+00:00"))
        except ValueError:
            dt = None
        if dt is not None:
            return int((dt.timestamp() + 3600) * 1000)

    return None


def _jwt_expiry_ms(access_token: str | None) -> int | None:
    payload = _decode_jwt_payload(access_token)
    exp = payload.get("exp")
    if isinstance(exp, int):
        return exp * 1000
    if isinstance(exp, float):
        return int(exp * 1000)
    return None


def _account_id_from_access_token(access_token: str | None) -> str | None:
    payload = _decode_jwt_payload(access_token)
    auth_block = payload.get(OPENAI_CODEX_PROVIDER.jwt_claim_path or "")
    if isinstance(auth_block, dict):
        account_id = _clean_str(cast(dict[object, object], auth_block).get(OPENAI_CODEX_PROVIDER.account_id_claim or ""))
        if account_id:
            return account_id
    return None


def _decode_jwt_payload(token: str | None) -> dict[str, object]:
    if not token:
        return {}
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    try:
        padding = "=" * (-len(parts[1]) % 4)
        payload_bytes = base64.urlsafe_b64decode(parts[1] + padding)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return {}
    # The default decoder creates string-keyed objects; retain the shape guard.
    return cast(dict[str, object], payload) if isinstance(payload, dict) else {}


def _clean_str(value: object) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _now_ms() -> int:
    return int(time.time() * 1000)


def should_refresh_access_token(token: str, threshold_seconds: int = _REFRESH_EARLY_SECONDS) -> bool:
    """Return True when the JWT is close to expiry."""
    exp_ms = _jwt_expiry_ms(token)
    if exp_ms is None:
        return False
    return exp_ms - _now_ms() <= threshold_seconds * 1000
