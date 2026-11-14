"""Instance-owned provider OAuth credentials (A3 E02-F16 acceptance cases).

Each case drives the real provider credential owners with two disposable
instance roots and synthetic credentials. No network access is used.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from nanobot.config import loader
from nanobot.providers import codex_auth, copilot_auth, xai_oauth
from nanobot.providers.oauth_credentials import (
    CredentialOwner,
    CredentialStatus,
    OAuthToken,
)

FUTURE_MS = 9_999_999_999_999
PROVIDERS = ("codex", "copilot", "xai")
_CODEX_LEGACY_JWT = (
    "eyJhbGciOiJub25lIn0."
    "eyJleHAiOjQxMDI0NDQ4MDAsImh0dHBzOi8vYXBpLm9wZW5haS5jb20vYXV0aCI6"
    "eyJjaGF0Z3B0X2FjY291bnRfaWQiOiJhY2NvdW50LUwifX0"
    ".sig"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(root: Path | CredentialOwner) -> dict[str, str]:
    base = root.data_root if isinstance(root, CredentialOwner) else root
    return {
        str(path.relative_to(base)): _sha256(path)
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


def _root(base: Path, name: str) -> CredentialOwner:
    (base / name).mkdir(parents=True, exist_ok=True)
    return CredentialOwner.from_config_path(base / name / "config.json")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


class CodexDriver:
    name = "codex"

    def sign_in(self, owner: CredentialOwner, account: str) -> None:
        authority = codex_auth.get_codex_authority(owner)
        generation = authority.begin_login()
        authority.commit_login(
            {
                "access": f"access-{account}",
                "refresh": f"refresh-{account}",
                "expires_at_ms": FUTURE_MS,
                "account_id": account,
            },
            generation,
        )

    def credential_path(self, owner: CredentialOwner) -> Path:
        return codex_auth.get_codex_authority(owner).paths.canonical

    def status(self, owner: CredentialOwner) -> tuple[str | None, bool]:
        token = codex_auth.get_codex_authority(owner).get_cached_or_import()
        return (None, False) if token is None else (token.account_id, True)

    def stored_access(self, owner: CredentialOwner) -> str | None:
        token = codex_auth.get_codex_authority(owner).get_cached_or_import()
        return None if token is None else token.access

    def account_state(self, owner: CredentialOwner) -> str | None:
        return self.stored_access(owner)

    def refresh(
        self,
        owner: CredentialOwner,
        gate: threading.Event | None = None,
        entered: threading.Event | None = None,
    ) -> None:
        authority = codex_auth.get_codex_authority(owner)

        def refresh(token: OAuthToken) -> OAuthToken:
            if gate is not None:
                if entered is not None:
                    entered.set()
                assert gate.wait(timeout=5)
            return OAuthToken(
                f"fresh-{token.account_id}", token.refresh, FUTURE_MS, token.account_id
            )

        authority.refresh_codex(10**10, refresh)

    def logout(self, owner: CredentialOwner) -> None:
        codex_auth.logout_codex(owner)

    def begin_pending(self, owner: CredentialOwner) -> Any:
        authority = codex_auth.get_codex_authority(owner)
        return authority, authority.begin_login()

    def cancel(self, pending: Any) -> None:
        authority, generation = pending
        assert authority.revoke_intent(generation) is True


class CopilotDriver:
    """Copilot keeps a long-lived token; its account cache is the derived bearer."""

    name = "copilot"

    def sign_in(self, owner: CredentialOwner, account: str) -> None:
        authority = copilot_auth.get_copilot_authority(owner)
        generation = authority.begin_login()
        authority.commit_login(
            OAuthToken(f"github-{account}", "", FUTURE_MS, account), generation
        )

    def credential_path(self, owner: CredentialOwner) -> Path:
        return copilot_auth.get_copilot_authority(owner).paths.canonical

    def status(self, owner: CredentialOwner) -> tuple[str | None, bool]:
        token = copilot_auth.get_copilot_identity(owner)
        return (None, False) if token is None else (token.account_id, True)

    def stored_access(self, owner: CredentialOwner) -> str | None:
        token = copilot_auth.get_copilot_identity(owner)
        return None if token is None else token.access

    def account_state(self, owner: CredentialOwner) -> str | None:
        path = copilot_auth.get_copilot_authority(owner).paths.derived
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        bearer = payload.get("bearer") if isinstance(payload, dict) else None
        return bearer if isinstance(bearer, str) else None

    def refresh(
        self,
        owner: CredentialOwner,
        gate: threading.Event | None = None,
        entered: threading.Event | None = None,
    ) -> None:
        authority = copilot_auth.get_copilot_authority(owner)

        def exchange(token: OAuthToken) -> dict[str, object]:
            if gate is not None:
                if entered is not None:
                    entered.set()
                assert gate.wait(timeout=5)
            return {
                "bearer": f"bearer-{token.account_id}-{time.monotonic_ns()}",
                "expires_at_ms": int(time.time() * 1000) + 3_600_000,
            }

        authority.get_copilot_bearer(7_200, exchange)

    def logout(self, owner: CredentialOwner) -> None:
        copilot_auth.logout_copilot(owner)

    def begin_pending(self, owner: CredentialOwner) -> Any:
        authority = copilot_auth.get_copilot_authority(owner)
        return authority, authority.begin_login()

    def cancel(self, pending: Any) -> None:
        authority, generation = pending
        assert authority.revoke_intent(generation) is True


class XaiDriver:
    name = "xai"

    def sign_in(self, owner: CredentialOwner, account: str) -> None:
        flow = xai_oauth.start_xai_oauth_login(timeout_s=5, owner=owner)
        try:
            token = xai_oauth.complete_xai_oauth_login(flow, f"access-{account}")
        finally:
            flow.cancel()
        assert token is not None and token.account_id == account

    def credential_path(self, owner: CredentialOwner) -> Path:
        return xai_oauth.get_xai_oauth_storage_path(owner)

    def status(self, owner: CredentialOwner) -> tuple[str | None, bool]:
        token = xai_oauth.get_xai_oauth_login_status(owner)
        return (None, False) if token is None else (token.account_id, bool(token.access))

    def stored_access(self, owner: CredentialOwner) -> str | None:
        token = xai_oauth.get_xai_oauth_login_status(owner)
        return None if token is None else token.access

    def account_state(self, owner: CredentialOwner) -> str | None:
        return self.stored_access(owner)

    def refresh(
        self,
        owner: CredentialOwner,
        gate: threading.Event | None = None,
        entered: threading.Event | None = None,
    ) -> None:
        xai_oauth.get_xai_oauth_token(force_refresh=True, owner=owner, min_ttl_ms=0)

    def logout(self, owner: CredentialOwner) -> None:
        xai_oauth.logout_xai_oauth(owner)

    def begin_pending(self, owner: CredentialOwner) -> Any:
        return xai_oauth.start_xai_oauth_login(timeout_s=5, owner=owner)

    def cancel(self, pending: Any) -> None:
        pending.cancel()


_DRIVERS: dict[str, Any] = {
    "codex": CodexDriver(),
    "copilot": CopilotDriver(),
    "xai": XaiDriver(),
}


@pytest.fixture(autouse=True)
def _offline_xai(monkeypatch: pytest.MonkeyPatch) -> dict[str, threading.Event | None]:
    discovery = xai_oauth._Discovery(  # pyright: ignore[reportPrivateUsage]
        authorization_endpoint="https://auth.x.ai/oauth2/authorize",
        token_endpoint="https://auth.x.ai/oauth2/token",
        userinfo_endpoint="https://auth.x.ai/userinfo",
    )
    monkeypatch.setattr(xai_oauth, "_discover", lambda _proxy: discovery)

    def exchange_code(token_endpoint: str, *, code: str, **_kwargs: object) -> dict[str, Any]:
        assert token_endpoint == discovery.token_endpoint
        return {
            "access_token": f"xai-{code}",
            "refresh_token": f"xai-refresh-{code}",
            "expires_in": 3_600,
        }

    monkeypatch.setattr(xai_oauth, "_exchange_code", exchange_code)
    monkeypatch.setattr(
        xai_oauth,
        "_fetch_account",
        lambda _endpoint, access, _proxy: str(access).removeprefix("xai-access-"),
    )

    gate_holder: dict[str, threading.Event | None] = {
        "gate": None,
        "entered": threading.Event(),
    }

    def refresh_token(token: Any, _proxy: str | None) -> Any:
        gate = gate_holder["gate"]
        if gate is not None:
            entered = gate_holder["entered"]
            assert entered is not None
            entered.set()
            assert gate.wait(timeout=5)
        return xai_oauth.XAIToken(
            access=f"xai-fresh-{token.account_id}",
            refresh=token.refresh,
            expires=int(time.time() * 1000) + 3_600_000,
            account_id=token.account_id,
        )

    monkeypatch.setattr(xai_oauth, "_refresh_token", refresh_token)
    return gate_holder


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("NANOBOT_CODEX_AUTH_PATH", raising=False)
    monkeypatch.delenv("NANOBOT_COPILOT_AUTH_PATH", raising=False)
    monkeypatch.delenv("OAUTH_CLI_KIT_TOKEN_PATH", raising=False)
    return home


@pytest.fixture
def roots(tmp_path: Path) -> tuple[CredentialOwner, CredentialOwner]:
    return _root(tmp_path, "instance-a"), _root(tmp_path, "instance-b")


def _legacy_profile(provider: str, home: Path, account: str) -> Path:
    if provider == "codex":
        path = home / ".codex" / "auth.json"
        _write_json(
            path,
            {
                "tokens": {
                    "access_token": _CODEX_LEGACY_JWT,
                    "refresh_token": "legacy-refresh-L",
                    "account_id": account,
                },
                "last_refresh": "2026-04-12T12:00:00Z",
            },
        )
        return path
    path = home / ".nanobot" / "copilot_auth.json"
    _write_json(path, {"github_token": f"github-{account}"})
    return path


def _assert_no_foreign_bytes(root: CredentialOwner, other_state: str) -> None:
    for path in root.data_root.rglob("*"):
        if path.is_file():
            assert other_state.encode() not in path.read_bytes(), path


@pytest.mark.parametrize("provider", PROVIDERS)
def test_c1_login_for_one_instance_leaves_the_other_untouched(
    provider: str, roots: tuple[CredentialOwner, CredentialOwner]
) -> None:
    driver = _DRIVERS[provider]
    root_a, root_b = roots
    driver.sign_in(root_b, "account-B")
    before = _snapshot(root_b)
    other_state = driver.stored_access(root_b)
    assert other_state is not None
    assert not driver.credential_path(root_a).exists()

    driver.sign_in(root_a, "account-A")

    assert driver.credential_path(root_a).exists()
    assert driver.status(root_a) == ("account-A", True)
    assert _snapshot(root_b) == before
    assert driver.status(root_b) == ("account-B", True)
    _assert_no_foreign_bytes(root_a, other_state)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_c2_late_refresh_for_one_instance_does_not_touch_the_other(
    provider: str,
    roots: tuple[CredentialOwner, CredentialOwner],
    request: pytest.FixtureRequest,
) -> None:
    driver = _DRIVERS[provider]
    root_a, root_b = roots
    driver.sign_in(root_a, "account-A")
    driver.sign_in(root_b, "account-B")
    snapshot_a = _snapshot(root_a)
    state_a = driver.account_state(root_a)
    state_b = driver.account_state(root_b)

    gate = threading.Event()
    entered = threading.Event()
    if provider == "xai":
        holder = request.getfixturevalue("_offline_xai")
        holder["gate"] = gate
        entered = holder["entered"]  # type: ignore[assignment]
    errors: list[BaseException] = []

    def late_refresh_a() -> None:
        try:
            driver.refresh(root_a, gate, entered)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=late_refresh_a)
    thread.start()
    assert entered.wait(timeout=5)

    if provider == "xai":
        holder["gate"] = None
    driver.refresh(root_b)
    snapshot_b_after = _snapshot(root_b)

    assert driver.status(root_b) == ("account-B", True)
    state_b_after = driver.account_state(root_b)
    assert state_b_after is not None and state_b_after != state_b
    assert _snapshot(root_a) == snapshot_a

    gate.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert errors == []

    assert driver.status(root_a) == ("account-A", True)
    state_a_after = driver.account_state(root_a)
    assert state_a_after is not None and state_a_after != state_a
    assert _snapshot(root_b) == snapshot_b_after
    assert state_a_after != state_b_after
    _assert_no_foreign_bytes(root_a, state_b_after)
    _assert_no_foreign_bytes(root_b, state_a_after)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_c3_logout_of_one_instance_leaves_the_other_working(
    provider: str, roots: tuple[CredentialOwner, CredentialOwner]
) -> None:
    driver = _DRIVERS[provider]
    root_a, root_b = roots
    driver.sign_in(root_a, "account-A")
    driver.refresh(root_a)
    driver.sign_in(root_b, "account-B")
    driver.refresh(root_b)
    before = _snapshot(root_b)

    driver.logout(root_a)

    assert driver.status(root_a) == (None, False)
    assert not driver.credential_path(root_a).exists()
    assert _snapshot(root_b) == before
    assert driver.status(root_b) == ("account-B", True)

    snapshot_a = _snapshot(root_a)
    driver.refresh(root_b)

    assert driver.status(root_b) == ("account-B", True)
    assert _snapshot(root_a) == snapshot_a


@pytest.mark.parametrize("provider", ("codex", "copilot"))
def test_c4_status_is_distinct_and_never_falls_back_to_a_shared_profile(
    provider: str,
    roots: tuple[CredentialOwner, CredentialOwner],
    _isolated_home: Path,
) -> None:
    driver = _DRIVERS[provider]
    root_a, root_b = roots
    driver.sign_in(root_a, "account-A")
    driver.sign_in(root_b, "account-B")
    legacy = _legacy_profile(provider, _isolated_home, "account-L")
    legacy_digest = _sha256(legacy)
    snapshot_a = _snapshot(root_a)
    snapshot_b = _snapshot(root_b)

    assert driver.status(root_a) == ("account-A", True)
    assert driver.status(root_b) == ("account-B", True)

    assert _snapshot(root_a) == snapshot_a
    assert _snapshot(root_b) == snapshot_b
    assert _sha256(legacy) == legacy_digest


def test_c5_legacy_profile_requires_an_explicit_owner_import(
    roots: tuple[CredentialOwner, CredentialOwner],
    _isolated_home: Path,
) -> None:
    driver = _DRIVERS["codex"]
    root_a, root_b = roots
    legacy = _legacy_profile("codex", _isolated_home, "account-L")
    legacy_bytes = legacy.read_bytes()

    assert driver.status(root_a) == (None, False)
    assert driver.status(root_b) == (None, False)
    assert not driver.credential_path(root_a).exists()
    assert not driver.credential_path(root_b).exists()
    assert legacy.read_bytes() == legacy_bytes

    with pytest.raises(codex_auth.CredentialError, match="credential needs refresh"):
        driver.refresh(root_a)
    assert not driver.credential_path(root_a).exists()
    assert not driver.credential_path(root_b).exists()

    with pytest.raises(TypeError):
        codex_auth.reimport_codex()  # type: ignore[call-arg]

    token = codex_auth.reimport_codex(root_a)
    assert token is not None and token.access == _CODEX_LEGACY_JWT
    assert driver.status(root_a) == ("account-L", True)
    assert not driver.credential_path(root_b).exists()
    assert driver.status(root_b) == (None, False)
    assert legacy.read_bytes() == legacy_bytes
    snapshot_a = _snapshot(root_a)

    token = codex_auth.reimport_codex(root_b)
    assert token is not None and token.access == _CODEX_LEGACY_JWT
    assert driver.status(root_b) == ("account-L", True)
    assert _snapshot(root_a) == snapshot_a
    assert legacy.read_bytes() == legacy_bytes


@pytest.mark.parametrize("provider", PROVIDERS)
def test_c6_cancelled_login_leaves_no_partial_credential(
    provider: str, roots: tuple[CredentialOwner, CredentialOwner]
) -> None:
    driver = _DRIVERS[provider]
    root_a, root_b = roots
    driver.sign_in(root_b, "account-B")
    before = _snapshot(root_b)

    pending = driver.begin_pending(root_a)
    driver.cancel(pending)

    assert driver.status(root_a) == (None, False)
    assert not driver.credential_path(root_a).exists()
    with pytest.raises((codex_auth.CredentialError, xai_oauth.XAIOAuthError)):
        driver.refresh(root_a)
    assert _snapshot(root_b) == before
    assert driver.status(root_b) == ("account-B", True)

    driver.sign_in(root_a, "account-A")

    assert driver.status(root_a) == ("account-A", True)
    assert _snapshot(root_b) == before


def test_settings_oauth_status_and_logout_use_the_selected_instance(
    roots: tuple[CredentialOwner, CredentialOwner],
    _isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.webui import settings_api
    from nanobot.webui.settings_services import WebUIOAuthFlowRegistry

    root_a, root_b = roots
    # A gateway process started on instance A.
    monkeypatch.setattr(loader, "_current_config_path", root_a.data_root / "config.json")
    CodexDriver().sign_in(root_a, "account-A")
    XaiDriver().sign_in(root_a, "account-A")
    before = _snapshot(root_a)

    payload = settings_api.settings_payload(config_path=root_b.data_root / "config.json")
    rows = {row["name"]: row for row in payload["providers"] if isinstance(row, dict)}
    for name in ("openai_codex", "xai_grok"):
        assert rows[name]["configured"] is False
        assert rows[name]["oauth_account"] is None
    assert _snapshot(root_a) == before

    settings_api.logout_oauth_provider(
        {"provider": ["openai_codex"]},
        oauth_flows=WebUIOAuthFlowRegistry(),
        config_path=root_b.data_root / "config.json",
    )

    assert _snapshot(root_a) == before
    assert codex_auth.get_codex_login_status(root_a) is CredentialStatus.AVAILABLE
    assert codex_auth.get_codex_login_status(root_b) is CredentialStatus.LOGGED_OUT

    payload_a = settings_api.settings_payload(config_path=root_a.data_root / "config.json")
    rows_a = {row["name"]: row for row in payload_a["providers"] if isinstance(row, dict)}
    assert rows_a["openai_codex"]["configured"] is True
    assert rows_a["xai_grok"]["oauth_account"] == "account-A"


def test_settings_explicit_import_writes_only_the_selected_instance(
    roots: tuple[CredentialOwner, CredentialOwner],
    _isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.webui import settings_api
    from nanobot.webui.settings_services import WebUIOAuthFlowRegistry

    root_a, root_b = roots
    monkeypatch.setattr(loader, "_current_config_path", root_a.data_root / "config.json")
    _legacy_profile("copilot", _isolated_home, "account-L")
    before = _snapshot(root_a)

    settings_api.login_oauth_provider(
        {"provider": ["github-copilot"], "mode": ["reimport"]},
        oauth_flows=WebUIOAuthFlowRegistry(),
        config_path=root_b.data_root / "config.json",
    )

    assert _snapshot(root_a) == before
    assert copilot_auth.get_copilot_login_status(root_b) is CredentialStatus.AVAILABLE
    assert copilot_auth.get_copilot_login_status(root_a) is CredentialStatus.ABSENT
    assert copilot_auth._auth_path(root_b).exists()  # pyright: ignore[reportPrivateUsage]
    assert not copilot_auth._auth_path(root_a).exists()  # pyright: ignore[reportPrivateUsage]
