from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from nanobot.providers import copilot_auth


@pytest.fixture(autouse=True)
def _isolated_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    auth_file = tmp_path / "copilot_auth.json"
    monkeypatch.setenv("NANOBOT_COPILOT_AUTH_PATH", str(auth_file))
    return auth_file


def test_is_authenticated_false_when_no_file(_isolated_auth: Path) -> None:
    assert copilot_auth.is_authenticated() is False


@pytest.mark.parametrize("use_override", [True, False])
def test_authority_path_rejects_final_symlink_without_touching_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_override: bool
) -> None:
    target = tmp_path / "target.json"
    target.write_text("sentinel", encoding="utf-8")
    before = target.read_bytes()
    link = (
        tmp_path / "copilot-link.json"
        if use_override
        else tmp_path / ".nanobot" / "auth" / "github-copilot.json"
    )
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    if use_override:
        monkeypatch.setenv("NANOBOT_COPILOT_AUTH_PATH", str(link))
    else:
        monkeypatch.delenv("NANOBOT_COPILOT_AUTH_PATH", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(copilot_auth.CredentialError, match="invalid credential store"):
        copilot_auth.get_copilot_authority()

    assert target.read_bytes() == before
    assert link.is_symlink()


def test_get_copilot_bearer_uses_generation_bound_cache_when_fresh(
    _isolated_auth: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = copilot_auth.get_copilot_authority()
    generation = authority.begin_login()
    authority.commit_login("gh_long_lived", generation)
    calls: list[int] = []

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        calls.append(1)
        return httpx.Response(
            200,
            json={"token": "tid=cached", "expires_at": int(time.time()) + 3600},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(copilot_auth.httpx, "get", fake_get)
    assert copilot_auth.get_copilot_bearer() == "tid=cached"
    assert copilot_auth.get_copilot_bearer() == "tid=cached"
    assert calls == [1]


def test_get_copilot_bearer_refreshes_when_expired(
    _isolated_auth: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = copilot_auth.get_copilot_authority()
    generation = authority.begin_login()
    authority.commit_login("gh_long_lived", generation)

    captured: dict[str, object] = {}

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        captured["url"] = url
        captured["headers"] = headers
        request = httpx.Request("GET", url)
        return httpx.Response(
            200,
            json={
                "token": "tid=fresh",
                "expires_at": int(time.time()) + 1800,
                "endpoints": {"api": "https://api.githubcopilot.com"},
            },
            request=request,
        )

    monkeypatch.setattr(copilot_auth.httpx, "get", fake_get)

    bearer = copilot_auth.get_copilot_bearer()
    assert bearer == "tid=fresh"
    assert captured["url"] == "https://api.github.com/copilot_internal/v2/token"
    assert captured["headers"]["Authorization"] == "token gh_long_lived"
    canonical = json.loads(_isolated_auth.read_text())
    assert canonical["token"]["access"] == "gh_long_lived"
    assert "tid=fresh" not in _isolated_auth.read_text()
    derived = json.loads((_isolated_auth.parent / f"{_isolated_auth.name}.derived.json").read_text())
    assert derived["bearer"] == "tid=fresh"
    assert derived["endpoint"] == "https://api.githubcopilot.com"


def test_get_copilot_bearer_raises_when_unauthenticated(_isolated_auth: Path) -> None:
    with pytest.raises(RuntimeError, match="credential needs refresh"):
        copilot_auth.get_copilot_bearer()


def test_get_copilot_bearer_surfaces_entitlement_error(
    _isolated_auth: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = copilot_auth.get_copilot_authority()
    generation = authority.begin_login()
    authority.commit_login("gh_token", generation)

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        return httpx.Response(403, text="no copilot", request=httpx.Request("GET", url))

    monkeypatch.setattr(copilot_auth.httpx, "get", fake_get)
    with pytest.raises(RuntimeError, match="credential exchange failed"):
        copilot_auth.get_copilot_bearer()


def test_copilot_request_headers_includes_editor_identity() -> None:
    headers = copilot_auth.copilot_request_headers()
    assert headers["Copilot-Integration-Id"] == "vscode-chat"
    assert headers["Editor-Version"].startswith("vscode/")
    assert "Editor-Plugin-Version" in headers


@pytest.mark.parametrize("superseder", ["logout", "new_login"])
def test_device_login_intent_is_staled_during_acquisition(
    _isolated_auth: Path, monkeypatch: pytest.MonkeyPatch, superseder: str
) -> None:
    class Response:
        status_code = 200

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def json(self) -> dict[str, object]:
            return self._payload

    class Client:
        def __init__(self, **_kwargs: object) -> None:
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, *_args: object, **_kwargs: object) -> Response:
            self.calls += 1
            if self.calls == 1:
                return Response(
                    {
                        "device_code": "device-secret",
                        "user_code": "user-secret",
                        "verification_uri": "https://github.com/login/device",
                        "interval": 1,
                    }
                )
            return Response({"access_token": "github-secret"})

    monkeypatch.setattr(copilot_auth.httpx, "Client", Client)
    monkeypatch.setattr(copilot_auth.time, "sleep", lambda _seconds: None)

    def supersede(_code: str, _url: str) -> None:
        if superseder == "logout":
            copilot_auth.logout_copilot()
        else:
            copilot_auth.get_copilot_authority().begin_login()

    with pytest.raises(copilot_auth.CredentialError, match="login attempt is stale"):
        copilot_auth.device_login(
            on_user_code=supersede,
            poll_timeout=5,
        )

    expected = "logged_out" if superseder == "logout" else "absent"
    assert copilot_auth.get_copilot_login_status().value == expected


@pytest.mark.parametrize(
    ("first_status", "first_payload", "expected"),
    [
        (500, {"secret": "response-body-secret"}, "request was rejected"),
        (200, {"device_code": "secret-only"}, "returned invalid data"),
    ],
)
def test_device_login_errors_are_bounded_and_redacted(
    _isolated_auth: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_status: int,
    first_payload: dict[str, object],
    expected: str,
) -> None:
    class Response:
        status_code = first_status
        text = "raw-response-secret"

        def json(self) -> dict[str, object]:
            return first_payload

    class Client:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, *_args: object, **_kwargs: object) -> Response:
            return Response()

    monkeypatch.setattr(copilot_auth.httpx, "Client", Client)

    with pytest.raises(RuntimeError, match=expected) as raised:
        copilot_auth.device_login(on_user_code=lambda _code, _url: None, poll_timeout=1)

    message = str(raised.value)
    assert "raw-response-secret" not in message
    assert "response-body-secret" not in message
    assert "secret-only" not in message


def _synthetic_device_flow(monkeypatch, device, polls):
    payloads = iter([device, *polls])
    waits = []

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def post(self, url, **_kwargs):
            return httpx.Response(
                200, content=json.dumps(next(payloads)), request=httpx.Request("POST", url))

    monkeypatch.setattr(copilot_auth.httpx, "Client", Client)
    monkeypatch.setattr(copilot_auth.time, "sleep", waits.append)
    monkeypatch.setattr(copilot_auth.time, "time", lambda: 0.0)
    monkeypatch.setattr(copilot_auth, "get_copilot_bearer", lambda: "synthetic-bearer")
    return waits


@pytest.mark.parametrize("interval,expected", [
    (None, 5), (0, 5), (False, 5), ("", 5), ([], 5), ({}, 5),
    ("2", 2), (2.9, 2), (-2, 1),
])
@pytest.mark.parametrize("uri", [None, 17, {"synthetic": "uri"}])
def test_device_flow_retains_interval_defaults_and_uri_conversion(
    monkeypatch, interval, expected, uri
) -> None:
    waits = _synthetic_device_flow(monkeypatch, {
        "device_code": "synthetic-device", "user_code": "synthetic-user",
        "verification_uri": uri, "interval": interval,
    }, [{"access_token": "synthetic-access"}])
    shown = []
    assert copilot_auth.device_login(
        on_user_code=lambda *values: shown.append(values), poll_timeout=5,
    ) == {"github_token": "synthetic-access"}
    assert waits == [expected]
    assert shown == [("synthetic-user", str(uri or "https://github.com/login/device"))]
    identity = copilot_auth.get_copilot_identity()
    assert identity is not None
    assert identity.access == "synthetic-access"


@pytest.mark.parametrize("device", [
    None, [], {"device_code": "synthetic-device"},
    {"device_code": 17, "user_code": "synthetic-user"},
    {"device_code": "synthetic-device", "user_code": "synthetic-user", "interval": "bad"},
    {"device_code": "synthetic-device", "user_code": "synthetic-user", "interval": [1]},
])
def test_device_flow_keeps_invalid_data_redacted_without_committing(monkeypatch, device) -> None:
    waits = _synthetic_device_flow(monkeypatch, device, [])
    shown = []
    with pytest.raises(RuntimeError, match="returned invalid data") as raised:
        copilot_auth.device_login(
            on_user_code=lambda *values: shown.append(values), poll_timeout=5,
        )
    assert "synthetic-device" not in str(raised.value)
    assert "synthetic-user" not in str(raised.value)
    assert waits == [] and shown == []
    assert copilot_auth.get_copilot_login_status().value == "absent"


def test_device_flow_preserves_pending_slowdown_and_invalid_candidate(monkeypatch) -> None:
    waits = _synthetic_device_flow(monkeypatch, {
        "device_code": "synthetic-device", "user_code": "synthetic-user", "interval": 2,
    }, [{"error": "authorization_pending"}, {"error": "slow_down"},
        {"access_token": 17}, {"error": 0, "access_token": "synthetic-access"}])
    assert copilot_auth.device_login(
        on_user_code=lambda *_values: None, poll_timeout=5,
    ) == {"github_token": "synthetic-access"}
    assert waits == [2, 2, 7, 7]


@pytest.mark.parametrize("poll,expected", [
    (None, "returned invalid data"),
    ({"error": "expired_token"}, "did not complete"),
    ({"error": "access_denied"}, "did not complete"),
    ({"error": "synthetic-secret"}, "authorization failed"),
    ({"error": ["synthetic-secret"]}, "network failure"),
])
def test_device_poll_keeps_failure_classification_and_redaction(monkeypatch, poll, expected) -> None:
    waits = _synthetic_device_flow(monkeypatch, {
        "device_code": "synthetic-device", "user_code": "synthetic-user", "interval": 1,
    }, [poll])
    with pytest.raises(RuntimeError, match=expected) as raised:
        copilot_auth.device_login(on_user_code=lambda *_values: None, poll_timeout=5)
    assert waits == [1]
    assert "synthetic-secret" not in str(raised.value)
    assert copilot_auth.get_copilot_login_status().value == "absent"


@pytest.mark.parametrize("second_value", ["https://example.invalid/synthetic", 17, None])
def test_bearer_endpoint_keeps_get_and_item_reads_distinct(monkeypatch, second_value) -> None:
    reads = []
    captured = {}
    marker = object()

    class Endpoints(dict):
        def get(self, key, default=None):
            reads.append(("get", key))
            return "https://example.invalid/checked"

        def __getitem__(self, key):
            reads.append(("item", key))
            return second_value

    class Authority:
        def get_copilot_bearer(self, min_ttl_seconds, exchange, *, expected_generation=None):
            assert min_ttl_seconds == 300
            assert expected_generation is None
            captured.update(exchange(copilot_auth.OAuthToken("synthetic-access", "", 1000)))
            return marker

    monkeypatch.setattr(copilot_auth, "get_copilot_authority", Authority)
    monkeypatch.setattr(copilot_auth, "_refresh_copilot_bearer", lambda _auth: {
        "copilot_token": "synthetic-bearer", "copilot_expires_at": 1000,
        "copilot_endpoints": Endpoints(),
    })
    assert copilot_auth.get_copilot_bearer_view() is marker
    assert reads == [("get", "api"), ("item", "api")]
    assert captured == {
        "bearer": "synthetic-bearer", "expires_at_ms": 1_000_000,
        "endpoint": second_value,
    }


@pytest.mark.parametrize("kind, expected_wait, error", [
    ("int", 3, None),
    ("index", 4, None),
    ("invalid", None, "returned invalid data"),
    ("overflow", None, "network failure"),
    ("boolean_failure", None, "returned invalid data"),
    ("lookup_failure", None, "network failure"),
])
def test_device_interval_keeps_exact_map_reads_coercion_and_failure_order(
    _isolated_auth, monkeypatch, kind, expected_wait, error,
):
    reads = []
    events = []
    waits = []
    shown = []
    committed = []
    generation = object()

    class IntValue:
        def __bool__(self):
            reads.append("bool")
            if kind == "boolean_failure":
                raise TypeError("synthetic-interval-secret")
            return True

        def __int__(self):
            reads.append("int")
            if kind == "invalid":
                raise ValueError("synthetic-interval-secret")
            if kind == "overflow":
                raise OverflowError("synthetic-interval-secret")
            return 3

    class IndexValue:
        def __bool__(self):
            reads.append("bool")
            return True

        def __index__(self):
            reads.append("index")
            return 4

    class Device(dict):
        def get(self, key, default=None):
            reads.append(("get", key))
            if key == "interval" and kind == "lookup_failure":
                raise LookupError("synthetic-interval-secret")
            return super().get(key, default)

    device = Device(
        device_code="synthetic-device", user_code="synthetic-user",
        verification_uri="https://github.com/login/device",
        interval=IndexValue() if kind == "index" else IntValue(),
    )

    class Authority:
        def begin_login(self):
            events.append("begin")
            return generation

        def commit_login(self, access, current_generation):
            committed.append((access, current_generation))
            events.append("commit")

    class Response:
        status_code = 200

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    class Client:
        def __init__(self, **_kwargs):
            self.calls = 0

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *_args):
            events.append("exit")

        def post(self, _url, **_kwargs):
            self.calls += 1
            events.append("device" if self.calls == 1 else "poll")
            return Response(device if self.calls == 1 else {"access_token": "synthetic-access"})

    def mint_bearer():
        events.append("bearer")
        return "synthetic-bearer"

    monkeypatch.setattr(copilot_auth, "get_copilot_authority", Authority)
    monkeypatch.setattr(copilot_auth, "get_copilot_bearer", mint_bearer)
    monkeypatch.setattr(copilot_auth.httpx, "Client", Client)
    monkeypatch.setattr(copilot_auth.time, "time", lambda: 0.0)
    monkeypatch.setattr(copilot_auth.time, "sleep", waits.append)

    if error is None:
        assert copilot_auth.device_login(
            on_user_code=lambda *values: shown.append(values), poll_timeout=5,
        ) == {"github_token": "synthetic-access"}
        assert waits == [expected_wait]
        assert shown == [("synthetic-user", "https://github.com/login/device")]
        assert committed == [("synthetic-access", generation)]
        assert events == ["begin", "enter", "device", "poll", "exit", "commit", "bearer"]
    else:
        with pytest.raises(RuntimeError, match=error) as caught:
            copilot_auth.device_login(on_user_code=lambda *values: shown.append(values), poll_timeout=5)
        assert "synthetic-interval-secret" not in str(caught.value)
        assert caught.value.__suppress_context__ is True
        assert waits == shown == committed == []
        assert events == ["begin", "enter", "device", "exit"]
    coercion = [] if kind == "lookup_failure" else ["bool"]
    if kind not in ("lookup_failure", "boolean_failure"):
        coercion.append("index" if kind == "index" else "int")
    assert reads == [("get", key) for key in (
        "device_code", "user_code", "verification_uri", "interval",
    )] + coercion
    assert not _isolated_auth.exists()
