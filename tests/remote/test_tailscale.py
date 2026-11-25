from __future__ import annotations

import pytest

from kohaku.remote import tailscale


def test_missing_tailscale_cli_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tailscale.shutil, "which", lambda _name: None)
    status = tailscale.tailscale_status()
    assert status["available"] is False
    assert status["connected"] is False
    assert "not installed" in status["error"]


def test_connected_serve_mapping_derives_https_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tailscale.shutil, "which", lambda _name: "/usr/bin/tailscale")
    responses = {
        ("status",): {
            "BackendState": "Running",
            "Self": {"DNSName": "desktop.tailnet.ts.net."},
        },
        ("serve",): {
            "TCP": {"443": {"HTTPS": True}},
            "Web": {
                "desktop.tailnet.ts.net:443": {
                    "Handlers": {
                        "/mobile/v1": {
                            "Proxy": "http://127.0.0.1:8767/mobile/v1",
                        },
                    },
                },
            },
        },
    }
    monkeypatch.setattr(
        tailscale,
        "_run_json",
        lambda command: responses.get(tuple(command[1:2])),
    )
    status = tailscale.tailscale_status()
    assert status["available"] is True
    assert status["connected"] is True
    assert status["serve_mapped"] is True
    assert status["serve_target"] == "http://127.0.0.1:8767/mobile/v1"
    assert status["dns_name"] == "desktop.tailnet.ts.net"
    assert status["origin"] == "https://desktop.tailnet.ts.net"
    assert status["error"] == ""


def test_connected_without_serve_mapping_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tailscale.shutil, "which", lambda _name: "/usr/bin/tailscale")
    responses = {
        ("status",): {"BackendState": "Running", "Self": {"DNSName": "desktop.ts.net"}},
        ("serve",): {
            "TCP": {"443": {"HTTPS": True}},
            "Web": {
                "desktop.ts.net:443": {
                    "Handlers": {
                        "/remote/v1": {"Proxy": "http://127.0.0.1:8900"},
                    },
                },
            },
        },
    }
    monkeypatch.setattr(
        tailscale,
        "_run_json",
        lambda command: responses.get(tuple(command[1:2])),
    )
    status = tailscale.tailscale_status()
    assert status["connected"] is True
    assert status["serve_mapped"] is False
    assert "exact /mobile/v1 target" in status["error"]


@pytest.mark.parametrize("path", ["", "/mobile/v1", "/mobile/v1/"])
def test_serve_mapping_preserves_mobile_path(
    monkeypatch: pytest.MonkeyPatch, path: str,
) -> None:
    monkeypatch.setattr(tailscale.shutil, "which", lambda _name: "/usr/bin/tailscale")
    responses = {
        ("status", "--json"): {
            "BackendState": "Running",
            "Self": {"DNSName": "desktop.ts.net"},
        },
        ("serve", "status", "--json"): {
            "Web": {"desktop.ts.net:443": {"Handlers": {
                "/mobile/v1": {"Proxy": f"http://127.0.0.1:18790{path}"},
            }}},
        },
    }
    monkeypatch.setattr(tailscale, "_run_json", lambda command: responses[tuple(command[1:])])

    status = tailscale.tailscale_status(expected_port=18790)

    assert status["serve_mapped"] is bool(path)
    assert status["expected_port"] == 18790
    assert status["recommended_command"] == (
        "tailscale serve --bg --https=443 --set-path=/mobile/v1 http://127.0.0.1:18790/mobile/v1"
    )
    if path:
        assert status["issue"] == ""
        assert status["error"] == ""
    else:
        assert status["issue"] == "serve_path_stripped"
        assert "strip" in status["error"] and "/mobile/v1" in status["error"]
