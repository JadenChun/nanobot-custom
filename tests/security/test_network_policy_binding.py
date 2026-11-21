"""Regression tests for turn-bound SSRF policy versus later config loads (A3-6)."""

from __future__ import annotations

from nanobot.security.network import (
    NetworkPolicy,
    bind_network_policy_for_cidrs,
    configure_ssrf_whitelist,
    current_network_policy,
    resolve_url_target,
)

_CGNAT_URL = "http://100.64.0.1/"
_CGNAT_CIDR = "100.64.0.0/10"


def _allowed(url: str) -> bool:
    return resolve_url_target(url)[0]


def test_bound_policy_wins_over_later_config_load() -> None:
    configure_ssrf_whitelist([_CGNAT_CIDR])
    try:
        assert _allowed(_CGNAT_URL)

        with bind_network_policy_for_cidrs([]):
            assert not _allowed(_CGNAT_URL)

            configure_ssrf_whitelist([_CGNAT_CIDR])
            assert not _allowed(_CGNAT_URL)

        assert (
            current_network_policy().allowed_networks
            == NetworkPolicy.from_cidrs([_CGNAT_CIDR]).allowed_networks
        )
        assert _allowed(_CGNAT_URL)
    finally:
        configure_ssrf_whitelist([])
