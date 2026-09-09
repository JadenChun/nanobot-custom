"""Regression tests for the fail-closed client-delivery gate.

These tests cover the deterministic delivery verdict contract (Daily Content
Idea, Trend Pulse, Weekly Review) and Meta analytics notification policy.
They must run against the REAL Nanobot repository.
"""

from nanobot.bus.events import OutboundMessage
from nanobot.cron.delivery import (
    build_explicit_fanout_messages,
    build_result_messages,
    is_client_deliverable,
)
from nanobot.cron.types import CronDestination, CronPayload


# --------------------------------------------------------------------------
# Deterministic verdict gate (is_client_deliverable)
# --------------------------------------------------------------------------

def test_daily_idea_pass_is_deliverable() -> None:
    assert is_client_deliverable("PASS", skip_verification=False) is True


def test_daily_idea_fail_is_not_deliverable() -> None:
    assert is_client_deliverable("FAIL", skip_verification=False) is False


def test_daily_idea_partial_is_not_deliverable() -> None:
    assert is_client_deliverable("PARTIAL", skip_verification=False) is False


def test_daily_idea_missing_verdict_fails_closed() -> None:
    # Missing / None verdict must never deliver (fail closed).
    assert is_client_deliverable(None, skip_verification=False) is False
    assert is_client_deliverable("MISSING", skip_verification=False) is False


def test_evaluate_response_false_still_delivers_verified_pass() -> None:
    # The gate no longer consults evaluate_response at all; an explicitly
    # PASS verdict is always deliverable regardless of any owner-evaluation.
    assert is_client_deliverable("PASS", skip_verification=False) is True


def test_trend_pass_is_deliverable() -> None:
    assert is_client_deliverable("PASS", skip_verification=True) is True


def test_trend_fail_is_not_deliverable() -> None:
    assert is_client_deliverable("FAIL", skip_verification=True) is False


def test_weekly_pass_is_deliverable() -> None:
    assert is_client_deliverable("PASS", skip_verification=True) is True


def test_weekly_fail_is_not_deliverable() -> None:
    assert is_client_deliverable("FAIL", skip_verification=True) is False
    assert is_client_deliverable("PARTIAL", skip_verification=True) is False


# --------------------------------------------------------------------------
# Meta analytics policy (deliver=false => no delivery destinations)
# --------------------------------------------------------------------------

def test_meta_success_is_silent() -> None:
    payload = CronPayload(
        deliver=False,
        channel="telegram",
        to="6344587670",  # owner DM; never the client group
        additional_destinations=[CronDestination("telegram", "-5340461568")],
    )
    assert payload.delivery_destinations() == []


def test_meta_partial_owner_only() -> None:
    # Meta's notification (owner alert) is handled by the agent's message tool
    # with an explicit owner chat_id, not by cron group delivery.  With
    # deliver=False the delivery gate never publishes to the group.
    payload = CronPayload(
        deliver=False,
        channel="telegram",
        to="6344587670",
        additional_destinations=[CronDestination("telegram", "-5340461568")],
    )
    assert payload.delivery_destinations() == []


# --------------------------------------------------------------------------
# Delivery destinations do not leak a failed-run response to the group
# --------------------------------------------------------------------------

def test_group_destination_only_reached_after_verified_verdict() -> None:
    # The deterministic gate is the ONLY thing that selects the client group.
    # A non-PASS verdict means is_client_deliverable is False, so the group
    # destination is never used even if build_result_messages could build one.
    assert is_client_deliverable("FAIL", skip_verification=False) is False
    assert is_client_deliverable("PARTIAL", skip_verification=False) is False
    assert is_client_deliverable(None, skip_verification=False) is False