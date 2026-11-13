"""Requirement-level tests for cancelling a device-pairing attempt.

The cancel contract must guarantee that no server authority survives a
successful cancel: issued credentials stop validating, the device is
revoked, the attempt is gone, and a new attempt for the same installation
is accepted. Wrong credentials must have no side effects. State must
survive a service restart over the same store.

Only the public surface from the task contract is used.
"""

import importlib

import pytest

from tests.remote.proof_helpers import collect_with_key, complete_proof

# ---------------------------------------------------------------------------
# Public-symbol resolution
# ---------------------------------------------------------------------------


def _load(attr, module_names):
    """Resolve a public symbol from the first importable candidate module."""
    for name in module_names:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        obj = getattr(module, attr, None)
        if obj is not None:
            return obj
    raise ImportError(f"could not resolve {attr} from {module_names}")


PAIRING_MODULES = (
    "nanobot.remote.pairing",
    "nanobot.remote.pairing_service",
    "nanobot.remote.service",
    "nanobot.remote",
)
STORE_MODULES = (
    "nanobot.remote.access_store",
    "nanobot.remote.store",
    "nanobot.remote.access",
    "nanobot.remote.pairing",
    "nanobot.remote",
)
TOKEN_MODULES = (
    "nanobot.remote.auth",
    "nanobot.remote.tokens",
    "nanobot.remote.token",
    "nanobot.remote.access",
    "nanobot.remote",
)
ERROR_MODULES = (
    "nanobot.remote.pairing",
    "nanobot.remote.errors",
    "nanobot.remote.exceptions",
    "nanobot.remote",
)


def pairing_service():
    return _load("PairingService", PAIRING_MODULES)


def remote_access_store():
    return _load("RemoteAccessStore", STORE_MODULES)


def token_service():
    return _load("TokenService", TOKEN_MODULES)


def pairing_rejected_error():
    return _load("PairingRejectedError", ERROR_MODULES)


def pairing_not_found_error():
    return _load("PairingNotFoundError", ERROR_MODULES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup(tmp_path):
    """Build a store, token service, and pairing service over a temp root."""
    store = remote_access_store()(tmp_path)
    store.load_or_create()
    store.set_enabled(True)
    tokens = token_service()(store)
    pairing = pairing_service()(store, tokens)
    return store, tokens, pairing


def offer(pairing, *, label="Test Device", chat_consent=True, media_consent=False):
    """Create a pairing challenge."""
    return pairing.create_challenge(
        origin="https://example.test",
        label=label,
        chat_consent=chat_consent,
        media_consent=media_consent,
    )


def collect_for(pairing, challenge, *, installation_id="install-1", label="Test Device"):
    """Collect a freshly generated device key for a challenge."""
    pending = collect_with_key(
        pairing,
        pairing_id=challenge.pairing_id,
        nonce=challenge.nonce,
        collector_secret=challenge.collector_secret,
        installation_id=installation_id,
        label=label,
    )
    assert pending is not None
    return pending


def pending_for(pairing, *, installation_id="install-1", label="Test Device"):
    """Create an offer and collect a device key for it."""
    challenge = offer(pairing, label=label)
    pending = collect_for(
        pairing, challenge, installation_id=installation_id, label=label
    )
    return challenge, pending


def approved_and_completed(pairing, *, installation_id="install-1", label="Test Device"):
    """Drive a pairing to issued credentials; return (challenge, access_token)."""
    challenge, _pending = pending_for(
        pairing, installation_id=installation_id, label=label
    )
    pairing.approve(
        pairing_id=challenge.pairing_id, comparison_code=challenge.comparison_code
    )
    issued = complete_proof(pairing, challenge.pairing_id)
    return challenge, issued.access_token


def fresh_collect(pairing, *, installation_id="install-1", label="Fresh Device"):
    """Start a fresh offer and collect it with a new device key."""
    challenge = offer(pairing, label=label)
    pending = collect_with_key(
        pairing,
        pairing_id=challenge.pairing_id,
        nonce=challenge.nonce,
        collector_secret=challenge.collector_secret,
        installation_id=installation_id,
        label=label,
    )
    assert pending is not None
    return challenge, pending


def cancel(pairing, challenge):
    return pairing.cancel_attempt(
        pairing_id=challenge.pairing_id,
        nonce=challenge.nonce,
        collector_secret=challenge.collector_secret,
    )


def assert_token_valid(tokens, access_token):
    assert tokens.validate_access(access_token) is not None


def assert_token_invalid(tokens, access_token):
    try:
        tokens.validate_access(access_token)
    except Exception:
        return
    pytest.fail("access token still validates after the pairing was cancelled")


def device_for(store, installation_id):
    devices = store.load().devices
    assert installation_id in devices
    return devices[installation_id]


# ---------------------------------------------------------------------------
# 1. Cancel after approval and proof completion
# ---------------------------------------------------------------------------


def test_cancel_after_completed_pairing_returns_true_and_revokes_authority(tmp_path):
    store, tokens, pairing = setup(tmp_path)
    challenge, access_token = approved_and_completed(pairing)
    assert_token_valid(tokens, access_token)

    assert cancel(pairing, challenge) is True

    assert_token_invalid(tokens, access_token)
    assert device_for(store, "install-1").revoked_at is not None


def test_cancel_after_completed_pairing_allows_fresh_attempt_for_same_installation(
    tmp_path,
):
    _store, _tokens, pairing = setup(tmp_path)
    challenge, _access_token = approved_and_completed(pairing)

    cancel(pairing, challenge)

    # A fresh offer plus collection for the same installation must not hit a
    # busy error left behind by the cancelled attempt.
    fresh_collect(pairing, installation_id="install-1")


# ---------------------------------------------------------------------------
# 2. Wrong nonce or collector secret is rejected without side effects
# ---------------------------------------------------------------------------


def test_cancel_with_wrong_nonce_is_rejected_without_revoking(tmp_path):
    store, tokens, pairing = setup(tmp_path)
    challenge, access_token = approved_and_completed(pairing)
    rejected = pairing_rejected_error()

    with pytest.raises(rejected):
        pairing.cancel_attempt(
            pairing_id=challenge.pairing_id,
            nonce="wrong-nonce",
            collector_secret=challenge.collector_secret,
        )

    assert_token_valid(tokens, access_token)
    assert device_for(store, "install-1").revoked_at is None


def test_cancel_with_wrong_collector_secret_is_rejected_without_revoking(tmp_path):
    store, tokens, pairing = setup(tmp_path)
    challenge, access_token = approved_and_completed(pairing)
    rejected = pairing_rejected_error()

    with pytest.raises(rejected):
        pairing.cancel_attempt(
            pairing_id=challenge.pairing_id,
            nonce=challenge.nonce,
            collector_secret="wrong-collector-secret",
        )

    assert_token_valid(tokens, access_token)
    assert device_for(store, "install-1").revoked_at is None


# ---------------------------------------------------------------------------
# 3. Cancel before proof removes the attempt and issues nothing
# ---------------------------------------------------------------------------


def test_cancel_before_proof_issues_no_device(tmp_path):
    store, _tokens, pairing = setup(tmp_path)
    challenge, _pending = pending_for(pairing)

    assert cancel(pairing, challenge) is True

    devices = store.load().devices
    assert "install-1" not in devices or devices["install-1"].revoked_at is not None


def test_cancel_before_proof_allows_fresh_attempt_for_same_installation(tmp_path):
    _store, _tokens, pairing = setup(tmp_path)
    challenge, _pending = pending_for(pairing)

    cancel(pairing, challenge)

    fresh_collect(pairing, installation_id="install-1")


# ---------------------------------------------------------------------------
# 4. Unknown pairing ID
# ---------------------------------------------------------------------------


def test_cancel_unknown_pairing_id_raises_pairing_not_found(tmp_path):
    _store, _tokens, pairing = setup(tmp_path)
    challenge = offer(pairing)
    not_found = pairing_not_found_error()

    with pytest.raises(not_found):
        pairing.cancel_attempt(
            pairing_id="unknown-pairing-id",
            nonce=challenge.nonce,
            collector_secret=challenge.collector_secret,
        )


# ---------------------------------------------------------------------------
# 5. Restart over the same store
# ---------------------------------------------------------------------------


def test_restart_keeps_cancelled_device_revoked(tmp_path):
    store, _tokens, pairing = setup(tmp_path)
    challenge, access_token = approved_and_completed(pairing)
    cancel(pairing, challenge)

    restarted_tokens = token_service()(store)
    pairing_service()(store, restarted_tokens)

    assert device_for(store, "install-1").revoked_at is not None
    assert_token_invalid(restarted_tokens, access_token)


def test_restart_keeps_freshly_collected_attempt_available_or_recollectable(tmp_path):
    store, _tokens, pairing = setup(tmp_path)
    challenge, _access_token = approved_and_completed(pairing)
    cancel(pairing, challenge)
    fresh, _pending = fresh_collect(pairing, installation_id="install-1")

    restarted = pairing_service()(store, token_service()(store))

    # The freshly collected attempt survived the restart: finish it, or if the
    # restart path requires a new collection, that collection must be accepted
    # for the same installation without a busy error.
    not_found = pairing_not_found_error()
    rejected = pairing_rejected_error()
    try:
        restarted.approve(
            pairing_id=fresh.pairing_id, comparison_code=fresh.comparison_code
        )
        issued = complete_proof(restarted, fresh.pairing_id)
        assert issued.access_token
    except (not_found, rejected):
        fresh_collect(restarted, installation_id="install-1")
