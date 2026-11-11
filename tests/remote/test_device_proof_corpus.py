"""Server consumes the fixed, shared native/server proof corpus."""
import hashlib
import json
from pathlib import Path

import pytest

from nanobot.remote.device_proof import (
    DeviceProofClaims,
    DeviceProofError,
    canonical_device_proof_bytes,
    decode_proof_base64url,
    verify_device_proof,
)

CORPUS_PATH = Path(__file__).resolve().parents[2] / "mobile/contracts/device-proof-v2-vectors.json"
CORPUS_BYTES = CORPUS_PATH.read_bytes()
assert hashlib.sha256(CORPUS_BYTES).hexdigest() == "270d3d2715258bb161c97eba520e445068f8fed6f06c01bc57076e4e18d4abfd"
CORPUS = json.loads(CORPUS_BYTES)


def verify(vector, *, expected=None):
    claims = DeviceProofClaims.from_payload(vector["claims"])
    binding = expected or vector["claims"]
    return verify_device_proof(claims, public_key_der=decode_proof_base64url(vector["public_key"]), signature_der=decode_proof_base64url(vector["signature"]), expected_instance_id=binding["instance_id"], expected_origin=binding["origin"], expected_installation_id=binding["installation_id"], expected_device_generation=binding["device_generation"], expected_challenge_id=binding["challenge_id"], expected_nonce=binding["nonce"], now=vector["now"])


@pytest.mark.parametrize("vector", CORPUS["valid"], ids=lambda v: v["id"])
def test_shared_valid_bytes_and_signatures(vector):
    claims = DeviceProofClaims.from_payload(vector["claims"])
    assert canonical_device_proof_bytes(claims).hex() == vector["canonical_utf8_hex"]
    assert verify(vector) == vector["claims"]["device_fingerprint"]


@pytest.mark.parametrize("vector", CORPUS["rejected_claims"], ids=lambda v: v["id"])
def test_shared_rejected_claims(vector):
    with pytest.raises(DeviceProofError):
        claims = DeviceProofClaims.from_payload(vector["claims"])
        canonical_device_proof_bytes(claims)


@pytest.mark.parametrize("vector", CORPUS["rejected_verification"], ids=lambda v: v["id"])
def test_shared_rejected_verification(vector):
    valid = CORPUS["valid"][0]
    with pytest.raises(DeviceProofError):
        verify(valid | vector, expected=valid["claims"])


@pytest.mark.parametrize("encoded", CORPUS["rejected_encodings"])
def test_shared_rejected_encodings(encoded):
    with pytest.raises(DeviceProofError):
        decode_proof_base64url(encoded)
