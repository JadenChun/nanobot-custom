"""Pure synthetic regressions for credential-detection output and traversal."""

import json

import pytest

from nanobot.utils.secret_scan import find_secrets, find_secrets_in_blob


@pytest.mark.parametrize("value", [None, 42, b"access_token=synthetic_only", {}, ""])
def test_blob_detector_retains_invalid_input_refusal(value):
    assert find_secrets_in_blob(value) == []


def test_nested_authorization_deduplicates_without_returning_full_value():
    token = "synthetic_study_token_0123456789"
    payload = {"headers": [{"Authorization": f"Bearer {token}"},
                           {"authorization": f"Bearer {token}"}]}

    hits = find_secrets(payload)

    assert hits == [{"kind": "plaintext_pattern", "path": "headers[0].Authorization",
                     "field": "Authorization", "fingerprint": "Bearer synt...6789"}]
    assert token not in json.dumps(hits)
    assert payload["headers"][0]["Authorization"] == f"Bearer {token}"


@pytest.mark.parametrize("key", [0, 42, None, ("source", 1)])
def test_nonstring_root_keys_retain_existing_paths_and_redaction(key):
    token = "synthetic_study_token_0123456789"
    hits = find_secrets({key: {"Authorization": f"Bearer {token}"}})

    expected_path = f"{key}.Authorization" if key else "Authorization"
    assert hits == [{"kind": "plaintext_pattern", "path": expected_path,
                     "field": "Authorization", "fingerprint": "Bearer synt...6789"}]
    assert token not in json.dumps(hits)


def test_unused_nonstring_path_is_not_stringified():
    class Key:
        def __str__(self):
            raise AssertionError("an unused path must not be stringified")

    assert find_secrets({Key(): None}) == []


def test_list_blob_hit_keeps_blob_label_and_redacted_fingerprint():
    token = "synthetic_study_token_0123456789"
    hits = find_secrets({42: [f"access_token={token}"]})

    assert hits == [{"kind": "url_query_param", "path": "blob", "field": "access_token",
                     "fingerprint": "access_token=synt...6789"}]
    assert token not in json.dumps(hits)


def test_short_authorization_value_is_fully_redacted():
    assert find_secrets({"Authorization": "Bearer abcd"}) == [
        {"kind": "plaintext_pattern", "path": "Authorization", "field": "Authorization",
         "fingerprint": "Bearer ****"},
    ]


def test_clean_nested_scalars_produce_no_hits():
    assert find_secrets({0: [None, False, 42, "ordinary study notes"]}) == []
