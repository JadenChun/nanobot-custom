"""Regression tests for A3-6 AC6.4: Bedrock bearer credential process gate."""

import os
from unittest.mock import MagicMock, patch

import pytest

from nanobot.providers.bedrock_provider import BedrockProvider

KEY = "AWS_BEARER_TOKEN_BEDROCK"


def test_ungated_configured_key_is_refused_without_writing_environment(monkeypatch):
    monkeypatch.delenv(KEY, raising=False)
    with patch("boto3.Session") as session:
        session.return_value.client.return_value = MagicMock()
        with pytest.raises(RuntimeError, match="process-identity gate"):
            BedrockProvider(api_key="secret", region="us-east-1")

    assert os.environ.get(KEY) is None


def test_explicit_gate_writes_environment_once(monkeypatch):
    monkeypatch.delenv(KEY, raising=False)
    with patch("boto3.Session") as session:
        session.return_value.client.return_value = MagicMock()
        BedrockProvider(
            api_key="secret",
            region="us-east-1",
            allow_process_bearer=True,
        )

    assert os.environ.get(KEY) == "secret"


def test_different_live_process_credential_is_not_replaced(monkeypatch):
    monkeypatch.setenv(KEY, "other")
    with patch("boto3.Session") as session:
        session.return_value.client.return_value = MagicMock()
        with pytest.raises(RuntimeError, match="Refusing to replace"):
            BedrockProvider(
                api_key="secret",
                region="us-east-1",
                allow_process_bearer=True,
            )

    assert os.environ.get(KEY) == "other"


def test_already_matching_process_credential_is_accepted_without_gate(monkeypatch):
    monkeypatch.setenv(KEY, "secret")
    with patch("boto3.Session") as session:
        session.return_value.client.return_value = MagicMock()
        BedrockProvider(api_key="secret", region="us-east-1")

    assert os.environ.get(KEY) == "secret"


def test_injected_client_does_not_touch_process_identity(monkeypatch):
    monkeypatch.delenv(KEY, raising=False)
    with patch("boto3.Session") as session:
        session.return_value.client.return_value = MagicMock()
        BedrockProvider(api_key="secret", region="us-east-1", client=object())

    assert os.environ.get(KEY) is None
