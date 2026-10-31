from __future__ import annotations

import pytest
from pydantic import ValidationError

from nanobot.config.schema import Config, RemoteAccessConfig


def test_remote_access_defaults_off() -> None:
    config = Config()
    assert config.remote_access.enabled is False
    assert config.remote_access.state_dir is None
    assert config.remote_access.socket_path is None
    assert config.remote_access.owner_timeout_seconds == 2.0


def test_remote_access_accepts_camel_case_and_bounds() -> None:
    config = Config.model_validate(
        {
            "remoteAccess": {
                "enabled": True,
                "stateDir": "/tmp/nanobot-remote-access",
                "socketPath": "/tmp/nanobot-remote-access/owner.sock",
                "ownerTimeoutSeconds": 1.5,
            }
        }
    )
    assert config.remote_access.enabled is True
    assert config.remote_access.state_dir == "/tmp/nanobot-remote-access"
    assert config.remote_access.socket_path == "/tmp/nanobot-remote-access/owner.sock"
    with pytest.raises(ValidationError):
        RemoteAccessConfig(owner_timeout_seconds=0)
    with pytest.raises(ValidationError):
        RemoteAccessConfig(owner_timeout_seconds=11)
