import json

import pytest

from nanobot.config.errors import ConfigLoadError
from nanobot.config.loader import load_config
from nanobot.config.schema import Config


@pytest.mark.parametrize(
    "gateway",
    [
        {"restartMode": "exit"},
        {"restartMode": "exit", "telegramBusy": {}},
        {"restartMode": "exit", "telegramBusy": {"enabled": False}},
        {"restartMode": "exit", "telegramBusy": {"enabled": False, "sessionKey": "telegram:42"}},
        {"restartMode": "exit", "telegram_busy": {}},
        {"restartMode": "exit", "telegram_busy": {"enabled": False}},
    ],
)
def test_retired_telegram_busy_disabled_forms_load_without_advertising_the_feature(gateway):
    config = Config.model_validate({"gateway": gateway})

    dumped = config.model_dump(by_alias=True)
    assert "telegramBusy" not in dumped["gateway"]
    assert "telegram_busy" not in dumped["gateway"]
    assert config.gateway.restart_mode == "exit"


@pytest.mark.parametrize(
    "block",
    [
        {"enabled": True},
        {"enabled": "false"},
        {"enabled": False, "unknownSetting": "value"},
        [],
    ],
)
def test_retired_telegram_busy_enabled_or_malformed_block_fails_closed(block):
    with pytest.raises(ValueError, match="retired"):
        Config.model_validate({"gateway": {"telegramBusy": block}})


def test_retired_telegram_busy_duplicate_aliases_fail_closed():
    with pytest.raises(ValueError, match="duplicate"):
        Config.model_validate({
            "gateway": {
                "telegramBusy": {"enabled": False},
                "telegram_busy": {"enabled": False},
            }
        })


def test_load_config_reports_retired_enabled_block_without_echoing_secrets(tmp_path):
    path = tmp_path / "config.json"
    secret = "sk-secret-sentinel-value"
    path.write_text(json.dumps({
        "gateway": {"telegramBusy": {"enabled": True}},
        "providers": {"openai": {"apiKey": secret}},
    }))

    with pytest.raises(ConfigLoadError) as error:
        load_config(path)

    rendered = str(error.value)
    assert error.value.kind == "invalid_schema"
    assert "retired" in rendered.lower()
    assert "H1" in rendered
    assert secret not in rendered


def test_load_config_accepts_disabled_retired_block_and_preserves_other_settings(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "gateway": {
            "restartMode": "exit",
            "telegramBusy": {"enabled": False, "sessionKey": "telegram:42"},
        },
    }))

    config = load_config(path)

    assert config.gateway.restart_mode == "exit"
    assert "telegramBusy" not in config.model_dump(by_alias=True)["gateway"]
