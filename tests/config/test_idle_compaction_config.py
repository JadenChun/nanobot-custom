"""Schema-only contract tests for dormant idle compaction configuration."""

from __future__ import annotations

import copy
import json
import math

import pytest
from pydantic import ValidationError

from nanobot.config.errors import ConfigLoadError
from nanobot.config.loader import load_config, save_config
from nanobot.config.schema import Config, IdleCompactionConfig

FULL_IDLE = {
    "enabled": True,
    "idleTtlSeconds": 30.0,
    "scanIntervalSeconds": 5.0,
    "queueCapacity": 2,
    "maxConcurrentJobs": 1,
    "maxAuxiliaryCalls": 3,
    "maxAuxiliaryTokens": 1000,
    "maxInputTokens": 12000,
    "maxOutputTokens": 2000,
    "maxChunks": 8,
    "maxAuxiliaryWallSeconds": 10.0,
    "initialBackoffSeconds": 2.0,
    "maxBackoffSeconds": 30.0,
    "summaryModelPreset": "idle-summary",
}


def _full_config() -> dict[str, object]:
    return {
        "modelPresets": {
            "idle-summary": {
                "model": "openai/gpt-4.1-mini",
                "provider": "openai",
            }
        },
        "agents": {"defaults": {"idleCompaction": copy.deepcopy(FULL_IDLE)}},
    }


def test_idle_compaction_defaults_are_dormant_and_schema_only() -> None:
    config = Config()
    idle = config.agents.defaults.idle_compaction

    assert idle.enabled is False
    assert idle.idle_ttl_seconds == 0
    assert idle.scan_interval_seconds is None
    assert idle.queue_capacity is None
    assert idle.summary_model_preset is None


@pytest.mark.parametrize("legacy_key", ["idleCompactAfterMinutes", "sessionTtlMinutes"])
def test_legacy_ttl_aliases_do_not_populate_idle_compaction(legacy_key: str) -> None:
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    legacy_key: 20,
                    "idleCompactCheckIntervalSeconds": 17,
                }
            }
        }
    )

    defaults = config.agents.defaults
    assert defaults.session_ttl_minutes == 20
    assert defaults.idle_compact_check_interval_seconds == 17
    assert defaults.idle_compaction.enabled is False
    assert defaults.idle_compaction.idle_ttl_seconds == 0

    dumped = config.model_dump(mode="json", by_alias=True)["agents"]["defaults"]
    assert dumped["idleCompactAfterMinutes"] == 20
    assert dumped["idleCompactCheckIntervalSeconds"] == 17
    assert dumped["idleCompaction"]["enabled"] is False


@pytest.mark.parametrize(
    ("legacy_key", "expected_dump_key"),
    [
        ("idleCompactAfterMinutes", "idleCompactAfterMinutes"),
        ("sessionTtlMinutes", "idleCompactAfterMinutes"),
        ("idleCompactCheckIntervalSeconds", "idleCompactCheckIntervalSeconds"),
        (
            "idle_compact_check_interval_seconds",
            "idleCompactCheckIntervalSeconds",
        ),
    ],
)
def test_each_legacy_ttl_or_scan_spelling_remains_supported(
    legacy_key: str, expected_dump_key: str
) -> None:
    config = Config.model_validate(
        {"agents": {"defaults": {legacy_key: 21}}}
    )

    defaults = config.agents.defaults
    if "Interval" in expected_dump_key:
        assert defaults.idle_compact_check_interval_seconds == 21
    else:
        assert defaults.session_ttl_minutes == 21
    assert config.model_dump(mode="json", by_alias=True)["agents"]["defaults"][
        expected_dump_key
    ] == 21


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("idleCompactAfterMinutes", "sessionTtlMinutes"),
        (
            "idleCompactCheckIntervalSeconds",
            "idle_compact_check_interval_seconds",
        ),
    ],
)
def test_ambiguous_legacy_ttl_or_scan_spellings_are_rejected(
    first: str, second: str
) -> None:
    with pytest.raises(ValidationError, match="Specify only one"):
        Config.model_validate(
            {"agents": {"defaults": {first: 10, second: 11}}}
        )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("idleCompactAfterMinutes", "sessionTtlMinutes"),
        (
            "idleCompactCheckIntervalSeconds",
            "idle_compact_check_interval_seconds",
        ),
    ],
)
def test_loader_rejects_ambiguous_legacy_ttl_or_scan_spellings(
    tmp_path, first: str, second: str
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"agents": {"defaults": {first: 10, second: 11}}}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError) as exc_info:
        load_config(path)
    assert exc_info.value.kind == "invalid_schema"
    assert exc_info.value.issues[0].path == ("agents", "defaults")


def test_disabled_idle_compaction_accepts_partial_dormant_values() -> None:
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "idleCompaction": {
                        "idleTtlSeconds": 60,
                        "queueCapacity": 2,
                    }
                }
            }
        }
    )

    idle = config.agents.defaults.idle_compaction
    assert idle.enabled is False
    assert idle.idle_ttl_seconds == 60
    assert idle.queue_capacity == 2
    assert idle.max_concurrent_jobs is None


def test_full_idle_compaction_config_round_trips_with_camel_case(tmp_path) -> None:
    config = Config.model_validate(_full_config())
    dumped = config.model_dump(mode="json", by_alias=True)
    idle_dump = dumped["agents"]["defaults"]["idleCompaction"]

    assert idle_dump == FULL_IDLE
    assert all("_" not in key for key in idle_dump)
    restored = Config.model_validate(dumped)
    assert restored.model_dump(mode="json", by_alias=True) == dumped

    path = tmp_path / "config.json"
    save_config(config, path)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["agents"]["defaults"]["idleCompaction"] == FULL_IDLE
    assert load_config(path).model_dump(mode="json", by_alias=True) == dumped


@pytest.mark.parametrize(
    "missing",
    [
        "scanIntervalSeconds",
        "queueCapacity",
        "maxConcurrentJobs",
        "maxAuxiliaryCalls",
        "maxAuxiliaryTokens",
        "maxInputTokens",
        "maxOutputTokens",
        "maxChunks",
        "maxAuxiliaryWallSeconds",
        "initialBackoffSeconds",
        "maxBackoffSeconds",
        "summaryModelPreset",
    ],
)
def test_enabled_idle_compaction_requires_every_operational_field(missing: str) -> None:
    payload = _full_config()
    idle = payload["agents"]["defaults"]["idleCompaction"]  # type: ignore[index]
    del idle[missing]  # type: ignore[index]

    with pytest.raises(ValidationError, match="enabled idle compaction"):
        Config.model_validate(payload)


def test_enabled_idle_compaction_requires_positive_ttl() -> None:
    payload = _full_config()
    payload["agents"]["defaults"]["idleCompaction"]["idleTtlSeconds"] = 0  # type: ignore[index]

    with pytest.raises(ValidationError, match="idle_ttl_seconds"):
        Config.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    [
        "queueCapacity",
        "maxConcurrentJobs",
        "maxAuxiliaryCalls",
        "maxAuxiliaryTokens",
        "maxInputTokens",
        "maxOutputTokens",
        "maxChunks",
    ],
)
def test_integer_limits_reject_bool_and_non_positive_values(field: str) -> None:
    for value in (True, 0, -1, "2"):
        with pytest.raises(ValidationError):
            IdleCompactionConfig.model_validate({field: value})


@pytest.mark.parametrize(
    "field",
    [
        "idleTtlSeconds",
        "scanIntervalSeconds",
        "maxAuxiliaryWallSeconds",
        "initialBackoffSeconds",
        "maxBackoffSeconds",
    ],
)
def test_duration_limits_reject_nan_zero_negative_and_strings(field: str) -> None:
    values = [float("nan"), float("inf"), 0, -1, "2"]
    if field == "idleTtlSeconds":
        values = [float("nan"), float("inf"), -1, "2"]
    for value in values:
        with pytest.raises(ValidationError):
            IdleCompactionConfig.model_validate({field: value})


def test_idle_ttl_allows_zero_but_rejects_non_finite_values() -> None:
    assert IdleCompactionConfig(idle_ttl_seconds=0).idle_ttl_seconds == 0
    for value in (math.nan, math.inf, -math.inf, -0.1):
        with pytest.raises(ValidationError):
            IdleCompactionConfig.model_validate({"idleTtlSeconds": value})


def test_enabled_idle_compaction_rejects_backoff_inversion() -> None:
    payload = _full_config()
    payload["agents"]["defaults"]["idleCompaction"].update(  # type: ignore[index]
        {"initialBackoffSeconds": 31.0, "maxBackoffSeconds": 30.0}
    )

    with pytest.raises(ValidationError, match="cannot exceed"):
        Config.model_validate(payload)


@pytest.mark.parametrize("value", [True, 1, "true"])
def test_enabled_is_strict_bool(value: object) -> None:
    with pytest.raises(ValidationError):
        IdleCompactionConfig.model_validate({"enabled": value})


@pytest.mark.parametrize("value", ["", "   "])
def test_summary_model_preset_must_be_non_empty_when_supplied(value: str) -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        IdleCompactionConfig.model_validate({"summaryModelPreset": value})


def test_enabled_idle_compaction_requires_an_existing_explicit_model_preset() -> None:
    payload = _full_config()
    payload["agents"]["defaults"]["idleCompaction"]["summaryModelPreset"] = "missing"  # type: ignore[index]

    with pytest.raises(ValidationError, match="not found in model_presets"):
        Config.model_validate(payload)


def test_snake_and_camel_duplicate_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="idleTtlSeconds"):
        IdleCompactionConfig.model_validate(
            {"idleTtlSeconds": 10, "idle_ttl_seconds": 20}
        )

    with pytest.raises(ValidationError, match="idleCompaction"):
        Config.model_validate(
            {
                "agents": {
                    "defaults": {
                        "idleCompaction": {},
                        "idle_compaction": {},
                    }
                }
            }
        )


def test_schema_construction_does_not_activate_idle_runtime(monkeypatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("idle configuration must not activate runtime work")

    monkeypatch.setattr("nanobot.agent.idle_compaction.IdleCompactionCoordinator", fail)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", fail)
    monkeypatch.setattr("nanobot.session.idle_checkpoint.capture_idle_checkpoint", fail)
    monkeypatch.setattr("nanobot.session.idle_checkpoint.publish_idle_checkpoint", fail)

    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "idleCompaction": {"idleTtlSeconds": 60, "queueCapacity": 2}
                }
            }
        }
    )

    assert config.agents.defaults.idle_compaction.enabled is False
