"""Requirements-only acceptance tests for source compaction configuration.

Contract under test (public interface only):

- ``nanobot.config.schema.SourceCompactionConfig`` (Pydantic model)
    - ``idle_timeout_seconds: float = 300.0``
    - ``overall_timeout_seconds: float = 900.0``
    - ``pressure_trigger_ratio: float = 0.9``
    - ``target_ratio: float = 0.6``
    - ``summary_generation_tokens: int | None = None``
- Accepted JSON aliases: ``idleTimeoutSeconds``, ``overallTimeoutSeconds``,
  ``pressureTriggerRatio``, ``targetRatio``, ``summaryGenerationTokens``.
- Nested as ``source_compaction`` (alias ``sourceCompaction``) inside
  ``nanobot.config.schema.AgentDefaults``.
- Invalid input raises Pydantic ``ValidationError``.
"""

import pytest
from pydantic import ValidationError

from nanobot.config.schema import AgentDefaults, SourceCompactionConfig


# Requirement: defaults preserve current behavior.
def test_defaults_preserve_current_behavior():
    config = SourceCompactionConfig()

    assert config.pressure_trigger_ratio == 0.9
    assert config.target_ratio == 0.6
    assert config.summary_generation_tokens is None


# Requirement: existing camelCase config spellings must work, mapping onto the
# snake_case fields.
def test_camel_case_spellings_parse_into_snake_case_fields():
    config = SourceCompactionConfig.model_validate(
        {
            "idleTimeoutSeconds": 120.0,
            "overallTimeoutSeconds": 450.0,
            "pressureTriggerRatio": 0.8,
            "targetRatio": 0.5,
            "summaryGenerationTokens": 4096,
        }
    )

    assert config.idle_timeout_seconds == 120.0
    assert config.overall_timeout_seconds == 450.0
    assert config.pressure_trigger_ratio == 0.8
    assert config.target_ratio == 0.5
    assert config.summary_generation_tokens == 4096


# Requirement: snake_case config spellings must also work directly.
def test_snake_case_spellings_parse_directly():
    config = SourceCompactionConfig.model_validate(
        {
            "idle_timeout_seconds": 60.0,
            "overall_timeout_seconds": 300.0,
            "pressure_trigger_ratio": 0.85,
            "target_ratio": 0.55,
            "summary_generation_tokens": 2048,
        }
    )

    assert config.idle_timeout_seconds == 60.0
    assert config.overall_timeout_seconds == 300.0
    assert config.pressure_trigger_ratio == 0.85
    assert config.target_ratio == 0.55
    assert config.summary_generation_tokens == 2048


# Requirement: supplying both spellings of the same field is an error.
def test_supplying_both_spellings_of_one_field_is_rejected():
    with pytest.raises(ValidationError):
        SourceCompactionConfig.model_validate(
            {
                "pressureTriggerRatio": 0.8,
                "pressure_trigger_ratio": 0.8,
            }
        )


# Requirement: target ratio must be strictly below the trigger ratio.
@pytest.mark.parametrize(
    "trigger_ratio, target_ratio",
    [
        pytest.param(0.9, 0.9, id="target-equals-trigger"),
        pytest.param(0.9, 0.95, id="target-above-trigger"),
    ],
)
def test_target_ratio_must_be_strictly_below_trigger_ratio(
    trigger_ratio, target_ratio
):
    with pytest.raises(ValidationError):
        SourceCompactionConfig.model_validate(
            {
                "pressure_trigger_ratio": trigger_ratio,
                "target_ratio": target_ratio,
            }
        )


# Requirement: an optional summary generation token override must be a positive
# integer when supplied; a positive value is accepted.
@pytest.mark.parametrize(
    "tokens",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative-one"),
        pytest.param(-4096, id="large-negative"),
    ],
)
def test_non_positive_summary_generation_tokens_is_rejected(tokens):
    with pytest.raises(ValidationError):
        SourceCompactionConfig.model_validate({"summary_generation_tokens": tokens})


def test_positive_summary_generation_tokens_is_accepted():
    config = SourceCompactionConfig.model_validate({"summary_generation_tokens": 1})

    assert config.summary_generation_tokens == 1


# Requirement: ratios must be strictly between 0 and 1; a non-finite ratio is
# invalid.
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_non_finite_ratio_is_rejected(value):
    with pytest.raises(ValidationError):
        SourceCompactionConfig.model_validate({"pressure_trigger_ratio": value})

    with pytest.raises(ValidationError):
        SourceCompactionConfig.model_validate({"target_ratio": value})


# Requirement: the config is nested as ``source_compaction`` (alias
# ``sourceCompaction``) inside the agent defaults configuration model.
def test_agent_defaults_nests_source_compaction_defaults():
    defaults = AgentDefaults()

    assert isinstance(defaults.source_compaction, SourceCompactionConfig)
    assert defaults.source_compaction.pressure_trigger_ratio == 0.9
    assert defaults.source_compaction.target_ratio == 0.6
    assert defaults.source_compaction.summary_generation_tokens is None


@pytest.mark.parametrize(
    "key",
    [
        pytest.param("source_compaction", id="snake-case-nested-key"),
        pytest.param("sourceCompaction", id="camel-case-nested-key"),
    ],
)
def test_agent_defaults_accepts_nested_source_compaction_under_either_spelling(key):
    defaults = AgentDefaults.model_validate(
        {
            key: {
                "pressureTriggerRatio": 0.75,
                "targetRatio": 0.4,
                "summaryGenerationTokens": 8192,
            }
        }
    )

    assert defaults.source_compaction.pressure_trigger_ratio == 0.75
    assert defaults.source_compaction.target_ratio == 0.4
    assert defaults.source_compaction.summary_generation_tokens == 8192
