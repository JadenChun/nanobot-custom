"""Foreground source-compaction timeout settings are explicit and validated."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from nanobot.config.schema import AgentDefaults, Config


def test_source_compaction_defaults_and_config_aliases() -> None:
    defaults = AgentDefaults()
    assert defaults.source_compaction.idle_timeout_seconds == 300.0
    assert defaults.source_compaction.overall_timeout_seconds == 900.0

    config = Config.model_validate({
        "agents": {"defaults": {"sourceCompaction": {
            "idleTimeoutSeconds": 12.5,
            "overallTimeoutSeconds": 45.0,
        }}},
    })
    assert config.agents.defaults.source_compaction.idle_timeout_seconds == 12.5
    assert config.agents.defaults.source_compaction.overall_timeout_seconds == 45.0

    snake = Config.model_validate({
        "agents": {"defaults": {"source_compaction": {
            "idle_timeout_seconds": 7.0,
            "overall_timeout_seconds": 21.0,
        }}},
    })
    assert snake.agents.defaults.source_compaction.idle_timeout_seconds == 7.0
    assert snake.agents.defaults.source_compaction.overall_timeout_seconds == 21.0


@pytest.mark.parametrize("field", ["idleTimeoutSeconds", "overallTimeoutSeconds"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan, True])
def test_source_compaction_rejects_nonpositive_or_nonfinite_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"agents": {"defaults": {"sourceCompaction": {field: value}}}})


@pytest.mark.parametrize(
    ("camel", "snake"),
    [
        ("idleTimeoutSeconds", "idle_timeout_seconds"),
        ("overallTimeoutSeconds", "overall_timeout_seconds"),
    ],
)
def test_source_compaction_rejects_conflicting_aliases(camel: str, snake: str) -> None:
    with pytest.raises(ValidationError, match="Specify only one"):
        Config.model_validate({"agents": {"defaults": {"sourceCompaction": {
            camel: 300.0,
            snake: 301.0,
        }}}})
