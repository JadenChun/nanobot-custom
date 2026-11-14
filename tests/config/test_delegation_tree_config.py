import pytest
from pydantic import ValidationError

from nanobot.config.schema import Config, DelegationTreeConfig


def test_delegation_tree_defaults_are_finite_and_distinct():
    limits = Config().agents.defaults.delegation_tree
    assert limits.max_depth == 2
    assert limits.max_live_children_per_parent == 2
    assert limits.max_live_descendants == 4
    assert limits.max_live_children_per_gateway == 4


@pytest.mark.parametrize(
    "payload",
    [
        {"maxDepth": 3},
        {"maxLiveChildrenPerParent": 3},
        {"maxLiveDescendants": 5},
        {"maxLiveChildrenPerGateway": 5},
        {"maxDepth": True},
        {"maxLiveChildrenPerParent": "2"},
    ],
)
def test_delegation_tree_limits_reject_unbounded_or_coerced_values(payload):
    with pytest.raises(ValidationError):
        DelegationTreeConfig.model_validate(payload)


def test_root_limit_must_hold_configured_parent_children():
    with pytest.raises(ValidationError):
        DelegationTreeConfig.model_validate(
            {"maxLiveChildrenPerParent": 2, "maxLiveDescendants": 1},
        )
