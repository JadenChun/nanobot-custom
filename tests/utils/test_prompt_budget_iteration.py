"""Preservation checks for the prompt reducer's two iteration spellings."""

import copy

import pytest

from nanobot.utils import prompt_budget


@pytest.mark.parametrize(
    "message",
    [
        {},
        {"tool_calls": None},
        {"tool_calls": False},
        {"tool_calls": 0},
        {"tool_calls": ""},
        {"tool_calls": []},
        {"tool_calls": ()},
        {"tool_calls": {}},
        {"tool_calls": "characters"},
        {"tool_calls": {"id": "a key is not a call"}},
        {"tool_calls": [None, "noise", 7, [], ()]},
    ],
)
def test_empty_or_non_dict_calls_keep_no_ids(message: dict[str, object]) -> None:
    original = copy.deepcopy(message)
    assert prompt_budget._tool_call_ids(message) == []
    assert message == original


@pytest.mark.parametrize(
    ("call_id", "expected"),
    [
        (None, []),
        ("", []),
        (0, []),
        (False, []),
        ([], []),
        ({}, []),
        ("call-a", ["call-a"]),
        (42, ["42"]),
        (True, ["True"]),
        ({"nested": 1}, ["{'nested': 1}"]),
    ],
)
def test_truthy_ids_keep_existing_string_conversion(
    call_id: object, expected: list[str],
) -> None:
    message = {"tool_calls": [{"id": call_id}]}
    original = copy.deepcopy(message)
    assert prompt_budget._tool_call_ids(message) == expected
    assert message == original


@pytest.mark.parametrize("container", ["list", "tuple", "iterator"])
def test_iterable_calls_keep_order_and_duplicate_ids(container: str) -> None:
    calls = [{"id": "a"}, "noise", {"id": "a"}, {"other": "ignored"}, {"id": 99}]
    original = copy.deepcopy(calls)
    values = tuple(calls) if container == "tuple" else iter(calls) if container == "iterator" else calls
    assert prompt_budget._tool_call_ids({"tool_calls": values}) == ["a", "a", "99"]
    assert calls == original
    if container == "iterator":
        assert list(values) == []


@pytest.mark.parametrize("calls", [1, True, 2.5, object()])
def test_truthy_non_iterable_calls_keep_type_error(calls: object) -> None:
    with pytest.raises(TypeError):
        prompt_budget._tool_call_ids({"tool_calls": calls})


def test_falsy_call_container_is_not_iterated() -> None:
    class FalsyCalls:
        def __bool__(self) -> bool:
            return False

        def __iter__(self):
            raise AssertionError("a falsy tool-call payload must not be iterated")

    assert prompt_budget._tool_call_ids({"tool_calls": FalsyCalls()}) == []


class _ContentCounter:
    @staticmethod
    def estimate_prompt_tokens(messages, tools, model):
        return sum(len(message.get("content", "")) for message in messages), "content"


@pytest.mark.parametrize("retain_count", [0, 1, 2, 3])
def test_no_turn_context_trims_oldest_first_and_preserves_system(retain_count: int) -> None:
    system = {"role": "system", "content": "policy", "metadata": {"source": "system"}}
    managed = [
        {"role": "user", "content": "[Past Knowledge] old"},
        {"role": "user", "content": "[Recent Scheduled Runs] middle"},
        {"role": "user", "content": "[Recent Scheduled Run] newest"},
    ]
    retained = managed[-retain_count:] if retain_count else []
    expected = [system, *retained]
    limit = sum(len(message["content"]) for message in expected)
    messages = [system, *managed]
    original = copy.deepcopy(messages)
    reduced = prompt_budget.reduce_messages_to_budget(
        messages, _ContentCounter(), "offline-counter", None,
        prompt_budget.PromptBudget(total_tokens=limit, safety_buffer=0),
    )
    assert reduced == expected
    assert messages == original
    assert reduced is not messages
    assert all(item is not source for item, source in zip(reduced, expected))
    reduced[0]["metadata"]["source"] = "detached"
    assert messages == original


def test_irreducible_system_policy_remains_required_and_error_is_sanitized() -> None:
    messages = [
        {"role": "system", "content": "private-system-policy"},
        {"role": "user", "content": "[Past Knowledge] private-managed-content"},
    ]
    original = copy.deepcopy(messages)
    with pytest.raises(prompt_budget.PromptBudgetExceeded) as caught:
        prompt_budget.reduce_messages_to_budget(
            messages, _ContentCounter(), "offline-counter", None,
            prompt_budget.PromptBudget(total_tokens=1, safety_buffer=0),
        )
    assert str(caught.value) == "Prompt exceeds the configured context budget."
    assert "private-system-policy" not in str(caught.value)
    assert "private-managed-content" not in str(caught.value)
    assert messages == original
