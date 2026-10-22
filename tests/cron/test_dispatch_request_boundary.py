"""Raw constructor inputs retain checked, immutable cron dispatch state."""
import inspect
from dataclasses import FrozenInstanceError, fields

import pytest

from nanobot.cron.delivery_runtime import DispatchMessage, DispatchRequest, DispatchRequestError


@pytest.mark.parametrize("kind", ("wire_tuple", "wire_list", "typed_tuple", "mixed"))
@pytest.mark.parametrize("positional", (False, True))
def test_constructor_checks_each_raw_item_once_and_keeps_detached_state(monkeypatch, kind, positional):
    first_wire = {"content": "first", "media": []}
    second_wire = {"content": "second", "media": []}
    typed_first, typed_second = DispatchMessage("first"), DispatchMessage("second")
    inputs = {
        "wire_tuple": (first_wire, second_wire),
        "wire_list": [first_wire, second_wire],
        "typed_tuple": (typed_first, typed_second),
        "mixed": (first_wire, typed_second),
    }
    messages = inputs[kind]
    seen: list[object] = []
    original = DispatchMessage.from_value

    def from_value(cls, value: object) -> DispatchMessage:
        assert cls is DispatchMessage
        seen.append(value)
        return original(value)

    monkeypatch.setattr(DispatchMessage, "from_value", classmethod(from_value))
    if positional:
        request = DispatchRequest("a" * 64, "b" * 64, 1, "c" * 64, "d" * 64, "cli", "test", messages)
    else:
        request = DispatchRequest(
            outbox_id="a" * 64, destination_id="b" * 64, attempt=1,
            action_id="c" * 64, request_digest="d" * 64, channel="cli", to="test",
            messages=messages,
        )
    assert len(seen) == 2
    assert all(actual is expected for actual, expected in zip(seen, messages, strict=True))
    assert isinstance(request.messages, tuple)
    assert all(type(message) is DispatchMessage for message in request.messages)
    assert [message.content for message in request.messages] == ["first", "second"]
    assert all(message.media == () for message in request.messages)
    if kind == "typed_tuple":
        assert request.messages[0] is not typed_first
        assert request.messages[1] is not typed_second
    first_wire["content"] = "changed"
    second_wire["media"].append("changed")
    assert [message.content for message in request.messages] == ["first", "second"]
    assert all(message.media == () for message in request.messages)
    with pytest.raises(FrozenInstanceError):
        request.messages = ()


@pytest.mark.parametrize("messages,error", (
    (None, "messages must be an ordered sequence"),
    ("text", "messages must be an ordered sequence"),
    ([], "messages must contain at least one text message"),
    (({"content": "text", "media": [], "extra": "denied"},), "dispatch messages must have exactly content and media fields"),
    (({"content": False, "media": []},), "message content must be text"),
    (({"content": "text", "media": ["denied"]},), "delivery messages must not carry media"),
))
def test_constructor_keeps_existing_raw_input_refusals(messages, error):
    with pytest.raises(DispatchRequestError, match=error):
        DispatchRequest(
            outbox_id="a" * 64, destination_id="b" * 64, attempt=1,
            action_id="c" * 64, request_digest="d" * 64, channel="cli", to="test",
            messages=messages,
        )


def test_runtime_retains_generated_dataclass_constructor_and_concrete_stored_annotation():
    expected = ["outbox_id", "destination_id", "attempt", "action_id", "request_digest", "channel", "to", "messages", "message_thread_id"]
    signature = inspect.signature(DispatchRequest)
    assert list(signature.parameters) == expected
    assert all(parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in signature.parameters.values())
    assert all(signature.parameters[name].default is inspect.Parameter.empty for name in expected[:-1])
    assert signature.parameters["message_thread_id"].default is None
    assert DispatchRequest.__init__.__code__.co_filename == "<string>"
    assert [field.name for field in fields(DispatchRequest)] == expected
    assert fields(DispatchRequest)[-2].type == "tuple[DispatchMessage, ...]"
    assert signature.parameters["messages"].annotation == "tuple[DispatchMessage, ...]"
