"""Preserve the dynamic keyword boundary used to load canonical run records."""

from collections.abc import Iterator, KeysView, Mapping
from datetime import datetime, timezone

import pytest

from nanobot.agent.turn import DeliveryTarget, RunRecord, ScheduledTurnLink


class ObservedMapping(Mapping[object, object]):
    def __init__(
        self,
        values: dict[object, object],
        events: list[tuple[str, str, object]],
        label: str,
        fault: BaseException | None = None,
        fault_at: str = "get",
    ) -> None:
        self.values = values
        self.events = events
        self.label = label
        self.fault = fault
        self.fault_at = fault_at

    def record(self, operation: str, key: object = None) -> None:
        self.events.append((self.label, operation, key))
        if self.fault is not None and operation == self.fault_at:
            raise self.fault

    def keys(self) -> KeysView[object]:
        self.record("keys")
        return super().keys()

    def __iter__(self) -> Iterator[object]:
        self.record("iter")
        yield from self.values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, key: object) -> object:
        self.record("get", key)
        return self.values[key]


@pytest.mark.parametrize(
    ("field", "values", "attribute"),
    [
        ("delivery_target", {"channel": "telegram", "to": "42"}, "channel"),
        ("scheduled_link", {"job_id": "job", "instruction": "private"}, "instruction"),
    ],
)
def test_mapping_is_copied_once_in_keyword_order(field, values, attribute) -> None:
    events: list[tuple[str, str, object]] = []
    raw_value = object()
    values[attribute] = raw_value
    mapping = ObservedMapping(values, events, field)

    record = RunRecord.from_dict({"run_id": "run", field: mapping})

    assert events == [
        (field, "keys", None),
        (field, "iter", None),
        *((field, "get", key) for key in values),
    ]
    assert getattr(getattr(record, field), attribute) is raw_value
    values[attribute] = "changed later"
    assert getattr(getattr(record, field), attribute) is raw_value


def test_delivery_conversion_precedes_scheduled_conversion() -> None:
    events: list[tuple[str, str, object]] = []
    delivery = ObservedMapping({"to": "42"}, events, "delivery")
    scheduled = ObservedMapping({"schedule_id": "job"}, events, "scheduled")

    record = RunRecord.from_dict(
        {"run_id": "run", "delivery_target": delivery, "scheduled_link": scheduled}
    )

    assert events == [
        ("delivery", "keys", None),
        ("delivery", "iter", None),
        ("delivery", "get", "to"),
        ("scheduled", "keys", None),
        ("scheduled", "iter", None),
        ("scheduled", "get", "schedule_id"),
    ]
    assert record.delivery_target == DeliveryTarget(channel="", to="42")
    assert record.scheduled_job_id == "job"


@pytest.mark.parametrize("field", ["delivery_target", "scheduled_link"])
@pytest.mark.parametrize("key", [1, None, ("channel",)])
def test_nonstring_keyword_refusal_is_not_filtered_or_coerced(field, key) -> None:
    events: list[tuple[str, str, object]] = []
    mapping = ObservedMapping({key: "raw"}, events, field)

    with pytest.raises(TypeError, match="keywords must be strings"):
        RunRecord.from_dict({"run_id": "run", field: mapping})

    assert events == [(field, "keys", None), (field, "iter", None), (field, "get", key)]


@pytest.mark.parametrize("field", ["delivery_target", "scheduled_link"])
def test_unknown_keyword_retains_constructor_refusal(field) -> None:
    events: list[tuple[str, str, object]] = []
    mapping = ObservedMapping({"unexpected": "raw"}, events, field)

    with pytest.raises(TypeError, match="unexpected keyword argument 'unexpected'"):
        RunRecord.from_dict({"run_id": "run", field: mapping})

    assert events == [
        (field, "keys", None), (field, "iter", None), (field, "get", "unexpected")
    ]


@pytest.mark.parametrize("field", ["delivery_target", "scheduled_link"])
@pytest.mark.parametrize("operation", ["keys", "iter", "get"])
@pytest.mark.parametrize("error_class", [RuntimeError, KeyboardInterrupt])
def test_mapping_fault_retains_exception_identity_and_short_circuit(
    field, operation, error_class
) -> None:
    events: list[tuple[str, str, object]] = []
    fault = error_class("mapping fault")
    keyword = "channel" if field == "delivery_target" else "job_id"
    failing = ObservedMapping({keyword: "raw"}, events, field, fault, operation)
    later = ObservedMapping({"job_id": "later"}, events, "later")
    payload: dict[str, object] = {"run_id": "run", field: failing}
    if field == "delivery_target":
        payload["scheduled_link"] = later

    with pytest.raises(error_class) as caught:
        RunRecord.from_dict(payload)

    assert caught.value is fault
    expected = [(field, "keys", None), (field, "iter", None), (field, "get", keyword)]
    assert events == expected[: ["keys", "iter", "get"].index(operation) + 1]


@pytest.mark.parametrize("alias", ["chat_id", "to", "recipient"])
def test_delivery_aliases_and_wire_projection_are_preserved(alias) -> None:
    record = RunRecord.from_dict(
        {"run_id": "run", "delivery_target": {"channel": "telegram", alias: "42"}}
    )

    assert record.delivery_target == DeliveryTarget(
        channel="telegram", chat_id="42", to="42", recipient="42"
    )
    assert record.to_dict()["delivery_target"] == {
        "channel": "telegram", "chat_id": "42", "thread_id": None, "message_id": None
    }


@pytest.mark.parametrize("alias", ["job_id", "schedule_id"])
def test_schedule_normalization_and_private_instruction_projection_are_preserved(alias) -> None:
    destination = DeliveryTarget(channel="telegram", chat_id="43")
    destinations = [destination]
    instant = datetime(2026, 10, 7, 9, 0, tzinfo=timezone.utc)
    record = RunRecord.from_dict({
        "run_id": "run",
        "scheduled_link": {
            alias: "job", "scheduled_at": instant, "additional_destinations": destinations,
            "instruction": "private instruction", "visible_session_key": "personal",
        },
    })

    assert record.scheduled_link == ScheduledTurnLink(
        job_id="job", schedule_id="job", scheduled_at=instant.isoformat(),
        additional_destinations=(destination,), instruction="private instruction",
        visible_session_key="personal",
    )
    assert record.scheduled_link.additional_destinations[0] is destination
    destinations.clear()
    assert record.scheduled_link.additional_destinations == (destination,)
    assert record.scheduled_job_id == "job"
    assert record.to_dict()["scheduled_link"] == {
        "job_id": "job", "job_name": "", "visible_session_key": "personal",
        "occurrence_id": None, "scheduled_at": instant.isoformat(),
    }


def test_existing_typed_values_are_not_reconstructed() -> None:
    delivery = DeliveryTarget(channel="telegram", chat_id="42")
    scheduled = ScheduledTurnLink(job_id="job")
    record = RunRecord.from_dict(
        {"run_id": "run", "delivery_target": delivery, "scheduled_link": scheduled}
    )

    assert record.delivery_target is delivery
    assert record.scheduled_link is scheduled


def test_nonmapping_delivery_value_is_not_reinterpreted() -> None:
    raw = [("channel", "telegram")]
    record = RunRecord.from_dict({"run_id": "run", "delivery_target": raw})

    assert record.delivery_target is raw


def test_failed_delivery_constructor_does_not_read_scheduled_mapping() -> None:
    events: list[tuple[str, str, object]] = []
    scheduled = ObservedMapping({"job_id": "job"}, events, "scheduled")

    with pytest.raises(TypeError, match="unexpected keyword argument 'unexpected'"):
        RunRecord.from_dict({
            "run_id": "run", "delivery_target": {"unexpected": "raw"},
            "scheduled_link": scheduled,
        })

    assert events == []
