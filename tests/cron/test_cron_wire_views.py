"""Stored cron/tool wire typing must preserve its actual dynamic behavior."""

from collections.abc import Mapping
from copy import deepcopy
from unittest.mock import Mock

import pytest

from nanobot.agent.tools.cron import CronTool
from nanobot.cron import types
from nanobot.cron.service import CronService
from nanobot.cron.types import CronDestination, CronJobState, CronPayload, CronRunRecord, CronStore


@pytest.mark.parametrize("container", [list, tuple, iter])
def test_destinations_keep_original_iterable_and_existing_record_semantics(container):
    existing = CronDestination("telegram", "42")
    raw = {"channel": "cli", "to": "direct"}
    data = {"additionalDestinations": container([existing, raw])}
    parsed = CronPayload.from_store_dict(data)
    assert parsed.additional_destinations == [existing, CronDestination("cli", "direct")]
    assert parsed.additional_destinations[0] is existing
    assert raw == {"channel": "cli", "to": "direct"}


@pytest.mark.parametrize("value", [None, [], (), {}, "", 0, False])
def test_falsey_wire_defaults_keep_empty_destinations_and_history(value):
    assert CronPayload.from_store_dict({"additionalDestinations": value}).additional_destinations == []
    assert CronJobState.from_store_dict({"runHistory": value}).run_history == []


def test_dynamic_destination_fields_are_not_newly_validated_or_coerced():
    payload = CronPayload.from_store_dict({"additionalDestinations": [{"channel": 7, "to": None}]})
    assert payload.additional_destinations[0].channel == 7
    assert payload.additional_destinations[0].to is None


class WireFaultError(RuntimeError):
    pass


class RecordedDestination(Mapping):
    def __init__(self, fault=None):
        self.reads = []
        self.fault = fault

    def __iter__(self):
        self.reads.append("keys")
        return iter(("channel", "to"))

    def __len__(self):
        return 2

    def __getitem__(self, key):
        self.reads.append(key)
        if self.fault is not None:
            raise self.fault
        return {"channel": "telegram", "to": "42"}[key]


def test_destination_kwargs_keep_mapping_read_order():
    wire = RecordedDestination()
    parsed = CronPayload.from_store_dict({"additionalDestinations": [wire]})
    assert parsed.additional_destinations == [CronDestination("telegram", "42")]
    assert wire.reads == ["keys", "channel", "to"]


def test_destination_mapping_fault_propagates_same_instance():
    fault = WireFaultError("sentinel")
    wire = RecordedDestination(fault)
    with pytest.raises(WireFaultError) as captured:
        CronPayload.from_store_dict({"additionalDestinations": [wire]})
    assert captured.value is fault
    assert wire.reads == ["keys", "channel"]


@pytest.mark.parametrize("record", [
    {"channel": "cli"}, {"channel": "cli", "to": "direct", "extra": True},
    {1: "cli", "to": "direct"}, "not a mapping", None,
])
def test_destination_kwargs_keep_native_constructor_refusals(record):
    original = deepcopy(record)
    with pytest.raises(TypeError):
        CronPayload.from_store_dict({"additionalDestinations": [record]})
    assert record == original


def test_destination_alias_precedence_and_default_allocation_remain(monkeypatch):
    assert CronPayload.from_store_dict({
        "additionalDestinations": [],
        "additional_destinations": [{"channel": "cli", "to": "ignored"}],
    }).additional_destinations == []
    original_get = types.get_camel_snake
    defaults = []

    def observe(data, camel, snake, default=None):
        if camel == "additionalDestinations":
            defaults.append(default)
        return original_get(data, camel, snake, default)

    monkeypatch.setattr(types, "get_camel_snake", observe)
    first = CronPayload.from_store_dict({})
    second = CronPayload.from_store_dict({})
    assert defaults == [[], []] and defaults[0] is not defaults[1]
    assert first.additional_destinations is not second.additional_destinations


def test_history_preserves_existing_record_and_decodes_aliases_without_mutation():
    existing = CronRunRecord(run_at_ms=1, status="ok")
    raw = {"runAtMs": "2", "run_at_ms": 99, "status": "error", "durationMs": "3"}
    data = {"runHistory": [existing, raw], "run_history": ["ignored"]}
    original = deepcopy(data)
    parsed = CronJobState.from_store_dict(data)
    assert parsed.run_history[0] is existing
    assert parsed.run_history[1] == CronRunRecord(run_at_ms=2, status="error", duration_ms=3)
    assert data == original


@pytest.mark.parametrize("value", [True, {"record": "bad"}, ("bad",), object()])
def test_non_list_history_keeps_exact_defensive_refusal(value):
    with pytest.raises(ValueError, match="^cron run history must be a list$"):
        CronJobState.from_store_dict({"runHistory": value})


@pytest.mark.parametrize("value", [None, "bad", 1, []])
def test_invalid_history_record_keeps_exact_defensive_refusal(value):
    with pytest.raises(ValueError, match="^cron run history contains an invalid record$"):
        CronJobState.from_store_dict({"runHistory": [value]})


def test_list_subclass_history_keeps_iterator_fault():
    fault = WireFaultError("sentinel")

    class BrokenList(list):
        def __iter__(self):
            raise fault

    with pytest.raises(WireFaultError) as captured:
        CronJobState.from_store_dict({"runHistory": BrokenList([{}])})
    assert captured.value is fault


@pytest.mark.parametrize("value", [None, object(), {}, {"jobs": None}, {"jobs": ()}, {"jobs": ""}])
def test_store_keeps_outer_object_list_refusal(value):
    with pytest.raises(ValueError, match="^legacy jobs store must be an object with a jobs list$"):
        CronStore.from_store_dict(value)


@pytest.mark.parametrize("value", [None, object(), [], "bad"])
def test_store_keeps_invalid_job_refusal(value):
    with pytest.raises(ValueError, match="^legacy jobs store contains an invalid job$"):
        CronStore.from_store_dict({"jobs": [value]})


def test_store_keeps_repeated_jobs_reads_and_native_key_fault():
    reads = []

    class Store(dict):
        def __contains__(self, key):
            reads.append(("contains", key))
            return super().__contains__(key)

        def __getitem__(self, key):
            reads.append(("read", key))
            return super().__getitem__(key)

        def get(self, key, default=None):
            reads.append(("get", key))
            return super().get(key, default)

    data = Store(jobs=[{}])
    with pytest.raises(KeyError) as captured:
        CronStore.from_store_dict(data)
    assert captured.value.args == ("id",)
    assert reads == [("contains", "jobs"), ("read", "jobs"), ("get", "version"), ("read", "jobs")]
    assert data == {"jobs": [{}]}


@pytest.mark.parametrize("field,value,fragment", [
    ("channel", " \n\t ", "channel must be nonblank"),
    ("to", " \n\t ", "to must be nonblank"),
    ("channel", "é" * 257, "channel exceeds the 512-byte limit"),
    ("to", "é" * 257, "to exceeds the 512-byte limit"),
])
def test_tool_wire_fields_keep_blank_and_utf8_refusals(field, value, fragment):
    service = Mock(spec=CronService)
    params = {"action": "list", "additional_destinations": [{"channel": "cli", "to": "direct", field: value}]}
    original = deepcopy(params)
    errors = CronTool(service).validate_params(params)
    assert f"additional_destinations[0] {fragment}" in errors
    assert params == original
    assert service.mock_calls == []


def test_tool_valid_wire_keeps_order_values_and_no_service_access():
    service = Mock(spec=CronService)
    params = {"action": "list", "additional_destinations": [
        {"channel": "cli", "to": " first "}, {"channel": "telegram", "to": "é" * 256},
    ]}
    original = deepcopy(params)
    assert CronTool(service).validate_params(params) == []
    assert params == original
    assert service.mock_calls == []
