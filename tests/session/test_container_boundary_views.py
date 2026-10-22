"""Preserve guarded container identity, native faults and repeated metadata reads."""

from __future__ import annotations

from collections import UserDict
from pathlib import Path

import pytest

import nanobot.cron.occurrence as occurrence
import nanobot.session.artifact_factory as artifact_factory
import nanobot.session.source_discovery as discovery
from nanobot.session.manager import JsonlSessionStore
from tests.session.test_source_discovery import _source_session


@pytest.mark.parametrize("value", [None, False, 1, "text", (), [], set(), object(), UserDict()])
def test_strict_object_retains_non_dict_refusal(value):
    assert discovery._strict_object(value) is None


@pytest.mark.parametrize("subclass", [False, True])
def test_strict_object_keeps_arbitrary_keys_and_exact_identity(subclass):
    class NoInspectionDict(dict):
        def __iter__(self):
            pytest.fail("dictionary projection iterated input")

        def get(self, *_args, **_kwargs):
            pytest.fail("dictionary projection read fields")

        def __repr__(self):
            pytest.fail("dictionary projection disclosed input")

    key = object()
    value = {key: object(), "nested": [object()]}
    if subclass:
        value = NoInspectionDict(value)
    assert discovery._strict_object(value) is value


@pytest.mark.parametrize("value", [None, False, 1, "text", set(), {}, UserDict()])
def test_path_items_does_not_iterate_non_sequence_values(value):
    assert artifact_factory._path_items(value) == ()


@pytest.mark.parametrize("container", [list, tuple])
def test_path_items_keeps_filter_order_duplicates_identity_and_one_iteration(container):
    events = []

    class IdentityPath(type(Path())):
        def __fspath__(self):
            pytest.fail("path projection resolved or converted a path")

    class ObservedList(list):
        def __iter__(self):
            events.append("iter")
            for item in super().__iter__():
                events.append(item)
                yield item

    class ObservedTuple(tuple):
        def __iter__(self):
            events.append("iter")
            for item in super().__iter__():
                events.append(item)
                yield item

    first, second = IdentityPath("first"), IdentityPath("second")
    sentinel = object()
    items = [first, "not-a-path", None, second, sentinel, first]
    value = ObservedList(items) if container is list else ObservedTuple(items)
    result = artifact_factory._path_items(value)
    assert len(result) == 3
    assert result[0] is first and result[1] is second and result[2] is first
    assert events[0] == "iter"
    assert len(events) == len(items) + 1
    assert all(observed is expected for observed, expected in zip(events[1:], items, strict=True))


@pytest.mark.parametrize("container", [list, tuple])
def test_path_items_propagates_the_original_iterator_exception(container):
    failure = RuntimeError("iteration fault")

    class BrokenList(list):
        def __iter__(self):
            raise failure

    class BrokenTuple(tuple):
        def __iter__(self):
            raise failure

    value = BrokenList() if container is list else BrokenTuple()
    with pytest.raises(RuntimeError) as caught:
        artifact_factory._path_items(value)
    assert caught.value is failure


@pytest.mark.parametrize("numeric_keys", [False, True])
def test_json_mapping_preserves_decoder_key_construction_and_detachment(numeric_keys):
    key = 7 if numeric_keys else "key"
    nested = {"values": [1, {"text": "é\n"}]}
    value = {key: nested}
    result = occurrence._copy_json_mapping(value, label="view")
    assert result == {str(key): nested}
    assert result is not value
    assert result[str(key)] is not nested
    result[str(key)]["values"][1]["text"] = "changed"
    assert nested["values"][1]["text"] == "é\n"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), {object()}])
def test_json_mapping_retains_native_failure_wrapping(value):
    with pytest.raises(occurrence.CronOccurrenceRuntimeError) as caught:
        occurrence._copy_json_mapping({"value": value}, label="view")
    assert str(caught.value) == "view must contain finite JSON data"
    assert isinstance(caught.value.__cause__, (TypeError, ValueError))


def test_json_mapping_cycle_retains_failure_before_decode(monkeypatch):
    value = {}
    value["cycle"] = value
    import json

    monkeypatch.setattr(json, "loads", lambda *_args, **_kwargs: pytest.fail("cycle decoded"))
    with pytest.raises(occurrence.CronOccurrenceRuntimeError) as caught:
        occurrence._copy_json_mapping(value, label="view")
    assert str(caught.value) == "view must contain finite JSON data"
    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize("decoded", [None, [], "text"])
def test_json_mapping_retains_non_dict_decoder_refusal(monkeypatch, decoded):
    import json

    monkeypatch.setattr(json, "loads", lambda *_args, **_kwargs: decoded)
    with pytest.raises(occurrence.CronOccurrenceRuntimeError) as caught:
        occurrence._copy_json_mapping({"value": 1}, label="view")
    assert str(caught.value) == "view must be a JSON object"
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, RecursionError])
def test_json_mapping_keeps_decoder_exception_identity_and_original_catch(monkeypatch, error_type):
    import json

    failure = error_type("decode fault")
    calls = []

    def fail_decode(encoded):
        calls.append(encoded)
        raise failure

    monkeypatch.setattr(json, "loads", fail_decode)
    if error_type is RuntimeError:
        with pytest.raises(RuntimeError) as caught:
            occurrence._copy_json_mapping({"z": "é", "a": 1}, label="view")
        assert caught.value is failure
    else:
        with pytest.raises(occurrence.CronOccurrenceRuntimeError) as caught:
            occurrence._copy_json_mapping({"z": "é", "a": 1}, label="view")
        assert str(caught.value) == "view must contain finite JSON data"
        assert caught.value.__cause__ is failure
    assert calls == ['{"a":1,"z":"é"}']


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_values,accepted", [
    ([1, 1], True),
    ([1, 2], False),
    ([True], False),
    ([1.0], False),
    (["1"], False),
    ([None], False),
    ([TypeError("first metadata fault")], False),
    ([1, TypeError("second metadata fault")], False),
])
async def test_discovery_metadata_keeps_two_reads_and_short_circuit(
    tmp_path, monkeypatch, schema_values, accepted,
):
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    key = "source:views"
    _source_session(backend, key)
    cursor = discovery.SourceDiscoveryCursor(
        backend, limits=discovery.SourceDiscoveryLimits(100, 100, 2000, 128),
    )
    events = []

    class Metadata(dict):
        def get(self, name, default=None):
            assert name == "source_schema_version"
            events.append("schema")
            value = schema_values[len(events) - 4]
            if isinstance(value, Exception):
                raise value
            return value

    class Header(dict):
        def get(self, name, default=None):
            events.append(name)
            return super().get(name, default)

    header = Header(_type="metadata", key=key, metadata=Metadata())
    try:
        with monkeypatch.context() as patch:
            patch.setattr(discovery.json, "loads", lambda *_args, **_kwargs: header)
            result = await cursor.scan_once()
        assert result.keys == ((key,) if accepted else ())
        assert events == ["_type", "key", "metadata", *(["schema"] * len(schema_values))]
    finally:
        await cursor.aclose()
