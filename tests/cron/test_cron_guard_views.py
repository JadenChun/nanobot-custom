"""Dormant cron guards reject malformed inputs before using authorities."""

from types import SimpleNamespace

import pytest

from nanobot.agent.turn import TurnResult
from nanobot.cron.callback_store import CronCallbackStore
from nanobot.cron.canonical_execution import execute_claimed_fresh_text
from nanobot.cron.canonical_result import complete_text_cron_result
from nanobot.cron.definition_store import CronDefinitionStore
from nanobot.cron.occurrence import CronOccurrenceRuntime
from nanobot.cron.occurrence_store import OccurrenceClaim


@pytest.mark.parametrize("value", [None, object(), "opaque", SimpleNamespace()])
@pytest.mark.parametrize("which", ["definition", "occurrence"])
def test_occurrence_constructor_refuses_before_namespace_reads(value, which):
    definition = value if which == "definition" else object.__new__(CronDefinitionStore)
    occurrence = None if which == "definition" else value
    expected = f"{which}_store must be a Cron{which.title()}Store"
    with pytest.raises(TypeError) as error:
        CronOccurrenceRuntime(definition, occurrence)
    assert str(error.value) == expected


@pytest.mark.parametrize("value", [None, 0, False, object(), ""])
def test_invalid_job_id_refuses_before_store_replay(value):
    runtime = object.__new__(CronOccurrenceRuntime)
    with pytest.raises(ValueError) as error:
        runtime._committed_definition(value)
    assert str(error.value) == "job_id must be a non-empty string"


@pytest.mark.parametrize("value", [None, object(), "opaque", SimpleNamespace()])
def test_callback_refuses_before_identity_and_lock(monkeypatch, value):
    store = object.__new__(CronCallbackStore)
    reads = []

    def forbidden(*args, **kwargs):
        reads.append("authority")
        raise AssertionError("malformed observation reached an authority")

    monkeypatch.setattr(CronCallbackStore, "callback_id", forbidden)
    monkeypatch.setattr(CronCallbackStore, "_locked", forbidden)
    with pytest.raises(TypeError) as error:
        store.publish(value)
    assert str(error.value) == "callback publication requires a typed observation"
    assert reads == []


@pytest.mark.parametrize("value", [None, object(), "opaque", SimpleNamespace()])
@pytest.mark.parametrize("which", ["claim", "result"])
def test_settlement_refuses_before_claim_result_or_service_reads(value, which):
    claim = value if which == "claim" else object.__new__(OccurrenceClaim)
    result = TurnResult() if which == "claim" else value
    with pytest.raises(TypeError) as error:
        complete_text_cron_result(object(), claim, result, object())
    assert str(error.value) == "cron result settlement requires a claim and canonical TurnResult"


@pytest.mark.parametrize("value", [None, object(), "opaque", SimpleNamespace()])
async def test_execution_retains_authority_check_order_before_claim_refusal(value):
    reads = []

    class ServiceProbe:
        def __getattribute__(self, name):
            reads.append(name)
            if name in {"occurrence_store", "completion_projector", "delivery_runtime"}:
                return object()
            raise AssertionError("malformed claim reached execution")

    with pytest.raises(TypeError) as error:
        await execute_claimed_fresh_text(ServiceProbe(), object(), value)
    assert str(error.value) == "claimed cron execution requires an exact occurrence claim"
    assert reads == ["occurrence_store", "completion_projector", "delivery_runtime"]
