from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from telegram.error import TimedOut

from nanobot.bus.queue import MessageBus
from nanobot.channels.delivery_contract import (
    ChannelDeliveryPreflightError,
    ChannelDeliveryRejectedError,
    ChannelTextPartReceipt,
    ChannelTextReceipt,
    PartialChannelTextDeliveryError,
    validate_receipt_binding,
)
from nanobot.channels.telegram.runtime import TelegramChannel, TelegramConfig
from nanobot.cron.action_store import CronActionStore
from nanobot.cron.channel_sender import CronChannelSender
from nanobot.cron.completion import CronCompletionProjector
from nanobot.cron.definition_store import CronDefinitionStore
from nanobot.cron.delivery_runtime import (
    ConfirmedDelivered,
    ConfirmedNotSent,
    CronDeliveryRuntime,
    CronDeliverySettlementError,
    DispatchMessage,
    DispatchRequest,
    PartialBundleOutcome,
    PermanentNotSent,
    UnknownOutcome,
)
from nanobot.cron.delivery_store import CronDeliveryOutboxStore, DeliveryOutboxPersistenceError
from nanobot.cron.occurrence_store import CronOccurrenceStore
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule


def _request(**changes: object) -> DispatchRequest:
    values = {
        "outbox_id": "0" * 64,
        "destination_id": "1" * 64,
        "attempt": 1,
        "action_id": "2" * 64,
        "request_digest": "3" * 64,
        "channel": "telegram",
        "to": "123",
        "messages": (DispatchMessage("first"), DispatchMessage("second")),
    }
    values.update(changes)
    return DispatchRequest(**values)


def _receipt(request: DispatchRequest) -> ChannelTextReceipt:
    return ChannelTextReceipt(
        channel=request.channel,
        target=request.to,
        action_id=request.action_id,
        request_digest=request.request_digest,
        parts=(
            ChannelTextPartReceipt(0, 0, 0, 8, 123),
            ChannelTextPartReceipt(1, 1, 0, 9, 123),
        ),
    )


def test_receipt_rejects_omitted_or_noncontiguous_evidence() -> None:
    invalid_parts = (
        (ChannelTextPartReceipt(0, 1, 0, 1, 123),),
        (
            ChannelTextPartReceipt(0, 0, 0, 1, 123),
            ChannelTextPartReceipt(1, 2, 0, 2, 123),
        ),
        (
            ChannelTextPartReceipt(0, 0, 0, 1, 123),
            ChannelTextPartReceipt(1, 0, 2, 2, 123),
        ),
        (
            ChannelTextPartReceipt(0, 0, 0, 1, 123),
            ChannelTextPartReceipt(1, 0, 1, 1, 123),
        ),
    )
    for parts in invalid_parts:
        with pytest.raises(ValueError):
            ChannelTextReceipt(
                "telegram", "123", "a" * 64, "b" * 64, parts
            )


def test_valid_split_multi_message_receipt_is_complete_and_partial_prefix_is_valid() -> None:
    parts = (
        ChannelTextPartReceipt(0, 0, 0, 10, 123),
        ChannelTextPartReceipt(1, 0, 1, 11, 123),
        ChannelTextPartReceipt(2, 1, 0, 12, 123),
    )
    receipt = ChannelTextReceipt(
        "telegram", "123", "a" * 64, "b" * 64, parts
    )
    binding = {
        "channel": "telegram",
        "target": "123",
        "action_id": "a" * 64,
        "request_digest": "b" * 64,
        "message_count": 2,
    }
    validate_receipt_binding(receipt, **binding)
    prefix = ChannelTextReceipt(
        "telegram", "123", "a" * 64, "b" * 64, parts[:2]
    )
    validate_receipt_binding(prefix, **binding, require_complete=False)


async def test_sender_resolves_exact_channel_and_forwards_frozen_request() -> None:
    request = _request()

    class Channel:
        async def send_text_with_receipt(
            self, target: str, messages: tuple[str, ...], *, action_id: str, request_digest: str
        ) -> ChannelTextReceipt:
            assert (target, messages, action_id, request_digest) == (
                "123", ("first", "second"), "2" * 64, "3" * 64
            )
            return _receipt(request)

    resolver_calls: list[str] = []

    def resolver(name: str) -> object:
        resolver_calls.append(name)
        return Channel()

    outcome = await CronChannelSender(resolver).send(request)
    assert isinstance(outcome, ConfirmedDelivered)
    assert outcome.receipt == _receipt(request).to_json()
    assert resolver_calls == ["telegram"]


@pytest.mark.parametrize(
    ("channel", "expected"),
    [(None, PermanentNotSent("unsupported_receipt_transport")), (object(), PermanentNotSent("unsupported_receipt_transport"))],
)
async def test_missing_or_unsupported_channel_fails_closed(channel: object, expected: object) -> None:
    assert await CronChannelSender(lambda name: channel).send(_request()) == expected


async def test_sender_maps_known_channel_errors() -> None:
    class Channel:
        def __init__(self, error: Exception) -> None:
            self.error = error

        async def send_text_with_receipt(self, *args: object, **kwargs: object) -> object:
            raise self.error

    request = _request()
    assert await CronChannelSender(lambda name: Channel(ChannelDeliveryPreflightError("offline"))).send(request) == ConfirmedNotSent("offline")
    assert await CronChannelSender(lambda name: Channel(ChannelDeliveryRejectedError("rejected"))).send(request) == PermanentNotSent("rejected")
    partial = PartialChannelTextDeliveryError(_receipt(request))
    assert await CronChannelSender(lambda name: Channel(partial)).send(request) == PartialBundleOutcome()


async def test_mismatched_or_malformed_receipt_propagates_unknown() -> None:
    request = _request()

    class Channel:
        async def send_text_with_receipt(self, *args: object, **kwargs: object) -> ChannelTextReceipt:
            return replace(_receipt(request), target="wrong")

    with pytest.raises(ValueError):
        await CronChannelSender(lambda name: Channel()).send(request)


def _prepared_runtime(
    tmp_path: Path, *, messages: list[dict[str, object]] | None = None
) -> tuple[CronDeliveryRuntime, object]:
    root = tmp_path / "cron"
    namespace = "cron:channel-sender-tests"
    definitions = CronDefinitionStore(CronActionStore(root, namespace=namespace))
    occurrences = CronOccurrenceStore(root / "occurrences", namespace=namespace)
    outbox = CronDeliveryOutboxStore(root / "outbox", namespace=namespace)
    service = CronService(
        root / "jobs.json",
        definition_store=definitions,
        occurrence_store=occurrences,
        completion_projector=CronCompletionProjector(occurrences, root / "runs"),
    )
    runtime = CronDeliveryRuntime(outbox, occurrence_store=occurrences)
    job = service.add_job(
        "delivery",
        CronSchedule(kind="every", every_ms=60_000),
        "computed",
        deliver=True,
        channel="telegram",
        to="123",
    )
    occurrence = service.claim_scheduled_occurrence(
        job.id,
        1_700_000_000_000,
        owner="worker",
        expected_job_instance_id=job.job_instance_id,
        expected_definition_revision=job.definition_revision,
    )
    with occurrences.acquire_execution_fence(occurrence) as fence:
        occurrences.begin_execution(occurrence, fence)
        service.complete_occurrence(
            occurrence,
            {
                "status": "ok",
                "result": "computed",
                "delivery": {
                    "version": 1,
                    "deliver": True,
                    "messages": messages or [{"content": "hello", "media": []}],
                },
            },
            fence=fence,
        )
    record = runtime.prepare_completed(occurrence.occurrence_id)
    destination = record.destinations[0]
    claim = runtime.claim(
        record.outbox_id, destination.destination_id, record.record_revision, owner="sender"
    )
    return runtime, claim


async def test_e3_success_persists_exact_receipt_and_settlement_failure_never_resends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Bot:
        def __init__(self) -> None:
            self.calls = 0

        async def send_message(self, **kwargs: object) -> object:
            self.calls += 1
            return SimpleNamespace(message_id=71, chat=SimpleNamespace(id=123), date=None)

    runtime, claim = _prepared_runtime(tmp_path / "success")
    bot = Bot()
    channel = TelegramChannel(TelegramConfig(enabled=True, token="unused"), MessageBus())
    channel._app = SimpleNamespace(bot=bot)
    channel._app_ready.set()
    sender = CronChannelSender(lambda name: channel if name == "telegram" else None)
    result = await runtime.dispatch(claim, sender=sender)
    stored = runtime.get(claim.outbox_id).destinations[0].attempts[-1].receipt
    assert isinstance(result, ConfirmedDelivered)
    assert stored == result.receipt
    assert stored["parts"][0]["provider_message_id"] == 71
    assert bot.calls == 1

    failing_runtime, failing_claim = _prepared_runtime(tmp_path / "failure")
    failing_bot = Bot()
    failing_channel = TelegramChannel(
        TelegramConfig(enabled=True, token="unused"), MessageBus()
    )
    failing_channel._app = SimpleNamespace(bot=failing_bot)
    failing_channel._app_ready.set()
    failing_sender = CronChannelSender(
        lambda name: failing_channel if name == "telegram" else None
    )

    def fail_settle(*args: object, **kwargs: object) -> object:
        raise DeliveryOutboxPersistenceError("settlement failed")

    monkeypatch.setattr(failing_runtime.outbox_store, "settle", fail_settle)
    with pytest.raises(CronDeliverySettlementError):
        await failing_runtime.dispatch(failing_claim, sender=failing_sender)
    with pytest.raises(CronDeliverySettlementError):
        await failing_runtime.dispatch(failing_claim, sender=failing_sender)
    assert failing_bot.calls == 1


async def test_e3_partial_bundle_reaches_reconciliation_and_invokes_sender_once(
    tmp_path: Path,
) -> None:
    runtime, claim = _prepared_runtime(
        tmp_path,
        messages=[
            {"content": "first", "media": []},
            {"content": "second", "media": []},
        ],
    )

    class Bot:
        def __init__(self) -> None:
            self.calls = 0

        async def send_message(self, **kwargs: object) -> object:
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(message_id=81, chat=SimpleNamespace(id=123), date=None)
            raise TimedOut("ambiguous")

    bot = Bot()
    channel = TelegramChannel(TelegramConfig(enabled=True, token="unused"), MessageBus())
    channel._app = SimpleNamespace(bot=bot)
    channel._app_ready.set()
    adapter = CronChannelSender(lambda name: channel if name == "telegram" else None)

    class CountingSender:
        def __init__(self) -> None:
            self.calls = 0

        async def send(self, request: DispatchRequest) -> object:
            self.calls += 1
            return await adapter.send(request)

    sender = CountingSender()
    result = await runtime.dispatch(claim, sender=sender)
    assert result == UnknownOutcome("partial_bundle")
    assert runtime.status(claim.outbox_id) == "reconciliation-required"
    assert sender.calls == 1
    assert bot.calls == 2
