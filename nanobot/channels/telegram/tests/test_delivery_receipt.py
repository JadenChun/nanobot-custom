import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from telegram.error import BadRequest, TimedOut

from nanobot.bus.queue import MessageBus
from nanobot.channels.delivery_contract import (
    ChannelDeliveryPreflightError,
    ChannelDeliveryRejectedError,
    ChannelTextPartReceipt,
    ChannelTextReceipt,
    PartialChannelTextDeliveryError,
)
from nanobot.channels.telegram.runtime import TelegramChannel, TelegramConfig

ACTION_ID = "a" * 64
REQUEST_DIGEST = "b" * 64


def _channel(bot: object | None = None) -> TelegramChannel:
    channel = TelegramChannel(TelegramConfig(enabled=True, token="unused"), MessageBus())
    if bot is not None:
        channel._app = SimpleNamespace(bot=bot)
        channel._app_ready.set()
    return channel


@pytest.mark.parametrize("reported_thread", [42, 43, None])
async def test_topic_delivery_checks_provider_thread_before_confirming_receipt(reported_thread):
    calls = []

    class Bot:
        async def send_message(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(message_id=1, chat=SimpleNamespace(id=-123), date=None,
                                   message_thread_id=reported_thread)

    channel = _channel(Bot())
    pending = channel.send_thread_text_with_receipt("-123", ("topic report",), message_thread_id=42,
                                                    action_id=ACTION_ID, request_digest=REQUEST_DIGEST)
    if reported_thread == 42:
        receipt = await pending
        assert receipt.parts[0].provider_chat_id == -123
    else:
        with pytest.raises(RuntimeError, match="thread_receipt_mismatch"):
            await pending
    assert len(calls) == 1 and calls[0]["message_thread_id"] == 42


async def test_complete_bundle_returns_ordered_content_free_receipt() -> None:
    class Bot:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def send_message(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            return SimpleNamespace(
                message_id=len(self.calls),
                chat=SimpleNamespace(id=123),
                date=datetime(2026, 9, 29, tzinfo=UTC),
            )

    bot = Bot()
    receipt = await _channel(bot).send_text_with_receipt(
        "123", ("first", "second"), action_id=ACTION_ID, request_digest=REQUEST_DIGEST
    )

    assert [part.index for part in receipt.parts] == [0, 1]
    assert [part.message_index for part in receipt.parts] == [0, 1]
    assert [part.part_index for part in receipt.parts] == [0, 0]
    assert [part.provider_message_id for part in receipt.parts] == [1, 2]
    assert all(part.provider_chat_id == 123 for part in receipt.parts)
    evidence = receipt.to_json()
    assert evidence["action_id"] == ACTION_ID
    assert evidence["request_digest"] == REQUEST_DIGEST
    assert "first" not in repr(evidence)
    assert "second" not in repr(evidence)


async def test_split_message_preserves_input_and_part_indices() -> None:
    class Bot:
        def __init__(self) -> None:
            self.calls = 0

        async def send_message(self, **kwargs: object) -> object:
            self.calls += 1
            return SimpleNamespace(
                message_id=self.calls, chat=SimpleNamespace(id=-123), date=None
            )

    receipt = await _channel(Bot()).send_text_with_receipt(
        "-123",
        ("word " * 1000, "tail"),
        action_id=ACTION_ID,
        request_digest=REQUEST_DIGEST,
    )

    assert len(receipt.parts) > 2
    assert [part.index for part in receipt.parts] == list(range(len(receipt.parts)))
    first_message_parts = [part for part in receipt.parts if part.message_index == 0]
    assert [part.part_index for part in first_message_parts] == list(
        range(len(first_message_parts))
    )
    assert receipt.parts[-1].message_index == 1
    assert receipt.parts[-1].part_index == 0


async def test_receipt_send_preflight_rejects_absent_app_and_invalid_target() -> None:
    with pytest.raises(ChannelDeliveryPreflightError) as absent:
        await _channel().send_text_with_receipt(
            "123", ("text",), action_id=ACTION_ID, request_digest=REQUEST_DIGEST
        )
    assert absent.value.reason_code == "telegram_app_unavailable"

    with pytest.raises(ChannelDeliveryPreflightError) as invalid:
        await _channel(SimpleNamespace()).send_text_with_receipt(
            "not-an-int", ("text",), action_id=ACTION_ID, request_digest=REQUEST_DIGEST
        )
    assert invalid.value.reason_code == "invalid_target"


async def test_parse_rejection_falls_back_once_but_other_bad_request_is_permanent() -> None:
    class Bot:
        def __init__(self, error: BadRequest) -> None:
            self.error = error
            self.calls = 0

        async def send_message(self, **kwargs: object) -> object:
            self.calls += 1
            if self.calls == 1:
                raise self.error
            return SimpleNamespace(message_id=9, chat=SimpleNamespace(id=123), date=None)

    parse_bot = Bot(BadRequest("Can't parse entities: unsupported start tag"))
    receipt = await _channel(parse_bot).send_text_with_receipt(
        "123", ("<bad>",), action_id=ACTION_ID, request_digest=REQUEST_DIGEST
    )
    assert parse_bot.calls == 2
    assert receipt.parts[0].provider_message_id == 9

    rejected_bot = Bot(BadRequest("chat not found"))
    with pytest.raises(ChannelDeliveryRejectedError):
        await _channel(rejected_bot).send_text_with_receipt(
            "123", ("text",), action_id=ACTION_ID, request_digest=REQUEST_DIGEST
        )
    assert rejected_bot.calls == 1


async def test_ambiguous_first_failure_propagates_and_acknowledged_prefix_is_partial() -> None:
    class Bot:
        def __init__(self) -> None:
            self.calls = 0

        async def send_message(self, **kwargs: object) -> object:
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(message_id=4, chat=SimpleNamespace(id=123), date=None)
            raise TimedOut("ambiguous")

    first_failure = SimpleNamespace(send_message=lambda **kwargs: None)

    async def timeout(**kwargs: object) -> object:
        raise TimedOut("ambiguous")

    first_failure.send_message = timeout
    with pytest.raises(TimedOut):
        await _channel(first_failure).send_text_with_receipt(
            "123", ("text",), action_id=ACTION_ID, request_digest=REQUEST_DIGEST
        )

    bot = Bot()
    with pytest.raises(PartialChannelTextDeliveryError) as partial:
        await _channel(bot).send_text_with_receipt(
            "123", ("first", "second"), action_id=ACTION_ID, request_digest=REQUEST_DIGEST
        )
    assert [part.provider_message_id for part in partial.value.receipt.parts] == [4]
    assert bot.calls == 2


async def test_malformed_first_acknowledgement_is_unknown_and_schema_fails_closed() -> None:
    class Bot:
        async def send_message(self, **kwargs: object) -> object:
            return SimpleNamespace(message_id=0, chat=SimpleNamespace(id=123), date=None)

    with pytest.raises(ValueError, match="message id"):
        await _channel(Bot()).send_text_with_receipt(
            "123", ("text",), action_id=ACTION_ID, request_digest=REQUEST_DIGEST
        )

    first = ChannelTextPartReceipt(0, 0, 0, 1, 123)
    duplicate = ChannelTextPartReceipt(0, 0, 1, 2, 123)
    with pytest.raises(ValueError, match="unique and contiguous"):
        ChannelTextReceipt(
            "telegram", "123", ACTION_ID, REQUEST_DIGEST, (first, duplicate)
        )


async def test_cancellation_holds_target_lock_until_provider_await_exits() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    cleanup = asyncio.Event()
    second_entered = asyncio.Event()

    class Bot:
        def __init__(self) -> None:
            self.calls = 0

        async def send_message(self, **kwargs: object) -> object:
            self.calls += 1
            if self.calls == 1:
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await cleanup.wait()
                    raise
            second_entered.set()
            return SimpleNamespace(message_id=2, chat=SimpleNamespace(id=123), date=None)

    channel = _channel(Bot())
    first = asyncio.create_task(channel.send_text_with_receipt(
        "123", ("first",), action_id=ACTION_ID, request_digest=REQUEST_DIGEST
    ))
    await entered.wait()
    second = asyncio.create_task(channel.send_text_with_receipt(
        "123", ("second",), action_id="c" * 64, request_digest="d" * 64
    ))
    first.cancel()
    await cancelled.wait()
    await asyncio.sleep(0)
    assert not second_entered.is_set()
    cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    assert second_entered.is_set()


async def test_same_target_bundles_do_not_interleave_and_different_targets_can_progress() -> None:
    entered: asyncio.Queue[int] = asyncio.Queue()
    releases = {1: asyncio.Event(), 2: asyncio.Event(), 3: asyncio.Event(), 4: asyncio.Event()}

    class Bot:
        def __init__(self) -> None:
            self.calls = 0

        async def send_message(self, **kwargs: object) -> object:
            self.calls += 1
            call = self.calls
            await entered.put(int(kwargs["chat_id"]))
            await releases[call].wait()
            return SimpleNamespace(
                message_id=call, chat=SimpleNamespace(id=kwargs["chat_id"]), date=None
            )

    channel = _channel(Bot())
    first = asyncio.create_task(channel.send_text_with_receipt(
        "1", ("a", "b"), action_id=ACTION_ID, request_digest=REQUEST_DIGEST
    ))
    assert await entered.get() == 1
    same = asyncio.create_task(channel.send_text_with_receipt(
        "1", ("c",), action_id="c" * 64, request_digest="d" * 64
    ))
    other = asyncio.create_task(channel.send_text_with_receipt(
        "2", ("d",), action_id="e" * 64, request_digest="f" * 64
    ))
    assert await entered.get() == 2
    releases[1].set()
    assert await entered.get() == 1
    releases[2].set()
    releases[3].set()
    assert await entered.get() == 1
    releases[4].set()
    await asyncio.gather(first, same, other)
