"""A3-8 A01-F08 requirements-only case U3 (local-runner restart recovery).

Contract capabilities are bound to the real symbols: ``recover_after_restart``,
``drain_local_trigger_queue_once``, the persisted delivery payload and its
``settlementId``, the owner query ``list_unknown_deliveries``, and the explicit
``resolve_unknown_delivery`` resolution. Scenario, inputs, and expected outcomes
are kept from ``req-a38-cron/CASES.md``.
"""

import json

from kohaku.bus.events import InboundMessage, OutboundMessage
from kohaku.triggers.local_runner import (
    drain_local_trigger_queue_once,
    recover_after_restart,
)
from kohaku.triggers.local_store import LocalTriggerStore


def _channel_is_enabled(_name: str) -> bool:
    return True


def _run_record(store: LocalTriggerStore, delivery_id: str) -> dict:
    return json.loads((store.runs_dir / f"{delivery_id}.json").read_text(encoding="utf-8"))


async def test_u3_restart_quarantines_started_attempt_and_drains_ready_once(tmp_path):
    store = LocalTriggerStore(tmp_path)
    trigger = store.create(
        name="Restart recovery",
        channel="websocket",
        chat_id="chat-1",
        session_key="websocket:chat-1",
    )

    # P: processing, attempt 1, settlement identity persisted before the turn.
    interrupted = store.enqueue(trigger.id, "P interrupted")
    claimed = store.claim_deliveries()
    assert len(claimed) == 1 and claimed[0].id == interrupted.id
    sid_p = store.begin_delivery_attempt(claimed[0])
    store.write_delivery_run_record(
        claimed[0], trigger=trigger, status="processing", settlement_id=sid_p,
    )
    # R: ready in the inbox, attempt 0, eligible for scheduling.
    ready = store.enqueue(trigger.id, "R ready")

    invocations: list[dict[str, object]] = []

    async def submit_turn(msg: InboundMessage) -> OutboundMessage:
        delivery_id = str(msg.metadata["_local_trigger"]["delivery_id"])
        record = _run_record(store, delivery_id)
        invocations.append({
            "delivery_id": delivery_id,
            "content": msg.content,
            "settlement_id": record.get("settlement_id"),
        })
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="done")

    # Simulate restart: new in-memory state, same persisted bytes.
    restarted = LocalTriggerStore(tmp_path)
    requeued = recover_after_restart(restarted)

    # 1-3: P is unknown, keeps attempt 1 and its settlement identity.
    assert requeued == 0
    unknown = restarted.list_unknown_deliveries()
    assert len(unknown) == 1
    interrupted_unknown = unknown[0]
    assert interrupted_unknown["delivery_id"] == interrupted.id
    assert interrupted_unknown["status"] == "unknown"
    assert interrupted_unknown["attempts"] == 1
    assert interrupted_unknown["settlement_id"] == sid_p
    assert list(restarted.processing_dir.glob("*.json")) == []
    assert [path.name for path in restarted.inbox_dir.glob("*.json")] == [ready.path.name]

    # 4: the interrupted effect is not re-executed by recovery or either drain.
    assert await drain_local_trigger_queue_once(
        store=restarted,
        submit_turn=submit_turn,
        is_channel_enabled=_channel_is_enabled,
    ) == 1
    assert await drain_local_trigger_queue_once(
        store=restarted,
        submit_turn=submit_turn,
        is_channel_enabled=_channel_is_enabled,
    ) == 0

    # 5-7: R ran exactly once with a settlement identity; P never ran.
    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation["delivery_id"] == ready.id
    assert invocation["content"] == "R ready"
    assert isinstance(invocation["settlement_id"], str) and invocation["settlement_id"]
    ready_record = _run_record(restarted, ready.id)
    assert ready_record["attempts"] == 1
    assert ready_record["settlement_id"] == invocation["settlement_id"]

    # 8: the owner still sees P as unknown with its settlement identity.
    still_unknown = restarted.list_unknown_deliveries()
    assert [entry["delivery_id"] for entry in still_unknown] == [interrupted.id]
    assert still_unknown[0]["settlement_id"] == sid_p

    # 9: resolution is explicit and moves P to an explicitly requeued state.
    assert restarted.resolve_unknown_delivery(interrupted.id, action="requeue") is True
    assert restarted.list_unknown_deliveries() == []
    requeued_claims = restarted.claim_deliveries()
    assert [claim.id for claim in requeued_claims] == [interrupted.id]
    assert requeued_claims[0].content == "P interrupted"
    assert len(invocations) == 1


def test_u3_unstarted_claim_returns_to_ready_once(tmp_path):
    store = LocalTriggerStore(tmp_path)
    trigger = store.create(
        name="Proof of absence",
        channel="websocket",
        chat_id="chat-1",
        session_key="websocket:chat-1",
    )
    delivery = store.enqueue(trigger.id, "claimed but never started")
    assert len(store.claim_deliveries()) == 1

    restarted = LocalTriggerStore(tmp_path)
    assert recover_after_restart(restarted) == 1

    assert restarted.list_unknown_deliveries() == []
    claims = restarted.claim_deliveries()
    assert [claim.id for claim in claims] == [delivery.id]
    assert claims[0].attempts == 0
    assert claims[0].last_error == "delivery was recovered from interrupted processing"
