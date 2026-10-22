"""Explicit Telegram authorization, admission, and exact-route H1 delivery."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from nanobot.agent.busy_control import BusyControlRouter, ControlDecision
from nanobot.agent.busy_dispatch import BusyInboxConsumer, BusyTurn
from nanobot.bus.events import OutboundMessage
from nanobot.channels.routed_delivery import (
    RoutedAddress,
    RoutedAttachment,
    RoutedBundle,
    RoutedDeliveryError,
    RoutedDeliveryReceipt,
    RoutedDeliveryRequest,
    validate_routed_receipt,
)
from nanobot.security.network import validate_url_target
from nanobot.session.busy_inbox import (
    InboxConflictError,
    InboxEntry,
    InboxNotice,
    InboxNoticeReceipt,
    InboxPayload,
    InboxResult,
    InboxResultReceiptAddress,
    InboxResultReceiptEnvelope,
    InboxResultReceiptPart,
    Origin,
    SessionBusyInbox,
    TelegramAlbumMember,
)

if TYPE_CHECKING:
    from nanobot.channels.telegram.runtime import TelegramChannel


_MAX_MEDIA_BYTES = 50 * 1024 * 1024


def _is_string(value: object) -> bool:
    return isinstance(value, str)


def _rethrow_delivery_cancellation(error: RoutedDeliveryError) -> None:
    cause = error.__cause__
    if isinstance(cause, asyncio.CancelledError):
        raise cause


def _numeric_sender(sender_id: str) -> str:
    return sender_id.split("|", 1)[0]


def _positive_id(value: object, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError(f"Telegram {name} must be a positive integer")
    return str(value)


class TelegramBusyInbox:
    """A single registered Telegram route into one AgentLoop-owned inbox."""

    def __init__(
        self,
        channel: TelegramChannel,
        inbox: SessionBusyInbox,
        consumer: BusyInboxConsumer,
        router: BusyControlRouter,
        *,
        account_id: str,
        owner_sender_ids: frozenset[str],
        owner_chat_ids: frozenset[str],
        tool_scope: tuple[str, ...],
        track_task: Callable[[asyncio.Task[None]], None],
    ) -> None:
        if inbox.temporary_lifetime_id is not None or consumer.inbox is not inbox:
            raise ValueError("Telegram H1 requires the registered regular inbox consumer")
        if not account_id.isdecimal() or int(account_id) <= 0 or str(int(account_id)) != account_id:
            raise ValueError("Telegram account identity must be a canonical bot ID")
        if not owner_sender_ids or any(
            not value.isdecimal() or int(value) <= 0 or str(int(value)) != value
            for value in owner_sender_ids
        ):
            raise ValueError("Telegram owner senders must be explicit numeric user IDs")
        if not owner_chat_ids or any(
            not value or value == "0" or str(int(value)) != value
            for value in owner_chat_ids
        ):
            raise ValueError("Telegram owner chats must be explicit numeric chat IDs")
        if type(tool_scope) is not tuple or len(set(tool_scope)) != len(tool_scope) or any(
            not _is_string(value) or not value or len(value) > 128 for value in tool_scope
        ):
            raise ValueError("Telegram tool scope must be a finite tuple of tool names")
        if channel.name != "telegram":
            raise ValueError("Telegram inbox requires a Telegram channel")

        self.channel = channel
        self.inbox = inbox
        self.consumer = consumer
        self.router = router
        self.account_id = account_id
        self.owner_sender_ids = owner_sender_ids
        self.owner_chat_ids = owner_chat_ids
        self.tool_scope = tool_scope
        self._track_task = track_task
        self._notice_event = asyncio.Event()
        self._notice_task: asyncio.Task[None] | None = None
        self._closed = False
        self._notice_waker = self.wake_notices
        consumer.add_delivery_waker(self._notice_waker)

    def matches_channel(self, channel: TelegramChannel) -> bool:
        return channel is self.channel

    def authorizes_address(self, address: RoutedAddress) -> bool:
        """Recheck the registered owner and current Telegram policy per send."""
        if (self._closed or not self.channel.owns_busy_inbox(self)
                or not self.channel.busy_account_matches(self.account_id)
                or address.channel != "telegram" or address.account_id != self.account_id
                or _numeric_sender(address.sender_id) not in self.owner_sender_ids
                or address.chat_id not in self.owner_chat_ids):
            return False
        return (self.channel.is_allowed(address.sender_id)
                and self.channel.is_chat_allowed(address.chat_id, address.chat_type))

    def _authorized_update(self, message: object, user: object) -> bool:
        if (not self.channel.busy_account_matches(self.account_id)
                or not self.channel.owns_busy_inbox(self)):
            return False
        sender_value = getattr(user, "id", None)
        chat_value = getattr(message, "chat_id", None)
        if type(sender_value) is not int or type(chat_value) is not int:
            return False
        chat_id = str(chat_value)
        sender_id = str(sender_value)
        if sender_id not in self.owner_sender_ids or chat_id not in self.owner_chat_ids:
            return False
        return self.channel.busy_update_allowed(message, user)

    def _origin(self, message: object, user: object) -> Origin:
        message_id = getattr(message, "message_id", None)
        chat_value = getattr(message, "chat_id", None)
        chat = getattr(message, "chat", None)
        chat_type = getattr(chat, "type", None)
        sender_value = getattr(user, "id", None)
        if (type(message_id) is not int or message_id <= 0 or type(chat_value) is not int
                or chat_value == 0 or type(sender_value) is not int or sender_value <= 0
                or chat_type not in {"private", "group", "supergroup", "channel"}):
            raise ValueError("Telegram update has incomplete authenticated identity")
        thread_id = _positive_id(getattr(message, "message_thread_id", None), "thread ID")
        reply = getattr(message, "reply_to_message", None)
        reply_id = _positive_id(getattr(reply, "message_id", None), "reply target")
        quote = getattr(message, "quote", None)
        forwarded = any(getattr(message, name, None) is not None for name in (
            "forward_origin", "forward_from", "forward_from_chat", "forward_sender_name",
        ))
        sender_id = self.channel.busy_sender_id(user)
        if sender_id is None:
            raise ValueError("Telegram update has an invalid sender identity")
        return Origin(
            channel="telegram",
            account_id=self.account_id,
            chat_id=str(chat_value),
            chat_type=chat_type,
            sender_id=sender_id,
            transport_id=str(message_id),
            thread_id=thread_id,
            reply_to_message_id=reply_id,
            output_reply_to_message_id=str(message_id),
            quoted_message_id=reply_id if quote is not None else None,
            forwarded=forwarded,
        )

    def _payload(
        self,
        message: object,
        user: object,
        *,
        content: str,
        media: tuple[str, ...],
        album_id: str | None,
        album_members: tuple[TelegramAlbumMember, ...],
        transport_id: str | None = None,
    ) -> InboxPayload:
        return InboxPayload(
            owner_id=self.inbox.owner_id,
            session_key=self.inbox.session_key,
            origin=self._origin(message, user).model_copy(update={
                "transport_id": transport_id or str(getattr(message, "message_id")),
            }),
            content=content,
            media=media,
            scope=self.tool_scope,
            album_id=album_id,
            album_members=album_members,
        )

    @staticmethod
    def _matches_clarification(row: InboxEntry, payload: InboxPayload) -> bool:
        route = row.clarification_receipt_origin
        return (
            row.state == "awaiting-clarification"
            and row.clarification_receipt_message_id is not None
            and payload.origin.reply_to_message_id == str(row.clarification_receipt_message_id)
            and route is not None
            and route.channel == payload.origin.channel
            and route.account_id == payload.origin.account_id
            and route.chat_id == payload.origin.chat_id
            and route.sender_id == payload.origin.sender_id
            and route.thread_id == payload.origin.thread_id
        )

    async def handle_update(
        self,
        message: object,
        user: object,
        *,
        content: str,
        media: tuple[str, ...] = (),
        album_id: str | None = None,
        album_members: tuple[TelegramAlbumMember, ...] = (),
        direct: bool = True,
        transport_id: str | None = None,
    ) -> bool:
        """Admit bound updates and consume owner-binding failures without legacy dispatch."""
        command = content.strip().split(maxsplit=1)[0].split("@", 1)[0] if content.strip() else ""
        if command in {"/start", "/help"}:
            return False
        if not self._authorized_update(message, user):
            return True
        payload = self._payload(
            message,
            user,
            content=content,
            media=media,
            album_id=album_id,
            album_members=album_members,
            transport_id=transport_id,
        )
        matches = tuple(row for row in self.inbox.entries() if self._matches_clarification(row, payload))
        if len(matches) == 1 and album_id is None:
            row = self.inbox.resume_clarification(matches[0].id, matches[0].revision, payload)
            duplicate = False
        else:
            admission = self.inbox.admit(payload)
            row = admission.entry
            duplicate = admission.duplicate
        self.wake_notices()
        if duplicate and row.state != "pending-routing":
            return True

        observed = self.consumer.current_run
        try:
            decision = await self.router.propose(
                content,
                observed,
                direct=direct and not payload.origin.forwarded and payload.origin.quoted_message_id is None,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            decision = ControlDecision("next-turn", observed, "error")

        current = self.consumer.current_run
        try:
            routed = self.inbox.route(
                row.id,
                row.revision,
                decision,
                current_run=current,
                active_scope=self.consumer.active_scope,
            )
        except InboxConflictError:
            latest = next((value for value in self.inbox.entries() if value.id == row.id), None)
            if latest is None or latest.state != "pending-routing":
                self.wake_notices()
                return True
            routed = self.inbox.route(
                latest.id,
                latest.revision,
                ControlDecision("next-turn", current, "error"),
                current_run=current,
                active_scope=self.consumer.active_scope,
            )
        self.wake_notices()
        if routed.state == "control":
            await self.consumer.apply_control(routed.id, routed.revision)
        elif routed.state in {"next-turn", "next-step"}:
            self.consumer.wake()
        return True

    @staticmethod
    def _notice_text(notice: InboxNotice) -> str:
        if notice.stage == "accepted":
            return "Received and saved for routing."
        if notice.stage == "queued":
            return "Queued for the next task turn."
        if notice.stage == "steered":
            return "Added as guidance for the current task; it may not have been consumed yet."
        if notice.stage == "clarify":
            return "I need clarification before routing this message. Reply to this notice."
        if notice.stage == "stopping":
            return "I asked the current task to stop."
        if notice.stage == "blocked":
            return "The current task is still running; its stop request is blocked."
        if notice.stage == "completed":
            report = notice.report
            if report is None:
                return "The task completed."
            active = report.active_run
            if active is not None:
                return f"The control request finished; the task is still {active.state}."
            status = report.target_status.value if report.target_status else None
            return f"The task was observed with status {status}." if status else "The control request finished."
        if notice.stage == "cancelled":
            return "The task was cancelled before dispatch."
        if notice.stage == "uncertain":
            return "The task outcome is uncertain; it will not be replayed automatically."
        if notice.stage == "control":
            if notice.intent == "status" and notice.report is not None:
                active = notice.report.active_run
                if active is not None:
                    return f"The task is {active.state}; {notice.report.queued} task turns are queued."
                if notice.report.recovery_required:
                    return "The task needs recovery before more work can run."
                return f"No task is running; {notice.report.queued} task turns are queued."
            return "Control request recorded."
        return "Task status updated."

    def _address(self, origin: Origin, *, reply_to_message_id: str | None = None) -> RoutedAddress:
        if origin.chat_type is None:
            raise RoutedDeliveryError("definitely-unsent", "telegram_chat_type_missing")
        try:
            return RoutedAddress(
                channel="telegram",
                account_id=origin.account_id,
                sender_id=origin.sender_id,
                chat_type=origin.chat_type,
                chat_id=origin.chat_id,
                thread_id=origin.thread_id,
                reply_to_message_id=reply_to_message_id or origin.output_reply_to_message_id,
            )
        except ValueError as exc:
            raise RoutedDeliveryError("definitely-unsent", "telegram_route_invalid") from exc

    async def _send(self, request: RoutedDeliveryRequest) -> RoutedDeliveryReceipt:
        if not self.authorizes_address(request.address):
            raise RoutedDeliveryError("definitely-unsent", "telegram_route_not_authorized")
        receipt = await self.channel.send_routed_with_receipt(request)
        try:
            validate_routed_receipt(request, receipt)
        except ValueError as exc:
            raise RoutedDeliveryError("uncertain", "telegram_receipt_invalid") from exc
        return receipt

    async def _flush_notices_once(self) -> None:
        for pending in self.inbox.notices():
            if pending.state != "pending":
                continue
            try:
                notice = self.inbox.claim_notice(pending.id)
            except InboxConflictError:
                continue
            try:
                text = self._notice_text(notice)
                bundle = RoutedBundle(content=text)
                action_id = hashlib.sha256(
                    f"telegram-notice:{notice.id}:{bundle.digest}".encode()
                ).hexdigest()
                request = RoutedDeliveryRequest(
                    address=self._address(notice.origin),
                    action_id=action_id,
                    bundle=bundle,
                )
                receipt = await self._send(request)
                part = receipt.parts[0]
                self.inbox.acknowledge_notice(notice.id, receipt=InboxNoticeReceipt(
                    notice_id=notice.id,
                    action_id=action_id,
                    bundle_digest=bundle.digest,
                    provider_message_id=part.provider_message_id,
                    provider_chat_id=part.provider_chat_id,
                    provider_thread_id=part.provider_thread_id,
                    provider_reply_to_message_id=part.provider_reply_to_message_id,
                ))
            except RoutedDeliveryError as exc:
                self._fail_notice(notice.id, definitely_unsent=exc.disposition == "definitely-unsent")
                _rethrow_delivery_cancellation(exc)
            except asyncio.CancelledError:
                self._fail_notice(notice.id, definitely_unsent=False)
                raise
            except Exception:
                self._fail_notice(notice.id, definitely_unsent=False)

    def _fail_notice(self, identity: str, *, definitely_unsent: bool) -> None:
        try:
            self.inbox.fail_notice(identity, definitely_unsent=definitely_unsent)
        except InboxConflictError:
            return

    def wake_notices(self) -> None:
        if self._closed:
            return
        self._notice_event.set()
        task = self._notice_task
        if task is None or task.done():
            task = asyncio.create_task(self._notice_pump())
            self._notice_task = task
            self._track_task(task)

    async def _notice_pump(self) -> None:
        try:
            while not self._closed:
                await self._notice_event.wait()
                self._notice_event.clear()
                await self._flush_notices_once()
        except asyncio.CancelledError:
            raise

    def _bundle(self, content: str, media: tuple[str, ...], buttons: tuple[tuple[str, ...], ...]) -> RoutedBundle:
        attachments: list[RoutedAttachment] = []
        for source in media:
            if self.channel.busy_is_remote_media_url(source):
                ok, _reason = validate_url_target(source)
                if not ok:
                    raise RoutedDeliveryError("definitely-unsent", "unsafe_output_media_url")
                media_type = self.channel.busy_media_type(source)
                attachments.append(RoutedAttachment(
                    source=source,
                    filename=source.rsplit("/", 1)[-1].split("?", 1)[0] or "attachment",
                    media_type=media_type,
                    reference_digest=hashlib.sha256(source.encode("utf-8")).hexdigest(),
                ))
                continue
            try:
                data = Path(source).read_bytes()
            except (OSError, ValueError) as exc:
                raise RoutedDeliveryError("definitely-unsent", "output_media_unavailable") from exc
            if len(data) > _MAX_MEDIA_BYTES:
                raise RoutedDeliveryError("definitely-unsent", "output_media_too_large")
            media_type = self.channel.busy_media_type(source)
            attachments.append(RoutedAttachment(
                source=source,
                filename=Path(source).name or "attachment",
                media_type=media_type,
                content_digest=hashlib.sha256(data).hexdigest(),
            ))
        try:
            bundle = RoutedBundle(content=content, attachments=tuple(attachments), buttons=buttons)
            text_parts = self.channel.busy_split_routed_text(bundle.rendered_text) if bundle.rendered_text else ()
            return RoutedBundle(
                content=content,
                attachments=tuple(attachments),
                buttons=buttons,
                text_parts=text_parts,
            )
        except ValueError as exc:
            raise RoutedDeliveryError("definitely-unsent", "output_bundle_invalid") from exc

    @staticmethod
    def _buttons(message: OutboundMessage) -> tuple[tuple[str, ...], ...]:
        rows: list[tuple[str, ...]] = []
        for row in message.buttons:
            if type(row) is not list or not row or any(not _is_string(label) for label in row):
                raise RoutedDeliveryError("definitely-unsent", "output_buttons_invalid")
            rows.append(tuple(row))
        return tuple(rows)

    def _request(self, origin: Origin, result: InboxResult) -> RoutedDeliveryRequest:
        bundle = self._bundle(result.content, result.media, result.buttons)
        address = self._address(origin)
        if (len(result.media_digests) != len(result.media)
                or self._result_digest(result.content, result.media, result.media_digests,
                                       result.buttons) != result.output_digest
                or tuple(attachment.content_digest or attachment.reference_digest
                         for attachment in bundle.attachments) != result.media_digests):
            raise RoutedDeliveryError("definitely-unsent", "stored_output_digest_changed")
        request = RoutedDeliveryRequest(
            address=address,
            action_id=result.action_id,
            bundle=bundle,
        )
        if result.bundle_digest is not None and (
            result.bundle_digest != request.bundle_digest
            or result.request_address != self._receipt_address(request.address)
        ):
            raise RoutedDeliveryError("definitely-unsent", "stored_request_digest_changed")
        return request

    @staticmethod
    def _receipt_address(address: RoutedAddress) -> InboxResultReceiptAddress:
        return InboxResultReceiptAddress.model_validate({
            "channel": address.channel,
            "account_id": address.account_id,
            "sender_id": address.sender_id,
            "chat_type": address.chat_type,
            "chat_id": address.chat_id,
            "thread_id": address.thread_id,
            "reply_to_message_id": address.reply_to_message_id,
        })

    @staticmethod
    def _result_digest(
        content: str,
        media: tuple[str, ...],
        media_digests: tuple[str | None, ...],
        buttons: tuple[tuple[str, ...], ...],
    ) -> str:
        encoded = json.dumps(
            {"content": content, "media": list(media), "media_digests": list(media_digests),
             "buttons": [list(row) for row in buttons]},
            ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _media_digests(media: tuple[str, ...]) -> tuple[str | None, ...]:
        digests: list[str | None] = []
        for source in media:
            if source.startswith(("http://", "https://")):
                digests.append(hashlib.sha256(source.encode("utf-8")).hexdigest())
                continue
            try:
                data = Path(source).read_bytes()
            except (OSError, ValueError):
                digests.append(None)
            else:
                digests.append(hashlib.sha256(data).hexdigest())
        return tuple(digests)

    @staticmethod
    def _result_receipt_parts(receipt: RoutedDeliveryReceipt | None) -> tuple[InboxResultReceiptPart, ...]:
        if receipt is None:
            return ()
        return tuple(InboxResultReceiptPart(
            index=part.index,
            message_index=part.message_index,
            part_index=part.part_index,
            kind=part.kind,
            provider_message_id=part.provider_message_id,
            provider_chat_id=part.provider_chat_id,
            provider_thread_id=part.provider_thread_id,
            provider_reply_to_message_id=part.provider_reply_to_message_id,
        ) for part in receipt.parts)

    @classmethod
    def _result_receipt_envelope(
        cls, receipt: RoutedDeliveryReceipt | None,
    ) -> InboxResultReceiptEnvelope | None:
        if receipt is None:
            return None
        return InboxResultReceiptEnvelope(
            address=cls._receipt_address(receipt.address),
            action_id=receipt.action_id,
            bundle_digest=receipt.bundle_digest,
            parts=cls._result_receipt_parts(receipt),
        )

    async def _deliver_persisted_result(self, identity: str, revision: int, origin: Origin, result: InboxResult) -> None:
        attempt = self.inbox.begin_result(
            identity,
            revision,
            run_id=result.run_id,
            origin=origin,
            action_id=result.action_id,
            output_digest=result.output_digest,
            content=result.content,
            media=result.media,
            buttons=result.buttons,
            media_digests=result.media_digests,
        )
        provider_started = False
        receipt_envelope: InboxResultReceiptEnvelope | None = None
        try:
            request = self._request(origin, attempt.result or result)
            attempt = self.inbox.bind_result_request(
                identity,
                attempt.revision,
                action_id=request.action_id,
                address=self._receipt_address(request.address),
                bundle_digest=request.bundle_digest,
            )
            provider_started = True
            receipt = await self._send(request)
            receipt_envelope = self._result_receipt_envelope(receipt)
            self.inbox.settle_result(
                identity,
                attempt.revision,
                disposition="delivered",
                receipt_parts=receipt_envelope.parts if receipt_envelope is not None else (),
                receipt_envelope=receipt_envelope,
            )
        except RoutedDeliveryError as exc:
            try:
                self.inbox.settle_result(
                    identity,
                    attempt.revision,
                    disposition=exc.disposition,
                    receipt_parts=self._result_receipt_parts(exc.receipt),
                    receipt_envelope=self._result_receipt_envelope(exc.receipt),
                )
            finally:
                _rethrow_delivery_cancellation(exc)
                raise
        except BaseException as exc:
            try:
                self.inbox.settle_result(
                    identity,
                    attempt.revision,
                    disposition="uncertain" if provider_started else "definitely-unsent",
                    receipt_parts=receipt_envelope.parts if receipt_envelope is not None else (),
                    receipt_envelope=receipt_envelope,
                )
            except Exception:
                pass
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise

    async def deliver_result(self, turn: BusyTurn, response: OutboundMessage) -> None:
        """Retain full task output, send once, and persist its exact receipt."""
        initial = turn.initial
        if initial is None:
            return
        current = next((row for row in self.inbox.entries() if row.id == initial.id), None)
        if (current is None or current.claimed_run_id != turn.run_id
                or current.origin != initial.origin):
            raise InboxConflictError("result ingress no longer belongs to this task run")
        if not response.content and not response.media and not response.buttons:
            return
        media = tuple(response.media)
        buttons = self._buttons(response)
        media_digests = self._media_digests(media)
        result_digest = self._result_digest(response.content, media, media_digests, buttons)
        action_id = hashlib.sha256(
            f"telegram-result:{initial.id}:{turn.run_id}:{result_digest}".encode()
        ).hexdigest()
        result = InboxResult(
            run_id=turn.run_id,
            origin=initial.origin,
            action_id=action_id,
            output_digest=result_digest,
            content=response.content,
            media=media,
            media_digests=media_digests,
            buttons=buttons,
            disposition="sending",
        )
        try:
            await self._deliver_persisted_result(current.id, current.revision, current.origin, result)
        except RoutedDeliveryError as exc:
            if exc.disposition == "definitely-unsent":
                latest = next((row for row in self.inbox.entries() if row.id == current.id), None)
                if latest is not None and latest.result is not None and latest.result.disposition == "sending":
                    self.inbox.settle_result(
                        latest.id, latest.revision, disposition="definitely-unsent",
                    )
            raise

    async def deliver_error(self, turn: BusyTurn) -> None:
        initial = turn.initial
        if initial is None:
            return
        response = OutboundMessage(
            channel="telegram",
            chat_id=initial.origin.chat_id,
            content="Sorry, I encountered an error.",
        )
        await self.deliver_result(turn, response)

    async def retry_result(self, identity: str) -> None:
        """Retry only a definitely-unsent immutable output; task work is never rerun."""
        current = next((row for row in self.inbox.entries() if row.id == identity), None)
        if (current is None or current.result is None
                or current.result.disposition != "definitely-unsent"):
            raise InboxConflictError("result is not eligible for output-only retry")
        await self._deliver_persisted_result(identity, current.revision, current.origin, current.result)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.channel.unregister_busy_inbox(self)
        self.consumer.remove_delivery_waker(self._notice_waker)
        task = self._notice_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


__all__ = ["TelegramBusyInbox"]
