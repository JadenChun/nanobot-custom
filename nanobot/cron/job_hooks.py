"""Configured evidence checks and once-only operational hooks for cron jobs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Awaitable, Callable, Literal, cast
from zoneinfo import ZoneInfo

from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.turn import TurnResult
from nanobot.cron.callback_store import (
    CronCallbackObservation,
    CronCallbackRecord,
    CronCallbackStore,
)
from nanobot.cron.channel_sender import CronChannelSender
from nanobot.cron.completion import CronCompletionProjection
from nanobot.cron.definition_store import CronDefinition
from nanobot.cron.delivery_runtime import (
    ConfirmedDelivered,
    ConfirmedNotSent,
    DispatchRequest,
    PermanentNotSent,
    SenderResult,
)
from nanobot.cron.occurrence_store import OccurrenceClaim
from nanobot.cron.types import CronPayload


@dataclass(frozen=True)
class AdapterResult:
    returncode: int | None
    stdout: bytes
    reason: str


async def run_adapter(argv: list[str], *, cwd: Path, timeout: float) -> AdapterResult:
    """Bound output/waiting; configured argv never passes through a shell."""
    try:
        process = await asyncio.create_subprocess_exec(
            *argv, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            start_new_session=os.name == "posix",
        )
    except OSError:
        return AdapterResult(None, b"", "exec_failed")

    async def read() -> AdapterResult:
        assert process.stdout is not None
        output = b""
        while chunk := await process.stdout.read(4096):
            output += chunk
            if len(output) > 65536:
                await ExecTool.kill_process_tree(process)
                return AdapterResult(None, b"", "output_limit")
        await process.wait()
        return AdapterResult(process.returncode, output, "ok" if process.returncode == 0 else "nonzero_exit")

    try:
        return await asyncio.wait_for(read(), timeout)
    except asyncio.TimeoutError:
        await ExecTool.kill_process_tree(process)
        return AdapterResult(None, b"", "timeout")
    except BaseException:
        await ExecTool.kill_process_tree(process)
        raise


def _render(argv: tuple[str, ...] | list[str], values: dict[str, str]) -> list[str]:
    result: list[str] = []
    for argument in argv:
        for key, value in values.items():
            argument = argument.replace("{" + key + "}", value)
        result.append(argument)
    return result


@dataclass(frozen=True)
class CronHookOutcome:
    ack_status: str | None
    bookkeeping: CronCallbackRecord | None = None
    owner_alert: CronCallbackRecord | None = None


class CronJobHooks:
    def __init__(self, root: Path, *, default_cwd: Path):
        self.root, self.default_cwd = root, default_cwd

    @staticmethod
    def _job(claim: OccurrenceClaim) -> tuple[CronDefinition, CronPayload, str]:
        definition = CronDefinition.decode_wire(claim.job_snapshot)
        payload = CronPayload.from_store_dict(definition.definition["payload"])
        zone = ZoneInfo(definition.definition["schedule"]["tz"] or "UTC")
        date = datetime.fromtimestamp(claim.identity.scheduled_fire_time_ms / 1000, UTC).astimezone(zone).strftime("%Y-%m-%d")
        return definition, payload, date

    async def verify(self, claim: OccurrenceClaim, turn: TurnResult) -> tuple[bool, str]:
        _definition, payload, date = self._job(claim)
        if not payload.verifiers:
            return True, "not_configured"
        if not isinstance(turn.content, str):
            return False, "non_text_result"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(dir=self.root, prefix="response-") as folder:
            response = Path(folder) / "response.txt"
            response.write_text(turn.content, encoding="utf-8")
            for verifier in payload.verifiers:
                cwd = Path(verifier.cwd).expanduser().resolve() if verifier.cwd else self.default_cwd
                values = {"date": date, "repo_root": str(cwd), "response_file": str(response), "status": ""}
                if verifier.status_file:
                    status_path = Path(_render([verifier.status_file], values)[0])
                    if not status_path.is_absolute():
                        status_path = cwd / status_path
                    try:
                        match = re.search(verifier.status_regex, status_path.read_text(encoding="utf-8"))
                        values["status"] = match.group(1) if match else ""
                    except OSError:
                        return False, "status_file_unavailable"
                outcome = await run_adapter(_render(verifier.argv, values), cwd=cwd, timeout=verifier.timeout)
                if outcome.returncode != 0:
                    return False, outcome.reason
                text = outcome.stdout.decode("utf-8", errors="replace").strip()
                try:
                    # Retain main's support for a log prefix before one JSON object.
                    decoded: object = json.loads(text[text.index("{"):text.rindex("}") + 1])
                except (ValueError, UnicodeError):
                    return False, "missing_verifier_verdict"
                if not isinstance(decoded, dict):
                    return False, "missing_verifier_verdict"
                verdict = cast(dict[str, object], decoded)
                if verdict.get("verified") is not True and verdict.get("ok") is not True:
                    return False, "verifier_did_not_pass"
            if response.read_text(encoding="utf-8") != turn.content:
                return False, "response_changed"
        return True, "verified"

    async def _once(self, purpose: str, observation: CronCallbackObservation,
                    callback: Callable[[CronCallbackObservation], Awaitable[object]]) -> CronCallbackRecord:
        store = CronCallbackStore(self.root / purpose, callback)
        record = store.publish(observation)
        return await store.invoke_once(observation) if record.state == "pending" else record

    async def after_delivery(self, claim: OccurrenceClaim, projection: CronCompletionProjection,
                             deliveries: tuple[SenderResult, ...], sender: CronChannelSender,
                             *, blocked_reason: str | None = None,
                             deferred_retry_at_ms: int | None = None) -> CronHookOutcome:
        definition, payload, date = self._job(claim)
        ack: Literal["success", "failed", "unknown", "blocked"] | None = None
        reason = blocked_reason
        if deferred_retry_at_ms is not None:
            reason = "quota_retry_scheduled"
        elif blocked_reason is not None:
            ack = "blocked" if payload.deliver else None
        elif payload.delivery_destinations():
            if deliveries and all(isinstance(item, ConfirmedDelivered) for item in deliveries):
                ack = "success"
            elif any(isinstance(item, (ConfirmedNotSent, PermanentNotSent)) for item in deliveries):
                ack, reason = "failed", "delivery_failed"
            else:
                ack, reason = "unknown", "delivery_unconfirmed"
        observation = CronCallbackObservation(claim.occurrence_id, claim.run_id, definition.job_id,
                                             definition.definition_revision, claim.generation, "result", projection.status)
        bookkeeping = None
        if ack is not None and payload.post_delivery_command:
            values = {"date": date, "ack_status": ack, "repo_root": str(self.default_cwd)}

            async def record_delivery(_observation: CronCallbackObservation) -> object:
                outcome = await run_adapter(_render(payload.post_delivery_command or [], values),
                                            cwd=self.default_cwd, timeout=120)
                if outcome.returncode != 0:
                    raise RuntimeError("post_delivery_failed")
                return None

            bookkeeping = await self._once("post-delivery", observation, record_delivery)
            if bookkeeping.state != "succeeded" and reason is None:
                reason = "delivery_state_update_failed"
        alert = None
        destination = payload.alert_destination()
        if reason is not None and destination is not None:
            content = (f"{definition.definition['name']}: quota retry scheduled at "
                       f"{datetime.fromtimestamp(deferred_retry_at_ms / 1000, UTC).isoformat()}."
                       if deferred_retry_at_ms is not None else
                       f"{definition.definition['name']}: {reason}. Inspect the retained run before retrying.")
            identity = hashlib.sha256((claim.occurrence_id + ":owner-alert").encode()).hexdigest()
            digest = hashlib.sha256(json.dumps([destination.channel, destination.to, content]).encode()).hexdigest()
            request = DispatchRequest(identity, identity, 1, identity, digest,
                                      destination.channel, destination.to, [{"content": content, "media": []}])

            async def notify_owner(_observation: CronCallbackObservation) -> object:
                result = await sender.send(request)
                if not isinstance(result, ConfirmedDelivered):
                    raise RuntimeError("owner_alert_unconfirmed")
                return None

            alert = await self._once("owner-alert", observation, notify_owner)
        return CronHookOutcome(ack, bookkeeping, alert)
