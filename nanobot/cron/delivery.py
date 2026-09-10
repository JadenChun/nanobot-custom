"""Build de-duplicated outbound deliveries for cron results."""

import asyncio
import json
import re
from collections.abc import Iterable
from pathlib import Path

from loguru import logger

from nanobot.bus.events import DeliveryResult, OutboundMessage
from nanobot.cron.types import CronDestination, CronVerifier


def build_result_messages(
    content: str,
    destinations: Iterable[CronDestination],
    sent_messages: Iterable[OutboundMessage] = (),
) -> list[OutboundMessage]:
    """Build final-result messages that were not already sent explicitly."""
    sent = {
        (message.channel, message.chat_id, message.content, tuple(message.media))
        for message in sent_messages
    }
    return [
        OutboundMessage(channel=destination.channel, chat_id=destination.to, content=content)
        for destination in destinations
        if (destination.channel, destination.to, content, ()) not in sent
    ]



def is_client_deliverable(verification_verdict, *, skip_verification: bool) -> bool:
    """Whether a client-facing scheduled result may be delivered to the group.

    FAIL CLOSED: for ANY client-facing job, group delivery requires an explicit
    PASS verification verdict.  ``skip_verification`` does not bypass the gate;
    it only controls whether the framework's internal verifier runs.  A
    missing/FAIL/PARTIAL verdict never delivers to the group.
    """
    return verification_verdict == "PASS"



def classify_delivery_results(results: Iterable[DeliveryResult]) -> str:
    """Aggregate per-destination transport results into one outcome.

    Returns "success" when every required destination was confirmed (or there
    were none), "failed" when any confirmed transport failure occurred, and
    "unknown" when nothing failed but at least one result was unconfirmed.
    """
    statuses = [r.status for r in results]
    if not statuses:
        return "success"
    if any(s == "failed" for s in statuses):
        return "failed"
    if any(s == "unknown" for s in statuses):
        return "unknown"
    return "success"


async def send_owner_alert(
    bus,
    *,
    channel: str | None,
    to: str | None,
    job_name: str,
    failure_stage: str,
    reason: str,
) -> DeliveryResult | None:
    """Send ONE concise owner operational-alert DM via the existing transport.

    Returns ``None`` when no owner alert destination is configured (no-op).
    The alert is sent with the delivery-acknowledged transport so its own
    transport result can be logged; a failed/unknown owner alert is only
    logged and NEVER re-alerts (no recursion).
    """
    destination = CronDestination(channel=channel or "telegram", to=str(to)) if to else None
    if destination is None:
        return None
    msg = OutboundMessage(
        channel=destination.channel,
        chat_id=destination.to,
        content=f"\u26a0\ufe0f {job_name} {failure_stage}\n\n{reason}",
        metadata={"_owner_alert": True},
    )
    result = await bus.publish_outbound_and_wait(msg)
    if result.status == "success":
        logger.info("Owner alert delivered: {} — {}", job_name, failure_stage)
    elif result.status == "failed":
        logger.error(
            "Owner alert transport FAILED: {} — {} ({})",
            job_name, failure_stage, result.error,
        )
    else:
        logger.warning(
            "Owner alert confirmation UNKNOWN: {} — {} ({})",
            job_name, failure_stage, result.error,
        )
    return result



def _resolve_status(verifier, *, date: str, repo_root: str | None) -> str:
    """Resolve a verifier's `{status}` input from its configured status file."""
    if not verifier.status_file:
        return ""
    path = str(verifier.status_file).replace("{date}", date).replace("{repo_root}", repo_root or "")
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""
    match = re.search(verifier.status_regex, text)
    return match.group(1) if match else ""


def _render_verifier_argv(
    argv: Iterable[str],
    *,
    date: str,
    repo_root: str | None,
    response_file: str | None = None,
    status: str = "",
) -> list[str]:
    """Substitute the run-scoped templates in a verifier argv.

    ``{date}``          - the run's date in the job timezone
    ``{repo_root}``     - the verifier working directory
    ``{response_file}`` - a file containing THIS run's exact client response
    ``{status}``        - an internal status resolved from ``status_file``
    """
    rendered = []
    for a in argv:
        value = str(a).replace("{date}", date).replace("{repo_root}", repo_root or "")
        if response_file is not None:
            value = value.replace("{response_file}", response_file)
        value = value.replace("{status}", status)
        rendered.append(value)
    return rendered


def _verifier_passed(returncode: int, stdout: bytes) -> str:
    """Classify one verifier process result as PASS / FAIL / MISSING.

    exit != 0 -> FAIL.  exit == 0 -> require a truthful JSON object on stdout
    (``verified`` or ``ok`` is true); otherwise MISSING (malformed/untruthful).
    """
    if returncode != 0:
        return "FAIL"
    text = (stdout or b"").decode("utf-8", "replace").strip()
    if not text:
        return "MISSING"
    # The verifiers print a single JSON object; tolerate leading noise.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return "MISSING"
    try:
        payload = json.loads(text[start:end + 1])
    except Exception:
        return "MISSING"
    if not isinstance(payload, dict):
        return "MISSING"
    if payload.get("verified") is True or payload.get("ok") is True:
        return "PASS"
    return "MISSING"


async def run_external_verifiers(
    verifiers: Iterable["CronVerifier"],
    *,
    date: str,
    default_cwd: str | None = None,
    response_file: str | None = None,
) -> tuple[str, list[dict]]:
    """Run authoritative external verifiers and aggregate the result.

    Each verifier runs via ``asyncio.create_subprocess_exec`` (no shell).  The
    per-verifier outcome is PASS / FAIL / MISSING (timeout, exec error, or
    malformed/untruthful output).  Aggregation:

        any MISSING -> MISSING
        else any FAIL -> FAIL
        else -> PASS

    Returns ``(aggregate, details)``.  Only PASS is eligible for client
    delivery.  Verifier stdout/stderr is NEVER returned as client content.
    """
    details: list[dict] = []
    for verifier in verifiers:
        cwd = verifier.cwd or default_cwd
        argv = _render_verifier_argv(
            verifier.argv,
            date=date,
            repo_root=cwd,
            response_file=response_file,
            status=_resolve_status(verifier, date=date, repo_root=cwd),
        )
        status = "MISSING"
        error = ""
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError, NotImplementedError) as exc:
            details.append({"name": verifier.name, "status": "MISSING", "error": str(exc)[:300]})
            continue
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=verifier.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            details.append({"name": verifier.name, "status": "MISSING", "error": "verifier timed out"})
            continue
        status = _verifier_passed(proc.returncode, stdout)
        if status != "PASS":
            error = (stderr or b"").decode("utf-8", "replace").strip()[-300:]
        details.append({"name": verifier.name, "status": status, "error": error})

    statuses = [d["status"] for d in details]
    if any(s == "MISSING" for s in statuses):
        aggregate = "MISSING"
    elif any(s == "FAIL" for s in statuses):
        aggregate = "FAIL"
    else:
        aggregate = "PASS"
    return aggregate, details


def build_explicit_fanout_messages(
    destinations: list[CronDestination],
    sent_messages: Iterable[OutboundMessage],
) -> list[OutboundMessage]:
    """Copy explicit primary-destination messages to missing destinations."""
    if len(destinations) < 2:
        return []

    messages = list(sent_messages)
    primary = destinations[0]
    primary_messages = [
        message
        for message in messages
        if message.channel == primary.channel and message.chat_id == primary.to
    ]
    sent = {
        (message.channel, message.chat_id, message.content, tuple(message.media))
        for message in messages
    }

    copies: list[OutboundMessage] = []
    for message in primary_messages:
        for destination in destinations[1:]:
            signature = (
                destination.channel,
                destination.to,
                message.content,
                tuple(message.media),
            )
            if signature in sent:
                continue
            sent.add(signature)
            copies.append(OutboundMessage(
                channel=destination.channel,
                chat_id=destination.to,
                content=message.content,
                media=list(message.media),
            ))
    return copies
