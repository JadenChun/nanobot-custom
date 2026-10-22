"""Gateway-owned activation of the qualified fresh-turn scheduler."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Callable

from nanobot.channels.delivery_contract import ReceiptCapableTextChannel
from nanobot.config.schema import GatewayCronConfig
from nanobot.cron.canonical_execution import SilentCronAgent
from nanobot.cron.channel_sender import ExactChannelAuthority
from nanobot.cron.job_hooks import CronJobHooks
from nanobot.cron.quota_recovery import CronQuotaRecovery, PendingQuotaRetry
from nanobot.cron.runtime import CronRuntime, GatewayCronOptions, SystemScheduleClock
from nanobot.cron.schedule import ScheduleClock
from nanobot.cron.service import CronService
from nanobot.cron.types import CronJob, cron_thread_id
from nanobot.session.manager import SessionManager


def validate_gateway_job(job: CronJob) -> None:
    """Reject unsupported contracts before any computation or delivery."""
    payload = job.payload
    if (payload.kind != "agent_turn" or payload.execution_mode != "legacy_fresh"
            or payload.session_key != f"cron:{job.id}"
            or payload.origin_channel != (payload.channel or "cli")
            or payload.origin_chat_id != str(payload.to or "direct")
            or not payload.message.strip()):
        raise ValueError(f"cron job {job.id} requires explicit fresh-turn migration before activation")
    cron_thread_id(payload.channel_meta)
    destinations = payload.delivery_destinations()
    if payload.deliver and not destinations:
        raise ValueError(f"cron job {job.id} has no delivery destination")
    if any(item.channel != "telegram" for item in destinations):
        raise ValueError(f"cron job {job.id} requires a qualified Telegram receipt transport")
    alert = payload.alert_destination()
    if alert is not None and alert.channel != "telegram":
        raise ValueError(f"cron job {job.id} has an unsupported alert transport")


class GatewayCron:
    def __init__(self, config: GatewayCronConfig, service: CronService,
                 agent: SilentCronAgent, sessions: SessionManager,
                 channels: ExactChannelAuthority, workspace: Path, runtime_root: Path,
                 *, clock: ScheduleClock | None = None,
                 is_codex: Callable[[], bool] = lambda: False) -> None:
        if not config.enabled:
            raise ValueError("gateway cron requires explicit enablement")
        self.config = config.model_copy(deep=True)
        self.service, self.channels = service, channels
        self._validate_jobs()
        cwd = Path(config.adapter_cwd).expanduser() if config.adapter_cwd else workspace
        if not cwd.is_absolute() or not cwd.is_dir():
            raise ValueError("gateway.cron.adapterCwd must name an existing absolute directory")
        identity = hashlib.sha256(b"nanobot-cron-runtime-root-v1\0" + str(workspace).encode()
            + b"\0" + service.action_store.namespace.encode() + b"\0" + str(runtime_root).encode()).hexdigest()
        root = runtime_root / "cron-runtime" / identity
        clock = clock or SystemScheduleClock()
        self.runtime = CronRuntime(service, agent, sessions, channels, root, identity,
            GatewayCronOptions(clock, config.drain_timeout_s, config.poll_interval_s),
            job_hooks=CronJobHooks(root / "job-hooks", default_cwd=cwd),
            quota_recovery=CronQuotaRecovery(root / "quota-recovery", clock, is_codex=is_codex)
                if config.codex_quota_retry else None)
        service.configure_gateway_execution(validate_gateway_job, self.status)
        self._state = "waiting_for_channels"

    def _import_retry_budget(self) -> None:
        quota = self.runtime.quota_recovery
        receipt = self.service.store_path.parent / "migration-receipt.json"
        if quota is None or not receipt.exists():
            return
        self.runtime.scheduler.require_lifetime_lease()
        from nanobot.cron.import_copy import CronImportReport

        if receipt.is_symlink() or receipt.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("invalid cron migration receipt")
        report = CronImportReport.model_validate_json(receipt.read_bytes())
        if report.namespace != self.service.action_store.namespace:
            raise ValueError("cron migration receipt namespace mismatch")
        digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
        for job in self.service.list_jobs(include_disabled=True):
            identity = report.definitions.get(job.id)
            state = report.legacy_states.get(job.id)
            if identity is None or state is None or not job.job_instance_id:
                continue
            instance, revision = identity.split(":", 1)
            if instance != job.job_instance_id:
                raise ValueError("cron migration retry identity mismatch")
            count, due = state.get("rateLimitRetries", 0), state.get("retryAtMs")
            if type(count) is not int or count < 0 or (due is not None and (type(due) is not int or due < 0)):
                raise ValueError("invalid legacy quota state")
            pending = PendingQuotaRetry(job_id=job.id, instance=instance, revision=revision,
                origin="legacy:" + digest, retry_at_ms=due, state="unknown") if due is not None else None
            quota.import_legacy(pending, instance=instance, retry_count=count)

    def _validate_jobs(self) -> None:
        for job in self.service.list_jobs():
            validate_gateway_job(job)

    def _channels_ready(self) -> bool:
        required = {destination.channel for job in self.service.list_jobs()
                    for destination in job.payload.delivery_destinations()}
        required.update(alert.channel for job in self.service.list_jobs()
                        if (alert := job.payload.alert_destination()) is not None)
        for name in required:
            channel = self.channels.get_channel(name)
            if channel is None or not isinstance(channel, ReceiptCapableTextChannel):
                raise ValueError(f"scheduled execution requires an enabled receipt channel: {name}")
            if getattr(channel, "receipt_transport_ready", False) is not True:
                return False
        return True

    async def run(self) -> None:
        try:
            async with asyncio.timeout(self.config.readiness_timeout_s):
                while not self._channels_ready():
                    await asyncio.sleep(0.05)
            self._validate_jobs()
            self.runtime.acquire()
            self._import_retry_budget()
            self._state = "running"
            # The gateway awaits this task alongside its channels and agent.
            # A failed timer therefore fails the owning gateway visibly.
            await self.runtime.run_timer()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._state = "failed"
            raise

    def status(self) -> dict[str, object]:
        result = self.runtime.status()
        result["state"] = self._state
        return result

    async def aclose(self) -> None:
        self._state = "stopping"
        await self.runtime.aclose()
        self._state = "closed"
