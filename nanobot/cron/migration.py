"""Non-destructive preparation of staging cron records for bound execution.

This module does not activate, save, or execute jobs. Callers must inspect the
admission diagnostic and install the legacy fresh-history execution adapter.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from nanobot.cron.types import CronJob


@dataclass(frozen=True)
class PreparedCronJob:
    job: CronJob
    execution_mode: str
    diagnostic: str | None = None


def prepare_staging_job(job: CronJob) -> PreparedCronJob:
    """Derive execution binding without changing delivery or historical state.

    Staging route defaults are cli/direct; its history namespace is cron:<id>.
    Existing complete upstream bindings keep their original session semantics.
    Partial explicit bindings are ambiguous and are preserved for reconciliation.
    """
    prepared = deepcopy(job)
    payload = prepared.payload
    if payload.kind != 'agent_turn':
        return PreparedCronJob(prepared, 'system_event')
    if payload.execution_mode not in ('session_bound', 'legacy_fresh'):
        return PreparedCronJob(prepared, 'unresolved', 'unsupported cron execution mode')
    explicit = (payload.session_key, payload.origin_channel, payload.origin_chat_id)
    if any(explicit):
        if not all(explicit):
            return PreparedCronJob(prepared, 'unresolved', 'partial explicit cron binding requires reconciliation')
        legacy_route = (payload.channel or 'cli', str(payload.to or 'direct'))
        if (payload.channel or payload.to) and legacy_route != (payload.origin_channel, payload.origin_chat_id):
            return PreparedCronJob(prepared, 'unresolved', 'legacy route conflicts with explicit cron origin')
        return PreparedCronJob(prepared, payload.execution_mode)
    payload.execution_mode = 'legacy_fresh'
    payload.session_key = f'cron:{job.id}'
    payload.origin_channel = payload.channel or 'cli'
    payload.origin_chat_id = str(payload.to or 'direct')
    if not payload.origin_metadata:
        payload.origin_metadata = deepcopy(payload.channel_meta)
    return PreparedCronJob(prepared, 'legacy_fresh')
