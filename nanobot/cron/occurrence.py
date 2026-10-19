"""Definition-bound occurrence admission for the staged cron runtime.

Package C owns the durable definition candidate.  This module is the narrow
adapter between that candidate and :class:`CronOccurrenceStore`: it admits a
claim only from a currently committed, enabled definition and routes every
source (timer, manual, and system) through the same occurrence identity.  It
does not schedule, execute, publish, or deliver a job.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Mapping, cast

from nanobot.cron.action_store import CronActionStoreError
from nanobot.cron.definition_store import (
    CronDefinition,
    CronDefinitionStore,
    DefinitionStoreError,
)
from nanobot.cron.occurrence_store import (
    CronOccurrenceStore,
    OccurrenceClaim,
    OccurrenceRecord,
    new_occurrence_owner,
)

OccurrenceSource = Literal["timer", "manual", "system"]
_OCCURRENCE_SOURCES = frozenset({"timer", "manual", "system"})


class CronOccurrenceRuntimeError(RuntimeError):
    """Base error for definition-bound occurrence admission."""


class CronOccurrenceRuntimeUnavailableError(CronOccurrenceRuntimeError):
    """The optional D authority was not configured on a management service."""


class CronOccurrenceDefinitionConflictError(CronOccurrenceRuntimeError):
    """The requested occurrence no longer matches committed definition state."""


class CronOccurrenceDefinitionUnavailableError(CronOccurrenceDefinitionConflictError):
    """The requested definition is missing or has not been durably committed."""


class CronOccurrenceDisabledError(CronOccurrenceDefinitionConflictError):
    """A disabled definition cannot be admitted for execution."""


@dataclass(frozen=True)
class CronOccurrenceAdmission:
    """A claim plus the exact definition and source used for admission."""

    claim: OccurrenceClaim
    definition: CronDefinition
    source: OccurrenceSource

    @property
    def occurrence_id(self) -> str:
        return self.claim.occurrence_id

    @property
    def run_id(self) -> str:
        return self.claim.run_id


def _copy_json_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Detach a JSON mapping without allowing a caller to mutate a claim."""
    import json

    try:
        copied = json.loads(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise CronOccurrenceRuntimeError(f"{label} must contain finite JSON data") from exc
    if not isinstance(copied, dict):  # pragma: no cover - mappings encode as objects
        raise CronOccurrenceRuntimeError(f"{label} must be a JSON object")
    # JSON decoding constructs string-keyed objects with opaque wire values.
    return cast(dict[str, Any], copied)


def _destination_policy(definition: CronDefinition) -> dict[str, Any]:
    """Freeze delivery context without granting any delivery authority."""
    payload = definition.definition["payload"]
    return _copy_json_mapping(
        {
            "deliver": payload["deliver"],
            "channel": payload["channel"],
            "to": payload["to"],
            "additionalDestinations": deepcopy(payload["additionalDestinations"]),
            "channelMeta": deepcopy(payload["channelMeta"]),
            "sessionKey": payload["sessionKey"],
            "originChannel": payload["originChannel"],
            "originChatId": payload["originChatId"],
            "originMetadata": deepcopy(payload["originMetadata"]),
        },
        label="cron destination policy",
    )


def destination_policy(definition: CronDefinition) -> dict[str, object]:
    """Return the existing detached policy without granting delivery authority."""
    return _destination_policy(definition)


def scheduler_destination_policy_conflicts(definition: CronDefinition, record: OccurrenceRecord) -> bool:
    return _destination_policy(definition) != record.destination_policy


class CronOccurrenceRuntime:
    """Admission adapter over committed definitions and durable occurrences."""

    def __init__(
        self,
        definition_store: CronDefinitionStore,
        occurrence_store: CronOccurrenceStore,
    ) -> None:
        if not isinstance(cast(object, definition_store), CronDefinitionStore):
            raise TypeError("definition_store must be a CronDefinitionStore")
        if not isinstance(cast(object, occurrence_store), CronOccurrenceStore):
            raise TypeError("occurrence_store must be a CronOccurrenceStore")
        if definition_store.action_store.namespace != occurrence_store.namespace:
            raise ValueError("definition and occurrence stores must share one namespace")
        self.definition_store = definition_store
        self.occurrence_store = occurrence_store
        self.owner = new_occurrence_owner()

    def _committed_definition(self, job_id: str) -> CronDefinition:
        if not isinstance(cast(object, job_id), str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        try:
            replay = self.definition_store.replay()
        except (DefinitionStoreError, CronActionStoreError) as exc:
            raise CronOccurrenceDefinitionConflictError(
                "cron definition authority could not be replayed"
            ) from exc
        if replay.pending_actions:
            raise CronOccurrenceDefinitionConflictError(
                "cron definition has a durable action awaiting checkpoint"
            )
        definition = next(
            (item for item in replay.candidate.definitions if item.job_id == job_id),
            None,
        )
        if definition is None:
            raise CronOccurrenceDefinitionUnavailableError(
                f"cron definition {job_id!r} does not exist"
            )
        return definition

    def committed_definition(self, job_id: str) -> CronDefinition:
        """Read through the original replay and pending-checkpoint fences."""
        return self._committed_definition(job_id)

    @staticmethod
    def _validate_expected_definition(
        definition: CronDefinition,
        *,
        expected_job_instance_id: str | None,
        expected_definition_revision: str | None,
    ) -> None:
        if (
            expected_job_instance_id is not None
            and expected_job_instance_id != definition.job_instance_id
        ):
            raise CronOccurrenceDefinitionConflictError(
                "cron definition job instance does not match committed definition"
            )
        if (
            expected_definition_revision is not None
            and expected_definition_revision != definition.definition_revision
        ):
            raise CronOccurrenceDefinitionConflictError(
                "cron definition revision does not match committed definition"
            )

    @staticmethod
    def _validate_source(source: str) -> OccurrenceSource:
        if source not in _OCCURRENCE_SOURCES:
            raise ValueError("occurrence source must be timer, manual, or system")
        return source  # type: ignore[return-value]

    @staticmethod
    def _validate_fire_time(scheduled_fire_time_ms: int) -> int:
        if type(scheduled_fire_time_ms) is not int:
            raise ValueError("scheduled_fire_time_ms must be an exact integer")
        return scheduled_fire_time_ms

    def claim(
        self,
        *,
        job_id: str,
        scheduled_fire_time_ms: int,
        source: OccurrenceSource,
        owner: str | None = None,
        expected_job_instance_id: str | None = None,
        expected_definition_revision: str | None = None,
    ) -> OccurrenceClaim:
        """Admit one source through the shared durable occurrence identity.

        The definition is read before and after the exclusive ledger claim.
        If either read does not agree, the claim is left in the explicit
        reconciliation-required state and no claim is returned to the caller.
        This avoids treating a cross-store race as successful execution.
        """
        source = self._validate_source(source)
        fire_time = self._validate_fire_time(scheduled_fire_time_ms)
        definition = self._committed_definition(job_id)
        self._validate_expected_definition(
            definition,
            expected_job_instance_id=expected_job_instance_id,
            expected_definition_revision=expected_definition_revision,
        )
        if not definition.definition["enabled"]:
            raise CronOccurrenceDisabledError(f"cron definition {job_id!r} is disabled")
        claim = self.occurrence_store.claim_scheduled(
            job_id=definition.job_id,
            job_revision=definition.definition_revision,
            scheduled_fire_time_ms=fire_time,
            job_snapshot=_copy_json_mapping(definition.to_wire(), label="cron job snapshot"),
            destination_policy=_destination_policy(definition),
            owner=self.owner if owner is None else owner,
        )

        try:
            confirmed = self._committed_definition(job_id)
            self._validate_expected_definition(
                confirmed,
                expected_job_instance_id=expected_job_instance_id,
                expected_definition_revision=expected_definition_revision,
            )
        except CronOccurrenceRuntimeError:
            raise
        if (
            confirmed.job_instance_id != definition.job_instance_id
            or confirmed.definition_revision != definition.definition_revision
        ):
            raise CronOccurrenceDefinitionConflictError(
                "cron definition changed while occurrence admission was being committed"
            )
        return claim

    def admit(
        self,
        *,
        job_id: str,
        scheduled_fire_time_ms: int,
        source: OccurrenceSource,
        owner: str | None = None,
        expected_job_instance_id: str | None = None,
        expected_definition_revision: str | None = None,
    ) -> CronOccurrenceAdmission:
        """Return claim and provenance for future runtime adapters."""
        claim = self.claim(
            job_id=job_id,
            scheduled_fire_time_ms=scheduled_fire_time_ms,
            source=source,
            owner=owner,
            expected_job_instance_id=expected_job_instance_id,
            expected_definition_revision=expected_definition_revision,
        )
        definition = self._committed_definition(job_id)
        if (
            definition.definition_revision != claim.identity.job_revision
            or definition.job_id != claim.identity.job_id
            or definition.job_instance_id != claim.job_snapshot.get("jobInstanceId")
        ):
            raise CronOccurrenceDefinitionConflictError(
                "cron definition changed after occurrence admission"
            )
        return CronOccurrenceAdmission(claim, definition, source)

    def claim_timer(
        self,
        job_id: str,
        scheduled_fire_time_ms: int,
        *,
        owner: str | None = None,
        expected_job_instance_id: str | None = None,
        expected_definition_revision: str | None = None,
    ) -> OccurrenceClaim:
        return self.claim(
            job_id=job_id,
            scheduled_fire_time_ms=scheduled_fire_time_ms,
            source="timer",
            owner=owner,
            expected_job_instance_id=expected_job_instance_id,
            expected_definition_revision=expected_definition_revision,
        )

    claim_scheduled = claim_timer

    def claim_manual(
        self,
        job_id: str,
        scheduled_fire_time_ms: int,
        *,
        owner: str | None = None,
        expected_job_instance_id: str | None = None,
        expected_definition_revision: str | None = None,
    ) -> OccurrenceClaim:
        return self.claim(
            job_id=job_id,
            scheduled_fire_time_ms=scheduled_fire_time_ms,
            source="manual",
            owner=owner,
            expected_job_instance_id=expected_job_instance_id,
            expected_definition_revision=expected_definition_revision,
        )

    def claim_system(
        self,
        job_id: str,
        scheduled_fire_time_ms: int,
        *,
        owner: str | None = None,
        expected_job_instance_id: str | None = None,
        expected_definition_revision: str | None = None,
    ) -> OccurrenceClaim:
        return self.claim(
            job_id=job_id,
            scheduled_fire_time_ms=scheduled_fire_time_ms,
            source="system",
            owner=owner,
            expected_job_instance_id=expected_job_instance_id,
            expected_definition_revision=expected_definition_revision,
        )

    def reconciliation_required(self) -> list[OccurrenceRecord]:
        """Expose claimed crash states without treating them as successful."""
        return self.occurrence_store.reconciliation_required()


# Compatibility aliases keep the error vocabulary discoverable to callers
# without creating a second authority or a second occurrence implementation.
OccurrenceDefinitionConflictError = CronOccurrenceDefinitionConflictError
OccurrenceDefinitionUnavailableError = CronOccurrenceDefinitionUnavailableError
CronOccurrenceAuthority = CronOccurrenceRuntime


__all__ = [
    "CronOccurrenceAdmission",
    "CronOccurrenceAuthority",
    "CronOccurrenceDefinitionConflictError",
    "CronOccurrenceDefinitionUnavailableError",
    "CronOccurrenceDisabledError",
    "CronOccurrenceRuntime",
    "CronOccurrenceRuntimeError",
    "CronOccurrenceRuntimeUnavailableError",
    "OccurrenceDefinitionConflictError",
    "OccurrenceDefinitionUnavailableError",
    "OccurrenceSource",
]
