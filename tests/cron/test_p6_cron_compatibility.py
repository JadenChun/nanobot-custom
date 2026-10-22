"""Non-waiver positive extensions paired with the retained legacy test nodes."""

import pytest

from tests.cron import p6_compat_fixtures as cases
from tests.cron.stage3_fixtures import offline_denial  # noqa: F401


async def test_reentrant_read_during_real_computation(tmp_path):
    await cases.reentrant(tmp_path)


async def test_distinct_manual_occurrences_overlap_without_duplicate_effects(tmp_path):
    await cases.overlap(tmp_path)


async def test_shutdown_fences_new_ingress_and_drains_admitted_work(tmp_path):
    await cases.overlap(tmp_path, closing=True)


@pytest.mark.parametrize("disable", [False, True])
async def test_external_commit_and_disable_under_real_scheduler_lease(tmp_path, disable):
    await cases.external_scheduler(tmp_path, disable=disable)


async def test_add_during_compute_preserves_frozen_work_and_new_commit(tmp_path):
    await cases.reentrant(tmp_path, add=True)


async def test_history_survives_revision_change_and_disable(tmp_path):
    await cases.history_mutation(tmp_path)


@pytest.mark.parametrize("field,value", [
    ("schedule.everyMs", "60000"), ("state.nextRunAtMs", "100"),
    ("state.lastRunAtMs", "90"), ("state.runHistory.0.runAtMs", None),
    ("state.runHistory.0.durationMs", None), ("createdAtMs", None),
    ("updatedAtMs", None),
])
def test_strict_conversion_checks_each_scalar_independently(tmp_path, field, value):
    cases.strict_scalar(tmp_path, field, value)


def test_routes_and_anonymous_legacy_history_are_lossless(tmp_path):
    cases.route_roundtrip(tmp_path / "new", update=True)
    cases.route_roundtrip(tmp_path / "old", legacy=True, anonymous=True)


async def test_canonical_completion_is_bound_and_reopened(tmp_path):
    await cases.completion(tmp_path)


async def test_forged_run_identity_is_rejected_without_send(tmp_path):
    await cases.completion(tmp_path, forge=True)


async def test_read_ring_bounded_without_pruning_occurrences(tmp_path):
    await cases.ring(tmp_path)
