"""Artifact owner operations preserve durable evidence and legacy fault seams."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from nanobot.session.artifact import ArtifactValidationError
from nanobot.session.artifact_grants import ArtifactGrantSet, ArtifactHandoffReceipt
from nanobot.session.artifact_reader import ArtifactReader
from tests.session.test_artifact_grants import _durable_child_evidence, published_artifact


async def test_wrong_merge_capability_rejected_before_receipt_iteration(tmp_path: Path) -> None:
    async with _durable_child_evidence(tmp_path) as (child, collector, receipt, _, _, _):
        before = child.owner.read_snapshot()
        observations: list[str] = []

        class ObservedReceipts(list[ArtifactHandoffReceipt]):
            def __iter__(self) -> Iterator[ArtifactHandoffReceipt]:
                observations.append("iterate")
                return super().__iter__()

        with pytest.raises(ArtifactValidationError, match="Artifact handoff is unavailable"):
            receipt.copy_tool_events_from(object(), ObservedReceipts([receipt]))

        assert observations == []
        assert receipt.belongs_to(collector)
        assert child.owner.read_snapshot() == before


async def test_finalization_rereads_receipt_collection_after_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _durable_child_evidence(tmp_path) as (child, collector, selected, _, _, _):
        before = child.owner.read_snapshot()
        original_seal = ArtifactHandoffReceipt._seal
        original_invalidate = ArtifactHandoffReceipt._invalidate
        trace: list[str] = []

        def seal(merged: ArtifactHandoffReceipt, capability: object) -> None:
            trace.append("seal")
            original_seal(merged, capability)
            collector._receipts = []

        def invalidate(receipt: ArtifactHandoffReceipt, capability: object) -> None:
            trace.append("invalidate")
            original_invalidate(receipt, capability)

        with monkeypatch.context() as patch:
            patch.setattr(ArtifactHandoffReceipt, "_seal", seal)
            patch.setattr(ArtifactHandoffReceipt, "_invalidate", invalidate)
            merged = collector.finalize(stop_reason="completed")

        assert merged is not None and merged.belongs_to(collector)
        assert trace == ["seal"]
        assert selected.belongs_to(collector)
        assert child.owner.read_snapshot() == before


async def test_grant_issuer_private_hook_remains_late_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with published_artifact(tmp_path) as (_, writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, "root_history", store, tool)
        child = writer.child_stream()
        before = writer.owner.read_snapshot()
        original = ArtifactReader._validate_grant_issuer
        failure = RuntimeError("controlled issuer fault")
        trace: list[ArtifactReader] = []

        def reject_after_validation(value: ArtifactReader) -> None:
            original(value)
            trace.append(value)
            raise failure

        with monkeypatch.context() as patch:
            patch.setattr(ArtifactReader, "_validate_grant_issuer", reject_after_validation)
            with pytest.raises(RuntimeError) as raised:
                ArtifactGrantSet.issue_downward(reader, child, [descriptor.reference])
            assert raised.value is failure

        assert trace == [reader]
        assert writer.owner.read_snapshot() == before
