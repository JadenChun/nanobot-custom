"""Pure artifact binding checks retain genuine recovery hooks and fault ordering."""

from pathlib import Path

import pytest

from nanobot.session.artifact_reader import ArtifactReader
from nanobot.session.artifact_rematerialization import ArtifactRematerializer
from tests.session.test_artifact_reader import setup_read
from tests.session.test_artifact_store import make_store


def test_store_identity_observation_is_pure_even_after_close(tmp_path: Path) -> None:
    store = make_store(tmp_path / "artifacts")
    foreign = make_store(tmp_path / "foreign")
    with store.transaction() as transaction:
        before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
        assert transaction.matches_store(store)
        assert not transaction.matches_store(foreign)
        assert not transaction.matches_store(None)
        transaction.assert_reachable()
        assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before

    assert transaction.matches_store(store)
    assert not transaction.matches_store(foreign)
    assert not transaction.matches_store(None)
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


async def test_rematerialization_root_fault_stays_late_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, content):
        reader = ArtifactReader(writer, "root_history", store, tool)
        before = writer.owner.read_snapshot()
        failure = RuntimeError("controlled root hook fault")
        trace: list[ArtifactReader] = []
        original = ArtifactReader._validate_private_root

        def reject_after_validation(value: ArtifactReader) -> None:
            original(value)
            trace.append(value)
            raise failure

        with monkeypatch.context() as patch:
            patch.setattr(ArtifactReader, "_validate_private_root", reject_after_validation)
            with pytest.raises(RuntimeError) as caught:
                ArtifactRematerializer(reader).rematerialize(
                    descriptor.reference, expected=before.state,
                )
            assert caught.value is failure

        assert trace == [reader]
        assert writer.owner.read_snapshot() == before
        assert next(store.root.rglob("*.blob")).read_bytes() == content


async def test_locked_native_source_fault_retains_exact_recovery_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, content):
        reader = ArtifactReader(writer, "root_history", store, tool)
        before = writer.owner.read_snapshot()
        original = ArtifactReader._verify_native_source_locked
        failure = RuntimeError("controlled locked source hook fault")
        trace: list[str] = []

        def verify(value, selected, snapshot):
            receipt = original(value, selected, snapshot)
            trace.append(selected.artifact_id)
            if len(trace) == 2:
                raise failure
            return receipt

        with monkeypatch.context() as patch:
            patch.setattr(ArtifactReader, "_verify_native_source_locked", verify)
            with pytest.raises(RuntimeError) as caught:
                ArtifactRematerializer(reader).rematerialize(
                    descriptor.reference, expected=before.state,
                )
            assert caught.value is failure

        assert trace == [descriptor.artifact_id, descriptor.artifact_id]
        assert writer.owner.read_snapshot() == before
        assert next(store.root.rglob("*.blob")).read_bytes() == content
