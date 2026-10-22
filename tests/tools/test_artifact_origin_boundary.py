"""Public artifact origin parsing preserves its original private owner algorithm."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from nanobot.agent.tools import read_provenance
from nanobot.session import artifact


def _origin_record(**changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "kind": "filesystem",
        "access_class": "workspace",
        "repository": None,
        "relative_path": "folder/file.txt",
    }
    result.update(changes)
    return result


def _receipt_record(**changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "source_access": _origin_record(),
        "root_binding": "0" * 64,
        "byte_length": 10,
        "sha256": "1" * 64,
        "schema_version": 1,
        "producer": "builtin_read_file_text",
        "block_index": None,
    }
    result.update(changes)
    return result


@pytest.mark.parametrize("access,repository", [
    ("workspace", None),
    ("context_repository", "context"),
    ("target_repository", '["context","target"]'),
])
def test_public_origin_parser_retains_real_locator_grammar(access, repository):
    raw = _origin_record(access_class=access, repository=repository)
    result = artifact.parse_filesystem_artifact_origin(raw)
    assert result == artifact._origin(raw)
    assert result.kind == "filesystem"
    assert result.access_class == access
    assert result.repository == repository
    assert result.relative_path == "folder/file.txt"


@pytest.mark.parametrize("raw,message", [
    (None, "Invalid artifact source locator"),
    ({}, "Invalid artifact source locator"),
    (_origin_record(extra=True), "Invalid artifact source locator"),
    (_origin_record(kind="other"), "Unsupported artifact provenance adapter"),
    (_origin_record(kind="other", relative_path="../file"),
     "Unsupported artifact provenance adapter"),
    (_origin_record(access_class="unknown"), "Invalid artifact source access class"),
    (_origin_record(repository="claimed"), "Workspace artifact cannot claim a repository"),
    (_origin_record(access_class="context_repository", repository=None),
     "Invalid artifact binding identity"),
    (_origin_record(access_class="target_repository", repository=" "),
     "Invalid artifact binding identity"),
    (_origin_record(relative_path=""), "Invalid relative artifact source locator"),
    (_origin_record(relative_path="../file"), "Invalid relative artifact source locator"),
    (_origin_record(relative_path="/file"), "Invalid relative artifact source locator"),
    (_origin_record(relative_path="a\\file"), "Invalid relative artifact source locator"),
    (_origin_record(relative_path="a:file"), "Invalid relative artifact source locator"),
    (_origin_record(relative_path="a//file"), "Invalid relative artifact source locator"),
    (_origin_record(relative_path="a/\x00file"), "Invalid relative artifact source locator"),
])
def test_public_origin_parser_retains_rejection_and_first_error(raw, message):
    with pytest.raises(artifact.ArtifactValidationError, match=message):
        artifact.parse_filesystem_artifact_origin(raw)
    with pytest.raises(artifact.ArtifactValidationError, match=message):
        artifact._origin(raw)


def test_receipt_parser_uses_the_public_origin_boundary(monkeypatch):
    raw = _receipt_record()
    expected = artifact.FilesystemArtifactOrigin("filesystem", "workspace", None, "returned.txt")
    calls: list[object] = []

    def parse(value: object) -> artifact.FilesystemArtifactOrigin:
        calls.append(value)
        return expected

    def select() -> Callable[[object], artifact.FilesystemArtifactOrigin]:
        return parse

    monkeypatch.setattr(artifact, "select_filesystem_artifact_origin_parser", select)
    result = read_provenance.parse_read_receipt(raw)
    assert calls == [raw["source_access"]]
    assert calls[0] is raw["source_access"]
    assert result.source_access is expected


def test_public_origin_and_receipt_keep_the_late_original_fault_seam(monkeypatch):
    fault = artifact.ArtifactValidationError("original origin seam")
    calls: list[object] = []

    def fail(value: object) -> artifact.FilesystemArtifactOrigin:
        calls.append(value)
        raise fault

    monkeypatch.setattr(artifact, "_origin", fail)
    raw = _origin_record()
    with pytest.raises(artifact.ArtifactValidationError) as direct:
        artifact.parse_filesystem_artifact_origin(raw)
    assert direct.value is fault and calls == [raw]
    receipt = _receipt_record(source_access=raw)
    with pytest.raises(artifact.ArtifactValidationError) as consumer:
        read_provenance.parse_read_receipt(receipt)
    assert consumer.value is fault and calls == [raw, raw]


@pytest.mark.parametrize("changes", [
    {"schema_version": True}, {"producer": "foreign"}, {"block_index": 0},
    {"byte_length": -1}, {"sha256": "invalid"}, {"extra": True},
])
def test_receipt_envelope_rejection_stays_before_origin_parsing(monkeypatch, changes):
    calls: list[object] = []

    def fail(value: object) -> artifact.FilesystemArtifactOrigin:
        calls.append(value)
        raise AssertionError("invalid envelope reached origin parser")

    monkeypatch.setattr(artifact, "_origin", fail)
    with pytest.raises(artifact.ArtifactValidationError, match="Invalid filesystem producer receipt"):
        read_provenance.parse_read_receipt(_receipt_record(**changes))
    assert calls == []


def test_receipt_retains_parser_selection_before_container_reads(monkeypatch):
    fault = artifact.ArtifactValidationError("rebound during envelope validation")

    def fail(value: object) -> artifact.FilesystemArtifactOrigin:
        raise fault

    class RebindOnRead(dict[str, object]):
        def __getitem__(self, key: str) -> object:
            if key == "schema_version":
                monkeypatch.setattr(artifact, "_origin", fail)
            return super().__getitem__(key)

    result = read_provenance.parse_read_receipt(RebindOnRead(_receipt_record()))
    assert result.source_access == artifact.FilesystemArtifactOrigin(
        "filesystem", "workspace", None, "folder/file.txt",
    )
