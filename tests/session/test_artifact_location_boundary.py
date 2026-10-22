"""Artifact location values and policy checks never substitute for read authority."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.context_repo import ContextRepoManager, ManagedContextRepo, ResourceAccessPolicy
from nanobot.session import artifact_factory, artifact_reader
from nanobot.session.artifact import ArtifactValidationError
from nanobot.session.artifact_reader import ArtifactReader
from nanobot.session.execution_owner import ExecutionOwnershipError
from nanobot.session.manager import JsonlSessionStore
from nanobot.session.source_store import SourceSessionStore
from tests.session.test_artifact_reader import setup_read


def test_source_location_metadata_retains_selected_backend_and_live_path_values(tmp_path, monkeypatch):
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    other = JsonlSessionStore(tmp_path / "other", sessions_root=tmp_path / "other-sessions")
    source = backend.source
    selected = source.select_artifact_location_metadata()
    assert selected.workspace == backend.workspace
    assert selected.sessions_dir == backend.sessions_dir
    assert selected.workspace_id == backend.sessions_dir.name
    monkeypatch.setattr(source, "_backend", other)
    assert selected.workspace == backend.workspace
    assert selected.workspace_id == backend.sessions_dir.name
    latest = source.select_artifact_location_metadata()
    assert latest.workspace == other.workspace
    assert latest.workspace_id == other.sessions_dir.name
    changed = tmp_path / "metadata-only-path"
    assert not changed.exists()
    monkeypatch.setattr(backend, "workspace", changed)
    assert selected.workspace == changed and latest.workspace == other.workspace
    assert not changed.exists()
    for name in ("backend", "fd", "authority", "store_identity", "write"):
        assert not hasattr(selected, name)
    with pytest.raises((AttributeError, FrozenInstanceError)):
        selected.workspace = tmp_path / "cannot-assign"


async def test_reader_keeps_initial_backend_selection_across_location_hook(tmp_path, monkeypatch):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, content):
        original = writer.store._backend
        other = JsonlSessionStore(tmp_path / "other", sessions_root=tmp_path / "other-state")
        assert other.sessions_dir.name != store.workspace_id
        calls = []

        with monkeypatch.context() as scoped:
            def scoped_hook():
                calls.append("hook")
                scoped.setattr(writer.store, "_backend", other)

            reader = ArtifactReader(writer, "root_history", store, tool,
                                    location_validator=scoped_hook)
            assert reader.store is store and calls == ["hook"]
            assert writer.store._backend is other
        assert writer.store._backend is original
        assert reader.store.workspace_id == original.sessions_dir.name


async def test_reader_delegates_private_location_with_lazy_exact_metadata(tmp_path, monkeypatch):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, content):
        original = ReadFileTool.validate_artifact_private_location
        observed = []

        def validate(**callbacks):
            observed.append(callbacks)
            assert callbacks["reader"].read_file is tool
            assert callbacks["reader"].store is store
            assert callbacks["source_location"].workspace == writer.store._backend.workspace
            assert callbacks["select_workspace_access"]() is artifact_reader.current_tool_workspace
            original(**callbacks)

        monkeypatch.setattr(ReadFileTool, "validate_artifact_private_location", staticmethod(validate))
        reader = ArtifactReader(writer, "root_history", store, tool)
        reader.revalidate_private_location()
        assert len(observed) == 2
        assert reader.read(descriptor.reference).text.encode() == content


async def test_reader_revalidation_selects_source_after_hook_before_workspace_lookup(tmp_path, monkeypatch):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, content):
        second = JsonlSessionStore(tmp_path / "second", sessions_root=tmp_path / "second-state")
        third = JsonlSessionStore(tmp_path / "third", sessions_root=tmp_path / "third-state")
        original_access = artifact_reader.current_tool_workspace
        original_backend_attribute = JsonlSessionStore.__getattribute__
        original_source_attribute = SourceSessionStore.__getattribute__
        source = writer.store
        trace = []
        armed = False

        def hook():
            if armed:
                trace.append("hook")
                object.__setattr__(source, "_backend", second)

        def access(*args, **kwargs):
            trace.append("workspace")
            object.__setattr__(source, "_backend", third)
            return original_access(*args, **kwargs)

        def backend_attribute(self, name):
            if self is second and name == "workspace":
                trace.append("selected-source-workspace")
            if self is third and name == "workspace":
                trace.append("wrong-source-workspace")
            return original_backend_attribute(self, name)

        def source_attribute(self, name):
            if self is source and name == "_backend":
                trace.append("source-selection")
            return original_source_attribute(self, name)

        reader = ArtifactReader(writer, "root_history", store, tool, location_validator=hook)
        with monkeypatch.context() as scoped:
            scoped.setattr(source, "_backend", source._backend)
            scoped.setattr(artifact_reader, "current_tool_workspace", access)
            scoped.setattr(JsonlSessionStore, "__getattribute__", backend_attribute)
            scoped.setattr(SourceSessionStore, "__getattribute__", source_attribute)
            armed = True
            reader.revalidate_private_location()
            assert trace == ["hook", "source-selection", "workspace", "selected-source-workspace"]


def test_factory_delegates_to_owned_root_operation_without_virtual_override(tmp_path, monkeypatch):
    tool = ReadFileTool(workspace=tmp_path)
    original = ReadFileTool.collect_artifact_policy_roots
    observed = []

    def collect(self, **selectors):
        observed.append(self)
        assert selectors["select_path_items"]() is artifact_factory._path_items
        assert selectors["select_workspace_access"]() is artifact_factory.current_tool_workspace
        return original(self, **selectors)

    def overridden(**kwargs):
        raise AssertionError("an instance override supplied location metadata")

    monkeypatch.setattr(ReadFileTool, "collect_artifact_policy_roots", collect)
    monkeypatch.setattr(tool, "collect_artifact_policy_roots", overridden)
    roots = artifact_factory.ConfiguredArtifactReaderFactory.read_policy_roots(tool)
    assert observed == [tool] and tmp_path in roots


def test_factory_path_parser_selection_precedes_each_original_attribute_lookup(tmp_path, monkeypatch):
    tool = ReadFileTool(workspace=tmp_path, extra_read_allowed_dirs=[tmp_path / "read"],
                        extra_write_allowed_dirs=[tmp_path / "write"],
                        extra_read_allowed_files=[tmp_path / "file"])
    original_attribute = ReadFileTool.__getattribute__
    original_parser = artifact_factory._path_items
    trace = []

    def first(value):
        trace.append("first-parser")
        return original_parser(value)

    def second(value):
        trace.append("second-parser")
        return original_parser(value)

    def attribute(self, name):
        if self is tool and name == "_extra_read_allowed_dirs":
            trace.append("read-dirs")
            monkeypatch.setattr(artifact_factory, "_path_items", second)
        return original_attribute(self, name)

    monkeypatch.setattr(artifact_factory, "_path_items", first)
    monkeypatch.setattr(ReadFileTool, "__getattribute__", attribute)
    roots = artifact_factory.ConfiguredArtifactReaderFactory.read_policy_roots(tool)
    assert trace == ["read-dirs", "first-parser", "second-parser", "second-parser"]
    assert tmp_path / "read" in roots and tmp_path / "write" in roots and tmp_path / "file" in roots


def test_factory_workspace_lookup_is_selected_after_resource_policy_observation(tmp_path, monkeypatch):
    context = tmp_path / "context"
    manager = ContextRepoManager([ManagedContextRepo(
        path=context, name="context", managed=True,
        manifest={"targetRepos": {"target": {"path": str(tmp_path / "target")}}},
    )])
    policy = ResourceAccessPolicy(workspace=tmp_path, context_manager=manager)
    tool = ReadFileTool(workspace=tmp_path, resource_policy=policy)
    original_roots = ResourceAccessPolicy.extra_read_dirs
    original_access = artifact_factory.current_tool_workspace
    trace = []

    def selected(*args, **kwargs):
        trace.append("selected-workspace")
        return original_access(*args, **kwargs)

    def roots(self):
        trace.append("policy-roots")
        monkeypatch.setattr(artifact_factory, "current_tool_workspace", selected)
        return original_roots(self)

    monkeypatch.setattr(ResourceAccessPolicy, "extra_read_dirs", roots)
    found = artifact_factory.ConfiguredArtifactReaderFactory.read_policy_roots(tool)
    assert trace == ["policy-roots", "selected-workspace"]
    assert context in found and tmp_path / "target" in found


@pytest.mark.parametrize("seam", ["parser", "workspace"])
def test_factory_retains_exact_original_helper_fault(tmp_path, monkeypatch, seam):
    fault = OSError("original factory metadata fault")

    def fail(*args, **kwargs):
        raise fault

    monkeypatch.setattr(artifact_factory, "_path_items" if seam == "parser" else
                        "current_tool_workspace", fail)
    with pytest.raises(OSError) as failure:
        artifact_factory.ConfiguredArtifactReaderFactory.read_policy_roots(ReadFileTool(workspace=tmp_path))
    assert failure.value is fault


@pytest.mark.parametrize("cause,message", [
    ("source", "Artifact store must be outside readable roots"),
    ("tool", "Artifact store must be outside readable roots"),
    ("extra-dir", "Artifact store must be outside readable roots"),
    ("policy", "Artifact store must be outside readable roots"),
    ("media", "Artifact store must be outside readable roots"),
    ("project", "Artifact store is inside the active project"),
    ("file", "Artifact store has a public file allowlist"),
])
async def test_reader_rejects_live_readable_root_changes_without_writes(
    tmp_path, monkeypatch, cause, message,
):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, content):
        reader = ArtifactReader(writer, "root_history", store, tool)
        path = writer.store._backend.get_session_path(writer.owner.key)
        before = path.read_bytes()
        blob = next(store.root.rglob("*.blob"))
        blob_before = blob.read_bytes()
        with monkeypatch.context() as scoped:
            if cause == "source":
                scoped.setattr(writer.store._backend, "workspace", store.root)
            elif cause == "tool":
                scoped.setattr(tool, "_workspace", store.root)
            elif cause == "extra-dir":
                scoped.setattr(tool, "_extra_read_allowed_dirs", [store.root])
            elif cause == "policy":
                manager = ContextRepoManager([ManagedContextRepo(
                    path=store.root, name="private-overlap", managed=True,
                )])
                scoped.setattr(tool, "_resource_policy", ResourceAccessPolicy(
                    workspace=tool._workspace, context_manager=manager,
                ))
            elif cause == "media":
                scoped.setattr(artifact_reader, "get_media_dir", lambda: store.root)
            elif cause == "project":
                original = artifact_reader.current_tool_workspace

                def project(default_workspace, **kwargs):
                    return original(store.root, **kwargs)

                scoped.setattr(artifact_reader, "current_tool_workspace", project)
            else:
                scoped.setattr(tool, "_extra_read_allowed_files", [store.root / "public.txt"])
            with pytest.raises(ArtifactValidationError, match=f"^{message}$"):
                reader.revalidate_private_location()
            assert path.read_bytes() == before and blob.read_bytes() == blob_before


@pytest.mark.parametrize("seam", ["hook", "workspace", "media", "policy"])
async def test_reader_preserves_fault_identity_and_first_error_before_overlap(
    tmp_path, monkeypatch, seam,
):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, content):
        armed = False
        fault = OSError("original private location fault")
        trace = []

        def hook():
            if armed and seam == "hook":
                trace.append("hook")
                raise fault

        def fail(*args, **kwargs):
            trace.append(seam)
            raise fault

        reader = ArtifactReader(writer, "root_history", store, tool, location_validator=hook)
        original_attribute = ReadFileTool.__getattribute__

        def attribute(self, name):
            if self is tool and name == "_resource_policy":
                trace.append("policy")
                raise fault
            return original_attribute(self, name)

        with monkeypatch.context() as scoped:
            scoped.setattr(tool, "_extra_read_allowed_dirs", [store.root])
            if seam == "workspace":
                scoped.setattr(artifact_reader, "current_tool_workspace", fail)
            elif seam == "media":
                scoped.setattr(artifact_reader, "get_media_dir", fail)
            elif seam == "policy":
                scoped.setattr(ReadFileTool, "__getattribute__", attribute)
            armed = True
            with pytest.raises(OSError) as failure:
                reader.revalidate_private_location()
            assert failure.value is fault and trace == [seam]


async def test_location_metadata_does_not_admit_a_quiesced_reader(tmp_path):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, content):
        reader = ArtifactReader(writer, "root_history", store, tool)
        metadata = writer.store.select_artifact_location_metadata()
        assert metadata.workspace_id == store.workspace_id
        await writer.owner.quiesce()
        reader.revalidate_private_location()
        with pytest.raises(ExecutionOwnershipError):
            reader.read(descriptor.reference)
