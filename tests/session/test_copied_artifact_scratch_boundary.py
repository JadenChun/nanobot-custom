"""Fresh copied scratch installation retains late hooks and exact first errors."""

from pathlib import Path

import pytest

from nanobot.session import artifact_operator
from nanobot.session.manager import JsonlSessionStore

KEY = "cli:copied-scratch"
WORKSPACE_ID = "a" * 32
COPIED = '{"role":"user","content":"copied 中文 🌟"}\n'.encode()
EVENTS = [
    "construct", "select_writer", "select_write_path", "write_path", "decode", "write",
    "select_read_path", "read_path", "read",
]


@pytest.fixture(autouse=True)
def operation():
    selected = getattr(JsonlSessionStore, "create_copied_artifact_scratch", None)
    assert callable(selected), "copied scratch operation is missing"
    return selected


def observed_copy(root, monkeypatch, *, fault=None, readback=COPIED, on_read=None):
    trace = []
    failure = OSError("injected copied scratch failure")

    def step(name):
        trace.append(name)
        if name == fault:
            raise failure

    class CopiedBytes(bytes):
        def decode(self, encoding="utf-8"):
            step("decode")
            assert encoding == "utf-8"
            backend.writer = lambda *_args: pytest.fail("selected private writer was replaced")
            return super().decode(encoding)

    class Readback:
        def read_bytes(self):
            step("read")
            if on_read is not None:
                on_read()
            return readback

    path = Readback()

    class Backend:
        def __init__(self):
            self.writer = self.write
            self.paths = 0

        @property
        def _write_text_atomic(self):
            step("select_writer")
            return self.writer

        @property
        def get_session_path(self):
            self.paths += 1
            phase = "write_path" if self.paths == 1 else "read_path"
            step("select_" + phase)

            def resolve(key):
                step(phase)
                assert key == KEY
                return path

            return resolve

        def write(self, selected_path, text):
            step("write")
            assert selected_path is path and text.encode() == COPIED

    backend = Backend()

    def construct(workspace, sessions_root):
        step("construct")
        assert workspace == root / "workspace" and sessions_root == root / "sessions"
        assert (workspace / ".nanobot" / "workspace-id").read_bytes() == (
            WORKSPACE_ID + "\n"
        ).encode("ascii")
        assert (sessions_root / WORKSPACE_ID / ".workspace").read_text() == str(workspace) + "\n"
        return backend

    return backend, CopiedBytes(COPIED), construct, trace, failure


def test_real_copy_preserves_bytes_and_private_layout(tmp_path, monkeypatch, operation):
    root = tmp_path / "scratch"
    root.mkdir()
    mkdir_calls, chmod_calls = [], []
    original_mkdir, original_chmod = Path.mkdir, Path.chmod

    def mkdir(path, mode=0o777, parents=False, exist_ok=False):
        mkdir_calls.append((path, mode))
        return original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    def chmod(path, mode, **kwargs):
        chmod_calls.append((path, mode))
        return original_chmod(path, mode, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(Path, "chmod", chmod)
    backend = operation(
        root, workspace_id=WORKSPACE_ID, key=KEY, source_bytes=COPIED,
        construct_backend=lambda workspace, sessions_root: JsonlSessionStore(
            workspace, sessions_root=sessions_root,
        ),
        mismatch_error=lambda: artifact_operator.ArtifactOperatorError(),
    )
    assert type(backend) is JsonlSessionStore
    assert backend.get_session_path(KEY).read_bytes() == COPIED
    assert backend.workspace == root / "workspace"
    assert backend.sessions_root == root / "sessions"
    for path in (
        root / "workspace", root / "workspace" / ".nanobot", root / "sessions",
        root / "sessions" / WORKSPACE_ID,
    ):
        assert (path, 0o700) in mkdir_calls
    assert (root / "workspace" / ".nanobot" / "workspace-id", 0o600) in chmod_calls
    assert (root / "sessions" / WORKSPACE_ID / ".workspace", 0o600) in chmod_calls
    assert not list(root.rglob("*.tmp"))


def test_existing_workspace_refuses_before_constructor_or_write(tmp_path, operation):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    retained = workspace / "retained"
    retained.write_bytes(b"unchanged")
    with pytest.raises(FileExistsError):
        operation(
            tmp_path, workspace_id=WORKSPACE_ID, key=KEY, source_bytes=COPIED,
            construct_backend=lambda *_args: pytest.fail("constructor ran after mkdir refusal"),
            mismatch_error=lambda: pytest.fail("mismatch error selected before a copy"),
        )
    assert retained.read_bytes() == b"unchanged" and not (tmp_path / "sessions").exists()


def test_private_writer_is_selected_before_path_and_decode(tmp_path, monkeypatch, operation):
    backend, copied, construct, trace, _failure = observed_copy(tmp_path, monkeypatch)
    result = operation(
        tmp_path, workspace_id=WORKSPACE_ID, key=KEY, source_bytes=copied,
        construct_backend=construct,
        mismatch_error=lambda: pytest.fail("mismatch error selected on equal bytes"),
    )
    assert result is backend and trace == EVENTS


@pytest.mark.parametrize("fault", EVENTS)
def test_exact_first_error_stops_later_observations(tmp_path, monkeypatch, operation, fault):
    _backend, copied, construct, trace, failure = observed_copy(tmp_path, monkeypatch, fault=fault)
    with pytest.raises(OSError) as caught:
        operation(
            tmp_path, workspace_id=WORKSPACE_ID, key=KEY, source_bytes=copied,
            construct_backend=construct,
            mismatch_error=lambda: pytest.fail("mismatch error selected after an earlier fault"),
        )
    assert caught.value is failure and trace == EVENTS[:EVENTS.index(fault) + 1]


def test_compatibility_constructor_selection_is_after_allocation(tmp_path, monkeypatch):
    backend, copied, construct, trace, _failure = observed_copy(tmp_path, monkeypatch)
    original_chmod = Path.chmod
    claim = tmp_path / "sessions" / WORKSPACE_ID / ".workspace"

    def late_constructor(workspace, *, sessions_root):
        return construct(workspace, sessions_root)

    def chmod(path, mode, **kwargs):
        result = original_chmod(path, mode, **kwargs)
        if path == claim:
            monkeypatch.setattr(artifact_operator, "JsonlSessionStore", late_constructor)
        return result

    monkeypatch.setattr(Path, "chmod", chmod)
    monkeypatch.setattr(
        artifact_operator, "JsonlSessionStore",
        lambda *_args, **_kwargs: pytest.fail("constructor was captured before allocation"),
    )
    result = artifact_operator._install_scratch_source(
        tmp_path, workspace_id=WORKSPACE_ID, key=KEY, source_bytes=copied,
    )
    assert result is backend and trace == EVENTS


def test_compatibility_mismatch_error_is_selected_only_after_readback(tmp_path, monkeypatch):
    failure = artifact_operator.ArtifactOperatorError()

    def change_error():
        monkeypatch.setattr(artifact_operator, "ArtifactOperatorError", lambda: failure)

    _backend, copied, construct, trace, _unused = observed_copy(
        tmp_path, monkeypatch, readback=b"mismatch", on_read=change_error,
    )
    monkeypatch.setattr(
        artifact_operator, "JsonlSessionStore",
        lambda workspace, *, sessions_root: construct(workspace, sessions_root),
    )
    monkeypatch.setattr(
        artifact_operator, "ArtifactOperatorError",
        lambda: pytest.fail("mismatch error was captured before readback"),
    )
    with pytest.raises(type(failure)) as caught:
        artifact_operator._install_scratch_source(
            tmp_path, workspace_id=WORKSPACE_ID, key=KEY, source_bytes=copied,
        )
    assert caught.value is failure and trace == EVENTS


def test_real_fsync_failure_keeps_replacement_and_cleans_temporary(tmp_path, monkeypatch, operation):
    failure = OSError("injected copied namespace fsync failure")
    selected = []

    def construct(workspace, sessions_root):
        backend = JsonlSessionStore(workspace, sessions_root=sessions_root)
        selected.append(backend)

        def refuse(path):
            assert path == backend.sessions_dir
            raise failure

        monkeypatch.setattr(JsonlSessionStore, "_fsync_directory", staticmethod(refuse))
        return backend

    with pytest.raises(OSError) as caught:
        operation(
            tmp_path, workspace_id=WORKSPACE_ID, key=KEY, source_bytes=COPIED,
            construct_backend=construct,
            mismatch_error=lambda: pytest.fail("mismatch error selected after fsync failure"),
        )
    assert caught.value is failure and len(selected) == 1
    assert selected[0].get_session_path(KEY).read_bytes() == COPIED
    assert not list(tmp_path.rglob("*.tmp"))
