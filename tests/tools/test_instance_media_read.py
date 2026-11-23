"""Author cases C3-C6 for R2 (instance media read).

Provenance: the case scenarios, inputs, and expected outcomes are copied from
`a3-evidence/a3-reconciliation-2026-11-14/resume2/req-next/CASES.md` (C3-C6) and
`test_cases_spec.py`, which were written from requirements/use cases and the
public interface contract only. This module binds the test-side ``Harness``
seam (gaps G7-G11) to the product API and changes no assertion.

C3=U3, C4=U4, C5=U5, C6=U6.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import pathlib
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.loader import set_config_path
from nanobot.config.paths import get_media_dir
from nanobot.security.workspace_policy import WorkspaceBoundaryError

INSTANCE_ARTIFACT_BYTES = b"\x89PNG\r\n\x1a\n" + b"instance-artifact-marker-c3"
AMBIENT_ARTIFACT_BYTES = b"ambient-artifact-marker-c5"
OUTSIDE_ARTIFACT_BYTES = b"outside-both-roots-marker-c6"

_FILE_TRAILER = "\n\n(End of file"


@dataclass(frozen=True)
class Refusal:
    """Typed refusal surfaced by the read_file boundary."""

    kind: str
    message: str


@dataclass(frozen=True)
class ProductHarness:
    """Harness binding of G7-G11 to the built-in read_file surface."""

    workspace: pathlib.Path
    _counter: list[int] = field(default_factory=lambda: [0])

    def _context(self, **kwargs: Any) -> RequestContext:
        self._counter[0] += 1
        return RequestContext(
            channel="probe",
            chat_id="probe",
            turn_id=f"case-{self._counter[0]}",
            **kwargs,
        )

    def request_context_with_media_root(self, root: str) -> RequestContext:
        return self._context(media_root=pathlib.Path(root))

    def ambient_request_context(self) -> RequestContext:
        return self._context()

    def ambient_media_root(self) -> str:
        return str(get_media_dir())

    async def read_file(self, path: str, context: RequestContext) -> Any:
        tool = ReadFileTool(workspace=self.workspace, allowed_dir=self.workspace)
        with request_context(context):
            try:
                result = await tool.execute(path=path, force=True)
            except WorkspaceBoundaryError as error:
                return Refusal("workspace_boundary", str(error))
            except PermissionError as error:
                return Refusal("permission", str(error))
        if getattr(result, "is_error", False):
            return Refusal("tool_error", str(result))
        return result

    @staticmethod
    def is_success(result: Any) -> bool:
        return not isinstance(result, Refusal)

    @staticmethod
    def is_refusal(result: Any) -> bool:
        return isinstance(result, Refusal)

    @staticmethod
    def bytes_of(result: Any) -> bytes:
        content = getattr(result, "content", result)
        if isinstance(content, list):
            for block in content:
                if block.get("type") != "image_url":
                    continue
                header, _, payload = block["image_url"]["url"].partition(",")
                assert "base64" in header, "image block did not carry base64 payload"
                return base64.b64decode(payload)
            raise AssertionError("image content carried no image block")
        assert isinstance(content, str), f"unsupported read result: {type(result)!r}"
        body, _, _ = content.partition(_FILE_TRAILER)
        lines = body.splitlines()
        return "\n".join(
            line.split("| ", 1)[1] if "| " in line else line for line in lines
        ).encode()


@pytest.fixture
def harness(tmp_path: pathlib.Path) -> ProductHarness:
    ambient = tmp_path / "ambient"
    set_config_path(ambient / "config.json")
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return ProductHarness(workspace=workspace)


def _read(harness: ProductHarness, path: str, context: RequestContext) -> Any:
    out = harness.read_file(path, context)
    if inspect.isawaitable(out):
        return asyncio.run(out)
    return out


def test_c3_instance_media_artifact_read(
    harness: ProductHarness, tmp_path: pathlib.Path,
) -> None:
    """C3 (U3/R2): an instance-bound read of its own artifact succeeds with exact bytes."""
    instance_root = tmp_path / "instance-media"
    instance_root.mkdir()
    artifact = instance_root / "artifact.png"
    artifact.write_bytes(INSTANCE_ARTIFACT_BYTES)

    context = harness.request_context_with_media_root(str(instance_root))
    result = _read(harness, str(artifact), context)

    assert harness.is_success(result), "C3.1 instance read was refused"
    assert harness.bytes_of(result) == INSTANCE_ARTIFACT_BYTES, "C3.2 content mismatch"


def test_c4_instance_context_does_not_fall_back_to_ambient(
    harness: ProductHarness, tmp_path: pathlib.Path,
) -> None:
    """C4 (U4/R2): with an instance root bound, an ambient path is refused."""
    ambient_root = pathlib.Path(harness.ambient_media_root())
    ambient_root.mkdir(parents=True, exist_ok=True)
    ambient_file = ambient_root / "ambient-only.bin"
    ambient_file.write_bytes(AMBIENT_ARTIFACT_BYTES)

    instance_root = tmp_path / "instance-media-2"
    instance_root.mkdir()
    context = harness.request_context_with_media_root(str(instance_root))

    result = _read(harness, str(ambient_file), context)

    assert ambient_file.read_bytes() == AMBIENT_ARTIFACT_BYTES, "C4.3 fixture precondition"
    assert harness.is_refusal(result), "C4.1 ambient path was not refused under an instance root"
    assert not harness.is_success(result), "C4.2 ambient fallback occurred"


def test_c5_no_instance_keeps_ambient_read(harness: ProductHarness) -> None:
    """C5 (U5/R2): with no instance bound, the legacy ambient read still succeeds."""
    ambient_root = pathlib.Path(harness.ambient_media_root())
    ambient_root.mkdir(parents=True, exist_ok=True)
    artifact = ambient_root / "ambient-artifact.bin"
    artifact.write_bytes(AMBIENT_ARTIFACT_BYTES)

    context = harness.ambient_request_context()
    result = _read(harness, str(artifact), context)

    assert harness.is_success(result), "C5.1 legacy ambient read was refused"
    assert harness.bytes_of(result) == AMBIENT_ARTIFACT_BYTES, "C5.2 legacy content mismatch"


def test_c6_outside_both_roots_is_refused(
    harness: ProductHarness, tmp_path: pathlib.Path,
) -> None:
    """C6 (U6/R2): a readable path outside both roots is refused in both contexts."""
    outside = tmp_path / "outside-roots.bin"
    outside.write_bytes(OUTSIDE_ARTIFACT_BYTES)

    instance_root = tmp_path / "instance-media-3"
    instance_root.mkdir()
    instance_context = harness.request_context_with_media_root(str(instance_root))
    ambient_context = harness.ambient_request_context()

    assert outside.read_bytes() == OUTSIDE_ARTIFACT_BYTES, "C6.3 fixture precondition"

    with_instance = _read(harness, str(outside), instance_context)
    assert harness.is_refusal(with_instance), "C6.1 outside path accepted under instance root"
    assert not harness.is_success(with_instance), "C6.1 outside bytes returned"

    without_instance = _read(harness, str(outside), ambient_context)
    assert harness.is_refusal(without_instance), "C6.2 outside path accepted with no instance"
    assert not harness.is_success(without_instance), "C6.2 outside bytes returned"


class _StubProvider:
    def get_default_model(self) -> str:
        return "synthetic-model"

    async def aclose(self) -> None:
        return None


async def test_c7_loop_registered_read_file_reads_instance_artifact(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C7: C3 through the loop-registered read_file, plus the C4/C6 controls.

    The loop is built from an explicit instance config while the ambient config
    path points at another instance. The turn context carries the instance media
    root, so the loop's own read_file must resolve that artifact and still
    refuse the ambient-only file, a path outside both roots, and a protected
    credential file.
    """
    import nanobot.migration_guard as guard
    import nanobot.providers.factory as factory
    from nanobot.agent.loop import AgentLoop
    from nanobot.config.schema import Config

    monkeypatch.setattr(guard, "require_runtime_ready", lambda: None)
    monkeypatch.setattr(factory, "make_provider", lambda _config: _StubProvider())
    monkeypatch.setattr("subprocess.Popen", Mock(side_effect=AssertionError("process execution")))

    ambient = tmp_path / "ambient"
    set_config_path(ambient / "config.json")
    instance = tmp_path / "instance"
    workspace = instance / "workspace"
    workspace.mkdir(parents=True)

    config = Config()
    config.bind_source_path(instance / "config.json")
    config.agents.defaults.workspace = str(workspace)
    config.agents.defaults.model = "synthetic-model"
    config.agents.defaults.model_preset = None
    config.tools.restrict_to_workspace = True
    config.tools.agent_browser.enabled = False
    config.tools.agent_device.enabled = False
    config.tools.desktop_use.enabled = False

    loop = AgentLoop.from_config(config, tool_registry=ToolRegistry())
    try:
        assert loop.media_root == instance / "media"
        artifact = loop.media_root / "generated" / "2026-11-14" / "artifact.png"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(INSTANCE_ARTIFACT_BYTES)
        ambient_file = get_media_dir() / "ambient-only.bin"
        ambient_file.parent.mkdir(parents=True, exist_ok=True)
        ambient_file.write_bytes(AMBIENT_ARTIFACT_BYTES)
        outside = tmp_path / "outside-roots.bin"
        outside.write_bytes(OUTSIDE_ARTIFACT_BYTES)
        blocked = loop.media_root / ".env"
        blocked.write_bytes(b"SECRET=c7")

        context = RequestContext(
            channel="probe", chat_id="probe", media_root=loop.media_root,
        )
        read_tool = loop.tools.get("read_file")
        assert read_tool is not None

        async def loop_read(path: pathlib.Path) -> Any:
            with request_context(context):
                return await ToolRegistry.execute_prepared(
                    read_tool, {"path": str(path), "force": True},
                    SimpleNamespace(lock_owner=None),
                )

        def refused(outcome: Any) -> bool:
            return bool(getattr(outcome, "is_error", False)
                        or getattr(outcome, "stop_reason", None))

        result = await loop_read(artifact)
        assert not refused(result), "C7.1 loop read was refused"
        assert ProductHarness.bytes_of(result) == INSTANCE_ARTIFACT_BYTES, "C7.2 content mismatch"

        assert refused(await loop_read(ambient_file)), "C4 control: ambient read"
        assert refused(await loop_read(outside)), "C6 control: outside read"
        credential = await loop_read(blocked)
        assert refused(credential), "credential read was not refused"
        assert "protected credential" in str(credential), "credential refusal changed"
    finally:
        await loop.aclose()
