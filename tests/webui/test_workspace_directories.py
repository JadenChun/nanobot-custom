from __future__ import annotations

import os
from pathlib import Path

import pytest

from nanobot.config.schema import DirectoryBrowserConfig, DirectoryBrowserRootConfig
from nanobot.webui.workspace_directories import DirectoryBrowserError, WorkspaceDirectoryBrowser


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def browser(root: Path, *, clock: Clock | None = None, roots: list | None = None) -> WorkspaceDirectoryBrowser:
    configured = roots if roots is not None else [
        DirectoryBrowserRootConfig(id="projects", label="Projects", path=str(root))
    ]
    config = DirectoryBrowserConfig(roots=configured)
    return WorkspaceDirectoryBrowser(lambda: config, clock=clock or Clock())


def test_empty_roots_are_explicit(tmp_path) -> None:
    configured = WorkspaceDirectoryBrowser(lambda: DirectoryBrowserConfig())
    payload = configured.roots()
    assert payload == {"roots": [], "policy_revision": payload["policy_revision"], "empty": True}


def test_roots_are_bounded_and_canonical(tmp_path) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    payload = browser(root).roots()
    assert payload["empty"] is False
    assert payload["roots"][0]["id"] == "projects"
    assert payload["roots"][0]["display_path"] == str(root.resolve())

    with pytest.raises(ValueError):
        DirectoryBrowserConfig(
            roots=[
                DirectoryBrowserRootConfig(id=f"r{index}", label="x", path=str(root))
                for index in range(33)
            ]
        )
    with pytest.raises(ValueError):
        DirectoryBrowserConfig(
            roots=[
                DirectoryBrowserRootConfig(id="same", label="x", path=str(root)),
                DirectoryBrowserRootConfig(id="same", label="y", path=str(root)),
            ]
        )


def test_list_returns_directories_only_without_contents(tmp_path) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    (root / "alpha").mkdir()
    (root / "beta").mkdir()
    (root / "notes.txt").write_text("secret contents", encoding="utf-8")
    os.symlink(root / "alpha", root / "alpha-link")
    payload = browser(root).list(root_id="projects", limit=10)
    names = {entry["name"] for entry in payload["entries"]}
    assert names == {"alpha", "beta", "alpha-link"}
    assert "secret contents" not in str(payload)
    assert all("directory_id" in entry for entry in payload["entries"])
    assert payload["next_cursor"] is None


def test_pagination_cursor_is_bound_to_root_and_path(tmp_path) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    for index in range(5):
        (root / f"dir-{index}").mkdir()
    (root / "child").mkdir()
    for index in range(3):
        (root / "child" / f"nested-{index}").mkdir()
    service = browser(root)

    first = service.list(root_id="projects", limit=2)
    assert len(first["entries"]) == 2
    assert first["next_cursor"]
    second = service.list(root_id="projects", limit=2, cursor=first["next_cursor"])
    assert len(second["entries"]) == 2
    assert not {entry["name"] for entry in first["entries"]} & {
        entry["name"] for entry in second["entries"]
    }

    with pytest.raises(DirectoryBrowserError) as exc:
        service.list(root_id="projects", relative_path="child", limit=2, cursor=first["next_cursor"])
    assert exc.value.code == "cursor_invalid"


def test_traversal_and_outside_symlinks_deny(tmp_path) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    (root / "alpha").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").mkdir()
    os.symlink(outside, root / "escape")
    service = browser(root)

    for bad in ("../outside", "a/../../outside", "a\\b", "a\x00b"):
        with pytest.raises(DirectoryBrowserError) as exc:
            service.list(root_id="projects", relative_path=bad)
        assert exc.value.code in {"invalid_path"}
    with pytest.raises(DirectoryBrowserError) as exc:
        service.list(root_id="projects", relative_path="/etc")
    assert exc.value.code == "outside_root"
    absolute_inside = service.list(root_id="projects", relative_path=str(root / "alpha"))
    assert absolute_inside["relative_path"] == "alpha"
    with pytest.raises(DirectoryBrowserError) as exc:
        service.list(root_id="projects", relative_path="escape/secret")
    assert exc.value.code == "outside_root"
    with pytest.raises(DirectoryBrowserError) as exc:
        service.list(root_id="projects", relative_path="a" * 5000)
    assert exc.value.code == "invalid_path"
    with pytest.raises(DirectoryBrowserError) as exc:
        service.list(root_id="projects", relative_path="/".join(["a"] * 65))
    assert exc.value.code == "path_too_deep"


def test_limits_are_enforced(tmp_path) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    service = browser(root)
    for limit in (0, 101, -1):
        with pytest.raises(DirectoryBrowserError) as exc:
            service.list(root_id="projects", limit=limit)
        assert exc.value.code == "invalid_limit"
    with pytest.raises(DirectoryBrowserError) as exc:
        service.list(root_id="missing")
    assert exc.value.code == "root_not_found"


def test_scan_budget_and_deadline_are_enforced(tmp_path) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    (root / "alpha").mkdir()
    for index in range(1001):
        (root / f"file-{index}.txt").write_text("x", encoding="utf-8")
    with pytest.raises(DirectoryBrowserError) as exc:
        browser(root).list(root_id="projects", limit=10)
    assert exc.value.code == "scan_too_large"

    clock = Clock()
    slow = browser(root, clock=clock)

    def tick() -> float:
        clock.advance(3)
        return float(clock())

    slow._clock = tick  # type: ignore[method-assign]
    with pytest.raises(DirectoryBrowserError) as exc:
        slow.list(root_id="projects", limit=10)
    assert exc.value.code == "directory_deadline"


def test_resolve_and_commit_are_single_use_and_revision_bound(tmp_path) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    (root / "alpha").mkdir()
    service = browser(root)
    policy = service.roots()["policy_revision"]

    resolved = service.resolve(
        root_id="projects",
        relative_path="alpha",
        policy_revision=policy,
        installation_id="install-owner",
    )
    assert resolved["root_id"] == "projects"
    token = resolved["commit_token"]

    with pytest.raises(DirectoryBrowserError) as exc:
        service.consume_commit_token(
            token,
            installation_id="install-other",
            root_id="projects",
            relative_path="alpha",
            extent="read",
        )
    assert exc.value.code == "resolve_other_device"

    with pytest.raises(DirectoryBrowserError) as exc:
        service.consume_commit_token(
            token,
            installation_id="install-owner",
            root_id="projects",
            relative_path="beta",
            extent="read",
        )
    assert exc.value.code == "resolve_path_changed"

    granted = service.consume_commit_token(
        token,
        installation_id="install-owner",
        root_id="projects",
        relative_path="alpha",
        extent="write",
    )
    assert granted["canonical_path"] == str((root / "alpha").resolve())
    assert granted["extent"] == "write"

    with pytest.raises(DirectoryBrowserError) as exc:
        service.consume_commit_token(
            token,
            installation_id="install-owner",
            root_id="projects",
            relative_path="alpha",
            extent="read",
        )
    assert exc.value.code == "resolve_reused"

    with pytest.raises(DirectoryBrowserError) as exc:
        service.resolve(
            root_id="projects",
            relative_path="alpha",
            policy_revision="stale",
            installation_id="install-owner",
        )
    assert exc.value.code == "policy_changed"


def test_expired_and_replaced_resolve_deny(tmp_path) -> None:
    clock = Clock()
    root = tmp_path / "projects"
    root.mkdir()
    (root / "alpha").mkdir()
    service = browser(root, clock=clock)
    resolved = service.resolve(
        root_id="projects",
        relative_path="alpha",
        policy_revision=service.roots()["policy_revision"],
        installation_id="install-owner",
    )
    clock.advance(301)
    with pytest.raises(DirectoryBrowserError) as exc:
        service.consume_commit_token(
            resolved["commit_token"],
            installation_id="install-owner",
            root_id="projects",
            relative_path="alpha",
            extent="read",
        )
    assert exc.value.code == "resolve_expired"

    resolved = service.resolve(
        root_id="projects",
        relative_path="alpha",
        policy_revision=service.roots()["policy_revision"],
        installation_id="install-owner",
    )
    replacement = root / "alpha.next"
    replacement.mkdir()
    replacement.replace(root / "alpha")
    with pytest.raises(DirectoryBrowserError) as exc:
        service.consume_commit_token(
            resolved["commit_token"],
            installation_id="install-owner",
            root_id="projects",
            relative_path="alpha",
            extent="read",
        )
    assert exc.value.code == "path_replaced"


def test_owner_grant_commit_uses_resolved_canonical_path(tmp_path) -> None:
    from types import SimpleNamespace

    from websockets.datastructures import Headers
    from websockets.http11 import Request as WsRequest

    from nanobot.webui.ws_http import GatewayHTTPHandler

    root = tmp_path / "projects"
    root.mkdir()
    (root / "alpha").mkdir()
    service = browser(root)
    resolved = service.resolve(
        root_id="projects",
        relative_path="alpha",
        policy_revision=service.roots()["policy_revision"],
        installation_id="install-owner",
    )
    stub = SimpleNamespace(
        directory_browser=service,
        _directory_installation=lambda _request: "install-owner",
    )
    request = WsRequest("/api/workspaces/grants/upsert", Headers())
    payload = {
        "expected_revision": 1,
        "path": "alpha",
        "file_access": "read_only",
        "commands": False,
        "commit_token": resolved["commit_token"],
        "root": "projects",
        "extent": "read",
    }
    cleaned = GatewayHTTPHandler._owner_grant_payload(stub, request, payload)
    assert cleaned == {
        "expected_revision": 1,
        "path": str((root / "alpha").resolve()),
        "file_access": "read_only",
        "commands": False,
    }

    with pytest.raises(DirectoryBrowserError) as exc:
        GatewayHTTPHandler._owner_grant_payload(stub, request, payload)
    assert exc.value.code == "resolve_reused"

    with pytest.raises(ValueError):
        GatewayHTTPHandler._owner_grant_payload(
            stub,
            request,
            {**payload, "is_local": True},
        )

    mismatch = service.resolve(
        root_id="projects",
        relative_path="alpha",
        policy_revision=service.roots()["policy_revision"],
        installation_id="install-owner",
    )
    with pytest.raises(ValueError):
        GatewayHTTPHandler._owner_grant_payload(
            stub,
            request,
            {**payload, "commit_token": mismatch["commit_token"], "extent": "read",
             "file_access": "read_write"},
        )
    read_with_commands = service.resolve(
        root_id="projects",
        relative_path="alpha",
        policy_revision=service.roots()["policy_revision"],
        installation_id="install-owner",
    )
    with pytest.raises(ValueError):
        GatewayHTTPHandler._owner_grant_payload(
            stub,
            request,
            {**payload, "commit_token": read_with_commands["commit_token"], "commands": True},
        )


def test_cursor_rejects_changed_directory_contents(tmp_path) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    for name in ("a", "b", "c"):
        (root / name).mkdir()
    service = browser(root)
    first = service.list(root_id="projects", limit=1)
    assert first["next_cursor"]
    (root / "0-new").mkdir()
    with pytest.raises(DirectoryBrowserError) as exc:
        service.list(root_id="projects", limit=1, cursor=first["next_cursor"])
    assert exc.value.code == "cursor_mismatch"


def test_resolve_and_cursor_stores_are_bounded(tmp_path) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    (root / "alpha").mkdir()
    service = browser(root)
    policy = service.roots()["policy_revision"]
    for index in range(400):
        service.resolve(
            root_id="projects",
            relative_path="alpha",
            policy_revision=policy,
            installation_id=f"install-{index}",
        )
    assert len(service._resolves) <= 256
    for index in range(400):
        (root / f"dir-{index}").mkdir(exist_ok=True)
        service._mint_cursor(
            installation_id="install-owner",
            root_id="projects",
            relative_path="",
            policy_revision=policy,
            offset=index,
            directory_snapshot="snapshot",
        )
    assert len(service._cursors) <= 256


def test_roots_response_is_bounded(tmp_path) -> None:
    deep = tmp_path
    for index in range(10):
        deep = deep / ("d" * 200 + str(index))
        deep.mkdir()
    path = str(deep)
    roots = [
        DirectoryBrowserRootConfig(id=f"r{index}", label="x" * 64, path=path)
        for index in range(32)
    ]
    service = WorkspaceDirectoryBrowser(lambda: DirectoryBrowserConfig(roots=roots))
    with pytest.raises(DirectoryBrowserError) as exc:
        service.roots()
    assert exc.value.code == "response_too_large"


def test_absolute_path_depth_is_bounded(tmp_path) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    service = browser(root)
    deep = root.joinpath(*(["a"] * 65))
    with pytest.raises(DirectoryBrowserError) as exc:
        service.list(root_id="projects", relative_path=str(deep))
    assert exc.value.code == "path_too_deep"
