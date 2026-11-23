"""E02-F15 FG-01..FG-04: explicit gateway instances own separate grant scope.

Authored from ``a3-evidence/a3-reconciliation-2026-11-14/req-a34p6/CASES.md``.
The cases use the existing public composition (``build_gateway_services``), the
store backing path (``FolderGrantStore.path``), the private-root mask
(``FolderGrantView.sandbox_private_roots``), and the store's ``upsert``/snapshot
grant listing. A grant identity is the (file access, commands) pair of one
grant record, so ``entry(w, g)`` is ``(canonical w, *g)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nanobot.bus.queue import MessageBus
from nanobot.channels.websocket.runtime import WebSocketConfig
from nanobot.security.folder_grants import FolderGrantStore
from nanobot.webui.gateway_services import GatewayServices, build_gateway_services

GrantIdentity = tuple[str, bool]
GrantEntry = tuple[str, str, bool]


@dataclass(frozen=True)
class _Instance:
    config_path: Path
    workspace: Path

    @property
    def root(self) -> Path:
        return self.config_path.parent.resolve(strict=False)


@dataclass(frozen=True)
class _Composition:
    services: GatewayServices
    store: FolderGrantStore

    @property
    def root(self) -> Path:
        return self.services.settings.config.path.parent.resolve(strict=False)

    @property
    def mask(self) -> tuple[Path, ...]:
        return self.store.view.sandbox_private_roots


def _instance(tmp_path: Path, name: str) -> _Instance:
    instance = tmp_path / name
    workspace = instance / "workspace"
    workspace.mkdir(parents=True)
    config_path = instance / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    return _Instance(config_path=config_path, workspace=workspace)


def _compose(config_path: Path | None, workspace: Path) -> _Composition:
    services = build_gateway_services(
        config=WebSocketConfig(),
        bus=MessageBus(),
        session_manager=None,
        static_dist_path=None,
        workspace_path=workspace,
        default_restrict_to_workspace=True,
        config_path=config_path,
        runtime_model_name=None,
        runtime_surface="webui",
        runtime_capabilities_overrides=None,
    )
    store = services.workspaces.grant_store
    assert store is not None
    return _Composition(services=services, store=store)


def _compose_instance(instance: _Instance) -> _Composition:
    return _compose(instance.config_path, instance.workspace)


def _default_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "legacy-workspace"
    workspace.mkdir(exist_ok=True)
    return workspace


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _entries(store: FolderGrantStore) -> set[GrantEntry]:
    return {
        (grant.path, grant.file_access, grant.commands)
        for grant in store.snapshot().grants
    }


def _entry(root: Path, identity: GrantIdentity) -> GrantEntry:
    return (str(root.resolve(strict=False)), identity[0], identity[1])


def _create(store: FolderGrantStore, root: Path, identity: GrantIdentity) -> None:
    root.mkdir()
    store.upsert(
        {"path": str(root), "file_access": identity[0], "commands": identity[1]},
        expected_revision=store.snapshot().revision,
    )


def test_fg01_two_explicit_instances_get_distinct_stores_and_roots(tmp_path: Path) -> None:
    a = _instance(tmp_path, "a")
    b = _instance(tmp_path, "b")
    composition_a = _compose_instance(a)
    composition_b = _compose_instance(b)

    assert composition_a.root != composition_b.root
    assert composition_a.store is not composition_b.store
    assert composition_a.store.path != composition_b.store.path
    assert _is_under(composition_a.store.path, composition_a.root)
    assert _is_under(composition_b.store.path, composition_b.root)

    before_b = _entries(composition_b.store)
    identity_a: GrantIdentity = ("read_write", False)
    _create(composition_a.store, tmp_path / "grant-a", identity_a)

    assert _entry(tmp_path / "grant-a", identity_a) in _entries(composition_a.store)
    assert _entry(tmp_path / "grant-a", identity_a) not in _entries(composition_b.store)
    assert _entries(composition_b.store) == before_b


def test_fg02_b_mask_and_store_derive_from_b_instance_root(tmp_path: Path) -> None:
    legacy = _compose(None, _default_workspace(tmp_path))
    legacy_root = legacy.root
    a = _instance(tmp_path, "a")
    b = _instance(tmp_path, "b")
    composition_a = _compose_instance(a)
    composition_b = _compose_instance(b)
    root_a = composition_a.root
    root_b = composition_b.root
    store_b = composition_b.store
    mask_b = composition_b.mask

    for path in mask_b:
        assert _is_under(path, root_b)
        assert not _is_under(path, root_a)
    assert mask_b != composition_a.mask
    for path in mask_b:
        matching = [
            root
            for root in (root_b, root_a, legacy_root)
            if _is_under(path, root)
        ]
        assert max(matching, key=lambda root: len(root.parts)) == root_b

    assert _is_under(store_b.path, root_b)
    assert not _is_under(store_b.path, root_a)
    assert not _is_under(store_b.path, legacy_root)

    identity_b: GrantIdentity = ("read_only", False)
    _create(store_b, tmp_path / "grant-b", identity_b)
    assert _entry(tmp_path / "grant-b", identity_b) in _entries(store_b)
    assert _entry(tmp_path / "grant-b", identity_b) not in _entries(composition_a.store)
    assert _entry(tmp_path / "grant-b", identity_b) not in _entries(legacy.store)


def test_fg03_grants_stay_independent_across_rebuild(tmp_path: Path) -> None:
    a = _instance(tmp_path, "a")
    b = _instance(tmp_path, "b")
    composition_a = _compose_instance(a)
    composition_b = _compose_instance(b)
    identity_a: GrantIdentity = ("read_write", False)
    identity_b: GrantIdentity = ("read_only", False)
    _create(composition_a.store, tmp_path / "grant-a", identity_a)

    assert _entry(tmp_path / "grant-a", identity_a) not in _entries(composition_b.store)

    _create(composition_b.store, tmp_path / "grant-b", identity_b)

    assert _entry(tmp_path / "grant-b", identity_b) not in _entries(composition_a.store)
    assert _entry(tmp_path / "grant-b", identity_b) in _entries(composition_b.store)
    assert _entry(tmp_path / "grant-a", identity_a) not in _entries(composition_b.store)

    rebuilt_a = _compose_instance(a)
    rebuilt_b = _compose_instance(b)

    assert rebuilt_a.root == composition_a.root
    assert rebuilt_b.root == composition_b.root
    assert rebuilt_a.store.path == composition_a.store.path
    assert rebuilt_b.store.path == composition_b.store.path
    assert _entry(tmp_path / "grant-a", identity_a) in _entries(rebuilt_a.store)
    assert _entry(tmp_path / "grant-b", identity_b) not in _entries(rebuilt_a.store)
    assert _entry(tmp_path / "grant-b", identity_b) in _entries(rebuilt_b.store)
    assert _entry(tmp_path / "grant-a", identity_a) not in _entries(rebuilt_b.store)


def test_fg04_default_caller_keeps_legacy_root_and_leaves_a_and_b_untouched(
    tmp_path: Path,
) -> None:
    a = _instance(tmp_path, "a")
    b = _instance(tmp_path, "b")
    composition_a = _compose_instance(a)
    composition_b = _compose_instance(b)
    identity_a: GrantIdentity = ("read_write", False)
    identity_b: GrantIdentity = ("read_only", False)
    _create(composition_a.store, tmp_path / "grant-a", identity_a)
    _create(composition_b.store, tmp_path / "grant-b", identity_b)
    legacy = _compose(None, _default_workspace(tmp_path))
    legacy_root = legacy.root

    before_root_a = composition_a.root
    before_root_b = composition_b.root
    before_entries_a = _entries(composition_a.store)
    before_entries_b = _entries(composition_b.store)
    before_mask_a = composition_a.mask
    before_mask_b = composition_b.mask

    composition_c = _compose(None, _default_workspace(tmp_path))

    assert composition_c.root == legacy_root
    assert composition_c.root != composition_a.root
    assert composition_c.root != composition_b.root
    assert composition_c.store is not composition_a.store
    assert composition_c.store is not composition_b.store

    identity_c: GrantIdentity = ("read_write", True)
    _create(composition_c.store, tmp_path / "grant-c", identity_c)

    assert _entry(tmp_path / "grant-c", identity_c) in _entries(composition_c.store)
    assert _entry(tmp_path / "grant-c", identity_c) not in _entries(composition_a.store)
    assert _entry(tmp_path / "grant-c", identity_c) not in _entries(composition_b.store)
    for path in composition_c.mask:
        assert _is_under(path, legacy_root)

    assert composition_a.root == before_root_a
    assert composition_b.root == before_root_b
    assert _entries(composition_a.store) == before_entries_a
    assert _entries(composition_b.store) == before_entries_b
    assert composition_a.mask == before_mask_a
    assert composition_b.mask == before_mask_b


def test_for_workspace_explicit_instance_roots_are_independent(tmp_path: Path) -> None:
    a = _instance(tmp_path, "a")
    b = _instance(tmp_path, "b")

    store_a = FolderGrantStore.for_workspace(
        a.workspace, instance_root=a.root, config_path=a.config_path
    )
    store_b = FolderGrantStore.for_workspace(
        b.workspace, instance_root=b.root, config_path=b.config_path
    )

    assert store_a is not store_b
    assert store_a.path != store_b.path
    assert store_a.view.sandbox_instance_root == a.root
    assert store_b.view.sandbox_instance_root == b.root
    for path in store_a.view.sandbox_private_roots:
        assert _is_under(path, a.root)
    for path in store_b.view.sandbox_private_roots:
        assert _is_under(path, b.root)
