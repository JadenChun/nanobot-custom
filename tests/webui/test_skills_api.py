import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from nanobot.agent.skills import SkillsLoader
from nanobot.channels.websocket.runtime import WebSocketConfig
from nanobot.config.loader import load_config, save_config
from nanobot.config.schema import Config
from nanobot.context_repo.manager import ContextRepoManager
from nanobot.context_repo.policy import ResourceAccessPolicy
from nanobot.security.folder_grants import FolderGrantStore
from nanobot.webui.gateway_services import build_gateway_services
from nanobot.webui.skills_api import (
    _RESTRICTED_CATALOG_NOTICE,
    SkillManagementError,
    delete_webui_skill,
    set_webui_skill_enabled,
    webui_skill_detail_payload,
    webui_skills_payload,
)


def _write_skill(workspace: Path, name: str, *, metadata: str = "") -> Path:
    directory = workspace / "skills" / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} description.\n{metadata}---\n",
        encoding="utf-8",
    )
    return directory


def _config(*disabled: str) -> SimpleNamespace:
    return SimpleNamespace(
        agents=SimpleNamespace(
            defaults=SimpleNamespace(disabled_skills=list(disabled)),
        )
    )


def test_disabled_skills_remain_visible_and_loadable(tmp_path: Path) -> None:
    _write_skill(tmp_path, "custom-skill")

    payload = webui_skills_payload(tmp_path, disabled_skills={"custom-skill"})
    skill = next(item for item in payload["skills"] if item["name"] == "custom-skill")

    assert skill["enabled"] is False
    assert skill["deletable"] is True
    detail = webui_skill_detail_payload(
        tmp_path,
        "custom-skill",
        disabled_skills={"custom-skill"},
    )
    assert detail is not None
    assert detail["enabled"] is False
    assert "custom-skill description" in detail["raw_markdown"]


def test_read_only_workspace_skill_is_not_deletable_or_removed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill = _write_skill(workspace, "read-only-skill")
    agent_home = tmp_path / "agent-home"
    agent_home.mkdir()
    instance_root = tmp_path / "instance"
    instance_root.mkdir()
    grants = FolderGrantStore(
        agent_home=agent_home,
        instance_root=instance_root,
        path=instance_root / "webui" / "folder-grants.json",
    )
    grants.upsert(
        {"path": str(workspace), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    manager = ContextRepoManager.from_config(
        context_repos=[{"path": workspace, "readOnly": True}],
    )
    policy = ResourceAccessPolicy(
        workspace=workspace,
        context_manager=manager,
        restrict_to_workspace=True,
        folder_grants=grants.view,
    )

    payload = webui_skills_payload(
        workspace,
        context_manager=manager,
        resource_policy=policy,
    )
    item = next(skill for skill in payload["skills"] if skill["name"] == "read-only-skill")
    assert item["deletable"] is False

    with pytest.raises(SkillManagementError) as exc_info:
        delete_webui_skill(
            workspace,
            "read-only-skill",
            disabled_skills=set(),
            context_manager=manager,
            resource_policy=policy,
        )
    assert exc_info.value.status == 403
    assert exc_info.value.message == "skill cannot be deleted with current access"
    assert (skill / "SKILL.md").is_file()


def test_protected_skill_resource_blocks_recursive_workspace_deletion(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill = _write_skill(workspace, "example-skill")
    resources = skill / "resources"
    resources.mkdir()
    private_resource = resources / "private.txt"
    private_resource.write_text("protected resource", encoding="utf-8")
    (workspace / "nanobot.context.json").write_text(
        json.dumps(
            {
                "entrypoints": {"skills": ["skills"]},
                "protected": ["skills/example-skill/resources/**"],
            }
        ),
        encoding="utf-8",
    )
    manager = ContextRepoManager.from_config(
        context_repos=[{"path": workspace, "readOnly": False}],
    )
    agent_home = tmp_path / "agent-home"
    agent_home.mkdir()
    instance_root = tmp_path / "instance"
    instance_root.mkdir()
    grants = FolderGrantStore(
        agent_home=agent_home,
        instance_root=instance_root,
        path=instance_root / "webui" / "folder-grants.json",
    )
    grants.upsert(
        {"path": str(workspace), "file_access": "read_write", "commands": False},
        expected_revision=0,
    )
    policy = ResourceAccessPolicy(
        workspace=workspace,
        context_manager=manager,
        restrict_to_workspace=True,
        folder_grants=grants.view,
    )

    payload = webui_skills_payload(
        workspace,
        context_manager=manager,
        resource_policy=policy,
    )
    item = next(skill for skill in payload["skills"] if skill["name"] == "example-skill")
    assert item["deletable"] is False

    with pytest.raises(SkillManagementError) as exc_info:
        delete_webui_skill(
            workspace,
            "example-skill",
            disabled_skills=set(),
            context_manager=manager,
            resource_policy=policy,
        )
    assert exc_info.value.status == 403
    assert exc_info.value.message == "skill cannot be deleted with current access"
    assert (skill / "SKILL.md").is_file()
    assert private_resource.read_text(encoding="utf-8") == "protected resource"


def test_plugin_helpers_do_not_read_paths_before_policy_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    plugin_root = workspace / "plugins" / "demo"
    (plugin_root / "skills" / "private-skill").mkdir(parents=True)
    (plugin_root / "plugin.json").write_text("{}", encoding="utf-8")
    private_skill_file = plugin_root / "skills" / "private-skill" / "SKILL.md"
    private_skill_file.write_text(
        "---\nname: private-skill\ndescription: Hidden plugin skill.\n---\n",
        encoding="utf-8",
    )
    (workspace / "nanobot.context.json").write_text(
        json.dumps(
            {
                "entrypoints": {"skills": ["skills"]},
                "protected": ["plugins/demo/skills/private-skill/SKILL.md"],
            }
        ),
        encoding="utf-8",
    )
    manager = ContextRepoManager.from_config(context_paths=[workspace])
    agent_home = tmp_path / "agent-home"
    agent_home.mkdir()
    instance_root = tmp_path / "instance"
    instance_root.mkdir()
    grants = FolderGrantStore(
        agent_home=agent_home,
        instance_root=instance_root,
        path=instance_root / "webui" / "folder-grants.json",
    )
    grants.upsert(
        {"path": str(workspace), "file_access": "read_write", "commands": False},
        expected_revision=0,
    )
    policy = ResourceAccessPolicy(
        workspace=workspace,
        context_manager=manager,
        restrict_to_workspace=True,
        folder_grants=grants.view,
    )
    policy.validate_path(plugin_root, "list")
    with pytest.raises(PermissionError):
        policy.validate_path(private_skill_file, "read")

    helper_calls: list[Path] = []

    def plugin_helper(path: Path) -> list[tuple[str, Path]]:
        helper_calls.append(path)
        raise AssertionError("plugin helper ran before every plugin path was authorized")

    monkeypatch.setattr("nanobot.webui.skills_api.enabled_agent_plugin_skills", plugin_helper)

    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve(strict=False) == private_skill_file.resolve(strict=False):
            raise AssertionError("restricted plugin skill body was read")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    payload = webui_skills_payload(
        workspace,
        context_manager=manager,
        resource_policy=policy,
    )

    assert helper_calls == []
    assert payload["catalog_complete"] is False
    assert all(skill["name"] != "private-skill" for skill in payload["skills"])
    assert str(plugin_root) not in json.dumps(payload)


def test_gateway_without_session_manager_keeps_protected_skill_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shared_repo = tmp_path / "shared"
    _write_skill(shared_repo, "secret-skill")
    (shared_repo / "nanobot.context.json").write_text(
        json.dumps(
            {
                "entrypoints": {"skills": ["skills"]},
                "protected": ["skills/**"],
            }
        ),
        encoding="utf-8",
    )
    manager = ContextRepoManager.from_config(context_paths=[shared_repo])
    agent_home = tmp_path / "agent-home"
    agent_home.mkdir()
    instance_root = tmp_path / "instance"
    instance_root.mkdir()
    grants = FolderGrantStore(
        agent_home=agent_home,
        instance_root=instance_root,
        path=instance_root / "webui" / "folder-grants.json",
    )
    grants.upsert(
        {"path": str(workspace), "file_access": "read_write", "commands": False},
        expected_revision=0,
    )
    monkeypatch.setattr(
        "nanobot.webui.gateway_services.FolderGrantStore.for_workspace",
        lambda _workspace, *, sessions_root=None: grants,
    )

    services = build_gateway_services(
        config=WebSocketConfig(),
        bus=MagicMock(),
        session_manager=None,
        static_dist_path=None,
        workspace_path=workspace,
        context_manager=manager,
        default_restrict_to_workspace=True,
        config_path=tmp_path / "config.json",
        runtime_model_name=None,
        runtime_surface="browser",
        runtime_capabilities_overrides=None,
    )
    assert services.http.skills_context_manager is manager
    assert services.http.skills_resource_policy is not None
    assert services.http.skills_resource_policy.context_manager is manager
    monkeypatch.setattr(services.http, "check_api_token", lambda _request: True)

    response = services.http._handle_webui_skills(SimpleNamespace(path="/api/webui/skills"))
    payload = json.loads(response.body)

    assert payload["catalog_complete"] is False
    assert payload["catalog_notice"] == _RESTRICTED_CATALOG_NOTICE
    assert all(skill["name"] != "secret-skill" for skill in payload["skills"])
    assert str(shared_repo) not in json.dumps(payload)


def test_shared_skills_are_visible_loadable_and_managed_without_path_leaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shared_repo = tmp_path / "shared"
    shared_root = shared_repo / "skills"
    _write_skill(shared_repo, "shared-skill")
    disabled = {"shared-skill"}
    config = _config("shared-skill")
    saved: list[object] = []
    monkeypatch.setattr("nanobot.webui.skills_api.load_config", lambda _path=None: config)
    monkeypatch.setattr(
        "nanobot.webui.skills_api.save_config",
        lambda value, _path=None: saved.append(value),
    )

    payload = webui_skills_payload(
        workspace,
        disabled_skills=disabled,
        extra_skill_paths=[shared_root],
    )
    skill = next(item for item in payload["skills"] if item["name"] == "shared-skill")
    assert skill["source"] == "library"
    assert skill["enabled"] is False
    assert "path" not in skill

    detail = webui_skill_detail_payload(
        workspace,
        "shared-skill",
        disabled_skills=disabled,
        extra_skill_paths=[shared_root],
    )
    assert detail is not None
    assert detail["source"] == "library"
    assert detail["enabled"] is False
    assert "shared-skill description" in detail["raw_markdown"]
    assert "path" not in detail

    action = set_webui_skill_enabled(
        workspace,
        "shared-skill",
        enabled=True,
        disabled_skills=disabled,
        config_path=tmp_path / "gateway.json",
        extra_skill_paths=[shared_root],
    )
    assert action["enabled"] is True
    assert disabled == set()
    assert config.agents.defaults.disabled_skills == []
    assert saved == [config]

    with pytest.raises(SkillManagementError) as exc_info:
        delete_webui_skill(
            workspace,
            "shared-skill",
            disabled_skills=disabled,
            extra_skill_paths=[shared_root],
        )
    assert exc_info.value.status == 403
    assert (shared_root / "shared-skill" / "SKILL.md").is_file()


def test_read_denied_shared_skills_are_excluded_from_settings(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shared_repo = tmp_path / "shared"
    _write_skill(shared_repo, "private-skill")

    class DenySharedReads(ResourceAccessPolicy):
        @property
        def managed_mode(self) -> bool:
            return True

        def validate_path(self, *_args: object, **_kwargs: object) -> None:
            raise PermissionError("shared skill is not readable")

    policy = DenySharedReads(workspace=workspace)
    payload = webui_skills_payload(
        workspace,
        extra_skill_paths=[shared_repo / "skills"],
        resource_policy=policy,
    )

    assert all(item["name"] != "private-skill" for item in payload["skills"])
    assert payload["catalog_complete"] is False
    builtin = next(item for item in payload["skills"] if item["source"] == "builtin")
    assert builtin["verification"] == "unverified"
    assert builtin["available"] is False
    assert payload["catalog_notice"] == _RESTRICTED_CATALOG_NOTICE


def test_protected_shared_root_is_not_enumerated_or_read_by_management(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shared_repo = tmp_path / "shared"
    shared_root = shared_repo / "skills"
    _write_skill(shared_repo, "secret-skill")
    (shared_repo / "nanobot.context.json").write_text(
        json.dumps(
            {
                "entrypoints": {"context": "AGENTS.md", "skills": ["skills"]},
                "protected": ["AGENTS.md", "nanobot.context.json", "skills/**"],
            }
        ),
        encoding="utf-8",
    )
    manager = ContextRepoManager.from_config(context_paths=[shared_repo])

    class ManagedPolicy(ResourceAccessPolicy):
        @property
        def managed_mode(self) -> bool:
            return True

    policy = ManagedPolicy(workspace=workspace, context_manager=manager)
    assert SkillsLoader(
        workspace,
        extra_paths=[shared_root],
        resource_policy=policy,
    ).load_skill("secret-skill") is None

    original_iterdir = Path.iterdir
    original_read_text = Path.read_text

    def guarded_iterdir(path: Path):
        if path.resolve(strict=False) == shared_root.resolve(strict=False):
            raise AssertionError("restricted shared root was enumerated")
        return original_iterdir(path)

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve(strict=False) == (shared_root / "secret-skill" / "SKILL.md").resolve(
            strict=False
        ):
            raise AssertionError("restricted skill body was read")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    payload = webui_skills_payload(
        workspace,
        context_manager=manager,
        resource_policy=policy,
    )

    serialized = json.dumps(payload)
    assert payload["catalog_complete"] is False
    assert payload["catalog_notice"] == _RESTRICTED_CATALOG_NOTICE
    assert "secret-skill" not in serialized
    assert str(shared_repo) not in serialized
    assert "path" not in serialized


def test_workspace_and_ordered_libraries_keep_source_precedence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_skill(workspace, "same-name", metadata="")
    first_repo = tmp_path / "first"
    second_repo = tmp_path / "second"
    first_dir = _write_skill(first_repo, "same-name")
    _write_skill(second_repo, "same-name")
    _write_skill(second_repo, "ordered-skill")
    _write_skill(first_repo, "ordered-skill")
    manager = ContextRepoManager.from_config(context_paths=[first_repo, second_repo])
    payload = webui_skills_payload(workspace, context_manager=manager)

    same = next(item for item in payload["skills"] if item["name"] == "same-name")
    ordered = next(item for item in payload["skills"] if item["name"] == "ordered-skill")
    assert same["source"] == "workspace"
    assert ordered["source"] == "library"
    assert "path" not in same and "path" not in ordered
    assert first_dir.is_dir()


def test_denied_known_library_name_reserves_precedence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first_repo = tmp_path / "first"
    second_repo = tmp_path / "second"
    first_skill = _write_skill(first_repo, "reserved-skill")
    _write_skill(second_repo, "reserved-skill")
    (second_repo / "skills" / "reserved-skill" / "SKILL.md").write_text(
        "---\nname: reserved-skill\ndescription: lower fallback.\n---\n",
        encoding="utf-8",
    )
    manager = ContextRepoManager.from_config(context_paths=[first_repo, second_repo])

    class DenyFirstSkill(ResourceAccessPolicy):
        @property
        def managed_mode(self) -> bool:
            return True

        def validate_path(self, path: Path, action: str, **kwargs: Any) -> None:
            if path.resolve(strict=False) == first_skill.resolve(strict=False):
                raise PermissionError("private candidate")
            super().validate_path(path, action, **kwargs)

    policy = DenyFirstSkill(workspace=workspace, context_manager=manager)
    payload = webui_skills_payload(
        workspace,
        context_manager=manager,
        resource_policy=policy,
    )

    assert payload["catalog_complete"] is False
    assert all(item["name"] != "reserved-skill" for item in payload["skills"])


def test_skill_directory_and_skill_file_symlinks_are_excluded(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shared_root = tmp_path / "shared" / "skills"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_skill = outside / "outside-skill"
    outside_skill.mkdir()
    outside_file = outside_skill / "SKILL.md"
    outside_file.write_text(
        "---\nname: private\ndescription: Must stay hidden.\n---\n",
        encoding="utf-8",
    )
    shared_root.mkdir(parents=True)
    try:
        (shared_root / "escaping-directory").symlink_to(
            outside_skill,
            target_is_directory=True,
        )
        linked_file = shared_root / "linked-file"
        linked_file.mkdir()
        (linked_file / "SKILL.md").symlink_to(outside_file)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    payload = webui_skills_payload(workspace, extra_skill_paths=[shared_root])

    names = {skill["name"] for skill in payload["skills"]}
    assert payload["catalog_complete"] is False
    assert "escaping-directory" not in names
    assert "linked-file" not in names
    assert "private" not in names
    assert str(outside) not in json.dumps(payload)


def test_hidden_name_management_actions_match_absent_skill_404(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shared_repo = tmp_path / "shared"
    _write_skill(shared_repo, "hidden-skill")

    class DenySharedReads(ResourceAccessPolicy):
        @property
        def managed_mode(self) -> bool:
            return True

        def validate_path(self, path: Path, action: str, **kwargs: Any) -> None:
            if path.resolve(strict=False).is_relative_to(shared_repo.resolve(strict=False)):
                raise PermissionError("hidden path")
            super().validate_path(path, action, **kwargs)

    policy = DenySharedReads(workspace=workspace)
    kwargs = {
        "extra_skill_paths": [shared_repo / "skills"],
        "resource_policy": policy,
    }
    absent = webui_skill_detail_payload(workspace, "absent-skill", **kwargs)
    hidden = webui_skill_detail_payload(workspace, "hidden-skill", **kwargs)
    assert hidden is absent is None

    actions = (
        lambda name: set_webui_skill_enabled(
            workspace,
            name,
            enabled=False,
            disabled_skills=set(),
            **kwargs,
        ),
        lambda name: delete_webui_skill(workspace, name, disabled_skills=set(), **kwargs),
    )
    for action in actions:
        errors = []
        for name in ("hidden-skill", "absent-skill"):
            with pytest.raises(SkillManagementError) as error:
                action(name)
            errors.append(error.value)
        assert errors[0].status == errors[1].status == 404
        assert errors[0].message == errors[1].message


def test_unsupported_skill_glob_is_incomplete_and_unverifies_builtins(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo = tmp_path / "shared"
    repo.mkdir()
    (repo / "nanobot.context.json").write_text(
        json.dumps({"entrypoints": {"skills": ["skills/**/SKILL.md"]}}),
        encoding="utf-8",
    )
    manager = ContextRepoManager.from_config(context_paths=[repo])

    payload = webui_skills_payload(workspace, context_manager=manager)

    assert payload["catalog_complete"] is False
    assert all(
        skill["verification"] == "unverified"
        for skill in payload["skills"]
        if skill["source"] != "workspace"
    )


def test_literal_skill_root_with_trailing_separator_is_supported(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo = tmp_path / "shared"
    _write_skill(repo, "shared-skill")
    (repo / "nanobot.context.json").write_text(
        json.dumps({"entrypoints": {"skills": ["skills/"]}}),
        encoding="utf-8",
    )
    manager = ContextRepoManager.from_config(context_paths=[repo])

    payload = webui_skills_payload(workspace, context_manager=manager)

    shared = next(skill for skill in payload["skills"] if skill["name"] == "shared-skill")
    assert shared["source"] == "library"
    assert shared["verification"] == "verified"
    assert payload["catalog_complete"] is True


def test_skill_detail_exposes_copyable_install_commands(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "custom-skill",
        metadata=(
            'metadata: {"nanobot":{"requires":{"bins":["demo"]},'
            '"install":[{"id":"brew","kind":"brew","formula":"acme/demo",'
            '"label":"Install demo"}]}}\n'
        ),
    )

    detail = webui_skill_detail_payload(tmp_path, "custom-skill")

    assert detail is not None
    assert detail["install_options"] == [
        {
            "id": "brew",
            "kind": "brew",
            "label": "Install demo",
            "command": "brew install acme/demo",
        }
    ]


def test_set_webui_skill_enabled_persists_and_updates_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_skill(tmp_path, "custom-skill")
    config = _config()
    saved: list[object] = []
    monkeypatch.setattr("nanobot.webui.skills_api.load_config", lambda _path=None: config)
    monkeypatch.setattr(
        "nanobot.webui.skills_api.save_config",
        lambda value, _path=None: saved.append(value),
    )
    disabled: set[str] = set()

    action = set_webui_skill_enabled(
        tmp_path,
        "custom-skill",
        enabled=False,
        disabled_skills=disabled,
    )

    assert action == {
        "name": "custom-skill",
        "enabled": False,
        "deleted": False,
    }
    assert config.agents.defaults.disabled_skills == ["custom-skill"]
    assert disabled == {"custom-skill"}
    assert saved == [config]


def test_skill_state_uses_explicit_gateway_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _write_skill(workspace, "custom-skill")
    default_path = tmp_path / "default.json"
    gateway_path = tmp_path / "gateway.json"
    save_config(Config(), default_path)
    save_config(Config(), gateway_path)
    monkeypatch.setattr("nanobot.config.loader._current_config_path", default_path)

    set_webui_skill_enabled(
        workspace,
        "custom-skill",
        enabled=False,
        disabled_skills=set(),
        config_path=gateway_path,
    )

    assert load_config(default_path).agents.defaults.disabled_skills == []
    assert load_config(gateway_path).agents.defaults.disabled_skills == ["custom-skill"]


def test_delete_webui_skill_only_deletes_workspace_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _write_skill(tmp_path, "custom-skill")
    config = _config("custom-skill")
    saved: list[object] = []
    monkeypatch.setattr("nanobot.webui.skills_api.load_config", lambda _path=None: config)
    monkeypatch.setattr(
        "nanobot.webui.skills_api.save_config",
        lambda value, _path=None: saved.append(value),
    )
    disabled = {"custom-skill"}

    action = delete_webui_skill(
        tmp_path,
        "custom-skill",
        disabled_skills=disabled,
    )

    assert action["deleted"] is True
    assert not directory.exists()
    assert disabled == set()
    assert config.agents.defaults.disabled_skills == []
    assert saved == [config]

    with pytest.raises(SkillManagementError) as exc_info:
        delete_webui_skill(tmp_path, "cron", disabled_skills=disabled)
    assert exc_info.value.status == 403


def test_delete_webui_skill_rejects_symlinked_skills_root(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    directory = outside / "custom-skill"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\nname: custom-skill\n---\n",
        encoding="utf-8",
    )
    try:
        (workspace / "skills").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")

    with pytest.raises(SkillManagementError) as exc_info:
        delete_webui_skill(workspace, "custom-skill", disabled_skills=set())

    assert exc_info.value.status == 403
    assert (outside / "custom-skill" / "SKILL.md").is_file()


def test_delete_webui_skill_restores_directory_when_config_save_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _write_skill(tmp_path, "custom-skill")
    config = _config("custom-skill")
    monkeypatch.setattr("nanobot.webui.skills_api.load_config", lambda _path=None: config)

    def fail_save(_config: object, _path: Path | None = None) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("nanobot.webui.skills_api.save_config", fail_save)
    disabled = {"custom-skill"}

    with pytest.raises(OSError, match="disk full"):
        delete_webui_skill(
            tmp_path,
            "custom-skill",
            disabled_skills=disabled,
        )

    assert directory.is_dir()
    assert (directory / "SKILL.md").is_file()
    assert config.agents.defaults.disabled_skills == ["custom-skill"]
    assert disabled == {"custom-skill"}
