"""Rules and refusals for managed macOS commands (A3-2M, AC5.1).

Pure generation: nothing is launched. Path spelling is faked off macOS.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from kohaku.agent.tools import sandbox
from kohaku.agent.tools.seatbelt_profile import (
    LAUNCH_TMP_PARAM,
    managed_cache_env,
    managed_profile,
    runtime_denies,
)
from kohaku.security import darwin_proc
from kohaku.security.folder_grants import FolderGrantStore

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX paths")


def _profile(**overrides: object) -> tuple[str, dict[str, str]]:
    arguments: dict[str, object] = {
        "hidden": (), "mounts": (), "protected": (), "input_files": (), "unix_sockets": (),
        "allow_network": False,
    }
    arguments.update(overrides)
    rules, params = managed_profile(**arguments)  # pyright: ignore[reportArgumentType]
    return rules, dict(params)


def _name(params: dict[str, str], path: str) -> str:
    return next(name for name, value in params.items() if value == path)


def _position(rules: str, params: dict[str, str], text: str, path: str) -> int:
    return rules.index(text.replace("PATH", f'(param "{_name(params, path)}")'))


_HIDE = "(deny file-read* file-write* (subpath PATH))"
_WRITE = "(allow file-read* file-write* (subpath PATH))"
_READ = "(allow file-read* (subpath PATH))"


def test_paths_reach_the_profile_only_as_parameters() -> None:
    hostile = '/w/a") (allow default) ("=b c'

    rules, params = _profile(
        hidden=(Path("/h/secret-dir"),), mounts=((Path(hostile), True),),
        protected=(Path("/p/private-dir"),),
    )

    assert hostile in params.values()
    for text in ("secret-dir", "private-dir", '/w/a', "(allow default)"):
        assert text not in rules
    assert len(set(params.values())) == len(params)


def test_the_most_specific_folder_rule_wins_and_protection_comes_last() -> None:
    rules, params = _profile(
        hidden=(Path("/data/instance"), Path("/work/project/hidden")),
        mounts=(
            (Path("/work/project"), True),
            (Path("/data/instance/workspace"), True),
            (Path("/work/project/readonly"), False),
        ),
        protected=(Path("/work/project/private"),),
        input_files=(Path("/data/instance/media/a.bin"),),
    )

    everything = rules.index("(allow file-read*)")
    instance = _position(rules, params, _HIDE, "/data/instance")
    workspace = _position(rules, params, _WRITE, "/data/instance/workspace")
    project = _position(rules, params, _WRITE, "/work/project")
    # A hidden folder inside a writable mount stays hidden.
    nested_hidden = _position(rules, params, _HIDE, "/work/project/hidden")
    read_only = _position(rules, params, _READ, "/work/project/readonly")
    protected = _position(rules, params, _HIDE, "/work/project/private")
    input_file = _position(
        rules, params, "(allow file-read* (literal PATH))", "/data/instance/media/a.bin",
    )

    assert everything < instance < workspace
    assert everything < project < nested_hidden < protected < input_file
    assert project < read_only
    assert _position(
        rules, params, "(deny file-write* (subpath PATH))", "/work/project/readonly",
    ) > project


def test_anchored_folders_and_their_ancestors_cannot_be_renamed() -> None:
    rules, params = _profile(
        mounts=((Path("/work/project"), True),), protected=(Path("/work/project/a/b"),),
    )

    for path in ("/work", "/work/project", "/work/project/a", "/work/project/a/b"):
        _position(
            rules, params,
            "(deny file-write-unlink (require-all (vnode-type DIRECTORY) (literal PATH)))", path,
        )
    # "/" may be named for ancestor metadata, but never as an anchored folder.
    anchored = re.findall(r'file-write-unlink [^\n]*?\(param "([^"]+)"\)', rules)
    assert anchored
    assert "/" not in {params[name] for name in anchored}


def test_network_rules_appear_only_when_allowed() -> None:
    offline, _ = _profile(allow_network=False)
    online, _ = _profile(allow_network=True)

    assert "(remote ip)" not in offline
    assert "mDNSResponder" not in offline
    assert "(allow network-outbound (remote ip))" in online


def test_other_processes_stay_private_and_the_default_is_deny() -> None:
    rules, _ = _profile(allow_network=True)

    deny = rules.index("(deny process-info*)")
    assert rules.index("(deny default)") < deny
    assert deny < rules.index("(allow process-info* (target same-sandbox))")
    assert rules.count("(allow process-info") == 1
    assert "(allow default)" not in rules


def test_every_parameter_reference_is_defined() -> None:
    rules, params = _profile(
        hidden=(Path("/h"),), mounts=((Path("/w"), True),), protected=(Path("/w/p"),),
        input_files=(Path("/h/i"),), unix_sockets=(Path("/s/sock"),), allow_network=True,
    )

    referenced = set(re.findall(r'\(param "([^"]+)"\)', rules))

    assert referenced == {*params, LAUNCH_TMP_PARAM}
    assert "(allow network-outbound (remote unix-socket (literal " in rules


def test_runtime_denies_use_their_own_parameters() -> None:
    rules, params = runtime_denies((Path("/state/receipts"), Path("/state/profiles")))

    assert all(name.startswith("R") for name, _ in params)
    assert {"/state/receipts", "/state/profiles", "/state"} <= {value for _, value in params}
    assert _HIDE.replace(
        "PATH", f'(param "{_name(dict(params), "/state/receipts")}")',
    ) in rules
    assert "(allow" not in rules


def test_cache_variables_stay_inside_the_temporary_folder() -> None:
    env = managed_cache_env("/private/tmp/kh-1")

    assert env["TMPDIR"] == "/private/tmp/kh-1"
    paths = [value for name, value in env.items() if name != "COREPACK_DEFAULT_TO_LATEST"]
    assert all(value.startswith("/private/tmp/kh-1") for value in paths)


@pytest.fixture
def macos(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prepare confinement as a ready macOS host would, on any POSIX host."""
    monkeypatch.setattr(sandbox, "managed_seatbelt_available", lambda: True)
    monkeypatch.setattr(darwin_proc, "canonical_path", lambda path: path.resolve(strict=False))
    monkeypatch.setattr(darwin_proc, "user_temp_dir", lambda: None)


def _store(tmp_path: Path) -> tuple[Path, Path, FolderGrantStore]:
    instance = tmp_path / "instance"
    home = instance / "workspace"
    project = tmp_path / "project"
    home.mkdir(parents=True)
    project.mkdir()
    store = FolderGrantStore(
        agent_home=home, instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    store.upsert(
        {"path": str(project), "file_access": "read_write", "commands": True},
        expected_revision=0,
    )
    return home.resolve(), project.resolve(), store


def test_prepared_command_keeps_the_real_cwd_and_runs_the_chosen_shell(
    tmp_path: Path, macos: None,
) -> None:
    home, project, store = _store(tmp_path)

    confinement = sandbox.prepare_managed_seatbelt(
        "echo 'hi there'", cwd=project, agent_home=home, grants=store.view,
        shell_program="/bin/zsh", login=True,
    )

    params = dict(confinement.params)
    assert confinement.cwd == str(project)
    assert confinement.argv[:2] == ("/bin/sh", "-c")
    assert confinement.argv[3:] == ("nanobot-sandbox", "/bin/zsh", "-l", "-c", "echo 'hi there'")
    instance = str((tmp_path / "instance").resolve())
    hidden = _position(confinement.rules, params, _HIDE, instance)
    assert hidden < _position(confinement.rules, params, _WRITE, str(home))
    _position(confinement.rules, params, _WRITE, str(project))
    assert "/private/tmp" in params.values()
    assert "(allow network-outbound (remote ip))" in confinement.rules


@pytest.mark.parametrize("root", ["/", "/usr"])
def test_a_repository_root_holding_the_runtime_is_refused(
    tmp_path: Path, macos: None, root: str,
) -> None:
    home, project, store = _store(tmp_path)

    with pytest.raises(ValueError, match="configured repository root cannot be masked"):
        sandbox.prepare_managed_seatbelt(
            "true", cwd=project, agent_home=home, grants=store.view,
            excluded_roots=(Path(root),),
        )


@pytest.mark.parametrize("root", ["/", "/dev"])
def test_a_task_root_covering_system_control_is_refused(
    tmp_path: Path, macos: None, root: str,
) -> None:
    home, project, store = _store(tmp_path)

    with pytest.raises(ValueError, match="Once permission root changed"):
        sandbox.prepare_managed_seatbelt(
            "true", cwd=project, agent_home=home, grants=store.view,
            task_roots=((root, "read_write", True),),
        )


def test_a_private_cwd_and_a_private_socket_are_refused(
    tmp_path: Path, macos: None,
) -> None:
    home, project, store = _store(tmp_path)

    with pytest.raises(PermissionError, match="cwd is inside private"):
        sandbox.prepare_managed_seatbelt(
            "true", cwd=tmp_path / "instance", agent_home=home, grants=store.view,
        )
    with pytest.raises(PermissionError, match="socket is inside a protected path"):
        sandbox.prepare_managed_seatbelt(
            "true", cwd=project, agent_home=home, grants=store.view,
            allow_unix_sockets=(str(tmp_path / "instance" / "owner.sock"),),
        )


def test_an_unready_host_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home, project, store = _store(tmp_path)
    monkeypatch.setattr(sandbox, "managed_seatbelt_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires a working macOS Seatbelt executor"):
        sandbox.prepare_managed_seatbelt(
            "true", cwd=project, agent_home=home, grants=store.view,
        )


@pytest.mark.parametrize("platform", ["linux", "other"])
def test_seatbelt_is_never_available_off_macos(
    monkeypatch: pytest.MonkeyPatch, platform: str,
) -> None:
    monkeypatch.setattr(sandbox, "current_platform", lambda: platform)
    monkeypatch.setattr(
        sandbox, "macos_seatbelt_ready", lambda: pytest.fail("the self-test must not run"),
    )

    assert sandbox.managed_seatbelt_available() is False
