from __future__ import annotations

import json
import multiprocessing
import os
import socket
import stat
from pathlib import Path
from typing import Any

import pytest

from nanobot.providers.oauth_credentials import (
    CredentialError,
    CredentialFlavor,
    CredentialNeedsRefreshError,
    CredentialStatus,
    OAuthCredentialCoordinator,
    OAuthToken,
)


@pytest.fixture(autouse=True)
def _deny_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is forbidden in credential coordinator tests")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def _codex_token(*, expiry: int | None = None, access: str = "access-test") -> OAuthToken:
    return OAuthToken(
        access=access,
        refresh="refresh-test",
        expires_at_ms=expiry if expiry is not None else 9_999_999_999_999,
        account_id="account-test",
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


def _race_import_worker(
    path: str, marker: str, barrier: Any, results: Any
) -> None:
    def importer() -> str:
        with Path(marker).open("ab") as stream:
            stream.write(b"i")
            stream.flush()
            os.fsync(stream.fileno())
        return "github-token-test"

    coordinator = OAuthCredentialCoordinator(
        Path(path), CredentialFlavor.COPILOT, importer=importer, clock_ms=lambda: 1_000
    )
    barrier.wait(timeout=10)
    credentials = coordinator.get_cached_or_import()
    results.put(None if credentials is None else credentials.access)



def test_json_null_canonical_is_invalid_and_never_triggers_import(tmp_path: Path) -> None:
    calls: list[int] = []
    canonical = tmp_path / "codex.json"
    coordinator = OAuthCredentialCoordinator(
        canonical,
        CredentialFlavor.CODEX,
        importer=lambda: calls.append(1) or _codex_token(access="external"),
    )
    generation = coordinator.begin_login()
    coordinator.commit_login(_codex_token(), generation)
    canonical.write_text("null", encoding="utf-8")
    canonical.chmod(0o600)

    assert coordinator.peek_status() is CredentialStatus.INVALID
    with pytest.raises(CredentialError, match="invalid credential store"):
        coordinator.get_cached_or_import()
    assert calls == []


def test_json_null_state_is_invalid_and_cannot_clear_logout_tombstone(
    tmp_path: Path,
) -> None:
    calls: list[int] = []
    coordinator = OAuthCredentialCoordinator(
        tmp_path / "codex.json",
        CredentialFlavor.CODEX,
        importer=lambda: calls.append(1) or _codex_token(access="external"),
    )
    coordinator.logout()
    state = tmp_path / "codex.json.state.json"
    state.write_text("null", encoding="utf-8")
    state.chmod(0o600)

    assert coordinator.peek_status() is CredentialStatus.INVALID
    with pytest.raises(CredentialError, match="invalid credential store"):
        coordinator.get_cached_or_import()
    assert calls == []
    assert not (tmp_path / "codex.json").exists()


def test_exact_maximum_encoded_token_payload_is_accepted(tmp_path: Path) -> None:
    size_limit = 65_536
    token = _codex_token()
    single_byte_payload = json.dumps(
        token.to_dict(), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    token = OAuthToken(
        access="x" * (size_limit - len(single_byte_payload) + len(token.access)),
        refresh=token.refresh,
        expires_at_ms=token.expires_at_ms,
        account_id=token.account_id,
    )
    payload = json.dumps(token.to_dict(), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    assert len(payload) == size_limit

    coordinator = OAuthCredentialCoordinator(tmp_path / "codex.json", CredentialFlavor.CODEX)
    generation = coordinator.begin_login()
    coordinator.commit_login(token, generation)

    assert coordinator.peek_status() is CredentialStatus.AVAILABLE


def test_oversized_utf8_token_is_rejected_before_temp_or_canonical_write(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "codex.json"
    coordinator = OAuthCredentialCoordinator(canonical, CredentialFlavor.CODEX)
    generation = coordinator.begin_login()
    temporary_files_before = set(tmp_path.glob(".codex.json.*.tmp"))
    token = _codex_token(access="é" * 32_768)
    payload = json.dumps(token.to_dict(), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    assert len(payload) > 65_536

    with pytest.raises(CredentialError, match="credential record exceeds size limit"):
        coordinator.commit_login(token, generation)

    assert not canonical.exists()
    assert set(tmp_path.glob(".codex.json.*.tmp")) == temporary_files_before


def test_token_json_is_exact_and_provider_validation_is_separate(tmp_path: Path) -> None:
    token = OAuthToken.from_json(
        '{"access":"a","refresh":"r","expires_at_ms":123,"account_id":"acct"}'
    )
    assert token == OAuthToken("a", "r", 123, "acct")
    assert '"a"' not in repr(token)
    assert '"r"' not in repr(token)

    for raw in (
        '{"access":"a","refresh":"r","expires_at_ms":123,"extra":true}',
        '{"access":"a","access":"b","refresh":"r","expires_at_ms":123}',
        '{"access":"a","refresh":"r","expires_at_ms":true}',
        '{"access":"a","refresh":"r","expires_at_ms":123.0}',
        '[]',
    ):
        with pytest.raises(CredentialError, match="invalid credential"):
            OAuthToken.from_json(raw)

    codex = OAuthCredentialCoordinator(tmp_path / "invalid-codex.json", CredentialFlavor.CODEX)
    codex_generation = codex.begin_login()
    with pytest.raises(CredentialError, match="invalid credential"):
        codex.commit_login(OAuthToken("a", "", 123, "acct"), codex_generation)
    copilot = OAuthCredentialCoordinator(
        tmp_path / "invalid-copilot.json", CredentialFlavor.COPILOT
    )
    copilot_generation = copilot.begin_login()
    with pytest.raises(CredentialError, match="invalid credential"):
        copilot.commit_login(
            OAuthToken("bearer-not-github", "refresh", 123), copilot_generation
        )


def test_constructor_requires_explicit_absolute_path_and_flavor(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        OAuthCredentialCoordinator()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="absolute credential path required"):
        OAuthCredentialCoordinator(Path("relative.json"), CredentialFlavor.CODEX)
    with pytest.raises(ValueError, match="reserved credential path"):
        OAuthCredentialCoordinator(tmp_path / "token.state.json", CredentialFlavor.CODEX)


def test_copilot_import_stores_only_github_token_with_long_expiry(tmp_path: Path) -> None:
    now_ms = 1_700_000_000_000
    calls: list[int] = []
    canonical = tmp_path / "copilot.json"
    coordinator = OAuthCredentialCoordinator(
        canonical,
        CredentialFlavor.COPILOT,
        importer=lambda: calls.append(1) or "github-long-lived-test-token",
        clock_ms=lambda: now_ms,
    )

    token = coordinator.get_cached_or_import()

    assert token == OAuthToken(
        "github-long-lived-test-token", "", now_ms + 315_360_000_000, None
    )
    assert calls == [1]
    assert json.loads(canonical.read_text(encoding="utf-8")) == {
        "access": "github-long-lived-test-token",
        "refresh": "",
        "expires_at_ms": now_ms + 315_360_000_000,
    }
    assert coordinator.get_cached_or_import() == token
    assert calls == [1]


def test_copilot_login_uses_github_token_and_never_a_cached_bearer(tmp_path: Path) -> None:
    now_ms = 500_000
    coordinator = OAuthCredentialCoordinator(
        tmp_path / "copilot.json", CredentialFlavor.COPILOT, clock_ms=lambda: now_ms
    )
    generation = coordinator.begin_login()

    token = coordinator.commit_login("github-login-token", generation)

    assert token.access == "github-login-token"
    assert token.refresh == ""
    assert token.expires_at_ms == now_ms + 315_360_000_000
    assert token.access != "cached-copilot-bearer"


def test_initial_import_is_not_repeated_and_external_source_is_read_only(tmp_path: Path) -> None:
    source = tmp_path / "synthetic-source.json"
    source.write_text('{"access":"source","refresh":"r","expires_at_ms":9999999999999}')
    before = source.read_bytes()
    calls: list[int] = []

    def importer() -> dict[str, object]:
        calls.append(1)
        value = json.loads(source.read_text(encoding="utf-8"))
        value["account_id"] = "source-account"
        return value

    coordinator = OAuthCredentialCoordinator(
        tmp_path / "codex.json", CredentialFlavor.CODEX, importer=importer
    )

    first = coordinator.get_cached_or_import()
    second = coordinator.get_cached_or_import()

    assert first == OAuthToken("source", "r", 9_999_999_999_999, "source-account")
    assert second == first
    assert calls == [1]
    assert source.read_bytes() == before


def test_importer_is_not_consulted_for_existing_malformed_or_expired_canonical(
    tmp_path: Path,
) -> None:
    calls: list[int] = []
    canonical = tmp_path / "codex.json"
    coordinator = OAuthCredentialCoordinator(
        canonical,
        CredentialFlavor.CODEX,
        importer=lambda: calls.append(1) or _codex_token(),
    )
    generation = coordinator.begin_login()
    coordinator.commit_login(_codex_token(), generation)
    _write_json(canonical, _codex_token(expiry=1).to_dict())

    with pytest.raises(CredentialNeedsRefreshError, match="credential needs refresh"):
        coordinator.get_cached_or_import()
    assert coordinator.peek_status() is CredentialStatus.NEEDS_REFRESH
    assert calls == []

    _write_json(canonical, {"access": "partial"})
    assert coordinator.peek_status() is CredentialStatus.INVALID
    with pytest.raises(CredentialError, match="invalid credential"):
        coordinator.get_cached_or_import()
    assert calls == []


def test_copilot_external_import_rejects_cached_bearer_object(tmp_path: Path) -> None:
    coordinator = OAuthCredentialCoordinator(
        tmp_path / "copilot.json",
        CredentialFlavor.COPILOT,
        importer=lambda: {
            "access": "cached-bearer",
            "refresh": "",
            "expires_at_ms": 999_999_999_999,
        },
    )

    with pytest.raises(CredentialError, match="credential import failed"):
        coordinator.get_cached_or_import()
    assert not (tmp_path / "copilot.json").exists()


def test_importer_exception_is_replaced_with_fixed_content_free_error(tmp_path: Path) -> None:
    coordinator = OAuthCredentialCoordinator(
        tmp_path / "codex.json",
        CredentialFlavor.CODEX,
        importer=lambda: (_ for _ in ()).throw(RuntimeError("secret value")),
    )
    with pytest.raises(CredentialError, match="credential import failed") as exc:
        coordinator.get_cached_or_import()
    assert "secret value" not in str(exc.value)


def test_logout_tombstone_precedes_deletion_and_suppresses_reimport(tmp_path: Path) -> None:
    calls: list[int] = []
    canonical = tmp_path / "codex.json"
    coordinator = OAuthCredentialCoordinator(
        canonical,
        CredentialFlavor.CODEX,
        importer=lambda: calls.append(1) or _codex_token(),
    )
    generation = coordinator.begin_login()
    coordinator.commit_login(_codex_token(), generation)
    original_lock = (tmp_path / "codex.json.lock").stat().st_ino

    coordinator.logout()

    assert not canonical.exists()
    assert coordinator.peek_status() is CredentialStatus.LOGGED_OUT
    assert coordinator.get_cached_or_import() is None
    assert calls == []
    assert (tmp_path / "codex.json.lock").stat().st_ino == original_lock
    state = json.loads((tmp_path / "codex.json.state.json").read_text(encoding="utf-8"))
    assert state["version"] == 1
    assert len(state["generation"]) == 32
    assert state["logged_out"] is True


def test_logout_unlink_failure_leaves_durable_tombstone_and_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / "codex.json"
    coordinator = OAuthCredentialCoordinator(canonical, CredentialFlavor.CODEX)
    generation = coordinator.begin_login()
    coordinator.commit_login(_codex_token(), generation)
    real_unlink = os.unlink

    def fail_canonical_unlink(path: str | bytes | os.PathLike[str] | os.PathLike[bytes], *args: object, **kwargs: object) -> None:
        if Path(path).name == canonical.name:
            raise OSError("synthetic unlink failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", fail_canonical_unlink)
    with pytest.raises(CredentialError, match="credential store unavailable"):
        coordinator.logout()

    assert canonical.exists()
    assert coordinator.peek_status() is CredentialStatus.LOGGED_OUT
    assert coordinator.get_cached_or_import() is None


def test_stale_login_commit_is_rejected_after_logout_or_new_login(tmp_path: Path) -> None:
    coordinator = OAuthCredentialCoordinator(tmp_path / "codex.json", CredentialFlavor.CODEX)
    stale = coordinator.begin_login()
    coordinator.logout()
    with pytest.raises(CredentialError, match="login attempt is stale"):
        coordinator.commit_login(_codex_token(), stale)

    first = coordinator.begin_login()
    second = coordinator.begin_login()
    with pytest.raises(CredentialError, match="login attempt is stale"):
        coordinator.commit_login(_codex_token(), first)
    assert coordinator.commit_login(_codex_token(), second) == _codex_token()


def test_generation_is_opaque_process_local_noncopyable_and_nonpickleable(
    tmp_path: Path,
) -> None:
    import copy
    import pickle

    coordinator = OAuthCredentialCoordinator(tmp_path / "codex.json", CredentialFlavor.CODEX)
    generation = coordinator.begin_login()
    assert "access" not in repr(generation)
    with pytest.raises(TypeError):
        copy.copy(generation)
    with pytest.raises(TypeError):
        copy.deepcopy(generation)
    with pytest.raises(TypeError):
        pickle.dumps(generation)

    other = OAuthCredentialCoordinator(tmp_path / "other.json", CredentialFlavor.CODEX)
    with pytest.raises(CredentialError, match="login attempt is stale"):
        other.commit_login(_codex_token(), generation)


def test_explicit_reimport_requires_current_generation_and_runs_only_on_request(
    tmp_path: Path,
) -> None:
    calls: list[int] = []
    coordinator = OAuthCredentialCoordinator(
        tmp_path / "codex.json",
        CredentialFlavor.CODEX,
        importer=lambda: calls.append(1) or _codex_token(access=f"import-{len(calls)}"),
    )
    initial = coordinator.begin_login()
    coordinator.commit_login(_codex_token(), initial)
    assert coordinator.get_cached_or_import() == _codex_token()
    assert calls == []

    current = coordinator.begin_login()
    with pytest.raises(CredentialError, match="login attempt is stale"):
        coordinator.explicit_reimport(initial)
    reimported = coordinator.explicit_reimport(current)
    assert reimported.access == "import-1"
    assert calls == [1]


def test_explicit_reimport_after_logout_requires_new_user_generation(tmp_path: Path) -> None:
    calls: list[int] = []
    coordinator = OAuthCredentialCoordinator(
        tmp_path / "codex.json",
        CredentialFlavor.CODEX,
        importer=lambda: calls.append(1) or _codex_token(),
    )
    coordinator.logout()
    assert coordinator.get_cached_or_import() is None
    assert calls == []

    generation = coordinator.begin_login()
    assert coordinator.peek_status() is CredentialStatus.LOGGED_OUT
    assert coordinator.explicit_reimport(generation) == _codex_token()
    assert coordinator.peek_status() is CredentialStatus.AVAILABLE
    assert calls == [1]


def test_invalid_or_missing_state_never_claims_canonical_as_active(tmp_path: Path) -> None:
    canonical = tmp_path / "codex.json"
    coordinator = OAuthCredentialCoordinator(canonical, CredentialFlavor.CODEX)
    _write_json(canonical, _codex_token().to_dict())

    assert coordinator.peek_status() is CredentialStatus.INVALID
    with pytest.raises(CredentialError, match="invalid credential"):
        coordinator.get_cached_or_import()

    state_path = tmp_path / "codex.json.state.json"
    state_path.write_text("{", encoding="utf-8")
    state_path.chmod(0o600)
    assert coordinator.peek_status() is CredentialStatus.INVALID


def test_copied_state_without_canonical_token_has_no_auth_claim(tmp_path: Path) -> None:
    source = OAuthCredentialCoordinator(tmp_path / "source.json", CredentialFlavor.CODEX)
    generation = source.begin_login()
    source.commit_login(_codex_token(), generation)
    target_path = tmp_path / "target.json"
    (tmp_path / "target.json.state.json").write_bytes(
        (tmp_path / "source.json.state.json").read_bytes()
    )
    (tmp_path / "target.json.state.json").chmod(0o600)

    target = OAuthCredentialCoordinator(target_path, CredentialFlavor.CODEX)
    assert target.peek_status() is CredentialStatus.ABSENT
    assert target.get_cached_or_import() is None


def test_canonical_state_and_lock_are_owner_private_and_regular(tmp_path: Path) -> None:
    coordinator = OAuthCredentialCoordinator(tmp_path / "codex.json", CredentialFlavor.CODEX)
    generation = coordinator.begin_login()
    coordinator.commit_login(_codex_token(), generation)

    for path in (
        tmp_path / "codex.json",
        tmp_path / "codex.json.state.json",
        tmp_path / "codex.json.lock",
    ):
        info = path.stat()
        assert stat.S_ISREG(info.st_mode)
        assert info.st_nlink == 1
        assert stat.S_IMODE(info.st_mode) == 0o600


def test_symlink_hardlink_and_nonprivate_canonical_are_rejected(tmp_path: Path) -> None:
    canonical = tmp_path / "codex.json"
    coordinator = OAuthCredentialCoordinator(canonical, CredentialFlavor.CODEX)
    generation = coordinator.begin_login()
    coordinator.commit_login(_codex_token(), generation)

    linked = tmp_path / "hardlink.json"
    os.link(canonical, linked)
    assert coordinator.peek_status() is CredentialStatus.INVALID
    linked.unlink()

    secret = tmp_path / "synthetic-target.json"
    _write_json(secret, _codex_token().to_dict())
    canonical.unlink()
    canonical.symlink_to(secret)
    assert coordinator.peek_status() is CredentialStatus.INVALID
    canonical.unlink()

    generation = coordinator.begin_login()
    coordinator.commit_login(_codex_token(), generation)
    canonical.chmod(0o644)
    assert coordinator.peek_status() is CredentialStatus.INVALID


def test_path_collision_between_flavors_fails_closed(tmp_path: Path) -> None:
    canonical = tmp_path / "shared.json"
    codex = OAuthCredentialCoordinator(canonical, CredentialFlavor.CODEX)
    generation = codex.begin_login()
    codex.commit_login(_codex_token(), generation)

    copilot = OAuthCredentialCoordinator(canonical, CredentialFlavor.COPILOT)
    assert copilot.peek_status() is CredentialStatus.INVALID
    with pytest.raises(CredentialError, match="invalid credential"):
        copilot.get_cached_or_import()


def test_lock_creation_failure_fails_closed_without_store_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / "codex.json"
    coordinator = OAuthCredentialCoordinator(
        canonical, CredentialFlavor.CODEX, importer=lambda: _codex_token()
    )
    lock_path = tmp_path / "codex.json.lock"
    real_open = os.open

    def fail_lock_open(path: str | bytes | os.PathLike[str] | os.PathLike[bytes], flags: int, *args: object, **kwargs: object) -> int:
        if Path(path).name == lock_path.name:
            raise OSError("synthetic lock failure")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", fail_lock_open)
    with pytest.raises(CredentialError, match="credential store unavailable"):
        coordinator.get_cached_or_import()
    assert not canonical.exists()
    assert not (tmp_path / "codex.json.state.json").exists()


def test_replace_failure_never_exposes_partial_canonical_and_leaves_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / "codex.json"
    coordinator = OAuthCredentialCoordinator(canonical, CredentialFlavor.CODEX)
    generation = coordinator.begin_login()
    real_replace = os.replace

    def fail_replace(src: str | bytes | os.PathLike[str] | os.PathLike[bytes], dst: str | bytes | os.PathLike[str] | os.PathLike[bytes], *args: object, **kwargs: object) -> None:
        if Path(dst).name == canonical.name:
            raise OSError("synthetic replace failure")
        real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(CredentialError, match="credential store unavailable"):
        coordinator.commit_login(_codex_token(), generation)

    assert not canonical.exists()
    assert (tmp_path / "codex.json.lock").exists()
    assert list(tmp_path.glob(".codex.json.*.tmp"))


def test_logout_directory_fsync_failure_keeps_canonical_and_tombstone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / "codex.json"
    coordinator = OAuthCredentialCoordinator(canonical, CredentialFlavor.CODEX)
    generation = coordinator.begin_login()
    coordinator.commit_login(_codex_token(), generation)
    real_fsync = os.fsync

    def fail_directory_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("synthetic directory fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)
    with pytest.raises(CredentialError, match="credential store unavailable"):
        coordinator.logout()

    assert canonical.exists()
    assert coordinator.peek_status() is CredentialStatus.LOGGED_OUT


def test_import_and_logout_serialize_to_a_final_tombstone(tmp_path: Path) -> None:
    canonical = tmp_path / "codex.json"
    importer_entered = multiprocessing.Event()
    release_importer = multiprocessing.Event()

    def importer() -> OAuthToken:
        importer_entered.set()
        assert release_importer.wait(timeout=10)
        return _codex_token()

    coordinator = OAuthCredentialCoordinator(
        canonical, CredentialFlavor.CODEX, importer=importer
    )
    result: list[OAuthToken | None] = []

    def import_once() -> None:
        result.append(coordinator.get_cached_or_import())

    thread = __import__("threading").Thread(target=import_once)
    thread.start()
    assert importer_entered.wait(timeout=10)

    logout_done = multiprocessing.Event()

    def logout_after_import() -> None:
        release_importer.set()
        coordinator.logout()
        logout_done.set()

    logout_thread = __import__("threading").Thread(target=logout_after_import)
    logout_thread.start()
    thread.join(timeout=10)
    logout_thread.join(timeout=10)

    assert not thread.is_alive()
    assert not logout_thread.is_alive()
    assert logout_done.is_set()
    assert result == [_codex_token()]
    assert not canonical.exists()
    assert coordinator.peek_status() is CredentialStatus.LOGGED_OUT


def test_multiprocess_import_is_single_and_uses_stable_flock(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    path = tmp_path / "copilot.json"
    marker = tmp_path / "import-count"
    barrier = context.Barrier(4)
    results = context.Queue()
    workers = [
        context.Process(target=_race_import_worker, args=(str(path), str(marker), barrier, results))
        for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=20)
    assert all(not worker.is_alive() for worker in workers)
    assert [worker.exitcode for worker in workers] == [0, 0, 0, 0]
    assert [results.get(timeout=2) for _ in workers] == ["github-token-test"] * 4
    assert marker.read_bytes() == b"i"

    lock = path.with_name(path.name + ".lock")
    first_inode = lock.stat().st_ino
    coordinator = OAuthCredentialCoordinator(path, CredentialFlavor.COPILOT)
    coordinator.logout()
    assert lock.stat().st_ino == first_inode


def test_atomic_import_directory_fsync_failure_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / "codex.json"
    coordinator = OAuthCredentialCoordinator(
        canonical, CredentialFlavor.CODEX, importer=lambda: _codex_token()
    )
    real_fsync = os.fsync

    def fail_directory_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("synthetic directory fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)
    with pytest.raises(CredentialError, match="credential store unavailable"):
        coordinator.get_cached_or_import()

    assert canonical.exists()
    assert coordinator.peek_status() is CredentialStatus.INVALID
    assert list(tmp_path.glob(".codex.json.*.tmp")) == []


def test_temp_file_fsync_failure_does_not_publish_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / "codex.json"
    coordinator = OAuthCredentialCoordinator(canonical, CredentialFlavor.CODEX)
    generation = coordinator.begin_login()
    real_fsync = os.fsync

    def fail_regular_file_fsync(fd: int) -> None:
        if stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("synthetic file fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_regular_file_fsync)
    with pytest.raises(CredentialError, match="credential store unavailable"):
        coordinator.commit_login(_codex_token(), generation)

    assert not canonical.exists()
    assert coordinator.peek_status() is CredentialStatus.ABSENT
    assert list(tmp_path.glob(".codex.json.*.tmp"))
