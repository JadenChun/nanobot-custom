"""Owner IPC admits kernel-authenticated peers before parsing their requests."""

from __future__ import annotations

import json
import os
import socket
import struct
import sys
from types import SimpleNamespace

import pytest

from nanobot.remote import ipc
from nanobot.remote.client import OwnerRequestError, RemoteAccessClient
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.remote.store import RemoteAccessStore

_TEST_UID = 4242


def _credentials(platform: str, uid: int, *, version: int = 0, groups: int = 1) -> bytes:
    if platform == "darwin":
        return struct.pack("=IIh2x16I", version, uid, groups, *([0] * 16))
    return struct.pack("3i", 1234, uid, 1234)


def _platform(monkeypatch, platform: str) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(os, "getuid", lambda: _TEST_UID, raising=False)
    monkeypatch.setattr(socket, "SO_PEERCRED", 17, raising=False)
    monkeypatch.setattr(socket, "LOCAL_PEERCRED", 1, raising=False)


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_same_user_uses_platform_kernel_credentials(tmp_path, monkeypatch, platform):
    _platform(monkeypatch, platform)
    calls = []

    def getsockopt(*args):
        calls.append(args)
        return _credentials(platform, os.getuid())

    server = ipc.OwnerIPCServer(RemoteAccessStore(tmp_path))
    server._check_peer(SimpleNamespace(getsockopt=getsockopt))
    assert calls == ([(0, 1, 76)] if platform == "darwin" else [(socket.SOL_SOCKET, 17, 12)])


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_other_user_is_denied_even_with_valid_credentials(tmp_path, monkeypatch, platform):
    _platform(monkeypatch, platform)
    peer = SimpleNamespace(getsockopt=lambda *_args: _credentials(platform, os.getuid() + 1))
    with pytest.raises(ipc.IPCDeniedError, match="selected OS user"):
        ipc.OwnerIPCServer(RemoteAccessStore(tmp_path))._check_peer(peer)


@pytest.mark.parametrize("platform,raw", [
    ("linux", b""),
    ("linux", b"\0" * 11),
    ("linux", b"\0" * 13),
    ("darwin", b""),
    ("darwin", b"\0" * 75),
    ("darwin", b"\0" * 77),
    ("darwin", _credentials("darwin", _TEST_UID, version=1)),
    ("darwin", _credentials("darwin", _TEST_UID, groups=-1)),
    ("darwin", _credentials("darwin", _TEST_UID, groups=17)),
])
def test_malformed_kernel_credentials_are_denied(tmp_path, monkeypatch, platform, raw):
    _platform(monkeypatch, platform)
    with pytest.raises(ipc.IPCDeniedError, match="invalid peer credentials"):
        ipc.OwnerIPCServer(RemoteAccessStore(tmp_path))._check_peer(
            SimpleNamespace(getsockopt=lambda *_args: raw),
        )


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_missing_credential_api_is_denied_without_reading_request(tmp_path, monkeypatch, platform):
    _platform(monkeypatch, platform)
    monkeypatch.delattr(socket, "SO_PEERCRED", raising=False)
    monkeypatch.delattr(socket, "LOCAL_PEERCRED", raising=False)
    with pytest.raises(ipc.IPCDeniedError, match="unavailable"):
        ipc.OwnerIPCServer(RemoteAccessStore(tmp_path))._check_peer(SimpleNamespace())


def test_unsupported_platform_is_denied_even_if_credential_constant_exists(tmp_path, monkeypatch):
    _platform(monkeypatch, "unsupported")
    with pytest.raises(ipc.IPCDeniedError, match="unavailable"):
        ipc.OwnerIPCServer(RemoteAccessStore(tmp_path))._check_peer(SimpleNamespace())


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_credential_read_failure_does_not_read_or_dispatch_request(tmp_path, monkeypatch, platform):
    _platform(monkeypatch, platform)
    replies = []
    closed = []

    def getsockopt(*_args):
        raise OSError("credential read failed")

    server = ipc.OwnerIPCServer(RemoteAccessStore(tmp_path))
    monkeypatch.setattr(server, "_dispatch", lambda *_args: pytest.fail("request dispatched"))
    server._handle_connection(SimpleNamespace(
        settimeout=lambda _timeout: None,
        getsockopt=getsockopt,
        recv=lambda *_args: pytest.fail("request read"),
        sendall=replies.append,
        close=lambda: closed.append(True),
    ))
    assert len(replies) == 1
    assert json.loads(replies[0])["error"]["code"] == "unavailable"
    assert closed == [True]


@pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"} or not hasattr(os, "getuid"),
    reason="requires Linux/Darwin kernel peer credentials and getuid",
)
def test_kernel_peer_is_checked_before_dispatch(tmp_path, monkeypatch):
    owner = RemoteAccessOwner(tmp_path / "remote-access")
    owner.start()
    dispatched = []
    try:
        client = RemoteAccessClient(owner.socket_path)
        assert client.ping()["version"] == 1
        actual_uid = os.getuid()
        with monkeypatch.context() as patches:
            patches.setattr(ipc.os, "getuid", lambda: actual_uid + 1)
            patches.setattr(owner.server, "_dispatch", lambda request: dispatched.append(request))
            with pytest.raises(OwnerRequestError, match="selected OS user") as rejected:
                client.ping()
            assert rejected.value.code == "denied"
        assert dispatched == []
    finally:
        owner.stop()
