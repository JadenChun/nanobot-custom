"""Image credential lookup keeps proxy-only kwargs and native thread/error phases."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from nanobot.providers import codex_auth
from nanobot.providers.image_generation import CodexImageGenerationClient, ImageGenerationError


@pytest.mark.parametrize("proxy", [None, "", "http://proxy.invalid:8080"])
@pytest.mark.parametrize("outcome", ["none", "missing_access", "error"])
async def test_native_token_thread_preserves_absent_or_proxy_only_keywords(
    monkeypatch, proxy, outcome,
) -> None:
    calls = []
    failure = ValueError("token acquisition failure")
    owner_thread = threading.get_ident()

    def lookup(**kwargs):
        calls.append((kwargs, threading.get_ident()))
        if outcome == "error":
            raise failure
        return None if outcome == "none" else SimpleNamespace(access="", account_id="")

    monkeypatch.setattr(codex_auth, "get_token", lookup)
    client = CodexImageGenerationClient(api_key=None, proxy=proxy)
    with pytest.raises(ImageGenerationError, match="OAuth token is unavailable") as raised:
        await client.generate(prompt="visible prompt", model="fixture")
    assert len(calls) == 1 and calls[0][0] == ({"proxy": proxy} if proxy else {})
    assert calls[0][1] != owner_thread
    assert raised.value.__cause__ is (failure if outcome == "error" else None)


@pytest.mark.parametrize("truthy", [False, True])
async def test_proxy_branch_retains_repeated_reads_and_value_identity(monkeypatch, truthy) -> None:
    marker = "http://proxy.invalid:8080" if truthy else ""

    class ObservedClient(CodexImageGenerationClient):
        def __getattribute__(self, name):
            values = object.__getattribute__(self, "__dict__")
            if name == "proxy" and "proxy_reads" in values:
                values["proxy_reads"].append("proxy")
                return marker
            return super().__getattribute__(name)

    calls = []

    def lookup(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(codex_auth, "get_token", lookup)
    client = ObservedClient(api_key=None)
    client.proxy_reads = []
    with pytest.raises(ImageGenerationError):
        await client.generate(prompt="visible prompt", model="fixture")
    assert client.proxy_reads == ["proxy"] * (2 if truthy else 1)
    assert calls == [{"proxy": marker} if truthy else {}]
    if truthy:
        assert calls[0]["proxy"] is marker


@pytest.mark.parametrize("fault_on", [1, 2])
async def test_proxy_read_fault_remains_wrapped_before_thread_dispatch(monkeypatch, fault_on) -> None:
    failure = RuntimeError("proxy read failure")

    class FaultClient(CodexImageGenerationClient):
        def __getattribute__(self, name):
            values = object.__getattribute__(self, "__dict__")
            if name == "proxy" and "proxy_reads" in values:
                values["proxy_reads"].append("proxy")
                if len(values["proxy_reads"]) == fault_on:
                    raise failure
                return "http://proxy.invalid:8080"
            return super().__getattribute__(name)

    async def unavailable(*args, **kwargs):
        pytest.fail("proxy read fault reached token dispatch")

    monkeypatch.setattr(asyncio, "to_thread", unavailable)
    client = FaultClient(api_key=None)
    client.proxy_reads = []
    with pytest.raises(ImageGenerationError) as raised:
        await client.generate(prompt="visible prompt", model="fixture")
    assert raised.value.__cause__ is failure
    assert client.proxy_reads == ["proxy"] * fault_on
