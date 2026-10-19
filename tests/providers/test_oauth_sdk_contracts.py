"""Real SDK acquisition supports save-only sinks, without activating OAuth I/O."""

from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from oauth_cli_kit import flow
from oauth_cli_kit.models import OAuthToken
from oauth_cli_kit.providers import OPENAI_CODEX_PROVIDER


class SaveOnlySink:
    def __init__(self):
        self.saved = []

    def save(self, token):
        self.saved.append(token)


def _offline_acquisition(monkeypatch, *, token, failure=None):
    outputs = []
    calls = []

    def no_file_storage(*args, **kwargs):
        pytest.fail("interactive acquisition must not construct SDK file storage")

    def no_browser(*args, **kwargs):
        pytest.fail("offline acquisition must not open a browser")

    def no_socket(state, *, on_code):
        calls.append(("callback_boundary", state))
        return None, "explicit offline unit dependency"

    def no_transport(code, verifier, provider, proxy):
        assert code == "unit-code"
        assert isinstance(verifier, str) and verifier
        assert provider is OPENAI_CODEX_PROVIDER
        assert proxy is None
        calls.append(("exchange_boundary", code))

        async def exchange():
            if failure is not None:
                raise failure
            return token

        return exchange

    def callback(prompt):
        assert prompt
        url = next(text for text in outputs if text.startswith(OPENAI_CODEX_PROVIDER.authorize_url))
        state = parse_qs(urlsplit(url).query)["state"][0]
        return OPENAI_CODEX_PROVIDER.redirect_uri + "?" + urlencode({"code": "unit-code", "state": state})

    monkeypatch.setattr(flow, "FileTokenStorage", no_file_storage)
    monkeypatch.setattr(flow, "_start_local_server", no_socket)
    monkeypatch.setattr(flow, "_exchange_code_for_token_async", no_transport)
    monkeypatch.setattr(flow.webbrowser, "open", no_browser)
    return outputs, calls, callback


@pytest.mark.parametrize("running_loop", [False, True])
async def test_real_interactive_sdk_accepts_save_only_sink(monkeypatch, running_loop):
    import asyncio

    token = OAuthToken("unit-access", "unit-refresh", 2_000_000_000_000, "unit-account")
    outputs, calls, callback = _offline_acquisition(monkeypatch, token=token)
    sink = SaveOnlySink()

    def acquire():
        return flow.login_oauth_interactive(
            print_fn=outputs.append, prompt_fn=callback,
            provider=OPENAI_CODEX_PROVIDER, storage=sink, open_browser=False,
        )

    acquired = acquire() if running_loop else await asyncio.to_thread(acquire)
    assert acquired is token
    assert sink.saved == [token]
    assert not hasattr(sink, "load") and not hasattr(sink, "get_token_path")
    assert [call[0] for call in calls] == ["callback_boundary", "exchange_boundary"]


def test_real_interactive_sdk_does_not_save_after_exchange_failure(monkeypatch):
    failure = RuntimeError("bounded unit exchange failure")
    token = OAuthToken("unit-access", "unit-refresh", 2_000_000_000_000, "unit-account")
    outputs, calls, callback = _offline_acquisition(monkeypatch, token=token, failure=failure)
    sink = SaveOnlySink()
    with pytest.raises(RuntimeError) as raised:
        flow.login_oauth_interactive(
            print_fn=outputs.append, prompt_fn=callback,
            provider=OPENAI_CODEX_PROVIDER, storage=sink, open_browser=False,
        )
    assert raised.value is failure
    assert sink.saved == []
    assert [call[0] for call in calls] == ["callback_boundary", "exchange_boundary"]
