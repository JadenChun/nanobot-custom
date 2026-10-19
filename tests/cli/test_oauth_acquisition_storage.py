"""Both interactive entry points leave persistence to the credential authority."""

from types import SimpleNamespace

import pytest

from nanobot.cli import onboard, provider
from nanobot.config.schema import Config


@pytest.mark.parametrize("entrypoint", ["provider", "onboard"])
def test_interactive_login_uses_acquisition_only_storage(monkeypatch, entrypoint):
    import oauth_cli_kit

    from nanobot.providers import codex_auth, oauth_model_catalog

    events = []
    generation = "login-generation"

    class Authority:
        def begin_login(self):
            events.append("begin")
            return generation

        def commit_login(self, token, actual_generation):
            assert actual_generation == generation
            assert token.access == "test-access"
            assert token.refresh == "test-refresh"
            events.append("commit")

    def login(**kwargs):
        assert events == ["begin"]
        storage = kwargs["storage"]
        assert isinstance(storage, provider.AcquisitionOnlyStorage)
        assert storage.load() is None
        storage.save(object())
        assert storage.load() is None
        with pytest.raises(RuntimeError, match="acquisition-only storage has no path"):
            storage.get_token_path()
        events.append("acquire")
        return SimpleNamespace(
            access="test-access", refresh="test-refresh", expires=2_000_000_000_000,
            account_id="test-account",
        )

    monkeypatch.setattr(codex_auth, "get_codex_authority", Authority)
    monkeypatch.setattr(oauth_cli_kit, "login_oauth_interactive", login)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: Config())
    monkeypatch.setattr(oauth_model_catalog, "invalidate_oauth_model_catalog",
                        lambda name: events.append(("invalidate", name)))

    if entrypoint == "provider":
        provider._login_openai_codex()
    else:
        assert onboard._quick_start_oauth_login(Config(), "openai_codex") is True

    assert events == ["begin", "acquire", "commit", ("invalidate", "openai_codex")]
