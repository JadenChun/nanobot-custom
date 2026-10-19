# oauth-cli-kit boundary types

These repository-local declarations cover the SDK exports Nanobot consumes.
They do not replace runtime modules, install a different dependency, change
checker configuration, or implement token storage. Models, default fields,
aliases, public argument order and persistent-storage contracts mirror the
installed SDK. Undeclared APIs are not claimed to be covered.

One annotation is intentionally more precise than the SDK's: interactive login
only invokes `storage.save(token)`. Its private `_InteractiveLoginSink` protocol
therefore does not require loading, locking or a token-file path. `get_token`
still requires the full `TokenStorage` contract, including `Path`, because that
separate refresh flow genuinely uses it. The private protocol is a stub-only
structural type, not an additional runtime SDK export.

Nanobot's acquisition-only sink stays non-persistent. The returned token is
committed by the existing canonical credential coordinator with the captured
login generation; revocation and commit ownership are not changed by typing.
No dummy path, fallback SDK persistence, suppression or cast is used.

The P6 qualification reads these exact installed source files through its
descriptor-relative reader and binds their SHA256 hashes:

- `__init__.py`: a84e59cd34f1284ed8f348caaab270a55c8c9a1e46c24bd9d0638ac046768e4c
- `models.py`: 166fb81b2596c8733860bfc44fc0e418a76fb2a8620f4e09176db2e388a5d629
- `storage.py`: b9c18a0f0cc172d1aae8a1bfe781772db8a47bcbdce7a98045e930c409794221
- `flow.py`: 7bea57fa845f2a5d3ce206f0e9d4402e3f08ddfff0240eda8eafb5c7735494cb
- `providers/__init__.py`: 4bcaabefdca941b6ba69e817fe2820df80cbd9ac2d1d259a78988ee9d103b95e
- `providers/openai_codex.py`: d50d960c0cfc3a25f515b9df5203bd2239c19c820686431147050d145a9e066e

`tests/providers/test_oauth_sdk_contracts.py` exercises real SDK interactive
control flow with a save-only sink. Socket, browser, token-file storage and HTTP
exchange boundaries have explicit local mocks; both synchronous and already-
running-loop paths are exercised. Existing canonical credential tests remain
unchanged. This is not native OAuth or credential/provider activation evidence.

When upgrading the SDK, requalify the source signatures, consumed storage
operations and real-flow tests. Do not extend the save-only contract to refresh
or persistence paths merely to make type checking pass.
