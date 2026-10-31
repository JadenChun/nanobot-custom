# M02 non-agent fixture

This fixture is a deterministic transport peer for the Android client. It is **not** a Nanobot
gateway: it imports no `nanobot` package, starts no agent, and uses synthetic credentials and
media only. The versioned contract is `../contracts/fixture-v1.json`.

## Run

```bash
source ../scripts/mobile-env.sh
uv run --frozen python run.py --port 8123
# prints: FIXTURE_READY http://127.0.0.1:8123
```

Loopback binds only; the runner exits with code 2 for non-loopback hosts. Pass `--tls-cert` and
`--tls-key` together for the TLS-negative case. The release app must not trust that test CA.

## Tests

```bash
uv run --frozen python -m pytest
```

The suite starts real subprocesses on free loopback ports, checks health/teardown, rotates access
and refresh credentials, proves one effect per `request_id` across lost responses, exercises byte
ranges and 416 responses, bounds attachments and WS frames, and verifies self-signed TLS is
rejected by default.

## Reset and effect count

`POST /fixture/v1/admin/reset` with `X-Fixture-Admin: fixture-admin` clears tokens, modes, handled
request IDs and the mutation counter. `GET /fixture/v1/counter` returns `effects`, `handled` and
`resets`. Tests use these endpoints; they are not production routes.

Live HTTPS enrollment, tailnet mappings, real device identities and backend media authorization
are explicitly outside this fixture. Those need the M05/M09/M11 setup gates.
