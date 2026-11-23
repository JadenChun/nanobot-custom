This file provides guidance to AI coding agents working with this repository.

## Project Overview

nanobot is a lightweight, open-source AI agent framework written in Python with a React/TypeScript WebUI. It centers around a small agent loop that receives messages from chat channels, invokes an LLM provider, executes tools, and manages session memory.

## Development Commands

```bash
# Python: focused test / lint
uv run --no-sync python -m pytest tests/webui/test_dev.py -v
ruff check nanobot/

# Strict type checking (CI uses Python 3.14)
uv sync --python 3.14 --all-extras --dev
uv run --no-sync python -m scripts.install_channel_dependencies --all-channels
uv run --no-sync basedpyright

# WebUI: HTTP proxy to gateway :8765; browser connects to the WebSocket directly
# Build outputs to ../nanobot/web/dist (bundled into the Python wheel)
(cd webui && bun run dev)    # or set NANOBOT_API_URL to a different gateway
(cd webui && bun run build)
(cd webui && bun run test)

# Gateway
nanobot gateway
```

## Architecture and entry points

Use the [architecture guide](docs/architecture.md) for the runtime diagram,
source ownership, and client feature links. The main entry points are
[`nanobot/cli/commands.py`](nanobot/cli/commands.py) and
[`nanobot/nanobot.py`](nanobot/nanobot.py). Keep provider, channel, and tool
features at their owning edges rather than adding them to the core turn loop.

## Project-Specific Notes

- Architecture constraints: [`.agent/design.md`](.agent/design.md)
- Security boundaries: [`.agent/security.md`](.agent/security.md)
- Common gotchas: [`.agent/gotchas.md`](.agent/gotchas.md)

## Contribution Flow

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for contribution flow and PR guidelines.

## Migration repair validation

While strict-type debt remains, follow `docs/migration/p6-error-attribution.md`.
Each bounded repair needs a frozen before/after full-source checker report under
the same dependency environment, passing affected tests, and zero newly introduced
diagnostic signatures across **all** files (not only outside the edited scope).
Use `scripts/check_type_regressions.py`; lower total counts alone do not pass.
Retain source hashes and command/environment evidence; the comparator does not
attest those inputs. Never silently refresh the baseline to absorb new errors.
A repair-batch PASS is not stage completion: full unchanged strict checking must
exit zero and the required local full test suite must pass before closing P6.

## Code Style

- Python 3.11+, asyncio throughout.
- Line length: 100.
- Linting: `ruff` with rules E, F, I, N, W (E501 ignored).
- pytest with `asyncio_mode = "auto"`.

Tests mirror the `nanobot/` package structure.
