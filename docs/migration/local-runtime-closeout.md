# Local runtime closeout

## Foreground scope

Normal local foreground startup is owned by `RuntimeComposition.define()` and
its read-only runtime preflight. The preflight retains configuration, startup
receipt, session identity, session-mode, and inactive-idle checks before the
runtime graph is acquired. Acquisition owns the session lease, bus, real tool
registry, provider-backed agent, and optional channel manager; close releases
those resources in reverse order.

The SDK and API direct-result paths continue to use the canonical turn runner
without bus rerouting. The local WebUI continues to route through its existing
WebSocket command router and message bus. Temporary WebUI chats use
server-selected transient sessions with restricted tools and workspace scope,
and do not create durable session history or transcripts.

## Deferred activation and integrations

Dream, heartbeat, scheduled callbacks, idle compaction, and other autonomous
activation remain disabled by the current composition. H1's authenticated
Telegram registration remains optional. This closeout does not qualify a live
provider or channel, start a network listener, import an old journal, run a
production canary, or make a whole-H1 verdict. Temporary H1 chats do not gain a
durable adapter here.

## Evidence status

The core foreground runtime passed direct local validation on 2026-10-19.
The checks use a scripted provider with the real composition, agent loop,
runner, filesystem tool, and canonical storage. They cover a completed tool
turn, history reopening, temporary-session isolation, busy input, recovery,
stop, and lifecycle cleanup. They do not measure live model answer quality.

| Final validation batch | Result |
| --- | --- |
| All session and command tests | 1,988 passed |
| Agent, runner, SDK/API, startup and gateway regression selection | 888 passed, 1 existing platform-specific skip |
| All migration tests | 761 passed |
| Full unchanged strict Python policy | 472 files, 0 errors, 0 warnings |
| Production and changed-test lint | Passed |

There are 3,637 passing tests in the final batches. The previous combined run
hit its time limit and is retained as incomplete; its entire selection is
covered by the final batches, with additional cancellation coverage. No test
exclusions or new skips were introduced. The type diagnostic comparison
resolved all 35 initial diagnostics and introduced none.

Busy WebUI messages now enter the canonical pending journal under the active
source owner before queueing. Consumption commits the original input and
removes its pending record atomically; the runner accepts the writer's bound
projection. Stop commits pending-input disposition before cancelling work.
A failed stop write returns a visible retry response and preserves the active
turn. Recovery requeues unconsumed input only when execution evidence permits
it, without replaying uncertain tool effects. Invalid gateway configuration is
rejected before creating a log file, and API CLI preflight errors produce a
clear nonzero exit.

Normal foreground use requires provider configuration and the supported idle
settings: `agents.defaults.sessionTtlMinutes` (also accepted as
`idleCompactAfterMinutes`) must be `0`, and
`agents.defaults.idleCompaction.enabled` must be `false`. These are the current
defaults. Live Telegram/provider qualification, background activation, CI,
deployment, and new package-build qualification remain separate from this
core-runtime verdict.

The source hashes, exact commands, raw results, and earlier failed attempts are
preserved in [the validation receipt](../../.assets/validation/core-runtime-2026-10-19.mv2Ywz/receipt.json).
The tested source matched the candidate before this reporting-only update.
