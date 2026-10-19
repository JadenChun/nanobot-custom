# P3 claimed cron composition checkpoint

This accepted dormant contract is part of the
[P3 closure checkpoint](p3-closure-checkpoint.md); acceptance does not wire or
authorize a runtime path.

`nanobot.cron.composition.compose_claimed_text_occurrence` is a dormant,
explicitly injected adapter for an already-issued occurrence claim. It accepts
only a fresh silent canonical turn, an exact Telegram receipt sender, optional
durable visible-session linking, and an optional private callback store. It is
not wired to the scheduler, gateway, channel manager, bus, cron tool, config,
or any production constructor.

The adapter validates the complete frozen destination set before consuming the
execution grant. Only Telegram text destinations or no-work are admitted;
channel metadata, unsupported channels, explicit send evidence, non-text or
empty results, and unsupported terminal states fail closed. Once execution has
begun, unsupported or interrupted computation remains claimed for explicit
reconciliation and is never replayed by this API.

Approval-required and policy-blocked silent runs execute no risky tool, create
no pending human approval, and settle as content-free skipped/no-work
completions. Successful bounded text settles completion, prepares the durable
outbox, optionally links the visible session, publishes a content-free callback
observation, and claims and dispatches each frozen destination once in order.
Ambiguous sends remain reconciliation-required and are never automatically
resent.

Callback records use deterministic occurrence/kind identities and transition
`pending -> invoking -> succeeded|failed`. Cancellation or crash after
`invoking` requires reconciliation; reconciliation may mark the record
`observed` or `abandoned` but never invokes the callback. Callback failure is
recorded without error text and does not block delivery.

The dormant P3 callback store requires POSIX `flock` and fails closed on
platforms without that primitive. Native cross-platform callback storage is a
later qualification task; this limitation does not permit fallback to weaker
locking or path-based storage.
