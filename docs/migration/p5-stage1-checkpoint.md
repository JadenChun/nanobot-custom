# P5 Stage 1 lifecycle checkpoint

Status as of 2026-09-30: **Stage 1 accepted for the guarded offline composition
scope; ready as the Stage 2 dependency**. Independent closure review and the
orchestrator's verification passed. P5 closure and D9 activation remain
unauthorized. D1-D8 are accepted. Every production migration guard remains in
place, and Stage 1 does not provide the Stage 2 durable session resolver.

`RuntimeComposition.define()` is synchronous and read-only. It retains typed
configuration, preflight facts, factories, and channel options only. Explicit
`await composition.acquire()` is the sole construction boundary and publishes
one frozen concrete `RuntimeHandle` after the caller's event loop has acquired
all requested components. The state machine is terminal:
`NEW → BUILDING → READY → CLOSING → CLOSED`, or `BUILDING → FAILED`.
Concurrent same-loop callers share one acquisition; foreign-loop access fails;
rollback and close are reverse, same-loop, and exactly once. There are no
transparent materializing proxies, helper threads, private event loops, or
synchronous async-cleanup shims.

Acquisition failure is terminal and stable: every later same-loop `acquire()`
re-raises the identical recorded exception object without retrying any factory.
FAILED and CLOSED compositions retain their owner-loop fence, including
`aclose()` from FAILED. Cleanup attempts every acquired component in exact
reverse order and propagates cleanup failures in ordered `ExceptionGroup`
evidence rather than swallowing them.

The default `Stage1UnavailableSessionAuthority` fails before bus, provider,
tools, MCP, cron, agent, plugin, channel, task, listener, or turn construction.
It never constructs `SessionManager` or `JsonlSessionStore` and does not select,
repair, migrate, compare, or convert session evidence. Offline positive tests
inject an explicitly borrowed, caller-owned manager. Direct `SessionManager`
construction remains eager for compatibility; Stage 2 must install the durable
resolver before guard removal.

Read-only preflight and `SourceSessionStore` share the public canonical-v1 file
validator. Definition and default failed acquisition preserve filesystem bytes,
inodes, and mtimes for absent, legacy, canonical, and mixed fixtures. Incomplete
canonical v1 is rejected. Classification is diagnostic only and never selects a
runtime authority.

`ChannelManager.__init__` is passive. Async acquisition reserves packaged names
before manifest success, enumerates legacy entry-point metadata only for unknown
configured and enabled names, imports targets after metadata/config/dependency
validation, constructs into an unpublished pending list, and rolls back on the
owning loop in reverse order. Cancellation or failure restores channels, owners,
runtime specs, and channel errors, clears the acquisition marker, rejects a
concurrent acquisition, and permits a clean explicit retry. Channel start remains
a separate transactional boundary. Startup owns its restart-notice task and
cancels/joins it before rollback. Every attempted concrete channel is stopped in
reverse exactly once; later composition shutdown stops only remaining owned
channels. Channel stop failures propagate after all reverse cleanup has been
attempted. Disabled or unconfigured legacy targets are neither enumerated nor
imported.

The closure package replaces raw object-ID tombstones with a single owned
shutdown task bound to the concrete channel in each runtime slot. Concurrent
shutdown callers join that task rather than invoking `stop()` twice. Owner
cancellation (including repeated cancellation) is reported only after channel
stop, outbound cleanup, and the cancelled channel-start task finish on the same
loop. Cancellation plus a cleanup failure retains both errors. A replacement at
the same runtime name receives independent ownership; removed/replaced slot
records are discarded. A channel awaiting an already-cancelled server during
stop retains its existing benign-cancellation behavior, while its start task is
still joined. No timer, listener, store, or authority is activated by this fix.

Acquisition and startup rollback also run in owned shielded tasks. Startup
supervision does not forward owner cancellation into channel starts: reverse
rollback invokes stop before joining start tasks, so stop can release a resistant
start finalizer. Repeated cancellation during either stop or start finalization
cannot detach rollback. A manager-initiated shutdown remains graceful only when
the startup owner has no cancellation request; genuine owner cancellation,
including a race with shutdown, remains observable. Acquisition rollback retains
the primary failure, reverse cleanup failures, and concurrent cancellation before
restoring all provisional fields.

Classic CLI and gateway define before entering their async owner loop and acquire
inside it. The API uses an aiohttp cleanup context, so later startup failure
closes the composition before listener ownership. `Nanobot.from_config()` is
definition-only; async entry/run/stream paths acquire before work or task spawn,
and close fences later use. The gateway contains no Dream, Heartbeat target,
evaluator, system-job, or cron callback execution path. Dream, Heartbeat,
callbacks, cron execution/status/start, legacy AutoCompact, new idle work,
timers, and rematerialization remain inactive.

After the composition is ready, the gateway restores only foreground-safe
coordination against the published identities: recovery admission and its
pre-listener scan, WebUI turn-event projection, provider observation/reload,
runtime/MCP/skill/recovery WebUI callbacks, the file-edit activity hook, the
config watcher, and the local-client lifetime monitor. These are acquired or subscribed before agent/channel
tasks and listener ownership, and are disconnected or closed on the same
owner loop. The top-level local-trigger enqueue command remains present and can
write its existing durable queue record; only the gateway queue runner is not
started in Stage 1, so enqueueing cannot become an alternate execution authority.

Verified offline evidence includes:

- 100 concurrent same-loop acquisitions return one identical handle and invoke
  each factory once;
- default acquisition cutoff and filesystem immutability, with explicit
  `SessionManager.__init__`, `JsonlSessionStore.__init__`, and
  `JsonlSessionStore._migrate_from_workspace` sentinels;
- identical terminal failure replay without retry, plus foreign-loop
  acquire/close rejection and post-close/post-failure fencing;
- cancellation and construction failure rollback on the owner loop, including
  acquisition-first cleanup `ExceptionGroup` ordering;
- shared concrete bus/session/tools/MCP/cron/delivery/agent identities;
- classic CLI and API mutation only after successful acquisition, with default
  cutoff tests proving no provider, plugin, listener, task, session store, or
  template mutation is constructed first;
- aiohttp later-startup unwind and gateway post-acquire/pre-task failure cleanup;
- SDK definition-only failure and injected positive execution;
- passive channel/plugin acquisition, complete provisional-field rollback,
  concurrent-acquisition rejection/retry, owned restart-notice cancellation, and
  reverse exactly-once start/shutdown cleanup with propagated failures;
- direct exact-one adapter evidence for SDK run, SDK stream, bus dispatch, and
  the real API handler; separate actual classic-CLI orchestration evidence using
  its real `AgentLoop`; and representative Telegram evidence through the real
  headless `_on_message`/manager/bus path. Gateway has no distinct turn adapter
  beyond its shared channel/bus ingress, so it is not mislabeled as a separate
  direct adapter case.

The complete channel suite currently reports 443 passed and 12 failed. Every
remaining failure is a CLI callback test blocked by the locally absent declared
`setproctitle` dependency; no remaining channel lifecycle or authorization
failure was observed. No dependency was installed for this checkpoint.

The pre-closure expanded historical entrypoint/channel diagnostic reported 636 passed,
93 failed, and no collection errors. On the same command this improved from
609 passed / 120 failed after constructor-era channel tests were moved to
explicit acquisition. The remaining failures are either locally absent
declared dependencies (`click`, `dulwich`, or `setproctitle`) or historical CLI
tests that patch removed eager-construction/module-global seams, default SDK or
gateway operability without a session authority, or the intentionally absent
Dream/Heartbeat/legacy cron/local-trigger runtime. They are not treated as
green evidence. The pre-closure requested Stage 1 selection reported 115 passed and two
load-sensitive API successor timeouts; those exact two cases passed 2/2 when
rerun alone. The focused lifecycle, adapter, default-cutoff, and corrected channel
selection passed 46/46.

The three API cancellation-successor tests which exceeded their 0.2-second outer
timeout in one combined run passed 3/3 when rerun alone; no lifecycle code lies
on their request execution path, so they are recorded as load-sensitive local
timing rather than weakened.

This checkpoint does not implement session mode records, conversion/export,
artifact/rematerialization activation, cron occurrence/scheduler/lease/cursor,
timers, live provider/channel/network/listener testing, P6/P7/H1, guard removal,
or deployment. Those remain separate accepted gates.

## Stage 1 closure validation (2026-09-30)

The shutdown regressions first failed on the original implementation: owner
cancellation detached cleanup, and 20 concurrent callers invoked `stop()` 20
times. Independent review then found two remaining rollback defects: startup
joined resistant starts before calling stop, and acquisition rollback could be
interrupted by repeated cancellation. Both were reproduced with deterministic
event-controlled tests before correction. The final 14 shutdown/rollback cases
cover manager/per-channel shutdown, concurrent callers, instance replacement,
benign already-cancelled server handling, single/repeated startup cancellation
during stop and start finalization, ordered acquisition failure evidence,
graceful-shutdown races, and gateway cancel/join/close ordering. The retained API
startup-admission test now injects a typed authority instead of requiring the
rejected `defer_storage` constructor.

The same independent reviewer narrowly rechecked both corrections and affected
compatibility: 17 focused lifecycle cases and 32 shutdown/hot-reload/reasoning
cases passed, with no material findings. The orchestrator issued **PASS** for
Stage 1 only after inspecting the actual changes and rerunning the checks below.

Independently executed offline commands use
`NANOBOT_DISABLE_NETWORK=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m pytest -q`:

| Selection (repository-relative arguments) | Result |
| --- | --- |
| `tests/migration/test_p5_stage1_shutdown.py tests/migration/test_p5_stage1_composition.py tests/migration/test_p5_stage1_adapter_matrix.py tests/channels/test_legacy_entrypoint_stage1.py tests/agent/test_migration_sdk.py tests/agent/test_migration_api_engine.py tests/cli/test_agent_interactive.py tests/channels/test_telegram_headless_composition.py` | 89 passed |
| `tests/cli/test_startup_admission_agent_gateway.py tests/cli/test_startup_admission_onboard_webui_api.py` | 29 passed |
| `tests/cli/test_startup_admission_agent_gateway.py tests/cli/test_startup_admission_onboard_webui_api.py tests/test_sdk_streaming.py tests/test_openai_api.py tests/test_openai_api_fork.py` | 50 passed |
| `tests/channels --tb=short` | 443 passed, 12 failed; all 12 fail on missing `setproctitle` |
| `tests/migration/test_p1_foundation.py tests/migration/test_headless_startup_compatibility.py tests/test_sdk_streaming.py tests/test_openai_api.py tests/test_openai_api_fork.py` | 47 passed, 8 failed; all 8 CLI subprocess cases fail on missing `click` |

Selections overlap and must not be summed. Scoped Ruff, Python 3.13 compileall,
and `git diff --check` passed. BasedPyright and native Windows remain unqualified.
Dependencies were not installed. Historical CLI reconciliation, installed-package
qualification and the full P5 startup matrix remain later integration/release
gates; dependency-blocked checks are not counted as passed and the broad suite is
not claimed green. Stage 2 may now build against this accepted guarded composition
seam. At this Stage 1 closure point, Stage 2 implementation had not begun; no persistent authority, automation,
guard removal, live-state change, or deployment is authorized by this checkpoint.

The subsequent [Stage 2 attempt](p5-stage2-checkpoint.md) is incomplete and rejected.
Its final safety correction restored the ordinary default cutoff, with 78
composition/startup cases independently passing. A fresh review verified that cutoff
but found a falsey injected-authority selection defect; this remains open and is not
Stage 2 acceptance. The original Stage 1 closure evidence above is retained as history.
