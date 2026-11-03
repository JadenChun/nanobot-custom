# Requirements-only input: H1 and concurrency retirement

Use this document's content as the authoring input, not the implementation plan.
Mode: **REQUIREMENTS_ONLY_TEST_AUTHOR**. Produce the smallest useful set of case
specifications with requirement ID, setup/input, public action, and exact
observable expected outcome. Executable cases are optional where the public
contract is sufficient. State any missing public contract rather than inventing
behaviour. Do not inspect a repository, source, diff, logs, existing tests,
running product, or parent history. Do not spawn agents. An orchestrator supplies
only an output directory outside the repository and the artifact format.

## Public product contract

- Configuration is JSON. `nanobot.config.schema.Config.model_validate(data)`
  validates a config object; `nanobot.config.loader.load_config(path)` loads a
  file and reports a typed configuration-load failure for invalid input. Public
  JSON output uses `Config.model_dump(by_alias=True)`.
- User turns can be submitted asynchronously with
  `AgentLoop.process_direct(content, session_key=..., channel=..., chat_id=...)`.
  The public result is an outbound response or an error. An offline controllable
  provider may hold one request until released and record request/session inputs.
  A test adapter may submit equivalent bus/channel turns. These are real product
  entry points; tests must not inspect or change private runtime fields.
- Session lifecycle actions are public reset/replacement/deletion operations
  on a stored session. `SessionManager.reset_session(key,
  replacement_events=...)` and `SessionManager.delete_session(key)` are public
  SDK operations. Seed representative legacy state through a public persistence
  fixture/import adapter. Request missing fixture-format details if executable
  tests need them; scenario specifications may state the meaningful legacy state.
- Ordinary Telegram is a message transport to the same canonical agent. Offline
  transport adapters supply authenticated input and observe routed responses;
  they must not call the real Telegram network or a real model.

## Accepted outcomes

**R1: no global admission cap.** The retired environment setting
`NANOBOT_MAX_CONCURRENT_REQUESTS`, including a positive value and a non-numeric
value, cannot restrict unrelated sessions or prevent loop construction. A second
session can reach its own provider while the first session's provider is held.
Two turns for the same session remain ordered; the second cannot execute a
model/tool effect before the first relinquishes ownership. Cancellation of a
waiting turn cannot execute that cancelled request later or block future work.

**R2: ordinary Telegram remains usable.** Normal authenticated messages and
responses keep their original session and reply destination. Ordinary stop and
scheduled execution retain their existing user outcomes. No H1 routing/control
model is called, and no H1 durable work item is newly created. This retirement
does not introduce automatic task-session routing or promise busy-input chat.

**R3: configuration retirement is explicit.** An absent `gateway.telegramBusy`
block or a valid disabled block loads. Missing `enabled` means disabled. The
supported snake-case form has the same behaviour. An enabled or malformed block
fails clearly as retired H1 before gateway work begins. Normal schema output
does not advertise or recreate the feature. Error text identifies retired H1
and does not echo secrets. Other settings retain their values.

**R4: accepted legacy work is preserved.** For a session with pending, accepted,
running, uncertain-effect, pending-delivery/receipt, unfinished temporary-lifetime,
or malformed/unrecognisable H1 state, a new ordinary turn is refused before any
model or tool effect. Reset, replacement, and deletion are refused without
discarding that state. The error identifies legacy H1 and the need for recovery.
Read-only inspection and unrelated sessions remain available. Fully settled
legacy state may remain inert. Private H1 recovery records do not appear as
ordinary user conversations. No automatic replay or effect retry is permitted.

**R5: process continuation retains its owner.** A completed external process
continues only its original session/project/reply route, waits behind that
session's foreground turn when necessary, and does not create an extra model
turn for already inline-consumed or cancelled work. This requirement preserves
existing behaviour; add cases only for a stated coverage gap.

## Public validation-tool contract for intentional source deletion

The offline type-report comparison CLI takes frozen before/after checker JSON,
absolute source roots, and optionally complete before/after source inventories
plus an explicit list of intentionally deleted in-scope paths. A report includes
checker version, `generalDiagnostics`, and summary counts including
`filesAnalyzed`. Each diagnostic has file, severity, rule, and message; repeated
diagnostics count separately. Source inventories map normalized relative paths
to SHA-256 hashes. The current source inventory must match the remaining
in-scope files. The strict checker policy stays identical.

**V1:** Without additional inventory/deletion evidence, a lower analyzed-file
count is invalid evidence. With complete consistent evidence exactly accounting
for the deleted files, it may be compared. Undeclared disappearance, inconsistent
counts/inventories, changed checker/policy, or invalid paths remain invalid.
New diagnostics in any retained/new file, including increased multiplicity,
fail even if total diagnostics fall. Moving line numbers alone is not new debt.
Exit statuses are 0 for no new signatures, 1 for new signatures, and 2 for invalid
evidence. A comparison pass does not assert a clean strict check when errors
remain. Ask for the chosen optional CLI argument names before executable CLI
cases; the stated observable outcomes are sufficient for case specifications.
