# P5 stopped-writer rollback and export runbook

2026-10-02: **ROOT TECHNICAL PASS for bounded rollback/export handoff;
no live operation authorized**.
Consumes accepted D1/D2/P4 and [Stage2](p5-stage2-acceptance-checkpoint.md).
The [closure checkpoint](p5-closure-checkpoint.md) records guarded-offline P5 DONE /
full Stage5 closure ACCEPTED / READY_TO_BEGIN_P6 on root technical PASS and fresh
integrated PASS and final root publication PASS accepting AC1–AC5/all nine statements.
[Permanent evidence](p5-compatibility-closure.md) retains exact coverage/history.
P6 NOT_STARTED; activation NOT_READY/NOT_AUTHORIZED and D9 NOT_AUTHORIZED remain. The [preflight contract](p5-stage5-implementation-contract.md) is an inventory,
not mode authority or a rollback executor.

## Stop, preserve, establish one authority

1. Operator identifies the exact config instance, workspace, sessions/runtime
   roots, current/rollback artifacts and owners in a private inventory. Stop old
   and new writers, gateway consumers, ingress, timer claims and background work;
   drain/join or fence resistant admitted effects. A lock or lease alone does not
   establish an old binary stopped. Unknown shutdown/effect state is a stop gate.
2. Preserve a readable stopped-state backup with mode records, witness/binding,
   source/checkpoints, transition journals, legacy projection, private artifact
   blobs, cron definitions, cursor/lease/occurrence/completion/outbox/callback
   evidence. Do not delete uncertain records or change their identity. Verify
   backup readability independently; make a distinct offline rehearsal copy.
3. On the copy only, use owning Stage2 session authority to validate exact
   incarnation/generation, mode record and journal; fail closed on conflict,
   unsupported schema, missing proof, quarantine or recovery residue. Never infer
   authority from freshness, filenames, a readable JSONL export or Stage5 counts.
   Mode resolution may lock/mutate bookkeeping: it is an explicit operator seam,
   never part of read-only Stage5 inventory.

| Exact durable point | Authority | Permitted recovery / rollback |
| --- | --- | --- |
| Valid legacy, no committed conversion | Legacy JSONL | Preserve snapshot; old compatible artifact only after readable backup/rehearsal and stopped-writer authorization. |
| Prepared/source installed but canonical mode not committed | Legacy predecessor | Resume/abort only the exact journal offline; staged target is ignored/quarantined, not selected by presence. |
| Canonical mode committed | Canonical source | Forward recovery and verified lossless/compatibility export; never rename JSONL into authority or restore old mode record. |
| Export prepared/interrupted | Existing committed mode remains owner | Finish/recover exact export journal; export publication cannot choose mode. |
| Conflict/partial/quarantined/unsupported | No runtime admission | Preserve evidence, stop; owning recovery/export contract/operator decides next exact action. |

## Explicit copied-state tooling

Existing `sessions bootstrap/convert/recover/export` in `nanobot/cli/commands.py`
and SessionManager's owning methods are the seams. Exact arguments are
`--copied-workspace`, `--copied-sessions-root`, `--offline-copy`, and `--config`;
convert/recover/export additionally take one exact session key. These commands
are not Stage5 report switches, do not prove old-writer exclusion and may require
P6's declared CLI dependencies. Do not invoke normal runtime or bypass the guard.
Bootstrap adopts valid copied namespaces; convert operates on one exact admitted
legacy key; recover resumes that exact journal; export calls
`export_canonical_session` to a new absolute destination. No bulk/live conversion,
automatic recovery or new downgrade writer is delivered here.

The operator `--config` is a **reference-instance config**, not the normalized
copied config used by the read-only inspector. `_copied_state_config` derives
reference/live workspace from its configured workspace and reference/live sessions
from its bound config parent plus `sessions`. `OfflineSessionOperator` rejects
copied workspace or sessions equal to, or samefile as, those respective reference
roots. Reusing the inspector's copied config unmodified therefore rejects: its
workspace and config-parent sessions identify the very copy being operated on.
Do not remove that exclusion, falsify reference roots, move the config merely to
evade the fence, or add a new executor/bypass.

Before any operator rehearsal P6 must qualify the existing loader in an approved
isolated environment: explicitly authorize the reference config and exact source/
copy path provenance, preserve truthful source-reference identities, deny unwanted
environment substitutions/NANOBOT_* overrides and default/fallback config, verify
binding to the explicit config instance, and prove no unauthorized state/effect
access. This is loader/path/environment qualification, not permission to access
live state during Stage5. If those facts cannot be proved through the existing
operator seam, stop and return to the owning P6/operator gate; do not use a copied
config trick or weaken the exclusion. The copy remains the only mutation target.

Illustrative existing-CLI argument templates below are **not executed or authorized
for live use**. All placeholders must become approved absolute paths in that P6
rehearsal; SESSION_KEY identifies one exact session and export destination is new:

```text
nanobot sessions bootstrap --copied-workspace /absolute/stopped-copy/workspace --copied-sessions-root /absolute/stopped-copy/instance/sessions --offline-copy --config /absolute/approved-reference-instance/config.json
nanobot sessions convert SESSION_KEY --copied-workspace /absolute/stopped-copy/workspace --copied-sessions-root /absolute/stopped-copy/instance/sessions --offline-copy --config /absolute/approved-reference-instance/config.json
nanobot sessions recover SESSION_KEY --copied-workspace /absolute/stopped-copy/workspace --copied-sessions-root /absolute/stopped-copy/instance/sessions --offline-copy --config /absolute/approved-reference-instance/config.json
nanobot sessions export SESSION_KEY --copied-workspace /absolute/stopped-copy/workspace --copied-sessions-root /absolute/stopped-copy/instance/sessions --offline-copy --config /absolute/approved-reference-instance/config.json --destination /absolute/new-export --include-compatibility
```

For committed canonical state request lossless canonical export plus explicit
compatibility JSONL export (`--include-compatibility`) into a new destination.
Verify the owning export proof and readable records with the intended rollback
reader on another stopped copy. Compatibility export can fail when canonical-only
state cannot be represented; that is a hard no-loss stop gate, not permission to
drop fields or revert mode. No release can be called rollback-capable until P6
proves that exact old artifact reads the verified target format. Keep canonical
inline/source evidence and blobs throughout mixed mode and at least one accepted
rollback-capable release; retention/removal needs separate post-P7 approval.

Only after copy rehearsal, exact backup/export verification and explicit later
operator authorization may a separately designed cutover restore the accepted
rollback reader/namespace. No live downgrade/import mechanism or deployment
command is authorized by this document. Never let old/new binaries write peers.

## External-effect and operational failure fences

Startup failure closes partial resources before provider/tool/send/listener.
Lease loss fences new claims; elapsed unclaimed fires become missed, not catch-up.
Started/uncertain computation never recomputes; sending/uncertain delivery never
resends automatically. Callback invoking/failed never reinvokes, independently
of settled delivery. Use accepted occurrence/outbox/callback reconciliation with
exact durable predecessor, offline/operator-only; preserve uncertainty until
resolved. Removing a ledger, resetting a lease, or retrying a request is not
rollback. Shutdown cancellation/timeouts remain visible and prevent promotion.

Artifact install failure publishes no dangling reference; retain inline evidence
or fail closed. Inspect/rematerialize one exact reference on copied state only via
accepted operator interfaces, never during reads/startup, and never delete inline
rollback evidence. Memory pending/quarantine is handled through owning P4 recovery
on copies after private path provenance review; Stage5 deliberately does not open
embedded persisted paths. Contents/IDs/paths stay out of routine diagnostics.

Operator records a private audit of exact artifact/root/authority, export proof,
effect uncertainty, backup readability, copy rehearsal and approval. P7 accepts
stop/rollback thresholds before a single-consumer canary; no automatic uncertain
retry, retention deletion, timer, Dream/Heartbeat/callback/idle enablement follows.
