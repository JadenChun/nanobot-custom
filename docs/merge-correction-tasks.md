# Merge corrections: mobile request authority and recovery

Review date: 2026-11-04, Asia/Kuala_Lumpur. Verdict: **CORRECT; hold staging merge**.
Reviewed implementation: `1689a4a06020c778a6e969b12bb19b27a1ac90f1` on
`feat/mobile-remote-context-lifecycle`. Fetched staging:
`aacca07cfadaf51fa1d83913f93ba2d797c0aa58`. Their merge tree was conflict-free;
combined-source acceptance and release qualification were not established.

These tasks close reachable gaps in [the mobile plan](mobile-remote-request-plan.md);
the [session context plan](session-recovery-architecture.md) remains controlling for its scope.
Use [the correction prompt](merge-correction-agent-prompt.md). Do not restart completed packages
or replace the accepted design. Source pointers below refer to the reviewed implementation.

## Sequence, ownership and evidence

Run MC01 through MC08 in order, with one writer for overlapping files. Work on isolated source
if the feature worktree is still used by the installed gateway. Preserve other agents' changes.
Keep corrections as small reviewable commits; no new permission store, routing framework,
dependencies, broad refactor, binary-retention migration or composer redesign is required.

The recorded full Python result (17,556 passed) and strict result (0 errors / 528 files) were
inspected during review. Nine existing authority checks also passed. They establish useful
primitive behavior, not the missing complete journeys below. Reuse valid unchanged evidence.

Before each behavior correction, identify existing coverage and obtain only necessary new or
changed cases from a fresh requirements-only author. Give that session accepted user outcomes
and the explicit public protocol/UI/process contract, never this document, source, repository
access, implementation plans, existing tests, diffs, logs or review analysis. Retain author
session/artifact IDs. Integrate cases mechanically; return semantic corrections to the author.
Each task is complete only when its observable outcomes and affected mandatory checks pass.

## MC01 — Enforce the negotiated request contract through admission

**Evidence / owner:** `nanobot/webui/inbound_commands.py:81` accepts absent version/source and
returns early for steering. Version/source/intent identity is lost before actual execution;
`nanobot/webui/workspaces.py:1022` rejects the documented v1 control payload fields.
Own those admission/parser seams and necessary existing request-context/loop wiring only.

**Required outcomes:**
- [ ] Negotiated v1 messages and permission controls enforce the documented version, source,
  intent/target and confirmation-correlation contract. Missing/unknown/stale input has the
  documented rejection/update-required outcome before media, transcript or defaults effects.
- [ ] Freeze submitted identity/settings and carry them to actual admission. Revalidate source,
  device consent and proof after waits; never mint permission against a replacement source
  for an older submission. Steering does not skip source validation.
- [ ] Keep only explicitly supported legacy-owner compatibility paths; they cannot bypass v1
  fences for negotiated clients. Authentication rotation alone does not mutate a queued snapshot.

**Validation:** public message/control requests with valid and stale/missing identity, including
a source change while waiting. Reuse affected cases in `tests/webui/test_permission_v2_api.py`
and `tests/webui/test_webui_mutation_replay_auth.py`. Exercise real admission, not only a parser.

## MC02 — Honor explicit Temporary Chat request settings

**Evidence / owner:** `nanobot/webui/inbound_commands.py:747` takes the live temporary policy's
scope instead of the submitted scope; `nanobot/webui/temporary_chats.py:132` derives current mode.
Own temporary message admission, its existing scope resolver and temporary lifecycle seam.

**Required outcomes:**
- [ ] An explicit Sandbox submission stays Sandbox even when temporary Full is enabled later
  or remains the chat default. Submitted project/mode wins over subsequent picker changes.
- [ ] Temporary Full still requires current device/source/request authority; both chat types
  use the same permission owner. No fake persistent session, source file or durable grant.
- [ ] Discard/disconnect retires temporary authority and prevents late reuse or replay.

**Validation:** send real temporary message envelopes with captured settings, execute through
the ordinary turn boundary and observe effective access. Reuse `tests/webui/test_temporary_full.py`
and affected temporary-session/permission cases. A controller reference lookup alone is insufficient.

## MC03 — Separate busy new submissions from exact-turn steering

**Evidence / owner:** `nanobot/webui/workspaces.py:1458` admits differing busy scopes;
`nanobot/agent/loop.py:2829` still injects ordinary busy messages into the current turn.
The steering comparison at `nanobot/webui/inbound_commands.py:763` uses chat preferences/key
rather than the exact active captured scope. Own ingress and the minimal existing admission seam;
preserve unrelated channel queues and do not activate a new busy-inbox subsystem.

**Required outcomes:**
- [ ] A busy `request_intent: new` returns `not_admitted` without media/transcript/defaults
  effects; it never becomes guidance executed under the active request's authority.
- [ ] The UI retains the original queued snapshot and stable submission identity for later
  admission. A never-admitted busy response does not poison its later valid retry. Uncertain
  already-sent outcomes require canonical reconciliation, not automatic new submission.
- [ ] Steering requires the exact active turn, current source and identical captured context;
  changed project/mode or target is explicitly not admitted. Check again at actual admission
  so an idle-to-busy race cannot create competing owners or inject the wrong context.

**Validation:** while Full work runs in project A, submit new Sandbox work in project B;
verify no effect until its own admission, then observe B/Sandbox execution. Also verify valid
same-context steering and rejected changed-context steering. Reuse relevant cases in
`tests/agent/test_workspace_scope.py`, `tests/webui/test_permission_v2_api.py` and
`nanobot/channels/websocket/tests/test_websocket_channel.py` through real ingress and execution.

## MC04 — Complete authenticated owner Full admission and peer consent

**Evidence / owner:** `nanobot/webui/workspaces.py:627` rejects Full using a localhost-only
availability check from `nanobot/webui/ws_http.py:736` before installation-bound proof lookup.
The active-Full early return at `workspaces.py:1094` skips another device's consent recording.
Own these existing owner/permission admission and consent paths, sequentially after MC01–MC03.

**Required outcomes:**
- [ ] A confirmed non-local owner phone can actually submit and execute Full through the
  authenticated mobile WS entry. Reaching an enabled menu or clicking confirmation is not enough.
- [ ] Keep local-only protections for untrusted/legacy connections. Use verified owner identity
  and device/source consent; do not spoof localhost or trust a client-provided owner flag.
- [ ] Device B receives its own consent after explicitly confirming, even if A already enabled
  Full. Before B confirms it remains denied; stale/revoked sources invalidate both appropriately.
- [ ] Preserve the running turn's captured scope, explicit Sandbox and temporary isolation.

**Validation:** the full owner message/control journey for two authenticated installations, with
an unconfirmed/legacy negative outcome. Reuse `tests/webui/test_permission_v2_api.py`,
`tests/remote/test_owner_permissions.py` and actual WS-entry coverage.

## MC05 — Retire native handles belonging to derived request proofs

**Evidence / owner:** `nanobot/security/desktop_permissions.py:559` derives distinct references;
`nanobot/agent/tools/native_process.py:307` registers native stdin against them, while
`desktop_permissions.py:743` retires the base reference. Own existing reference/retirement
bookkeeping and immediate native registration consumers only.

**Required outcomes:**
- [ ] Revocation/downgrade retires the source/epoch's native stdin and accounting across its
  derived request proofs, including multiple owned processes; unrelated owners stay unaffected.
- [ ] Keep request-specific authorization and source/epoch fences. Do not fix aggregation by
  making different request proofs interchangeable or removing the request-binding checks.
- [ ] Preserve retiring/active/unknown process semantics and the distinction between Stop and
  revocation; later writes remain fenced and closure/count failures remain visible.

**Validation:** launch an owned native command through the existing execution boundary, revoke
its authority and observe stdin closure/retirement, not merely reference invalidity. Reuse
`tests/security/test_native_admission.py` and `tests/security/test_permission_v2_authority.py`.
Retain applicable platform qualification/skips without claiming a skipped native case executed.

## MC06 — Use independent owner control from the actual phone client

**Evidence / owner:** `NativeNetworkPlugin.kt:1065` still refreshes via gateway-dependent GET
`/mobile/v1/refresh`. Broker allowlisting of POST `/remote/v1/refresh` alone does not wire a client.
Own `mobile/android/app/src/main/java/ai/nanobot/mobile/network/NativeNetworkPlugin.kt`, existing
custody/HTTP adapters, bootstrap/status presentation and exact broker operation contracts as needed.

**Required outcomes:**
- [ ] The paired phone uses the independent owner's status/refresh/list/revoke operations with
  correct methods/payloads while the gateway is stopped. Remote Start remains unsupported.
- [ ] Refresh stays single-flight and preserves custody/rotation/revoke fences; an absent gateway
  is not interpreted as revoked pairing. Preserve explicit compatibility negotiation.
- [ ] Existing owner scope checks and exact allowlists remain intact; no arbitrary URL/path proxy.

**Validation:** actual native-to-owner requests against isolated owner/gateway fixtures with
the gateway absent; assert routes and end-user outcomes. Reuse existing `tests/remote` owner
checks, broker tests and `NativeNetworkRefreshInstrumentedTest` as applicable. Merely testing
that Go accepts a path does not verify the phone uses it. No live service shutdown is authorized.

## MC07 — Recover in-session after gateway restart/token expiry

**Evidence / owner:** `docs/implementation-progress.md` records a real `1008 access expired`
closure after gateway restart, with the composer stuck at Sending until cold restart/Retry.
`webui/src/lib/mobile-runtime.ts:263` skips retired sockets. Own the existing shared
runtime/client/bootstrap recovery path and closest UI status integration; build on MC06.

**Required outcomes:**
- [ ] With the owner available, gateway restart and access expiry recover the existing persistent
  chat in-session or show an actionable retry state. Sending/Connecting cannot remain indefinitely.
- [ ] Preserve draft, attachments, screen/scroll and pairing; reconcile canonical turn outcomes.
  Single-flight recovery has one current socket and no duplicate callbacks or message effects.
- [ ] Same-gateway rotation, owner unreachability and confirmed revoke have distinct outcomes.
  Do not erase pairing on generic 401, disguise authentication failure as Tailscale sign-in,
  auto-resubmit an unknown mutation or resurrect a disconnected temporary chat.

**Validation:** realistic close/expiry/refresh/reopen journeys through the installed runtime's
public transport contract, then applicable disposable native/phone qualification. Reuse
`webui/src/tests/mobile-runtime.test.ts`, `webui/src/tests/nanobot-client.test.ts` and
`webui/src/tests/bootstrap.test.ts`. A healthy-Ping-only test does not cover this failure.

## MC08 — Validate the corrected candidate with current staging

- [ ] Fetch staging again, record exact parents and integrate into the isolated correction branch
  without rewriting shared history. Preserve staging notifications, scroll/gallery improvements
  and settings changes; keep unrelated dirty original-checkout files out of the candidate.
- [ ] Preserve frozen before/after source/environment/checker evidence and run the mandatory
  full-source no-new-signature comparison from `docs/migration/p6-error-attribution.md`.
  Dependency changes need a comparable environment; never absorb errors into a refreshed baseline.
- [ ] Run focused checks for each correction first. Run remaining mandatory integrated gates
  once on the final relevant source/assets; build matching native/shared assets where changed.
  Record outcomes, exact candidate hashes and unresolved phone/signing/radio/release qualification.
- [ ] Update progress with MC01–MC07 acceptance evidence and requirements-only author provenance.
  Revise superseded status summaries without erasing historical results. Obtain a bounded review
  of the findings and correction delta; retain required M38 review at the actual release gate.
- [ ] Produce a merge-readiness report tied to exact corrected and staging SHAs. A clean merge
  tree or old green totals alone is not completion. No push, staging merge or installation occurs
  under this correction handoff; prepare a clean local candidate for the next merge decision.

Stop expanding each task once its outcomes and required checks pass. Optional hardening,
unrelated defects and unverified broader architecture questions are separate follow-ups.
