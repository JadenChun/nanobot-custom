# Consolidated synthesis — active, pending all modules

The audit is **ACTIVE after user-requested resumption**. The whole-project architecture/
code-quality judgement and simplification discussion remain **pending**. [S01](reports/S01.md)
and [S02](reports/S02.md) retain coordinator **PASS** for their 21 assigned primary bodies.
[H01](reports/H01.md) now also has coordinator **PASS** for its source-review publication:
all 29 primary bodies remain credited to the supplied ledger, H01-F1–F7 retain their stated
conditional/structural classifications, H01-F8 remains an open contract, and six questions
remain deferred. Fresh continuation checks initially found `staging` unchanged at `9562e57...`. During R01-A and rechecked at the R01-B boundary, `staging` is `29612c3...`; only the K01 skill-display source/test changed, so R01 remains byte-current while K01 current-origin refresh is deferred. The
H01 acceptance publication is checkpointed at `f6a41b840c7e50c8f757e8889a880bc3ce77d47d`.
Integrated review remains pending and no repair/design is accepted. [C00](reports/C00.md) is
now coordinator-checked from its supplied 70/70 primary-body handoff with seven qualified P2
findings and seven retained D1–D7 questions; its acceptance publication is checkpointed at
`85c59330c91e7982574d79f92602cc7e605acb4c`. See
[H02](reports/H02.md) is coordinator-checked at final F74/P0/R0 with six promoted findings,
one unpromoted title-edge observation, and seven retained questions; its acceptance publication
is checkpointed at `08ba3c3d48c9c96c1d04c25f5815a5efd17e20f4`. [H03](reports/H03.md) is now
coordinator-checked at final F45/P0/R0 with seven promoted findings and seven questions. [U01](reports/U01.md) is now coordinator-checked at final F203/P0/R0 with six qualified P2 findings and six retained questions; its acceptance publication is checkpointed at `b53644a2e20086a5d6a48e237e571d1720151270`. [D01](reports/D01.md) is now coordinator-checked at final F47/P0/R0 with two qualified P2 findings and five retained questions; its acceptance publication is checkpointed at `9db80a30be08be3428ca33fda1532031dabd6a7f`.
[status.json](status.json) for the eighteen streams and [scope.json](scope.json)
for the 919 primary assignments. The earlier [33-row comparison](../architecture-review-2026-11-06/README.md)
retains its targeted historical boundary; it is not the new deep-review synthesis.

The coordinator accepted the publication foundation and committed/pushed checkpoint
`5f0d14479b6ecd7a334ae72476a7c0ad5065be66`. [C00/H02/H03/U01/D01/T01/K01/E02 handoffs](handoffs/index.json)
remain preserved as provenance; C00, H02, H03, U01, D01, T01 and K01 are now published/checked. [A01](reports/A01.md) is now coordinator-checked at final F47/P0/R0 with nine promoted findings and seven retained questions; F06/F08/F09 remain explicitly conditional. K01 is coordinator-checked PASS at F56/P0/R0 after K01-A/B/C/D/E with thirteen promoted findings and eight final questions. [E02](reports/E02.md) is coordinator-checked PASS at F68/P0/R0 with twenty-three promoted findings and nine final questions. E01 is now closed as an explicit **limited-depth/non-priority exception** by the 2026-11-07 user scope decision. The prior E01-A..F deep work remains preserved at F40/P0/R55 with twenty-six findings across the shared framework, Telegram, DingTalk, Discord, Email and Feishu control/setup/instance/UI packages. The remaining 55 E01 primaries received a quick high-signal sanity screen only; no new finding was promoted from that shallow pass, and Weixin pathless config loading only corroborated E01-F02. E01 contributes no coordinator-checked deep body credit and channel functionality remains intentionally untouched. R01-A/B/C/D are now active at F30/P0/R40 with eight provisional findings (one P1, seven P2) and zero publication credit; four later primary streams remain queued. Published coordinator-checked deep evidence
still covers 704 primary bodies. T01 is closed at F44/P0/R0 with twelve promoted findings and eight retained questions. Across the twelve coordinator-checked reports there are 99 promoted finding entries and 76 retained/final questions; these remain module-level evidence rather than becoming
a whole-project conclusion; H01-F8 is explicitly open.
S01-F03 is cross-referenced by S02 and counted once. H02 supplements5A/5B supply current F74/P0/R0, superseding initial F33/P4/R37; the saved handoffs remain provenance snapshots rather than additional body credit.

The user-authorized sequence now prioritizes core/non-channel modules: preserve per-module Git status
checkpoints, reconcile shared contracts and then hold the consolidated discussion. The frozen 919-target inventory remains provenance, but E01's 95 targets are excluded from the current deep-review-required denominator, leaving 824 deep-review-required targets. Source
inspection has resumed. The latest staging check found a two-commit delta limited to `nanobot/agent/skills.py` and `tests/agent/test_skills_loader.py`; R01 is unaffected and K01 refresh is pending before final closure. H01/C00/H02/H03/T01 checks are complete. A01 is published and coordinator-checked. K01-A/B/C/D/E are published and coordinator-checked. E02-A/B/C/D/E/F/G/H/I/J/K/L/M are published and coordinator-checked at F68/P0/R0. E01-A/B/C/D/E/F remain saved at F40/P0/R55. The former E01-G-and-later deep adapter sequence is cancelled by explicit scope decision; the 55 remaining E01 primaries are recorded as quick-screened/not-deeply-reviewed. R01-A/B/C/D are saved at F30/P0/R40. Next is R01-E for full-body NativeNetworkPlugin.kt only, followed by later JNI/Go/helper/tooling R01 substages and then M01/X01/Q01/P01. Integrated review and D1–D7 remain deferred.
Fresh integrated review remains pending. No open decision blocks later source inspection. Continuation is stage-by-stage: each session should finish one bounded stage/substage, preserve exact evidence and gaps, and checkpoint before the next stage.

| Prior decision area | Current review-sequencing disposition |
| --- | --- |
| D1 — Supported executor and client platforms | Explicitly deferred to final consolidated discussion |
| D2 — Full authority and private boundaries | Explicitly deferred to final consolidated discussion |
| D3 — Managed task sessions and delegation | Explicitly deferred to final consolidated discussion |
| D4 — Compatibility and retirement | Explicitly deferred to final consolidated discussion |
| D5 — Request/tool registration lifetime | Explicitly deferred to final consolidated discussion |
| D6 — Attempt outcomes and recovery | Explicitly deferred to final consolidated discussion |
| D7 — Completion and resource policy | Explicitly deferred to final consolidated discussion |

These are deferrals of discussion, **not acceptance of implementation**, platform promises,
authority semantics, schemas, migrations, compatibility removals or resource budgets. The
original [decision evidence](../architecture-review-2026-11-06/decisions.md) remains unchanged.
Findings and alternatives may be recorded conditionally while requirements remain unsettled.

When evidence arrives, consolidate preserved strengths; current architecture judgement;
deduplicated conditional findings; concrete removal/combination/move/retirement options;
relevant prior architectural changes; compatibility, authority and durability consequences;
reference mechanisms and justified differences; conflicts, unread gaps and final questions.
Coordinator reconciliation and acceptance remain separate from publication and Git preservation.

The review supports source-based design and efficiency conclusions. No measured benefit,
runtime stability, sandbox enforcement, operational parity or successful delivery is claimed.
No fixes are authorized by this package.

## 2026-11-07 E01 scope decision

Chat channels are not a current audit/product priority. E01 is intentionally left functionally untouched. Existing E01 findings remain evidence for later channel work, but unresolved adapter-specific semantics are not blockers for the current architecture audit. Re-open deep E01 review only if channel functionality becomes a product priority, a channel change is planned, or final cross-module reconciliation exposes a material dependency that cannot be resolved without deeper channel tracing.


## 2026-11-07 R01-A checkpoint

R01-A established the remote owner/device authority split across fourteen fresh primary-body reads. Durable device/scopes/revocation and refresh hashes live in the independent owner store; short-lived access credentials are memory-only; owner pairing requires local approval; and gateway owner-permission mutations use a typed authorize→single-use-consume admission before the existing workspace permission owner acts.

One provisional P2 finding is retained: desktop approval persists the device and stores only the refresh hash before the phone has confirmed custody of the plaintext access/refresh pair. `PairingService.poll()` also marks that in-memory result delivered before IPC/HTTP response delivery. Owner restart or response loss in that window can therefore strand a durable paired device with no recoverable client credential, requiring local revoke/re-pair. R01 transport/native conclusions remain pending.


## 2026-11-07 R01-B checkpoint

R01-B adds seven full primary-body reads for owner IPC, the private listener, production mobile edge, operation registry, in-memory replay, staging helpers and the isolated data server. Cumulative R01 source review is **F21/P0/R49**, still unpublished and coordinator-unaccepted.

Four new provisional findings are retained. **R01-F02 (P1)** establishes that the production mobile WebSocket validates credential currency but does not enforce the declared/per-command chat and attachment scopes before shared WebUI commands hydrate sessions, send messages or commit attachments. **R01-F03 (P2)** extends the credential-custody concern to ordinary refresh rotation: the owner invalidates the old credential before the replacement is safely in client custody. **R01-F04 (P2)** records that the versioned mobile route registry declares HTTP bootstrap/mutation/attachment/logout endpoints the production edge does not serve. **R01-F05 (P2)** records that the isolated data server's refresh/mutation/logout authorization semantics differ materially from production, so it cannot by itself qualify production transport behavior.

Counterevidence narrowed several leads: production standard GET dispatch rechecks the operation registry; private HTTP reads/media revalidate authority immediately before delivery; the in-memory `OperationReplayCache` is not the normal production mutation owner; and the broken delegated `device_list` plus `delegate_execute` shortcut have no established first-party production caller. R01-C is limited to `remote/supervision.py` and `remote/tailscale.py`.


## 2026-11-07 R01-C checkpoint

R01-C closes the final two Python-side R01 primaries: opt-in systemd user supervision and read-only Tailscale/Serve readiness. Cumulative R01 source review is **F23/P0/R47**, still unpublished and coordinator-unaccepted.

Two new provisional P2 findings are retained. **R01-F06** records that the supervision adapter runs systemctl with `check=False`, ignores nonzero results, can continue deleting the unit on a failed disable/stop, and still reports `applied: true` / exits zero. The finding is bounded to the explicit local supervision CLI; normal gateway/owner composition does not call this path. **R01-F07** records that Tailscale readiness treats any `"/mobile/v1"` substring anywhere in Serve JSON as a valid mapping, while the desktop pairing UI falls back to `currentHttpsOrigin()` and creates an owner-phone offer even when Tailscale readiness is negative. That contradicts the implemented Tailscale-only v1 design requiring exact Serve→gateway verification before offer creation.

Counterevidence narrows both results: supervision has prefix/path/socket containment, dry-run by default and one historical successful live systemd cycle; Tailscale probes are read-only and owner/device tokens remain the application authority. No P1/P0 was added in R01-C. R01-D is limited to the seven Android shell/lifecycle/keepalive/custody primitives listed in the R01 handoff.


## 2026-11-07 R01-D checkpoint

R01-D adds seven full Android primary-body reads for the activity/WebView shell, fixture probe, foreground run keepalive, encrypted credential/native-state custody and the narrow JNI façade. Cumulative R01 source review is **F30/P0/R40**, still unpublished and coordinator-unaccepted.

One new provisional P2 finding is retained. **R01-F08** records that QR pairing persists its pending poll challenge to encrypted custody, while the 8-digit manual-code path keeps the server-issued nonce/collector secret only in memory. If Android recreates the process/plugin while the request is waiting for desktop approval, `pendingPairing()` cannot restore it and the server's single-winner collector binding rejects recollection of the same offer; the user must create a new desktop offer.

Counterevidence narrowed the rest of this slice: corruption/missing Keystore keys fail closed, refresh persistence is fenced against logout races, app backup is disabled, keepalive is explicitly process-local and Android-bounded, and the JNI surface itself exposes no arbitrary dial/URL authority. Stale native state directories and a transient stale media handle after logout are retained only as cleanup/UX observations, not promoted defects. R01-E is limited to a fresh full-body review of the 1,359-line `NativeNetworkPlugin.kt`.
