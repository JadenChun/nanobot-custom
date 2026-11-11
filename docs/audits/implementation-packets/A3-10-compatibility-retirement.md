# A3-10 implementation packet — Compatibility and retirement

**Status.** Ready for packet review. Dispatch after the A3-9 candidate and compatibility evidence are accepted.
[Release qualification](A3-9-release-qualification.md) · [Retirement decision](../architecture-review-2026-11-06/decisions.md#d4-compatibility-and-retirement) · [Integrated findings](../module-review-2026-11-06/integrated/I05-final-reconciliation.md)

## Objective and boundary

Remove obsolete live paths after the replacement and its rollback are proven.
Keep supported imports, data readers, export paths, and historical display.
Use the compatibility evidence to select each removal. The absence of a registry entry is insufficient evidence.

## 1. Record a disposition for each retirement candidate — R18

Trace the registration, in-tree callers, public exports, configuration, installed integrations, persisted records, and recovery tools.
Record a named owner and supporting evidence for each disposition.
Choose retain, forward to the current owner, isolate offline, or remove.
If a supported caller or stored format still needs a path, keep the smallest compatible reader or adapter.
If the disposition changes a public contract or data format, record the accepted compatibility decision before changing it.

| Candidate | Required evidence or disposition |
| --- | --- |
| Legacy delegation | A3-5 owns all live children. Remove old manager construction, task lifecycle, and text-derived controls after checking public imports and plugin callers. Keep legacy announcement rendering as long as retained history needs it. |
| Mobile v1 paths | A3-7 owns current device access. Remove unused refresh writers and duplicate pairing UI. Preserve revocation markers, the version refusal, and the supported re-pair path. |
| Gated onboarding and dormant idle compaction | Keep each gate until a supported replacement or an explicit retirement is accepted. If the wizard is retained, use the current safe config and secret inputs before enabling it. |
| Legacy session recovery | Keep the sealed read, export, and rollback proofs. Move an offline compatibility path only after old fixtures recover with equivalent results. |
| Private workspace focus and older model setters | Trace direct imports, plugins, and automation before removing a callable. Preserve public wrappers that still forward to an authoritative owner. |
| Goals, local triggers, and inactive tools | Match discovery to the accepted activation state. Keep the recovery and permission gates for any future activation. |
| Source-backed artifact alternative | Retain the current byte, grant, and restore contract. Treat this alternative as separate work if evidence later justifies it. |

**Edit boundary.** Start with the proven obsolete adapter and its direct wiring.
Candidates include `nanobot/agent/subagent.py`, its exports in `nanobot/agent/__init__.py`, the legacy pairing bridge, and gated paths in `nanobot/cli/onboard.py` and `nanobot/agent/idle_compaction.py`.
Keep changes to the reviewed callers, compatibility tests, and current operator guidance.

**AC10.1.** Every proposed removal has an accepted disposition and evidence for its callers and stored state.
A supported legacy import or retained session continues to work through the approved reader or adapter.
No removal depends only on a search that found no in-tree caller.

## 2. Remove redundant live ownership

Remove the retired registration, startup hook, background task, and writer together.
Retain a read-only historical projection when old history needs it.
Keep new delegation, pairing, and runtime controls bound to their current owners.
Remove stale flags or guidance only when the current product no longer accepts or advertises them.
Leave measured efficiency work with its owning packet.

**AC10.2.** A normal startup creates one live runtime for each supported capability.
The legacy path cannot launch a child, re-enable a revoked credential, or rewrite old history.
The supported compatibility fixtures can still read and export their data.

## 3. Requalify the changed candidate

Run focused compatibility and startup checks after each removal.
Carry forward earlier evidence only when its source, inputs, and dependencies still match.
Repeat the affected A3-9 checks at the new source and artifact digests.
Keep an executable rollback that uses the current durable authority.
Restore the retired read path if compatibility fails. Keep the old live authority closed.

**AC10.3.** The final candidate passes the applicable release checks.
Rollback retains post-cutover histories and device revocation.
The handoff lists all retained compatibility and separately deferred changes.

Start with `tests/test_agent_import_order.py`, `tests/agent/test_subagent_lifecycle.py`, `tests/utils/test_subagent_channel_display.py`, and `tests/agent/test_migration_sdk.py`.
Use `tests/session/test_session_mode_export.py`, `tests/session/test_legacy_mode_journal_compatibility.py`, and `tests/cli/test_startup_admission_onboard_webui_api.py`.
Return the final supported surface, the removed and retained paths, the verification evidence, and the rollback receipt.
