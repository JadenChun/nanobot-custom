# macOS correction qualification — 2026-11-16

**Subsequent Git checkpoint:** these completed fixes and records are now committed
locally as `Jaden <changherng.jc@gmail.com>`. Local/uncommitted status below describes
the original qualification time; see the [current handoff](implementation-handoff.md#git-checkpoint-completed-fixes-committed--2026-11-16).
No push, staging merge or deployment occurred, and qualification limits are unchanged.

## Status and scope

**PASS for the bounded Stop/test/tooling corrections; not full A3 or merge acceptance.**
Work remains on `codex/staging-audit-implementation`, based on
`de2e0fec517203d0f2c45fa83ecf7c591fa2ab4b` (also the last fetched origin head).
Staging remains separate and unchanged at
`f1a6bb85588ba044fc2ca50e3e9f41eb2f13b6f9`.

These corrections and this record are local, uncommitted work at this checkpoint;
no commit, push, merge, PR or deployment was performed. Effective repository author
and committer identity is `Jaden <changherng.jc@gmail.com>`. The frozen
`A3-PLAN-2026-11-11` architecture was not changed.

## Corrections

- Ordinary admitted cancellation previously omitted `TurnCompleted`. The GUI could
  retain “Working” and disabled project controls after backend Stop settlement.
  Cancellation now joins stream abort, its original-root terminal and idle in order.
  No terminal is synthesized from a passive null Stop target, and no Stop-attempt
  identity replaces the original root identity.
- Each cancellation-cleanup operation receives one cancellation-request deadline.
  Repeated root cancellation cannot abandon it. If a collaborator resists cancellation,
  the admitted scope/source remains owned until actual settlement; there is no detached
  late terminal. This preserves existing uncertainty semantics, not unconditional
  source release or atomic network delivery. The existing Stop settlement budget is
  separate; it is not a universal request deadline.
- Fixed stale test doubles for `reconcile_stale` and absent `remote_owner`, and updated
  the revoked-device test to assert removal **and** rejection of previously issued access.
- Fixed stock macOS Bash 3.2 Android system-image path normalization without changing
  the Linux-only Android build tuple or missing-tool refusals.
- Removed the five reported test lint diagnostics. No authorization guard was bypassed.

## Qualification evidence

macOS 27.0.1 / Apple Silicon; checkout-local Python 3.13.12; 216 frozen distributions;
BasedPyright 1.40.2; Ruff 0.17.5. WebUI Bun 1.4.2 and TUI Bun 1.3.13 were invoked
with their pinned versions, without replacing global tools.

| Executed check | Result |
| --- | --- |
| Full Python, four workers, CLI rendering module separately excluded | 18,471 passed, 68 skipped, 6 subtests passed; exit 0 |
| CLI rendering module, serial, `COLUMNS=100` | 129 passed; exit 0 |
| Independent affected Stop modules | 178 passed |
| Final strengthened ingress module | 55 passed |
| Independent critical fixture/tooling checks | 12 passed |
| CI-scope Ruff | passed |
| Unchanged full-source strict checking | 547 files, zero diagnostics; exit 0 |
| Official frozen before/after type comparator | zero new signatures |
| WebUI full coverage suite / production build / lint | 1,530 tests in 99 files passed; build/lint passed |
| TUI tests / TypeScript | 241 tests passed; type check passed |
| Clean wheel install / dependency check | passed; 99 compatible installed distributions |
| Public installed gateway and remote-owner restart | passed; identity/history retained, owned listener closed |
| Installed ordinary macOS exec / hard deadline / session close | passed |
| Actual macOS Full request | safely refused `supervisor_unavailable`; no authority issued |

The full-suite snapshot was followed by **test-only** strengthening of two assertions:
observing the actual correlated Stop reply while terminal publication remains held,
and checking canonical source absence after deletion/drain. Only
`tests/webui/test_child_approval_ingress.py` changed afterward; its final 55-case module
and full-scope lint were independently rerun. Production bytes, packaged bytes and
strict-checker inputs remained identical. The fresh integrated review passed after
those two evidence gaps were closed; the earlier full-suite receipt was not overwritten.

Useful reproduction commands (use this checkout's existing dependency environment):

```bash
.venv/bin/python -m pytest -q -n 4 --dist loadfile \
  --ignore=tests/cli/test_commands.py --basetemp=/tmp/nbmac-full
COLUMNS=100 .venv/bin/python -m pytest -q tests/cli/test_commands.py \
  --basetemp=/tmp/nbmac-cli
.venv/bin/basedpyright --outputjson
.venv/bin/ruff check nanobot tests conftest.py
```

Keep the macOS temporary path short for AF_UNIX limits. Do not sync dependencies
between the frozen before/after checks or refresh the baseline to absorb diagnostics.

## Real installed GUI reproduction

An unpatched installed public gateway and real browser used only the local scripted
provider `custom/qualification-offline`; no external or paid model was called.
Under one held, partially streaming parent, approving one child changed only
`build_config.txt`; denying its sibling left `test_plan.txt` unchanged. Original-chat
Stop then stopped both children and the parent without releasing provider gates.

The byte recorder observed stream end → exactly one matching original-root terminal
→ idle. The terminal arrived 0.876 seconds after the Stop frame. Child capacity and
approvals cleared, the canonical Stop target became null, and the live GUI changed
to “Worked” with the project selector restored **without reloading**.

Wheel SHA-256: `83f2d2698e39b1d8ea9ab9e7398804598627eaa0ff71e9ccc7f844fb69d113cb`.
All 547 packaged Python files, 590 WebUI files and the operation contract matched.
The accepted code/test patch, before these documentation changes, has SHA-256
`d0ca65d0e8092ff492470a1cbd33f787eda01f00cae08ffd7aeea1f34281f71a`.

Local raw receipts, manifests, reports, frames and screenshots are under
`/private/tmp/nb-a3-mac-fixes-20261116-1crdvf`. They are local temporary evidence,
not Git-published artifacts and not automatically available on another device.
All disposable provider/proxy/gateway processes exited successfully; operator
services and state were not touched.

## Still open

- Native macOS Full/whole-process-tree supervision remains unavailable. Ordinary
  execution and safe refusal do not qualify that capability or close A3 release gates.
- No new physical Android, native desktop signing, OS-notification or full release
  qualification is claimed here; historical Linux/Android receipts retain their scope.
- The five newly reported client issues are separate, sequential work in
  [client-ux-followups.md](client-ux-followups.md). Restoring the selector after Stop
  does not establish that every during-run folder-selection report is resolved.
