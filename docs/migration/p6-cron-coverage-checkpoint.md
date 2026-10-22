# P6 cron coverage closure and next source qualification

2026-10-02: **bounded follow-up ACCEPTED / root PASS**. Both prior P2 AC3
cron coverage findings are closed. **P6 strict-source qualification IN_PROGRESS**;
whole P6 **NOT_ACCEPTED**, runtime **NOT_READY / NOT_AUTHORIZED**, D9 and P7
**NOT_AUTHORIZED**. The [single completion checklist](p6-completion-checklist.md)
still owns all20 gates. No production behavior defect was established by the
earlier two coverage findings.

## Delivered and verified

- `changed_definition` now asserts exact requested names/messages, agent-turn
  interval120000 and cron expression `0 18 * * *`/UTC against the returned job,
  durable reopen and scheduling commit. A detached expected job checks every
  unrelated field, including non-default primary/secondary routes, session/origin
  binding and nested metadata. Instance identity, revision change, token-required
  rejection, receipt count and definition-only fences remain checked.
- `route_roundtrip(update=True)` now requests a real exact-token target change
  to `discord`/`updated-primary`, rather than changing an unrelated name. Returned
  and reopened full-job assertions preserve the original session/origin, secondary
  destination, metadata and all unspecified fields. Missing-token rejection and
  detached metadata checks remain. No automatic binding inference or delivery-field
  clearing was restored; production cron code is unchanged.
- Six temporary, test-local fault probes cover a wrong valid interval, a wrong
  valid cron expression, unintended message/name/delete-after-run changes and an
  ignored delivery-target update. All six escaped the frozen old helper; all six
  are caught by the strengthened helper. Faults pass through the real definition
  owner, so returned/reopened definitions and commits remain internally consistent.
  Probes exist only in disposable evidence snapshots, not application code.
- The next source stage begins with truthful `NoReturn` declarations for checked
  session `_fail` and two serialization-refusal hooks; the unused pickle protocol
  argument follows its base `SupportsIndex` contract. Normalized runtime AST is
  identical. No control flow, storage schema, lock, permission, error, guard or
  operation changes. Scoped lint and in-memory compilation pass.

Root final combined selection: **943 passed,11 retained fork warnings,zero
exclusions**. It includes full `tests/cron`, Stage3 composition/shutdown, checked
session lock/read/replace, recovery locator and three recovery consumer modules.
Earlier764/178 runs overlap this selection and must not be added to it. The
original test-node files/assertions are byte-identical; only the cron helper and
checked-session annotations changed before documentation publication. Root scope
audit preserves1876 other rg-visible baseline files.

One original sticky-root test initially failed because its own TemporaryDirectory
uses `/private/tmp/checked-source-*`, outside the runner's write subtree. The
same denial reproduces on frozen pre-edit source. The final run allows writes
only to that transient `[a-z0-9_]` prefix and descendants; home, network, process,
production/protected-original restrictions remain. The original test/assertions
are unmodified. A rejected first regex profile and all earlier red runs are
retained; they are not counted as successful test execution.

## Strict gate and next bounded work

Unchanged full BasedPyright analyzes463 files and still **FAILS at3977 errors**,
down from4122. Signature comparison removes146 and introduces one inside the
owned checked-session module, with zero introduced outside it. That module's
diagnostics drop176→31; no diagnostics are waived. Full strict, dependencies,
CI and runtime guard policy are not weakened.

The remaining source stage must repair truthful first-party interfaces and
validated dynamic boundaries in bounded, regression-tested packages. The next
candidate boundary is checked-session validation/exception typing:31 diagnostics
remain, including path-protocol calls, failure-code unions, optional descriptors
and source-evidence pairs. Preserve fail-closed runtime checks; do not remove
guards or replace unknown values with `Any` merely to make the checker green.
Data/schema/locking changes are outside that typing scope and need a separate
decision if evidence requires them. Other source packages remain enumerated in
the complete strict report; no arbitrary clean-check waiver or automatic P7
promotion is permitted.

After source qualification, the unchanged native/full-source/UI jobs and corrected
artifact qualification remain required; production-profile, stopped-state,
previous-reader and operating-threshold inputs still gate their specific rows.
These two bounded PASS packages do not complete those stages.

## Evidence and review boundary

Private ignored [QA metadata](../../.assets/releases/p6-ac3-followup.QnTGmA/manifest.json),
[checksums](../../.assets/releases/p6-ac3-followup.QnTGmA/SHA256SUMS),
[root audit](../../.assets/releases/p6-ac3-followup.QnTGmA/ac3-followup-audit.json),
[final run](../../.assets/releases/p6-ac3-followup.QnTGmA/ac3-followup-final-integrated-two.json)
and [full strict report](../../.assets/releases/p6-ac3-followup.QnTGmA/ac3-followup-source-strict.json)
retain exact commands, OS profiles, source inventories, raw red/green outcomes
and the two-file diff. This is pre-publication local QA, not a final source freeze,
release artifact, portable install or activation authority. Earlier bundles remain
unchanged. Main performed the intentionally selected direct implementation and
verification loop; no new independent subagent/fresh whole-source review is claimed.
The orchestration skill informed the bounded contract and acceptance evidence.
No live config/state/credentials, transport/service/background activation, commits,
deployment or external publication occurred.
