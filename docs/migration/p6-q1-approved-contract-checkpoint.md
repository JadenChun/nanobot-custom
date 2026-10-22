# P6 Q1 approved-contract checkpoint — 2026-10-05

**Q1 NOT_ACCEPTED / P6 NOT_ACCEPTED**. Runtime **NOT_READY / NOT_AUTHORIZED**;
D9/P7 **NOT_AUTHORIZED**. The [single current checklist](p6-completion-checklist.md)
retains all20 gate IDs/statuses, including **S2/S3/V1 FAIL**. This checkpoint
records bounded implementation acceptance and private evidence publication,
not a source freeze, release artifact, all-green whole-source qualification,
production profile, native-platform result, or authority to activate anything.
The [Q1–Q5 execution plan](p6-completion-execution-plan.md), historical
[3,977-diagnostic starting inventory](p6-source-qualification-inventory.json),
and [earlier dated source checkpoint](p6-q1-source-checkpoint.md) are unchanged.

## Accepted bounded implementation and exact scope

The fresh independent [integrated review](../../.assets/releases/p6-q1-approved-oct05.nlkXsj/q1-approved-integrated-review.md)
recommends **PASS for the bounded package**, with no material defect or required
correction. The independently executed [root verdict](../../.assets/releases/p6-q1-approved-oct05.nlkXsj/q1-approved-root-verdict.json)
supplies **PASS_BOUNDED_APPROVED_CONTRACT_IMPLEMENTATION_ONLY**. Neither is a
Q1/P6 or all20-row release verdict. Independent root verification of this later
documentation/private-QA publication remains required before final handoff.

The approved behavior repairs portable Responses ID normalization and preserves
exact server-native identity/replay. Legacy memory tests now exercise approved
current fail-closed owners instead of restoring removed mutation/recovery APIs.
The [eleven-file audit](../../.assets/releases/p6-q1-approved-oct05.nlkXsj/q1-approved-root-audit.json)
and [complete diff](../../.assets/releases/p6-q1-approved-oct05.nlkXsj/q1-approved-root-audit.diff)
record1,883→1,888 rg-visible files, exactly11 approved changed/added files,
eight changed Python files compiled, no removal, and all unowned rg-visible bytes
unchanged. Compilation and hash checks are structural evidence; the fresh review
separately assesses semantic assertions and current owner behavior.

Only `nanobot/providers/openai_responses/converters.py` and `state.py` change
production behavior: request-local bounded ASCII portable allocation and explicit
native pending-delta nonnormalization. Native parsing/capture bytes, transport,
schema, provider error policy and all current memory/governor/source/session/
guard/policy files remain unchanged. No dual persistence, global normalization,
suppression, test exclusion, trap relaxation or permission bypass is introduced.

Assertion authority remains explicit:

- [Memory dispositions](p6-q1-memory-dispositions.md) and the
  [machine-readable ledger](p6-q1-memory-dispositions.json) cover40 original
  methods/173 predicates:74 retained,99 approved replacements, and14 nonliteral
  retained fixture/accessor ports. The current three modules run67 cases.
  Replacement predicates positively test current-owner one-attempt summaries,
  impossible-budget zero dispatch, exact canonical prefixes/groups/checkpoints,
  private fail-closed recovery/finalization, durable ring and claimed-cron behavior.
  They do not claim removed automatic writes/retries/RAW fallback or rollback.
- [Provider-ID dispositions](p6-q1-provider-id-dispositions.md) retain ten original
  assertions exactly in text/AST; the two approved native length predicates become
  exact raw-ID equality. Portable safety/bounds/pairing remain supported customs.
  Current tests cover ASCII versus Unicode, collision allocation/reserved safe
  IDs, nonmutation, exact native SSE/SDK capture/state/pending output replay,
  and portable fallback when native state is absent or mismatched.

## Executed evidence and limitations

Raw records retain exact argv, source SHA256 inventories, isolated environment/
sandbox profiles, stdout/stderr and exit codes. The unchanged audited runner
denies home/network access and retains prohibited-effect traps. Counts overlap
and must never be added together. No suite was rerun merely to publish these docs.

| Evidence | Actual result | Exact boundary |
| --- | --- | --- |
| [Final root package](../../.assets/releases/p6-q1-approved-oct05.nlkXsj/q1-approved-root-final-package.json) | **270 passed**, exit0 | Exact eight modules listed below; no exclusions. |
| [Final full collection](../../.assets/releases/p6-q1-approved-oct05.nlkXsj/q1-approved-root-final-collection.json) | **12,142 collected / zero collection errors**, exit0 | Full `--collect-only`; one retained pytest_asyncio plugin-rewrite warning. Discovery is not execution qualification. |
| [Final full Ruff](../../.assets/releases/p6-q1-approved-oct05.nlkXsj/q1-approved-root-final-lint.json) | **PASS**, exit0 | Exact `ruff check nanobot tests conftest.py`; no fixes/formatting. |
| [Final unchanged full strict](../../.assets/releases/p6-q1-approved-oct05.nlkXsj/q1-approved-root-final-strict.json) | **FAIL:3,880 errors /463 analyzed files**, exit1 | All-extras/channel environment and unchanged checker policy. Audit signature delta versus prior checkpoint: introduced0/resolved0; checker-source mismatch list empty. No missing-dependency waiver. |
| [Current-owner regression](../../.assets/releases/p6-q1-approved-oct05.nlkXsj/q1-approved-root-owner-regressions.json) | **149 passed / one failed / two retained fork warnings**, exit1 | Exact nine modules listed below. Non-green result is retained. |
| [Approved before-image reproduction](../../.assets/releases/p6-q1-approved-oct05.nlkXsj/q1-approved-before-canonical-trap.json) | **Same failing node/cause**, exit1 | Cold croniter import invokes `platform.architecture` subprocess; unchanged fixture traps process execution. No trap/sandbox weakening or exclusion. |
| [Whole-suite diagnostic](../../.assets/releases/p6-q1-approved-oct05.nlkXsj/q1-approved-root-full-diagnostic.json) | **FAIL / INCOMPLETE: exit124 at600seconds**, last printed87% | Multiple failures/errors and existing skips in partial output; no final summary or defensible final totals. Predates final doc/import-spacing edits; its exact raw inventory is authoritative, not final-frozen whole-source qualification. |

The final package, collection, lint and strict records all carry the same1,888
source hashes, independently checked against the candidate before publication.
This later documentation publication changes only the new checkpoint and the
checklist's latest paragraph/S2/S3/V1 evidence cells; implementation bytes stay
frozen. These pre-publication run inventories are not relabeled as a later
complete source freeze or immutable release bundle.

Exact final package module selection:

```text
tests/agent/test_correction_round.py
tests/agent/test_memory_budget_hardening.py
tests/agent/test_memory_consolidation_types.py
tests/providers/test_openai_codex_provider_fork.py
tests/providers/test_portable_responses_ids.py
tests/providers/test_openai_responses.py
tests/providers/test_openai_codex_provider.py
tests/providers/test_conversation_state.py
```

Exact current-owner regression module selection:

```text
tests/agent/test_source_summary.py
tests/agent/test_source_pressure_compaction.py
tests/agent/test_source_compaction_publication.py
tests/agent/test_memory_recovery_coordinator.py
tests/agent/test_memory_recovery_receipt_finalizer.py
tests/agent/test_memory_recovery_transaction_prepare.py
tests/providers/test_final_request_budget.py
tests/cron/test_canonical_execution.py
tests/cron/test_visible_completion_link.py
```

The reproduced failing node is
`tests/cron/test_canonical_execution.py::test_real_runner_claim_computes_once_and_prepares_durable_outbox`.
Its failure is not silently dismissed as an obsolete assertion or credited as
green. The retained before-image reproduction establishes its specific existing
trap cause; no broader test failure is assumed to share that explanation.

## Private QA publication and remaining authority

The unique ignored PRIVATE local [QA README](../../.assets/releases/p6-q1-approved-oct05.nlkXsj/README.md),
[manifest](../../.assets/releases/p6-q1-approved-oct05.nlkXsj/manifest.json), and
[SHA256SUMS](../../.assets/releases/p6-q1-approved-oct05.nlkXsj/SHA256SUMS) bind only
explicitly allowlisted direct regular raw records/stdout/stderr/diffs/reports/
helpers, including red/intermediate runs, original assertion proof, accepted
mapping, before-image hash manifest, fresh review and actual root verdict.
Every copied raw file remains byte-identical; directory0700/files0600.
SHA256SUMS covers the raw files plus generated README/manifest, excluding itself.

No source/state snapshot, credential, config-state fixture, dependency environment,
live data or unlisted directory is archived. Source hashes and sanitized run
environment/profile fields are provenance only. Absolute old temporary replay
paths are not portable installation/replay claims. Older dated checkpoints and
raw bundles remain immutable. Publication audit/helper records remain in the
private QA directory for the root's separate independent publication check.

Strict-green and whole-source execution acceptance remain absent. Native
Linux/Windows, frontend/TUI/Docker, corrected release freeze/artifacts, production
profile/stopped-copy/old-reader/resource/operating inputs and final all20-row
acceptance are not waived. No credentials/live state, CI dispatch, service,
provider transport, network, deployment, commit/upload or background activation
is authorized or performed by this publication. Q1/P6 remain **NOT_ACCEPTED**;
D9/P7 retain their separate **NOT_AUTHORIZED** authorities.
