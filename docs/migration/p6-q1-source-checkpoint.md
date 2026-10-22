# P6 Q1 source checkpoint — 2026-10-05

**Q1 NOT_ACCEPTED / P6 NOT_ACCEPTED**. Runtime **NOT_READY / NOT_AUTHORIZED**;
D9/P7 **NOT_AUTHORIZED**. This is dated progress evidence, not a source freeze,
release artifact, whole-source qualification, or permission to activate anything.
The [single current checklist](p6-completion-checklist.md) retains all20 gates.
The [Q1–Q5 execution plan](p6-completion-execution-plan.md) and historical
[3,977-diagnostic starting inventory](p6-source-qualification-inventory.json)
are unchanged. No Q2–Q5 acceptance is claimed.

## Executed evidence and exact boundaries

Raw records below preserve exact argv, source inventories, isolated profiles,
stdout/stderr and exit codes. Home/network denial remains in place. No broad
suite was rerun for this docs publication; its source/test bytes are unchanged
from the final strict inventory. Counts overlap and must never be summed.

| Evidence | Actual result | Boundary |
| --- | --- | --- |
| [Full strict](../../.assets/releases/p6-q1-source-oct05.f71c9a/q1-oct05-final-strict.json) | FAIL: 3,880 diagnostics, 463 analyzed files, exit1 | Unchanged full BasedPyright policy and all-extras/channel environment; not green. |
| [Final audit](../../.assets/releases/p6-q1-source-oct05.f71c9a/q1-oct05-final-audit.json) / [diff](../../.assets/releases/p6-q1-source-oct05.f71c9a/q1-oct05-final-audit.diff) | 97 resolved signatures, zero introduced; 91 changed Python files compiled; rg-visible 1,881→1,882, no removals | Pre-publication implementation comparison; checker/CI/policy/initial inventory/execution plan unchanged. Not semantic acceptance by compilation alone. |
| [Root lint](../../.assets/releases/p6-q1-source-oct05.f71c9a/q1-oct05-root-lint.json) | PASS, exit0 | Exact `ruff check nanobot tests conftest.py`, not formatting or an inferred pass. |
| [Session/config/foreground/target-scope](../../.assets/releases/p6-q1-source-oct05.f71c9a/q1-session-config-final-check.json) | 1,264 passed, nine retained fork warnings, zero exclusions, exit0 | Exact bounded selection in raw argv; not all repository tests or installed-loader qualification. |
| [Root token/lease](../../.assets/releases/p6-q1-source-oct05.f71c9a/q1-oct05-root-token-lease.json) | 54 passed, zero exclusions, exit0 | Current token fixture and target-scope selection; overlaps the previous selection. |
| [Full discovery](../../.assets/releases/p6-q1-source-oct05.f71c9a/q1-full-collection-current.json) | 12,058 collected, four errors, exit2 | Predates one subsequently added token test; retained count is not relabeled as final-source collection. |
| [Full provider diagnostic](../../.assets/releases/p6-q1-source-oct05.f71c9a/q1-oct05-full-provider-diagnostic.json) | 39 failed, 1,358 passed, three skipped, 12 warnings, one collection error, exit1 | Exact `tests/providers --continue-on-collection-errors --tb=short`. Continuation is diagnostic only, not acceptance or an exclusion workaround. |

Full discovery fails in three modules importing `MemoryConsolidator`
(`test_correction_round.py`, `test_memory_budget_hardening.py`,
`test_memory_consolidation_types.py`) and one importing removed Codex private
helpers (`test_openai_codex_provider_fork.py`). Three provider xAI callback
failures are local-loopback OS denials. Other provider failures involve retained
legacy/API mismatches requiring explicit dispositions or investigation; they
are not all assumed obsolete. No runtime restoration or expectation weakening
is authorized here.

Earlier [whole-source tests](../../.assets/releases/p6-q1-source-oct05.f71c9a/q1-all-source-tests.json)
stopped with 49 collection errors (exit2), and the earlier
[executable diagnostic](../../.assets/releases/p6-q1-source-oct05.f71c9a/q1-all-executable-tests.json)
timed out (exit124) before config isolation. Red/intermediate outputs are retained
with their historical source inventories, not promoted to current green results.
S2 and S3 remain **FAIL**; V1 remains **FAIL** without all-row release acceptance.

## Bounded implementation review, not Q1 acceptance

The fresh [independent review](../../.assets/releases/p6-q1-source-oct05.f71c9a/q1-oct05-integrated-review.md)
recommends PASS strictly for the bounded checkpoint. The supplied
[root verdict](../../.assets/releases/p6-q1-source-oct05.f71c9a/q1-oct05-root-verdict.json)
accepts that bounded implementation only: truthful refusal/capability/session
typing, an informational workspace lease accessor, qualified imports/lint fixes,
temporary test config/home fixtures, and current-owner token fixtures. Neither
verdict accepts Q1/P6 or authorizes activation.

Original assertions are retained except two documented helper-call-reference
ports from `_is_reasoning_model` to `_requires_max_completion_tokens`, with
semantic parity of all 13 positive and nine negative cases. Other assertion
changes are additions only (reported local alpha-renames normalized). The
[fixture proof](../../.assets/releases/p6-q1-source-oct05.f71c9a/q1-token-kwargs-fixture-proof.json),
[worker diff](../../.assets/releases/p6-q1-source-oct05.f71c9a/q1-token-kwargs-fixture-worker.diff)
and [completion report](../../.assets/releases/p6-q1-source-oct05.f71c9a/q1-token-kwargs-fixture-completion.md)
retain their exact boundary. Production provider/registry bytes are unchanged
by that fixture package; the Path accessor exposes no authority and does not
replace `require_current()`.

Two review gaps are nonblocking only for this bounded verdict: no direct
foreign-process/fork assertion for the new accessor (existing pre-lock owner
checks and related lease regressions were inspected), and no dedicated
bound/unbound fixture-teardown regression (monkeypatch undo and the bounded
selection were inspected). These are not silently asserted as proven tests.

Explicit approval and per-assertion dispositions for obsolete fork predicates,
with positive current-owner replacement coverage, remain an **OPEN material
gate**. This factual documentation package does not settle that gate. Strict,
full-source/memory/provider compatibility, native/frontend checks, corrected
immutable artifacts, production/stopped-copy/old-reader/resource inputs, and
final all20 acceptance remain required.

## Private QA publication

The ignored local [README](../../.assets/releases/p6-q1-source-oct05.f71c9a/README.md),
[manifest](../../.assets/releases/p6-q1-source-oct05.f71c9a/manifest.json) and
[SHA256SUMS](../../.assets/releases/p6-q1-source-oct05.f71c9a/SHA256SUMS) bind
exactly the explicit allowlisted top-level raw records/stdout/stderr/helpers/
diffs/reports, including relevant red runs and the actual reviewer/root verdict.
Raw files are copied unchanged with byte-hash verification. Directory0700 /
files0600. Earlier archives remain untouched. No baseline/source/state snapshot,
dependency environment, credentials or live state is archived.

`q1-retained-contract-analysis-summary.md` is a coordinator transcription, not a
raw agent transcript. Absolute temporary paths and sanitized environment/profile
fields in raw records are provenance, not portable install/replay claims. The
bundle binds reviewed pre-publication inventories; this later docs publication
does not manufacture a final source freeze or release PASS.
