# A3 architecture record

This is the current, reviewable record of the A3 audit and candidate refactor, as of
2026-11-11. It collects the design, reasons, implementation order and evidence in the
repository. **The implementation merge is on HOLD.** Documentation integrity can pass
while architecture, security and release acceptance remain open.

| Read | Purpose |
| --- | --- |
| [Design and alternatives](design.md) | Ownership, data flow, required invariants, observed mechanisms and smaller options. |
| [D1–D7 decisions](decisions.md) | Historical pending choices, observed candidate choices and missing acceptance provenance. |
| [Implementation plan](implementation-plan.md) | A3-0 plus ten stages, dependencies, owner boundaries, acceptance and rollback. |
| [Review and merge gates](review.md) | Independent findings, inspected evidence, reported claims and conditions for acceptance. |
| [Evidence and finding ledger](evidence.json) | Machine-readable pins, 136 finding/package/stage assignments, qualifications and evidence states. |
| [Archive manifest](audit-manifest.json) | Exact Git blob and SHA-256 identities for the 138 imported audit files. |

## Version boundary

[The main architecture page](../../architecture.md) describes the staging baseline. This
record describes a separately pinned implementation candidate; publishing these documents
does not make that candidate merged behavior.

| Identity | Durable reference and meaning |
| --- | --- |
| Frozen deep-source audit | [`9562e57f8333d7f14b6aaeae837660723a9b2650`](https://github.com/JadenChun/nanobot-custom/tree/9562e57f8333d7f14b6aaeae837660723a9b2650), tree `bbe2f237efe0cd7b935946b533f66ed0097a3c93`. Findings describe this source and their stated conditions. |
| Original staging and packet base | [`29612c3df01dc3dce3501fe349bcf555cd6c58b6`](https://github.com/JadenChun/nanobot-custom/tree/29612c3df01dc3dce3501fe349bcf555cd6c58b6). The audit records the bundled skill-path delta separately as K01-STG-F01. |
| Updated pre-refactor code baseline | [`ab9598217f7bfe230466bc4c5513f188e869fced`](https://github.com/JadenChun/nanobot-custom/tree/ab9598217f7bfe230466bc4c5513f188e869fced), tree `74ba3b4f3183dc8e735f7e0a423a86893b839b54`. Includes integration `cea61e0598fba5784237fdaab21c448fb9b2a9bf` of the original remote staging and local staging `4e2bf4d8d0b34f133c01591b67826ca956fed384`, all six feature-repair commits, the architecture records, and a bounded cancellation/recovery correction. No refactor implementation is included. |
| Immutable documentation archive | [`a14aefab0235dd49ed2b579d448e2c47c75e3860`](https://github.com/JadenChun/nanobot-custom/tree/a14aefab0235dd49ed2b579d448e2c47c75e3860/docs/audits), tree `9a2d0c5ef834bd06cc95dd22c9e9724eb7740524`. The 138 files under `docs/audits/` are copied byte for byte from this revision. |
| Reviewed implementation | [`0518de5f5013b77cbc20d2c5b456e973f33c68f5`](https://github.com/JadenChun/nanobot-custom/tree/0518de5f5013b77cbc20d2c5b456e973f33c68f5), tree `26fae6f933d1dad3e8ae586c847b3168a4571a28`; [PR #27 into staging](https://github.com/JadenChun/nanobot-custom/pull/27). Source links in this record use this full SHA. |
| CI snapshot | [Run 42972763133](https://github.com/JadenChun/nanobot-custom/actions/runs/42972763133) at the implementation SHA completed with overall conclusion **cancelled**. Several jobs passed; full Windows Python 3.14 was cancelled. |
| Migration CI exercise | [`dee68697157dccb0b49a6782c8dd13a594c9cf92`](https://github.com/JadenChun/nanobot-custom/commit/dee68697157dccb0b49a6782c8dd13a594c9cf92) descends from `980b491d01cdf308041bd9ba7a2ccdd5998d6a99` and adds only two lines/comment to the baseline script. It lacks the later `7fbb00c2a` and `0518de5f` fixes and is a CI exercise, not a separate architectural implementation. |

The six repairs ending at [`d3380eb69d3c8416a9ca72522001234ce30a8eeb`](https://github.com/JadenChun/nanobot-custom/tree/d3380eb69d3c8416a9ca72522001234ce30a8eeb)
are included in the updated pre-refactor staging baseline. Their
[reconciliation dispositions](review.md#repair-branch-reconciliation) still apply to the
separate refactor candidate. Commit ancestry alone does not establish whether that
candidate preserves each correction semantically.

Future refactor acceptance must compare against the latest `staging`, including these
pre-refactor repairs and the [integration validation](review.md#pre-refactor-staging-validation).
The existing refactor review and its source links remain pinned to
`0518de5f5013b77cbc20d2c5b456e973f33c68f5`; they do not attest reconciliation with the
updated baseline. The implementation branch and PR #27 remain separate. The immutable
138-file manifest covers the architecture audit history; the feature-repair records
under `docs/audits/feature-implementation-2026-11-05/` retain their separate provenance.

## Evidence vocabulary

| State | What it establishes |
| --- | --- |
| `SOURCE_OBSERVED` | A named body or format exists in the pinned source. It does not prove the invariant on every path. |
| `REPRODUCED` | An independent reviewer exercised a bounded public or owning boundary on isolated state, with the recorded result. |
| `HOSTED_LOG_INSPECTED` | Exact-head hosted status or log output was inspected. Its environment and covered boundary remain part of the claim. |
| `REPORTED` | A PR or stage report makes the claim; the supporting receipt has not been independently inspected here. |
| `UNVERIFIED` | No exact acceptance evidence is attached to this item. Every original finding defaults to this state. |
| `READY_FOR_REVIEW` | The implementer requests review. It does not mean accepted, fixed, closed or release qualified. |
| `ACCEPTED` | A named reviewer has accepted an exact criterion at matching source and evidence inputs. No blanket A3 acceptance is recorded. |

The PR reports all ten implementation stages `READY_FOR_REVIEW`. That status and a global
passing test count cannot close 136 different conditional findings. The current [review](review.md)
separates concrete counterexamples from platform, provenance and human-decision gaps.

## Immutable audit and packet history

Start historical reading with the [source comparison](../../audits/architecture-review-2026-11-06/README.md),
[module synthesis](../../audits/module-review-2026-11-06/synthesis.md),
[I05 reconciliation](../../audits/module-review-2026-11-06/integrated/I05-final-reconciliation.md)
and [original implementation packets](../../audits/implementation-packets/README.md).
The [discovery checkpoint](../../audits/codebase-discovery/completion-2026-11-06.md) supplies
the coverage basis; [the original audit plan](../../audits/project-audit-plan.md) supplies
the earlier sequencing.

Their dates, pending states, source pins, historical links and claims are preserved.
This A3 record supersedes their planning status **for current navigation only**. It does
not rewrite their contents, validate old links or turn historical recommendations into
accepted product requirements. The linked ChatGPT roadmap is supplemental; this record
and the repository packets contain the reviewable plan.

I05 preserves 136 original findings assigned exactly once to 22 repair packages, 105
module questions, zero whole-finding deduplications and one partial factual overlap.
Its 824 deep-required bodies and 95 E01 channel bodies have different coverage depth.
The 26 retained E01 findings stay outside the 136 denominator; K01-STG-F01 stays a
separate staging overlay. Source audit completion means reconciliation under that audit's
definition, not execution, platform or architecture acceptance.
