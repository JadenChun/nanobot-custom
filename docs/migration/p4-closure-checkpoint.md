# Final P4 durable context/session closure checkpoint

Status: **accepted 2026-09-30**. This is the authoritative current-status
record for P4 and supersedes older P4 open, partial, and work-in-progress status
statements. The linked package documents retain their incremental evidence as
historical chronology.

P4 accepts the guarded durable context/session substrate: fail-closed memory
recovery and startup admission, canonical source events and governed checkpoints,
artifact offloading/retrieval and dormant rematerialization, bounded provider
conversation-state settlement, and the optional injection-only idle seams. The
accepted implementation remains behind its existing activation fences.

## Closure matrix

| Surface | Accepted P4 result | Boundary retained at closure |
| --- | --- | --- |
| Recovery and startup admission | Terminal receipt finalization/replay, durable quarantine seals, historical semantic validation, constructor checks, and CLI/gateway/session startup admission are wired through `memory_recovery_startup_admission.py`. | Recovery remains fail-closed and content-free. Offline finalization/replay is explicit operator action; P4 does not authorize live-state migration. |
| Canonical source and checkpoints | Durable source transactions, execution ownership, canonical runner recording, checkpoint publication/admission, governance, pressure projection, and restart behavior are accepted. | Canonical source remains authoritative; retained source is not rewritten or discarded merely because a compact projection exists. |
| Artifact lifecycle | Qualified `read_file` and single-file `grep` output offloading, scoped retrieval, dependency-aware checkpoints, lifecycle/failure handling, and explicit canonical rematerialization are accepted. | `ArtifactRematerializer` has no production caller and is not wired into reads, projection, startup, loop, CLI, or configuration. The reviewed producer set is not implicitly widened. |
| Provider conversation state | Settlement and restart behavior are accepted for ordinary legacy sessions. | Provider conversation state is deliberately not persisted for canonical-source, fresh, or ephemeral sessions. |
| Idle I1/I2/I3 | The durable activity, dependency-aware checkpoint, scanner/runtime binding, and lifecycle seams are accepted as optional substrate. | Idle remains disabled, dormant, injection-only, and without production construction, cadence, numeric policy, background scheduling, or activation. Those choices are not P4 closure requirements. |
| Activation fences | Existing migration, runtime, scheduler, configuration, and production-call-site fences remain effective. | P4 acceptance grants no authority to remove a guard, schedule cron/timers, activate idle/rematerialization, deploy, migrate live state, or serve live traffic. |

## Validation and independent review

The accepted evidence is recorded by suite, without summing overlapping selections:

- recovery and admission: **279 passed**;
- provider-state: **176 passed**;
- broad canonical session/source/governance/artifact/idle selection: **1,158 passed**;
- P4-H artifact evidence: **292 integrated**, **862 full session**, and
  **14 correction recheck** passed;
- fresh independent closure review: **206 passed**, with a **PASS** recommendation;
- scoped Ruff, Python 3.13 `compileall`, and `git diff --check` passed.

BasedPyright was unavailable in the validation environment. Windows is
intentionally unqualified; POSIX descriptor-relative guarantees are not claimed
for it. These limitations do not authorize weakening a fail-closed platform
boundary.

## Explicit non-scope and known limitations

P4 is a substrate acceptance checkpoint, not runtime activation, guard removal,
cron scheduling, deployment, live provider/channel traffic, live-state migration,
release qualification, or Windows qualification. Artifact rematerialization has
no production caller; idle has no production policy or construction path; the
artifact producer set remains the explicitly reviewed set; and no production
retention, cadence, or rollout setting is selected here.

P5, P6, P7, H1, and P8 remain separate gates under the migration plan. The next
bounded stage is **P5 integration/activation planning**, not activation itself.
P5 must preserve these fences unless and until its own acceptance authorizes a
specific change; P6 owns release and native-platform qualification, P7 owns live
canary/promotion, H1 owns the busy-input harness, and P8 remains optional.
