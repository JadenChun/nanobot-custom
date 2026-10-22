# P6 strict-zero checkpoint — 2026-10-13

**Full strict PASS; P6 remains NOT_ACCEPTED.** Runtime/D9/P7 are not authorized.

The unchanged full checker reports **0 errors, 0 warnings, 464 files analyzed**.
The global comparator against the approved immutable 20-error checkpoint reports
zero introduced diagnostics. This is not a fresh reproduction of the historical
488-file environment. Full lint passes. The current source inventory contains
1,993 files, with 1,213 Python sources compiled.

## Repairs and evidence

- Shared pinned-directory and process-tree operations have explicit public names.
  Eight production/test AST comparisons verify mechanical renames, including
  fault-injection seams. No process-cleanup algorithm changed.
- The foreground manifest now registers the existing exec-session interaction
  and listing tools alongside `exec`. All three retain the existing enable flag
  and per-agent manager; isolation and disabled-exec tests cover the handoff.
- Quarantine owns its own no-follow workspace descriptor. It compares that pin
  with the locked workspace identity and retains live lock/operation checks.
  It no longer reads another object's private descriptor. New tests cover
  ownership, foreign-directory rejection and failed-entry cleanup; no raw owner
  descriptor getter, admission token or recovery policy was added.
- The gateway uses one typed cleanup helper containing its previous actual
  `finally` sequence. AST comparison verifies that extraction. The unused legacy
  timeout/detach branch is gone; replacement tests use a real acquired,
  effect-fenced composition and verify task joining, exactly-once resource close,
  background-work settlement and error propagation.

Recorded passing runs (overlapping counts, not a whole-suite result):

| Scope | Result |
| --- | --- |
| Shared interfaces and registration, earlier snapshot | 269 passed / 13 modules |
| Quarantine owning tests | 125 passed / 4 modules |
| Broader recovery paths | 485 passed / 8 modules |
| Gateway lifecycle and admission | 63 passed / 4 modules |
| Shell/process tests, current source | 60 passed / 2 modules |

All listed runs finished without timeout or forced process-group cleanup.
The shell suite uses a separately qualified finite route permitting only bash,
sh, sleep and touch in addition to the frozen runtime executables. Its preflight
verifies denied network bind/connect and denied unlisted execution. The default
runner/profile, dependencies, shared fixtures and production guards are unchanged.
Historical local-process runner files were missing; their old evidence was not
relabelled as current validation.

QA authority: `p6-strict-zero-r1-root-publication.json` in
`/private/tmp/nanobot-p6-completion.rk7wje`, chaining the previous accepted
`classic-entry-contract-r1-root-publication.json`. Raw results retain source,
environment, output and dependency hashes. Failed intermediate attempts remain.

## Work still required

Fresh `p6-strict-zero-cli-baseline-r1`: **47 failed, 151 passed** across complete
`test_commands.py` and `test_desktop_target.py` modules. These include removed
Dream/Heartbeat helpers, obsolete eager runtime-construction fixtures,
WebUI/desktop admission expectations, a hardcoded `/tmp` socket location and
terminal/process permissions. Diagnose and migrate these tests against the
accepted composition/foreground contract; do not resurrect disabled automation
or weaken guards merely to make old tests pass.

There is still no current passing single default whole-suite run. The previous
whole-suite timeout remains incomplete. Full-suite validation, source/stage
review, applicable local Q2-Q5 and retained native/request gates remain open.
Prior deferrals and the unaccepted model overlay are unchanged. CI remains
deferred; no deployment, provider/channel activation or Git publication occurred.
