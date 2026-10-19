# P6 release and copied-state rehearsal

Current authoritative verdict and finite gates:
[P6 completion checklist](p6-completion-checklist.md), **P6 NOT_ACCEPTED**.
Dated evidence below is retained history, not a current corrected-source release
or production qualification. The test-only foreground slice is described next;
it does not add a runtime bypass to the release helper.

Latest [authorized follow-up](p6-cron-coverage-checkpoint.md) closes both cron
coverage findings and begins the next strict-source qualification stage with a
behavior-neutral checked-session annotation repair. Root final selection passes
943 tests with 11 retained warnings; unchanged strict still FAILS at 3,977.
Earlier CORRECT/4,122 records below remain historical, not current blockers of
those two now-closed findings. Full P6 and all activation gates remain unaccepted.

## Dedicated isolated foreground rehearsal (2026-10-02)

[`test_p6_foreground_rehearsal.py`](../../tests/migration/test_p6_foreground_rehearsal.py)
loads synthetic JSON through real `load_config`/`set_config_path` with a sanitized
environment, explicit workspace/source-parent runtime paths, exact Config class,
Heartbeat/Dream/idle/TTL disabled and no real user config. An absent synthetic home
is used only for default-workspace comparisons; production paths remain unread.
Real RuntimeComposition, AgentLoop, runner and durable session ownership perform
clean acquire, exactly one scripted foreground turn, fake completion-sink delivery,
cooperative close and a new composition reopening the same accepted source history
without re-execution. Config bytes remain unchanged.

The positive test temporarily injects only the migration guard and empty
ToolLoader discovery, with invocation/empty-registry and restoration assertions.
The scripted provider supplies its supported local token estimator and synthetic
usage so no optional tokenizer assets are fetched. Ordinary guard-negative startup
runs before/after that context and in a separate unpatched test. Effect traps and
OS network/home denial remain active; positive effect-attempt inventory is empty.
Cron is definition-only, no runtime timer is injected, no channels are constructed,
and legacy AutoCompact, idle integration and agent background work remain absent.

This is **not** actual CLI/systemd or Telegram transport activation, real
provider/tool/plugin startup, native/production-target qualification, truthful
stopped-production-copy rehearsal, or immutable previous-reader rollback proof.
The existing state helper still proves relocation/export only. The release script
has no public guard-patch/foreground-bypass flag. Exact current evidence and all
unpassed gates are recorded once in the completion checklist.

The bounded existing API/CLI compatibility fixture repairs were conditional on
an independent frozen replay of the same six failures. Real temporary config JSON
loading and public source-path binding restore the missing fixture context; all
original assertions and production admission remain intact. The complete seven-file
rehearsal/composition/API/CLI selection now passes100; the prior94pass/sixfail result
and separate frozen-sixfail/sixpass evidence remain recorded, not overwritten.

Status: **rehearsal tooling implemented; release gate remains open**. This
document records the bounded checks that can run without credentials, live
state, service control, network access, or a production deployment. It does
not authorize removing the migration guard or activating a canary.

The executable helper is
[`scripts/p6_release_rehearsal.py`](../../scripts/p6_release_rehearsal.py). It
has two deliberately separate responsibilities:

1. inspect a wheel or sdist without extracting or executing it, and record its
   SHA-256, size, required package assets, dependency inputs, and forbidden
   local-state members;
2. run a disposable copied-state rehearsal using the existing session and
   stopped-writer cron conversion APIs.

The state rehearsal imports its session and cron dependencies before entering a
temporary effect guard. The guard blocks and records network sockets, process
launch, browser launch, and shell/system seams; a recorded attempt makes
`external_effects` nonempty and fails report validation. Provider and channel
namespaces are outside the rehearsal import allowlist, so an import attempted
after the guard is also recorded and stopped. `--state-rehearsal` creates a
temporary directory and removes it when the process exits. Archive inspection
is read-only.

## Release inspection contract

`inspect_archive()` accepts `.whl`, `.zip`, `.tar`, `.tgz`, and `.tar.gz`
archives. It rejects duplicate names, absolute or parent-traversing paths,
symlinks in wheels, symlinks/hard links in sdists, credentials and environment
files, tests, and runtime state directories. It reports missing required
members instead of silently treating a partial archive as complete.

Both archive types must contain the Python package, retained templates, the
five custom skills and the video extractor, crawler worker, channel-owned
WebUI sources, and `nanobot/web/dist/index.html`. Sdists additionally require
the build hook, project manifest, README, license, and third-party notices.
Wheels require `METADATA` and `RECORD` below the canonical
`<distribution>-<version>.dist-info/` root derived from the wheel filename.
The inventory and SHA-256 are taken from one bound file snapshot; before and
after checks reject concurrent archive replacement or mutation.

`inspect_dependency_inputs()` records every dependency and optional extra from
`pyproject.toml`, plus a SHA-256 of that manifest. It reports whether `uv.lock`
exists and whether the repository ignores it. A local ignored lockfile is not
treated as a reproducible release input: the accepted artifact still needs a
tracked or separately archived resolved dependency inventory with hashes for
the production and enabled channel requirements.

`inspect_service_unit()` checks the existing headless service contract without
starting it. The unit must retain a foreground `nanobot gateway`, explicit
`--config`, `--log-file`, `EnvironmentFile`, and working directory. Frontend
setup, package installation, WebUI startup, service-time mutation markers, and
wrapper commands that merely pass `nanobot gateway` as an argument are
rejected. Empty `EnvironmentFile=` and `WorkingDirectory=` assignments reset
their effective values.

Run the read-only inspection after building an artifact in a separate release
workspace:

```sh
python scripts/p6_release_rehearsal.py \
  --artifact dist/nanobot_ai-<version>-py3-none-any.whl \
  --artifact dist/nanobot-ai-<version>.tar.gz \
  --service-unit deploy/systemd/nanobot-gateway.service \
  --project-root . --json
```

The command exits nonzero when a required archive member or service contract
is missing. It does not build the frontend or install Python dependencies.

## Copied-state and rollback contract

The temporary rehearsal covers the migration boundaries that must be safe to
repeat:

- a legacy in-workspace session is copied to the private canonical session
  root, and a second construction observes the same digest;
- a new message written after conversion is exported back to the legacy
  workspace for a bounded rollback rehearsal while the canonical copy remains
  present;
- the pre-conversion session backup remains byte-for-byte unchanged;
- legacy cron action and job files are converted through
  `CronActionStore.convert_legacy()`, content-addressed backups match the
  original bytes, and a second conversion returns the same result;
- the legacy jobs snapshot remains available as conversion evidence; no
  delivery, scheduler, repository push, or process starts.

Run it with:

```sh
./.venv/bin/python scripts/p6_release_rehearsal.py --state-rehearsal --json
```

The current deterministic result is:

```text
session.conversion_runs_equal       true
session.rollback_retained_new_write true
session.canonical_copy_retained     true
session.rollback_restored           true
session.backup_unchanged            true
cron.conversion_runs_equal          true
cron.legacy_backup_matches_source   true
cron.source_jobs_retained            true
external_effects                    []
```

The rehearsal validates only session relocation/export while retaining a new
write. It does not prove that a previous application version can load the
exported state, perform a schema downgrade, or restore an old snapshot over
active state. Cron rollback remains open, as does post-conversion retention of
occurrence and receipt records. Restoring an old snapshot over active state
would lose accepted writes and could repeat external work.

## Evidence run on 2026-09-15

The new focused tests pass:

```text
./.venv/bin/pytest -q tests/migration/test_p6_release_rehearsal.py
20 passed
```

The combined release boundary suite also passes:

```text
./.venv/bin/pytest -q \
  tests/webui/test_build.py \
  tests/cli/test_startup_service_compatibility.py \
  tests/deploy/test_deployment_bundle.py \
  tests/migration/test_p1_foundation.py \
  tests/migration/test_headless_startup_compatibility.py
81 passed
```

Static validation passes:

```text
./.venv/bin/ruff check \
  scripts/p6_release_rehearsal.py tests/migration/test_p6_release_rehearsal.py
All checks passed
```

The focused and combined commands were also run under the repository's
macOS `sandbox-exec` profile denying non-loopback network access:

```text
focused P6 suite: 20 passed
combined release boundary suite: 81 passed
```

The helper itself reports the current dependency-manifest SHA-256 as
`9762830b66b6aae2946144178cd1ed601520fa26ff83acb828d46eedc0d039b1`.
`uv.lock` exists in this checkout but is ignored by `.gitignore` and is not
tracked (`git check-ignore -v uv.lock` confirms the rule). Its presence does
not close the dependency-freeze gate.

## Checks still blocked by this offline environment

The real archive and clean-environment checks have not run:

- `.venv/bin/python -m build --version` fails because the `build` module is not
  installed;
- `webui/node_modules` and `nanobot/web/dist` are absent;
- Bun and npm executables are present, but installing frontend dependencies
  would require a network-backed mutation, so neither was run;
- no wheel or sdist SHA-256, resolved wheelhouse, channel-manifest lock, or
  clean `--no-index` installation evidence exists yet;
- consequently installed-wheel skill discovery, crawler entrypoint imports,
  GUI asset presence, and runtime-network independence remain release gates.

When an offline wheelhouse and frontend dependency tree are supplied, run the
Hatch build from the accepted SHA, inspect both archives with the helper, then
install the wheel in an empty temporary environment using only local artifact
paths and `--no-index`. Exercise `nanobot`, `nanobot gateway --help`,
`nanobot agent --help`, custom skill discovery from outside the repository,
and the guarded headless startup matrix before recording those results here.

## Documentation and source contracts consulted

- [Nanobot upstream migration plan](../nanobot-upstream-migration-plan.md),
  especially P6, release reproducibility, copied-state rehearsal, and rollback
  sections;
- [`pyproject.toml`](../../pyproject.toml) and
  [`hatch_build.py`](../../hatch_build.py) for package inclusion and WebUI
  build-hook behavior;
- [`deploy/systemd/nanobot-gateway.service`](../../deploy/systemd/nanobot-gateway.service)
  and [`deploy/README.md`](../../deploy/README.md) for the retained service
  command and deployment boundary;
- [P4 artifact contract](p4-artifact-offloading-contract.md) and [P3 cron
  action-store contract](p3-cron-action-store-contract.md) for private state,
  content-addressed conversion evidence, and rollback constraints;
- existing session-location, cron-conversion, WebUI-build, deployment-bundle,
  and startup compatibility tests.

Assumptions are limited to the accepted source contract: the release owner
will provide a separately frozen dependency artifact before release, and the
actual live installation remains unknown until a read-only production
preflight is authorized. No live canary or deployment was performed.

## 2026-10-02 — P6 IN_PROGRESS — P6-A ACCEPTED

The preceding 2026-09-15 build-blocked observations and old state-rehearsal output
are historical, not current installed/runtime qualification. **P6 IN_PROGRESS —
P6-A ACCEPTED**:
actual immutable dirty-source, frontend, wheel and sdist build/inspection have
root bounded AC1–AC5 PASS and fresh independent review PASS/no findings. Exact
review/root verdict and pre-final scope/publication evidence are retained; root final
publication recheck is not yet recorded. See [checkpoint](p6-artifact-checkpoint.md) and
[permanent source/artifact/command evidence](p6-artifact-evidence.json).

Fresh isolated pinned build-tool acquisition and frozen-lock `npm ci --ignore-scripts`
preceded real OS-network-denied TypeScript/Vite and original Hatch hook builds.
The first sdist's 910 npm-member packaging leak remains retained red; one bounded
root-anchoring correction to its seven existing include entries and one added
red-green regression produced actual archives with zero missing/forbidden members.
No dependency cleanup, synthetic asset, lock/hook/runtime/default change or artificial
VCS tree bypassed the test. The exact wheel and corrected sdist, complete manifests/
hashes and source snapshot are retained read-only in the fresh private bundle.

Corrected source tests passed 32 under full OS network/process denial; the exact
version subprocess node passed separately under the unchanged local-process profile
with all network denied. Root repeated both, separately from historical 31+1 evidence.
Service inspection is read-only; no deploy/update/service operation ran.

Full P6 **NOT_ACCEPTED**, runtime **NOT_READY/NOT_AUTHORIZED**, D9
**NOT_AUTHORIZED**; P5 stays DONE and P7/H1/P8 are not delivered. Old state rehearsal
remains **UNQUALIFIED/NOT_RUN** here. Clean installed artifact/dependency closure,
channel extras, typing/DOCX, current copied-state/config/service-start/old-reader
rollback, native targets and live inventory remain distinct required gates.

## 2026-10-02 — host installation qualification accepted

The preceding body is immutable historical evidence. Current
[host installation checkpoint](p6-installed-checkpoint.md) and
[hash-bound evidence](p6-installed-evidence.json) record a fresh macOS arm64/
Python3.13 core+api+dev114-wheel freeze, offline hash-required install, pip check,
installed imports/package bytes/assets/skills/channel inventory/help and unchanged
guard-negative startup. Exact unchanged installed DOCX mock-agent fixture passes
under separately authorized localhost-only isolation and public disposable instance
binding, with worker/root runs recorded separately. Strict BasedPyright ran and
FAILS with7420 errors;56 direct missing optional imports do not waive the other7364
diagnostics. Failed profile/launcher/home-denial attempts remain retained red.

HOST_QUALIFICATION_ACCEPTED: root bounded AC1–AC5 PASS and fresh review PASS/no
findings are retained in `final_acceptance_evidence`; final root publication recheck
is not yet performed. Full P6 NOT_ACCEPTED, P5/P6-A
unchanged, runtime NOT_READY/NOT_AUTHORIZED, D9 NOT_AUTHORIZED. No current-state/
operator/init-only-loader/old immutable rollback reader, production subset, native
platform or positive service-start qualification is supplied; no background work
or activation occurred. Historical denied-loopback/type/install records above are
not retroactively green.

## 2026-10-02 — full extras/channel host closure; strict source blocker

[Full host environment evidence](p6-full-environment-checkpoint.md) advances the
dependency gate:210 wheels cover all ten extras and17 channel manifests; offline
hash-required install, dependency consistency and independent root input verification
PASS. Unchanged strict checking reports5970 errors/zero missing imports; identical
pinned-upstream checking reports zero. These need substantive bounded source repairs;
no runtime source fix or new artifact acceptance is claimed. Earlier results remain
history, not relabeled green.

Full P6 stays NOT_ACCEPTED. Native/platform qualification, owner-selected production
profile, approved immutable old rollback release, truthful stopped-state copies/
backups/reference config/artifact provenance, actual loader policy and positive
isolated rehearsal remain required. No live access, guard change, activation or
deployment occurred. See the checkpoint's repair scope and bound raw records.

## 2026-10-02 — bounded source type/call repairs accepted

[Repair checkpoint](p6-type-root-repair-checkpoint.md) records root/fresh review
PASS for journal/decorator declarations, eight throw-only recovery annotations,
provider positional-call parity and one independently reproduced stale symlink-test
expectation. Root409 session/recovery and207 provider cases pass; unchanged strict
checking still FAILS at4815 errors (net1155 fewer; zero new outside owned files).
Only the bounded source wave is accepted. Prior exact artifacts remain historical,
not rebuilt or relabeled as the current corrected source. Full P6 NOT_ACCEPTED;
native/production/stopped-copy/loader/rollback/rehearsal gates remain open. No guard,
activation, live-state, deployment or external publication action occurred.

The later root current-source fan-in is recorded in the
[completion checklist](p6-completion-checklist.md): unchanged strict FAIL at4,122
raw diagnostics, not4,122 independent defects. The newly passing synthetic-loader/
foreground-close/restart slice narrows two offline rehearsal gaps; it does not
retroactively qualify prior artifacts or waive native/current-production/old-reader
inputs. Production guard and checker policy remain byte-identical.

## 2026-10-02 — final bounded diagnosis review CORRECT

The [single completion checklist](p6-completion-checklist.md#final-bounded-root-verdict-and-private-qa-publication)
links the private allowlisted root records: fullcron+Stage3 764PASS/eight retained
warnings/zero exclusions, offline rehearsal100PASS, recovery930PASS/two existing
skips, scopedlintPASS and 20-file compilation STRUCTURAL_PASS. Integrated review
is **CORRECT**, not PASS: requested schedule/unaffected-field assertions and a
positive exact-token delivery-target mutation remain two P2 AC3 test-only gaps.
No third correction is authorized. QA records describe pre-publication reviewed
implementation inventories, not a final source freeze/release artifact. Strict
FAIL4122, S3 FAIL and full P6 NOT_ACCEPTED remain; runtime/D9 stay unauthorized.
