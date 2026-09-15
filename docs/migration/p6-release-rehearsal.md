# P6 release and copied-state rehearsal

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
