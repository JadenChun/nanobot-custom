# Assistant prototype validation

Date: 2026-10-23. Checkout: attached `codex/complete-remaining-work` worktree,
based on `79934a5630ca419e02e124693efa938d9d3407f0`, including its preserved
pre-existing namespace-inventory repair. These results describe the working
files and packaged artifacts, not a production deployment.

| Check | Result |
| --- | --- |
| Full Python suite, four workers, excluding the separately run CLI output tests | 16,441 passed, 22 skipped, 6 subtests passed; 235.03 seconds |
| Serial CLI output tests | 128 passed; 2.98 seconds |
| Strict BasedPyright | 483 files; zero errors and warnings |
| Full lint / whitespace | Passed |
| WebUI tests | 1,186 passed across 75 files |
| WebUI production build | Passed |
| Wheel and source distribution | Built successfully; required assets present, forbidden entries absent |
| Package/source comparison | All 483 packaged Python files and the agent tool contract match the validated source; WebUI bundled |
| Fresh isolated wheel install | CLI help and new module imports passed; dependencies checked and all 17 discovered channel classes loaded |
| Copied-state rehearsal | Repeat cron conversion and retained backups passed; session conversion/rollback preserved the original backup and retained a newer write |
| Runtime boundary checks | Normal gateway due job used the canonical runner and local receipt adapter; default automation remained inactive |

The checker environment is Python 3.14.8 with BasedPyright 1.40.2. Its frozen
420-entry distribution inventory is unchanged from the initial baseline.
Each bounded batch retained before/after full-source checker reports and source
hashes. The final comparison reports `PASS_NO_NEW_DIAGNOSTICS` with zero introduced
signatures. Initial failures and subsequent repairs remain in the local evidence.

The full Python commands were:

```bash
# uv is available on PATH; COLUMNS makes exact Rich-output checks stable in
# the longer xdist temporary paths. The dependency environment is not resynced.
COLUMNS=240 .venv/bin/python -m pytest -q -p no:cacheprovider \
  -n 4 --dist loadfile --ignore=tests/cli/test_commands.py \
  -o faulthandler_timeout=90
COLUMNS=240 .venv/bin/python -m pytest -q -p no:cacheprovider tests/cli/test_commands.py
.venv/bin/basedpyright --outputjson
.venv/bin/ruff check nanobot tests conftest.py scripts/import_cron_copy.py
git diff --check
```

Frontend and package checks used `bun install --frozen-lockfile`, `bun run test`,
`bun run build`, and `uv build`. The built wheel was installed in a separate
environment with the channel-manifest dependencies, then checked from outside
the source checkout. `python -m scripts.p6_release_rehearsal --state-rehearsal`
checked both final archives and exercised copied state. Configuration compatibility
and memory recovery are also covered by the full Python suite.

The source-manifest SHA-256 is
`f6b3bf2c98ef251409afb7f14a6fcb97520315d659055860c8884342da43b4d2`.
The final wheel SHA-256 is
`24de202fe62ab9b41302a97ee0a37cfc5363d6c18e464e623dc2ae603599ac64`.
The detailed local record is under
`.assets/validation/remaining-work-2026-10-22/`, including full reports, source
manifests, frozen environment, test logs, package inventory and rehearsal output.

The first broad run exposed failures also reproduced on the original checkout:
access-time-sensitive assertions, a container-specific traceback path, obsolete
registration-guard expectations, shell-path fixture assumptions and follow-up
tests that no longer isolated their intended republication boundary. Repairs
preserve the content/identity checks, exact shell command allowlist and existing
activation guards. The small-context replay test now pressures the current-user
delta while allowing the archive request to fit; it still proves a real checkpoint
and exact preservation of the current question. Crawler tests use the explicit
extension runtime interface. A local CLI preflight now reports unsupported idle
configuration before attempting provider setup.

No live provider, production client delivery, production service change or
production repository push was used to establish these results. Live deployed
identity/configuration drift and each actual marketing workflow still require
the isolated rehearsal and cutover procedure in the
[integration guide](assistant-prototype-integration.md). Native Windows and
Python 3.11 runs were not performed in this Linux/Python 3.14 environment. No
production performance improvement is claimed from synthetic/local checks.
