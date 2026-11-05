# macOS reliability repair contract

User-authorized follow-up: repair F-005, triage the 22 retained backend failures,
and validate pairing, revocation, restart, permissions and the full backend suite.
Earlier batches and unrelated changes remain intact on `codex/feature-audit-repairs`.

Implementation uses the direct loop. A fresh read-only review is required after
integration because peer authentication changes. No material product decision
is pending: same-user authentication remains mandatory, and Linux-only native
supervision/confinement remains unavailable on macOS.

## Acceptance

1. Read the peer UID from kernel credentials on Linux and macOS. Reject another
   UID, malformed/unsupported credentials and failed credential reads before
   dispatch. Keep socket privacy and admin/pairing role checks.
2. Real local IPC must pass pairing, token validation/revocation, owner loss and
   restart tests, including remote WebUI settings and linked-file workflows.
3. Classify every retained failure. Correct platform assumptions in tests at
   their owning fixtures; never introduce a production fallback for unavailable
   whole-tree proof or sandboxing. Linux-specific process tests must say why
   they require Linux. Cross-platform denial tests continue running.
4. Make response-bound and binary-preview fixtures test their actual contracts
   without exceeding host filesystem limits or treating supported images as
   unsupported binary files.
5. Run the full backend suite without failure-based exclusions. Record skips,
   failures and any limits honestly. No clean-code or P6 claim follows from
   repairing only this selection.
6. Preserve a frozen before/after full-source checker report, source copies and
   hashes, identical packages/checker policy, zero new diagnostic signatures
   across all files, affected tests, lint and diff checks.
7. Obtain a bounded fresh review of authentication, test fidelity and relevant
   failure handling before final acceptance.

## Evidence and implementation boundaries

Baseline: `/tmp/nanobot-macos-repairs-20261105.s00w7c6h`; pointer:
`.git/macos-repairs-evidence`. Full unchanged BasedPyright 1.40.1 reports
**1,001 errors over 537 files**. The original report is immutable. Full test
collection succeeds with **17,039 tests** before this batch.

Production changes belong to the owner IPC boundary and any demonstrated defects
found by validation. Tests may use explicit capability/host fixtures when they
exercise policy serialization without launching a process. Actual process
lifecycle tests retain real supervision requirements. Platform support must not
be fabricated by globally enabling capabilities in shared fixtures.

Authoritative Darwin ABI evidence: the installed MacOSX27.0 SDK's `sys/un.h`
defines `SOL_LOCAL=0`, `LOCAL_PEERCRED=1`; `sys/ucred.h` defines version-0
`xucred` with effective UID, group count and 16 group IDs. A real local socket
returns the expected 76-byte structure. The Python runtime exposes
`socket.LOCAL_PEERCRED` but no `socket.getpeereid`.

No new native sandbox/supervisor backend, persistent schema migration, broad
refactor, dependency update, commit or publication is part of this batch.

## Fresh review routing

Role: `development_analysis_medium`; model/effort: `gpt-6.1-sol` / `max`.
Authentication changed, so the verification reference requires an isolated,
read-only review. Its bounded output is authentication/test-fidelity findings
and AC1–AC7 coverage. Reviewing the completed implementation while the full
suite runs overlaps independent checks and avoids delaying the final gate.

Implementation is complete under the direct loop. The review starts graph mode;
the root is read-only for task files until its terminal verdict. Corrections,
if required, receive an explicit write owner. Final acceptance waits for full
suite completion and the fresh review; no passing selection substitutes for it.

## Final bounded acceptance

AC1–AC7: **PASS**; root verdict `root-final-verdict.json` closes the graph as
`CLOSED_PASS`. The fresh reviewer reports no material unresolved findings.
The exact retained selection is **15 passed, 7 Linux-only skips**. The default
backend suite completed with **17,001 passed, 75 skipped, 42 warnings and
6 passing subtests**, exit 0. The first failed full run and its transient
dependency mutation remain retained; the original 183-package inventory was
restored and unchanged throughout the accepted run.

The full run preceded the final peer-test portability correction. That fixture
subsequently passed 19 normal cases and 18 cases with one real-IPC skip when
`getuid` was absent; root independently repeated all 19 normal cases. All 537
production checker inputs, other sources, 590 assets and 183 packages still
match the passing run. The full suite was not repeated after this test-only
change. Full strict remains 1,001 errors with zero new diagnostic signatures.

See [results and limits](./macos-repair-results.md) and the
[evidence snapshot](./macos-repair-snapshot.json). Acceptance evidence was
captured before commit packaging. P6, native platform qualification and the
architecture audit remain open.
