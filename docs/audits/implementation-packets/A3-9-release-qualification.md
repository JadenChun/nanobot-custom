# A3-9 implementation packet — Release and production qualification

**Status.** Ready for packet review. Dispatch after A3-8 passes.
[Storage and rollback contract](A3-8-durability-retention.md) · [Release findings](../module-review-2026-11-06/integrated/I04-release-residuals.md) · [Deployment guide](../../guides/deploy-nanobot-gateway.md)

## Objective and boundary

Qualify the exact executable, bundle, mobile package, and deployment that users will receive.
Use the existing install, build, and deployment scripts.
Record one provenance tuple per artifact. Link the source, lockfiles, bundle, toolchain, artifact digest, upgrade evidence, and rollback target.
Run qualification in isolated environments before any live switch.

## 1. Verify the installed executable and built artifacts — R20

Resolve the persistent pipx executable that the installer created. Use it for version, setup, and readiness verification.
Keep the existing uv and managed-environment paths.
Install frontend dependencies from the lockfile in release builds.
Require a fresh WebUI output for the pinned source and lock. Record its digest before packaging.
Reject a successful command exit when the expected output is absent or stale.
Keep local development build modes explicit.

Validate cached mobile tools and installed Android components against the toolchain lock.
Verify the Gradle archive against the official checksum for the wrapper version.
Build native code and shared assets before exposing signing inputs.
Give signing inputs only to the signing phase. Record the signed APK digest and signer fingerprint.

**Edit boundary.** `scripts/install.sh`, `scripts/install.ps1`, `nanobot/webui/build.py`, and `hatch_build.py`.
Use `mobile/toolchain.lock.json`, `mobile/scripts/bootstrap-toolchain.sh`, `mobile/scripts/build-release.sh`, and `mobile/android/gradle/wrapper/gradle-wrapper.properties`.
Keep toolchain evidence separate from signing and bundle evidence.

**AC9.1.** Install into a clean environment. The installed executable reports the intended version and can start with the packaged bundle.
A stale or missing bundle, mismatched toolchain, or bad archive digest fails qualification.
The unsigned build cannot read signing credentials. The final signed artifact has a verified digest and signer.

## 2. Stage the deployment and prove rollback — R04, R09, R16, R21

Build the candidate in a separate release directory from its immutable source.
Verify dependencies, bundle, config, state compatibility, and readiness before switching.
Use isolated state during candidate checks. Keep a single live writer and a single active effect owner.
Retain the previous verified release pointer and its compatible state or backup.
Stop or drain the old owner before switching. Verify the live target after the switch.
If verification fails, restore the verified previous target and recheck it.
Report a failed rollback as a failure that requires operator action.

Use one local readiness contract for gateway and CLI. Reject malformed or non-object health responses.
Use the existing direct local probe for managed loopback checks.
Require actual successful service-manager results before reporting an applied operation.
Redact audit diagnostics by field. Remove credential-bearing URL data, secret arguments, and token values.

**Edit boundary.** `deploy/scripts/deploy_nanobot_gateway.sh`, `deploy/scripts/update_nanobot_gateway.sh`, and `scripts/audit_deployed_instance.sh`.
Use the owning readiness and lifecycle adapters in `nanobot/gateway/runtime.py` and `nanobot/cli/webui_support.py`.
Keep systemd outcome reporting in `nanobot/remote/supervision.py`.
Keep the real deployment behind the accepted candidate and rollback evidence.

**AC9.2.** Inject a failure before and after the switch on an isolated target.
A pre-switch failure leaves the live release intact. A post-switch failure restores the previous verified executable and compatible state.
Readiness failures and service-manager failures produce truthful results. A synthetic credential does not appear in the audit output.

## 3. Qualify the real upgrade and supported matrix — R03, R19, R21

Pin the historical verifier and dependency inputs as well as the historical source.
Run applicable CI when executable migration helpers change.
Make the parity checker cover synchronous and asynchronous production dispatch.
Verify a known command and route through the self-test.
Use two distinct APKs for upgrade. Record both hashes and verify that they differ.
Seed the baseline, install the candidate over it, and verify retained state and the approved v1-to-v2 re-pairing behavior.
Use the actual production gateway, edge, and shared router. Substitute only bounded external effects.

Qualify runtime and client targets separately.

| Target | Required evidence |
| --- | --- |
| Linux, macOS, Windows runtime | Startup, supported confinement, session reopen, native execution, child join and Stop, and applicable recovery. |
| Browser and desktop clients | Exact gateway identity, two-client control, thread and child navigation, replay, and settings readiness. |
| TUI | Admission and recovery through the production gateway. Record the host PTY or ConPTY evidence. |
| Android | Exact signed candidate, distinct APK upgrade, physical-device pairing and revoke, transport recovery, and child navigation. |
| Optional Windows crawler | Enable only after the worker authenticates the actual caller. Exercise a denied unrelated local principal. |

**Edit boundary.** `.github/workflows/ci.yml`, `docs/migration/run-p0-baseline.sh`, and the historical verifier inputs.
Use `mobile/scripts/check-desktop-parity.py`, `mobile/scripts/test-update-persistence.sh`, and the existing emulator and physical instrumentation.
Keep the optional crawler authentication slice with `deploy/scripts/setup_local_crawl4ai_profile.ps1` and `nanobot/crawler/worker.py`.

**AC9.3.** An executable migration-only change exercises CI. Altered verifier inputs cannot silently reuse historical evidence.
The parity test detects a missing async route. The upgrade test rejects identical APKs and verifies the actual state transition.

**AC9.4.** Every advertised target has evidence at the artifact digest and source revision being qualified.
An unavailable platform or skipped required check remains an explicit release gap.
The ordinary A3-0 regression comparison does not replace the full checks required by the release and migration gates.

## Verification and handoff

Start with `tests/deploy/test_deployment_bundle.py`, `tests/test_build_status.py`, `tests/cli/test_gateway_runtime.py`, and `tests/migration/test_p6_release_rehearsal.py`.
Use the actual installer, bundle, Android, and deployment commands on isolated targets.
Retain full CI and platform results, provenance tuples, upgrade and rollback receipts, and any unsupported scope.
Keep release qualification separate from live deployment authorization.
Return the accepted candidate and compatibility evidence to A3-10.
