# A3-2 implementation packet — Runtime and process ownership

**Status.** Ready for packet review. Dispatch after the A3-1 authority and outcome criteria pass.
[A3-1 handoff](A3-1-authority-foundations.md) · [Sandbox and process findings](../module-review-2026-11-06/reports/S02.md) · [Lifetime findings](../module-review-2026-11-06/integrated/I02-durability.md)

## Objective and boundary

Give every admitted command one owner from reservation through cleanup.
Make a hard timeout independent of polling, streaming, and notifications.
Keep application-level path policy, sandbox confinement, and process-tree ownership as separate checks.
Use the existing `RuntimeComposition`. Keep `ManagedProcessRuntime` responsible for detached gateway services.

## 1. Cut over the canonical exec path — R01, R09

Start with the real `exec` and `exec_session` tools. Apply the changes to foreground commands and yielded sessions.

- Take ownership immediately after native admission reserves a launch. Release a reservation when the owner can prove the process did not start.
- Keep an uncertain launch under the same owner. Transfer the reservation once to the live handle when the launch succeeds.
- Count native work once per durable owner epoch. Keep each admitted handle's stdin and cleanup proof.
- Preserve the input snapshot and its second policy check. Refuse a denied command before opening the launch gate.
- Schedule the deadline when the launch becomes live. Use the same deadline for output collection and yielded sessions.
- Retain one termination task for concurrent Stop, deadline, and shutdown requests. Keep notification as an optional consumer of its result.
- Distinguish launch rejection, running, root exit, timed out, and tree quiescence. Keep ownership when the backend cannot prove quiescence.
- Use bounded chunk reads for stdout and stderr before buffering. Preserve the current head/tail output and truncation contract.
- Keep `timeout=0` unlimited. Keep the caller's yield and poll timeout independent of the command deadline.

**Edit boundary.** `nanobot/agent/tools/shell.py`, `nanobot/agent/tools/exec_session.py`, `nanobot/agent/tools/process.py`, `nanobot/agent/tools/native_process.py`, and `nanobot/agent/tools/native_process_supervisor.py`.
Use the existing admission owner in `nanobot/security/native_admission.py` for reservation transfer and settlement.
Use `nanobot/security/desktop_permissions.py` only for the owner-epoch count projection.
Keep the handle at this boundary. Add a terminal-specific entry point only with a current terminal caller.

**AC2.1.** A denial during input revalidation starts no process and leaves no starting reservation.
A spawn failure releases ownership only if the owner can prove that no child started.
Cancellation after a possible launch preserves recoverable ownership.
Two derived requests for the same owner see one aggregate count without losing per-handle cleanup.

**AC2.2.** A yielded process with notifications disabled reaches its hard deadline without a later call.
Concurrent Stop and shutdown requests settle it once. A failed kill leaves an inspectable owned result.
The unlimited and ordinary poll paths retain their behavior.

**AC2.3.** Flood both pipes with a fixture command. The returned output reports truncation and the retained buffers stay within the configured limits.
Run the test with a child that outlives the root. Completion is quiescent only after the backend proves that the owned tree has stopped.

## 2. Admit the supported confinement modes

Record the required modes and the proof for each executor OS before implementing a new backend.
Qualify native execution, descendant settlement, and managed confinement separately.
Keep a missing required backend as a stage gap.

Resolve the executor OS from the running gateway. Select the sandbox from the admitted request and the backend's actual capability.
Reuse the managed Linux bwrap path and its live probe. Preserve its private-root masks and admitted mount list.
Keep native POSIX process groups for commands that use native execution. Use the Linux supervisor for the existing supervised managed modes.
On Windows, use a Job Object or an equivalent owned-tree backend for any new whole-tree guarantee.
Require process assignment before user code can start. Verify descendant cleanup after the root exits.
For a requested confinement mode, refuse launch if the installed backend cannot provide that mode.
Qualify native macOS and Windows separately from managed confinement. A successful native command does not qualify a sandbox.
[Sandbox implementation](../module-review-2026-11-06/reports/S02.md) · [Windows Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)

**Edit boundary.** Sandbox selection in `nanobot/agent/tools/sandbox.py` and the shell's existing admission branch. Put OS-specific launch code in the process owner.
Keep compatibility for explicitly native callers. Preserve live grant checks before writes and stdin delivery.

**AC2.4.** The same managed request uses the configured sandbox when its probe passes.
When the backend is unavailable, the request returns a refusal and launches no command.
Verify POSIX and Windows claims with real host-specific process tests.

## 3. Bind the owner to composition and move the bounded consumers

Acquire the exec owner through the existing graph used by gateway, CLI, and SDK. Register its close action before a later acquisition can fail.
Preserve reverse-order cleanup and the original error with any cleanup failure.
Keep failed cleanup reachable for the owner to retry.
Make SDK close calls join the same close attempt and retain a failed close for retry.
Place the interactive CLI cleanup scope immediately after acquisition.

After exec passes its criteria, migrate only the process paths with the same ownership gap.

| Consumer | Cutover |
| --- | --- |
| MCP stdio | Keep the MCP connection owner and protocol. Use the process owner where the transport exposes a supported launch and close seam. A3-6 owns publication and draining of connection generations. |
| WebUI skill installer | Use the owned process path for installer timeout and cancellation. Preserve the existing install authorization. |
| CLI Apps | Use bounded reads and owned cleanup for execution. Keep the existing CLI Apps adapter and output contract. Enable restricted use only with qualified confinement. |

Keep VPN thread lease cleanup with its VPN owner. Keep systemd status reporting with its deployment owner.
These paths have separate owners even when a failure looks like process cancellation.

**Edit boundary.** `nanobot/runtime_composition.py`, `nanobot/agent/loop.py`, `nanobot/nanobot.py`, and the interactive acquisition in `nanobot/cli/commands.py`.
The adapter write set is limited to the launch/close paths in `nanobot/agent/tools/mcp.py`, `nanobot/webui/skills_marketplace.py`, and `nanobot/apps/cli/service.py`.
If the installed MCP transport lacks the required seam, retain its existing close owner and record the limitation before changing that transport.

**AC2.5.** A later startup failure closes each acquired resource in reverse order.
Concurrent SDK close callers observe one result. A failed cleanup remains retryable by the same owner.
Each migrated adapter cleans up on timeout and cancellation. A CLI Apps flood stays within its output limit.

## Verification and rollback

Use `tests/security/test_native_admission.py`, `tests/agent/test_native_process_io.py`, `tests/tools/test_exec_session_tools.py`, `tests/tools/test_process.py`, `tests/tools/test_exec_platform.py`, and `tests/tools/test_sandbox.py`.
Use the composition and shutdown cases in `tests/migration/test_p5_stage3_composition.py` and `tests/migration/test_p5_stage3_shutdown.py`.
For the adapter slices, use `tests/webui/test_skills_marketplace.py` and `tests/cli_apps/test_service.py` with the existing MCP connection tests.

Record the backend, OS, root result, and tree proof for each claimed capability.
Keep S02-F03 lock-scope cleanup separate unless the change needs it or measurement shows contention.
Revert each adapter independently. Stop new launches and settle existing handles before reverting the owning runtime.
Return the qualified confinement matrix and the process handle contract to A3-3, A3-5, and A3-6.
