# A3 rollback on a stopped copy

The previous release reads scheduler state versions 1 and 2. A3 can write versions 3 and 4.
Keep the previous release and its compatible state snapshot before the switch.
Keep the newer scheduler, occurrence, completion, and delivery evidence for reconciliation.

1. Stop the gateway and all cron writers. Wait for admitted work to settle or retain its uncertain disposition.
2. Verify a backup of the stopped state. Prepare a separate copy for rollback.
3. Run the candidate's session rollback command on the copy. Use the reference configuration to identify the original instance.
4. Export the copied scheduler with the candidate's offline command.
5. Verify the export digest and inspect it with the previous release. Use the matching copied occurrence store.

```text
nanobot sessions prepare-rollback --offline-copy --config /reference/config.json --copied-workspace /copy/workspace --copied-sessions-root /copy/sessions
nanobot cron-runtime scheduler-rollback-export --offline-copy --root /copy/cron/scheduler --occurrence-root /copy/cron/occurrences --namespace NAMESPACE --runtime-identity IDENTITY --destination /copy/scheduler-export
previous-nanobot cron-runtime scheduler-inspect --root /copy/scheduler-export --namespace NAMESPACE --runtime-identity IDENTITY
previous-nanobot cron-runtime occurrence-inspect --root /copy/cron/occurrences --namespace NAMESPACE
```

The export retains the generation, cursors, request order, and reconciliation bridges.
The command returns a SHA-256 digest for the exported `state.json`.
Keep the exported scheduler and its matching occurrence records together.
Preserve the newer snapshot before you use the export in the previous runtime.

If the previous format cannot represent all evidence, keep cron disabled in the rollback configuration.
Retain both the verified previous snapshot and the newer stopped copy.
Keep completed receipts, uncertain work, and device revocation records.
Use the current reader to reconcile the newer evidence before you enable cron again.
If any evidence fails validation, stop and keep the copy for inspection.

Rehearse the rollback on isolated state before you switch a live instance.
Verify that the previous release can read post-cutover histories and that revoked devices remain revoked.
Verify that the scheduler does not dispatch a completed occurrence and dispatches the next due occurrence once.
