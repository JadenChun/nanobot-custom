# P3 cron migration checkpoint

The data adapter restores ordered/deduplicated additional destinations, historical run/occurrence identifiers, and canonical run status. Store/action decoding and explicit service serialization preserve those fields and execution mode. Action decoding does not mutate its input.

Service preparation now uses `prepare_staging_job`: a legacy record without an explicit binding gets `cron:{id}`, staging cli/direct route defaults, and `legacy_fresh` mode. Complete existing bindings retain their mode and origin. Partial/conflicting bindings and unsupported modes remain unchanged with a separate admission diagnostic. Migration does not rewrite enabled/next-fire/timestamps/history/delivery fields. Explicit user updates and enable/disable operations retain their ordinary scheduling changes.

Binding completeness is distinct from execution admission. Until canonical fresh-history execution and delivery adapters are installed, legacy fresh jobs and retained legacy delivery policy are fenced. Diagnostics do not replace historical last_error. Recompute, next-wake, timer selection, manual/forced run, direct execution, gateway callback, and bound runner enforce admission. WebUI run returns409 with the diagnostic instead of reporting a task accepted when it cannot execute. Complete modern jobs and system events retain execution.

Validation: new admission regressions first reproduced destructive load behavior. The final focused admission/field selection passes25 tests with network denied; Ruff passes for cron, gateway runtime, WebUI HTTP handler and the new tests. Coverage includes repeated load, one-shot preservation, no callback/run records, mixed blocked/system due jobs, action merge/update/enable, each retained delivery category, unsupported modes, runner rejection before side effects, and actual HTTP handler409, and extracted real gateway callbacks named dream/heartbeat rejecting before any agent/model/delivery access. Review found and closed the callback ordering defect by moving admission before all name-based branches.

The wider selection of service/persistence/session-delivery/delivery/admission/field tests yielded89 passed and8 failed before the final12 admission cases were added. The unchanged upstream baseline failures are:

- test_unbound_agent_jobs_are_disabled_on_add
- test_unbound_agent_jobs_are_disabled_on_load
- test_add_job_migrates_legacy_delivery_context
- test_load_store_migrates_legacy_delivery_context
- test_load_store_disables_malformed_legacy_payload
- test_list_bound_agent_jobs_includes_migrated_legacy_delivery_payloads
- test_channel_meta_and_session_key_survive_store_reload
- test_update_job_migrates_legacy_delivery_target

These assert destructive disable/route clearing or origin-session adoption that conflicts with preservation requirements. They remain recorded failures, not a green full-suite result. Their migration-aware replacements and final baseline reconciliation remain required.

Remaining implementation: canonical fresh-session restore under the turn lock, no prior checkpoint/followup reuse, canonical outcome and explicit-message ledger, suppressed ordinary streaming/final publication, one computation per stable occurrence, destination fanout/dedup, durable receipts and uncertain-send reconciliation, concurrent edits/deletes without resurrection, callback lifecycle propagation and complete cron-tool fields. Fresh input must still preserve run detail history. Do not create a parallel legacy runner or activate legacy jobs before these contracts are integrated. Full migration remains guarded. No live cron state was read or changed.
