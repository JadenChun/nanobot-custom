# Offline pending-memory-receipt coordinator

`nanobot.agent.memory_recovery_coordinator.reconcile_pending_memory_receipt_offline(
workspace, *, sessions_root)` is an explicit operator-only API. It is not called
from gateway/CLI startup, `MemoryStore`, or `SessionManager`, and must not be
used while a live writer owns the workspace. A pending receipt still blocks
ordinary construction after this API returns.

The API validates the existing no-follow receipt, obtains `session_key` from
that receipt, locates the existing canonical recovery session, and uses the
checked workspace marker's `workspace_id`; callers cannot supply a namespace
ID. It validates the checked source before invoking stages. An absent or
invalid receipt, missing/unsafe/mismatched session, malformed or conflicting
manifest, and unsupported `preserve_nonempty_jsonl` disposition raise a fixed
content-free `OfflineMemoryRecoveryError`. Such cases require operator
reconciliation; the coordinator neither unlinks nor quarantines anything.

The durable manifest prefix selects the next stage: conversion preparation,
target application, archive/finalization, transaction preparation, history,
memory, source offset, then terminal claims. A completed conversion alone
never authorizes receipt disposal. Each stage revalidates its evidence and
holds its own workspace lock before checked-session locks. The coordinator
releases its short inspection locks before calling a stage, performs a
bounded number of advances, and uses the terminal stage's full proof even on
transaction `COMPLETE` replay. A successful call returns the transaction
`COMPLETE` record with the pending receipt retained unchanged.

Receipt finalization remains a separate decision gate: before any deletion,
the receipt-disposition policy and crash durability of that disposition must
be specified and tested. This package does not activate recovery at startup.
