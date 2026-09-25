# Forward upstream refresh trial (candidate, not accepted)

The protected integration checkout remains at `6c4f2df574c10eea814b8ceaa139d6126bd3847d`. The independent `codex/harness-migration-candidate` began there and carried the two previously untracked terminal-recovery files byte-for-byte, plus the accepted H1 conversational-control design without replacing newer integration test evidence. Its first checkpoint is `36df65200a3e8e2366eba3041c2ef374e22d9240`. The candidate also fixes a source `/new` quiescence deadlock and updates a synthetic turn-bridge fixture in `49f4600e`; this is separate from, and does not accept, the upstream merge. Neither original checkout was edited for this trial.

## Frozen trial inputs and result

- Integrated migration base: `f47b8f23155ddec23cfb80ec001b544d495df89c` from official upstream.
- Forward snapshot: `9756468a8cb475dbf685f64837c0e0ecdac9ad10` from official upstream `main`, verified by `git ls-remote` before fetch. This is a snapshot, not a moving target or an approved replacement for the pinned base.
- Delta from the pinned base: 255 upstream commits; 862 changed paths with about 71,661 inserted and 22,066 deleted lines. The integrated candidate has 264 commits not reachable from that upstream snapshot.
- `git merge-tree --write-tree --name-only HEAD upstream/main` predicted 35 content conflicts. A real `git merge --no-commit --no-ff upstream/main` in separate worktree `codex/harness-upstream-trial` reproduced all 35. The trial is deliberately uncommitted and unresolved; the candidate branch remains on the prior base.
- The merge staged 827 automatically combined paths. **Automatic merging is not behavioral acceptance.** It includes prompt, channel, provider, session, CI, build and WebUI changes that still need contract review even if they have no textual conflict. Do not use the trial as a runtime or deploy it.

Conflict owners for review (35 paths total):

| Boundary | Conflicted paths | Required review |
| --- | ---: | --- |
| Repository docs/ignore | 2 | Preserve fork artifact exclusions and correct current deployment guidance. |
| Agent context/loop/memory/runner | 4 | One turn owner, mandatory policy, source-event and recovery integrity, context-budget semantics. |
| Agent tools | 4 | Registry/dispatch capability checks, shell and filesystem scopes, cron delivery. |
| API | 1 | Preserve public request/response and stream rejection contracts. |
| Telegram and WebSocket | 3 | Sender/chat authorization, media/reply routing and channel compatibility. |
| CLI/startup | 3 | Preserve headless defaults and no implicit listeners/installers. |
| Cron | 2 | Origin/destination conversion and occurrence/delivery identity. |
| Providers | 7 | Config/fallback/auth/state compatibility and no duplicated retries/effects. |
| Session | 2 | Source/cursor CAS, queued work and recovery ownership. |
| Utility | 1 | Preserve evaluator contract. |
| Tests | 6 | Reconcile both sides' assertions; do not drop failing fork contracts. |

No blanket `ours`/`theirs` resolution is permitted. Resolve a boundary against the 50-finding audit and 158-entry inventory, add differential tests for semantics, then run both imported upstream and fork tests. Review nonconflict paths too. A successful merge commit alone is not a source-migration milestone.

## Candidate evidence before accepting an update

- `pytest -q tests/agent/test_memory_recovery*.py`: **691 passed, 1 skipped**. The terminal stage is directly tested but is not yet wired into the complete startup recovery path; these results do not prove end-to-end conversion or live-data safety.
- `pytest -q tests/agent/test_migration_source_engine.py tests/agent/test_migration_bus_admission.py tests/agent/test_migration_foreground.py tests/agent/test_migration_canonical_admission.py`: **95 passed** after closing the owner-registered activity-admission worker before `/new` calls `quiesce()`. Previously the source-command test hung while the worker waited on a queue that only closed after the command returned.
- `pytest -q tests/agent/test_migration_turn_bridge.py`: **13 passed** after updating an `object.__new__` test double for the runner's optional artifact reader. This is a fixture alignment, not a behavior relaxation.
- A broader migration/runner/Telegram/provider/API selection, capped at five failures, reached **206 passed, 5 errors**. All five errors were in `test_runner_context_trim.py`: the test constructs the removed `AgentRunner(provider)` / `max_input_tokens` API. Map its preservation assertions to the new context governor; do not merely skip the file.
- The old `tests/channels/test_telegram_channel.py` fails collection because it imports legacy exports (`TELEGRAM_MAX_CAPTION_LEN`, `TelegramChannel`, etc.) from the new Telegram package. Establish the actual package adapter and differential behavior before changing the tests.
- Ruff passed for the carried recovery files and the `/new` loop change; `git diff --check` passed. No full CI, strict type check, release build, copied-state rehearsal, canary or live inventory has run.

## Decision and next gates

Keep `codex/harness-migration-candidate` as the implementation base while the custom source and channel contracts are completed. Keep `codex/harness-upstream-trial` as a conflict/evaluation worktree, not an accepted merge. In order: (1) finish checked receipt recovery through startup and crash-boundary tests; (2) map legacy Telegram and trimming assertions to their current interfaces; (3) resolve upstream changes by the ownership boundaries above and test every imported behavior, including nonconflict paths; (4) qualify P5/P6 before any activation. The H1 busy-input inbox/control-model work remains a separate gated package after stable turn/session/delivery interfaces. It must not be represented as already present in upstream's mid-turn queue or in this candidate.

There has been **no push, deployment, production state migration or production cutover**. A forward-refresh target is accepted only after its conflict ledger, preservation audit, cross-platform tests and copied-state rollback evidence are complete.
