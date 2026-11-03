# Coding handoff — prepared 2026-11-03–04

Implement [the cleanup plan](../docs/telegram-h1-concurrency-cleanup-plan.md).
The plan is the scope/acceptance contract; the SuperAgent research is rationale,
and the mobile/session-recovery documents are separate context only.

- Work only in `/home/jaden/.codex/worktrees/agent-session-cleanup/nanobot-custom`,
  branch `codex/telegram-h1-concurrency-cleanup`.
- Product code starts at `a90191a41b025884e76ce029b98a5d5349171f2e`. Preparation
  commits contain documentation only; cleanup is not implemented.
- You are not alone in the codebase. Preserve other edits and sibling worktrees.
  The original checkout has an unrelated `nanobot/webui/ws_http.py` edit that
  is deliberately excluded. Do not copy it or change the original checkout.
- Read `AGENTS.md`, `.agent/design.md`, `.agent/security.md`, `.agent/gotchas.md`,
  the cleanup plan, and the applicable shared development skill first.
- Own W1/W2 production changes sequentially. Resolve W0's deletion-aware type
  evidence before deleting modules. Keep one writer on shared runtime files.
- For new/changed cases, dispatch a fresh requirements-only author using only
  [the public contract](cleanup-test-requirements.md) and external scratch output
  metadata. Do not give that author this handoff, the plan, or repository access.
- Use `.venv/bin/python -m pytest`, `.venv/bin/python -m basedpyright`, and
  `.venv/bin/ruff check`. `uv` is unavailable here; do not silently change the
  frozen environment or use `ruff format`.
- Evidence and exact commands are under
  `.assets/validation/cleanup-2026-11-03/`. Inspect `readiness.json` first.
  Compare later reports against the final frozen `baseline-strict.json`, not
  the retained preliminary `pre-extras-*` reports. Save after-state hashes and
  dependency evidence before claiming acceptance.
- Do not run the real gateway, alter live settings/state, call real providers,
  send Telegram messages, start/stop local model servers, or merge/push/deploy.

When dispatching an implementation worker, perform the development skill's
current availability preflight; prefer the available Q6 base worker or its
specified Luna fallback. No worker has been started and no model capacity has
been reserved by this preparation. Keep final review in the main agent unless
a specific unresolved risk requires an independent session.

Return reviewable commits, the deleted-module/retained-legacy-guard inventory,
focused test/lint results, full-source type comparison, strict exit status,
and any remaining limitation. This task cannot declare P6 or a production
cutover complete.
