# Runtime usability evidence — 2026-11-07

**Status: runtime evidence only. This document adds no source-review credit, changes no stream status, and accepts no repair.**

- Audit source pin: `9562e57f8333d7f14b6aaeae837660723a9b2650`
- Current staging: `29612c3d` (`origin/staging` synced)
- Post-pin production delta: `6556d394`, merged by `29612c3d`
- Session evidence captured at the audit pin, before the post-pin skill fix

## 1. Session evidence

| Item | Value |
|---|---|
| Session | `5859a5af-2d61-42d2-ab47-40f09c41cf00` |
| Title | Nanbeige vs Qwen 27B Coding Test |
| Model preset | DeepSeek V4.1 Flash (CommandCode) |
| Messages | 624 |
| Session file | 1,010,287 bytes |
| Context tokens | 150,124 |
| Total reported tokens | 5,201,872 |
| Requests | 37 |
| Last TTFT | 60,366 ms |
| Last generation | 70,372 ms |
| Long turn 1 | 1,622,742 ms, 39 iterations |
| Long turn 2 | 499,279 ms, 36 iterations |

Evidence paths:

- `/home/jaden/.nanobot-staging/sessions/5b9b211f4e59619f7a0d6503f8ab0f38/d2Vic29ja2V0OjU4NTlhNWFmLTJkNjEtNDJkMi1hYjQ3LTQwZjA5YzQxY2YwMA.jsonl`
- `/home/jaden/.nanobot-staging/logs/gateway.6e8d9acab7b33d62.log`

The final turn ended with:

```text
Error: File not found: /home/jaden/.agents/skills/cron/SKILL.md
```

The agent tried the workspace skills copy and the context-repo copy. The built-in cron skill is packaged at `nanobot/skills/cron/SKILL.md`. The post-pin fix makes the exact readable path `skills/cron/SKILL.md`.

## 2. Findings

### U1 — Long work blocks the chat

**Severity:** high.

**Evidence:** One turn ran 27 minutes. Another ran 8 minutes. The agent used foreground `exec` with `sleep 90` and `sleep 120`.

**User impact:** The chat is unusable for other work during the turn. The user cannot see progress without asking. Long commands can hit provider timeouts.

**Audit streams:** H01, A01, S02, U01, D01.

**Architectural question:** What owns long-task execution, progress, cancellation, and completion notification?

**Smallest direction:** A task contract that starts long work in a background job, returns a job ID, exposes status, and notifies on terminal state. Forbid foreground sleeps in agent guidance.

### U2 — Session growth has no visible budget

**Severity:** high.

**Evidence:** 624 messages, 1 MB, 150k context, 60-second TTFT. The UI thread and file-preview routes take 1 to 3 seconds.

**User impact:** Every request is slower and more expensive. The UI becomes sluggish. The model may hit a context limit.

**Audit streams:** H02, H03, U01, H01.

**Architectural question:** When should a session compact, split, refuse a turn, or show context pressure?

**Smallest direction:** Define a visible session budget. Trigger compaction or split at a threshold. Show pressure in the UI. Keep the current session as history and start a new chat for new work.

### U3 — Recovery exhaustion returns a raw tool error

**Severity:** medium.

**Evidence:** The recovery budget reached 2 of 2. The assistant returned the raw file-not-found error.

**User impact:** The user sees an internal error instead of a cause and a next step. The agent appears broken.

**Audit streams:** H01, S02, U01.

**Architectural question:** What is the structured outcome when recovery is exhausted?

**Smallest direction:** Define a recovery-exhausted response contract with cause, attempted action, and next step. Do not expose raw tool errors as the assistant answer.

### U4 — Remote owner does not autostart

**Severity:** medium for mobile use.

**Evidence:** After a desktop restart, the phone showed “Reconnect this phone”. The owner process had to be started manually.

**User impact:** Mobile pairing is fragile. The user must know the owner command.

**Audit streams:** R01, D01, E02.

**Architectural question:** Who owns the remote owner lifecycle: desktop, systemd, or both?

**Smallest direction:** One owner per desktop. Supervise it with the desktop app or a systemd user unit. Show owner status.

### U5 — SSRF guard blocks local service checks

**Severity:** medium, policy decision.

**Evidence:** The agent tried `curl http://127.0.0.1:8082`. The SSRF guard blocked it.

**User impact:** The agent cannot inspect the local inference manager over HTTP.

**Audit streams:** S01, S02, T01, H03.

**Architectural question:** Which local services may the agent access directly?

**Smallest direction:** An explicit local-service allowlist if this is a normal workflow. Otherwise require the local manager CLI. Do not add a generic SSRF bypass.

### U6 — Title generation fails on long sessions

**Severity:** low.

**Evidence:** The log reports “WebUI title generation returned no usable title”.

**User impact:** Minor. The session already has a title.

**Audit streams:** E02, U01.

**Smallest direction:** Cleanup only. Do not prioritize over U1 and U2.

## 3. Post-pin staging delta

The audit source pin is `9562e57f`. Staging now has `29612c3d`.

| Commit | Change | Affected streams |
|---|---|---|
| `6556d394` | Built-in skills display `skills/<name>/SKILL.md`; `read_file` maps it to the packaged skill | T01, K01, H01 |
| `29612c3d` | Merge of the skill-path fix | T01, K01, H01 |

The delta changes `nanobot/agent/skills.py` and `tests/agent/test_skills_loader.py` only. It does not change the audit source reports or stream status.

The audit should treat this as a named staging delta. It should not refresh the source pin silently.

## 4. Validation evidence

| Check | Result |
|---|---|
| WebUI tests | 1457 passed |
| TypeScript `tsc --noEmit` | clean |
| `pytest tests/webui` | 503 passed |
| WebSocket channel tests | 211 passed, 1 pre-existing failure |
| Agent tests | 114 passed |
| Provider and budget tests | 176 passed |
| Ruff | clean |
| BasedPyright | 803 baseline, 803 current, 0 new |

The pre-existing WebSocket failure is `test_handle_file_preview_probe_reports_binary_file_as_unavailable`. It also fails on unchanged staging.

The strict BasedPyright CI job fails on the same pre-existing debt. Staging CI is already red for the same reason. This branch adds no new diagnostics.

## 5. Design questions for consolidated review

1. Should internal recovery notes use `user` role, a normalized `developer` role, or a provider adapter rule?
2. What is the maximum wall-clock budget for one interactive turn?
3. When should a session compact, split, or refuse a new turn?
4. Should long tasks be owned by a job service instead of the chat turn?
5. What structured error should the agent return when recovery is exhausted?
6. Should the remote owner be desktop-managed, systemd-managed, or both?
7. Which local services may the agent access directly?
8. Should fallback budgeting always replace the primary window with the fallback window?
9. Should the UI show explicit context-pressure and task-progress state?

## 6. Limits

This document records runtime observations. It does not prove a source defect, a measured frequency, or a user-wide impact. It does not execute source review, tests, builds, checkers, services, providers, or benchmarks.

The session evidence was captured at `9562e57f`. The skill-path fix landed after the session. The audit should analyze the delta separately.
