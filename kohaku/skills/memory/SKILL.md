---
name: memory
description: Find durable project facts, lessons, and decisions from past sessions.
---

# Memory

- `memory/memory_summary.md` is an index of durable topics.
- `memory/MEMORY.md` contains project context, decisions, and technical lessons.
- `memory/rollout_summaries/` contains reusable lessons from past sessions.
- `USER.md` contains personal facts and communication preferences.
- `SOUL.md` contains agent behavior, interaction rules, and tool-use strategy.
- `skills/<name>/SKILL.md` contains repeatable procedures.

Use `memory_search` with the terms you expect to find. For example, search for
`auth redirect bug` to find those terms in the same passage. Set the scope to
`memory`, `rollouts`, `sessions`, or `knowledge` to narrow the results.

Use `read_file` to follow a result with a file path. Use `read_session` to follow
a result with a session key. If you need an exact line from older memory history,
use `grep` with the History log path in the system prompt.

Dream manages the index, handbook, profile, and style files. Run `/dream-log` to
see the changes Dream recorded.
