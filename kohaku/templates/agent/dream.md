Maintain the workspace's durable memory. Treat conversation history and rollout files as
evidence. Keep confirmed facts and reusable lessons. Remove superseded information and
duplicated details. Never save credentials or secrets.

## File routing

| File | Content |
|---|---|
| `memory/memory_summary.md` | Index of durable topics, with links to handbook sections and relevant rollout files |
| `memory/MEMORY.md` | Durable project context, decisions, and technical lessons by topic |
| `USER.md` | Confirmed personal facts and communication preferences |
| `SOUL.md` | Agent behavior, interaction rules, and tool-use strategy |
| `skills/<name>/SKILL.md` | Repeatable procedures with steps and examples |

Keep each fact in its most specific file. The index may name and point to the topic.
Put language, tone, and response length preferences in `USER.md`. Put rules for agent
behavior in `SOUL.md`. Reuse an existing skill if it covers the procedure. For a new
skill, follow `{{ skill_creator_path }}`.

## Consolidate a rollout batch

Read the pending rollout files and use the supplied extracts to select durable facts.
Read the current handbook and index before editing them. Read the current profile or
style file before updating it.

1. Update `memory/MEMORY.md` with verified project facts and reusable technical lessons.
2. Write `memory/memory_summary.md` with the exact first line `<!-- kohaku-memory-summary: v1 -->`.
3. Keep the index within 1,500 tokens and use exact paths and handbook section names.
4. Keep `USER.md` within 500 tokens and `SOUL.md` within 400 tokens.
5. Preserve valid entries that new evidence does not replace.

## Consolidate older conversation history

When the input contains conversation history without a rollout batch, update the
durable files with confirmed changes. Create the index only if you have enough
evidence to make it useful. Use correction tags to replace conflicting entries.
Remove all routing and retention tags from saved text.

## Verification

Report the changes that successful tool results confirm. Leave source rollouts and
raw extracts in place. The memory pipeline records the Git diff and clears processed
extracts after a successful run.
