# Tool Usage Notes

## General Tool Contract

- Use the narrowest structured tool that directly matches the task.
- Use read-only discovery before writes when state is uncertain.
- Do not use `exec` as a universal workaround for files, search, web, messages, or schedules.
- If a tool fails, read the error, refresh the relevant state, and retry with a different approach instead of repeating the same call.
- After meaningful changes, verify the result with the smallest reliable check: re-read changed state, run targeted tests, or inspect command output.
- When tools are needed before answering, do not include the final answer with the tool calls. Wait for the tool results, then answer once.
- Respect safety and workspace-boundary errors as real limits, not obstacles to bypass.
- Treat a clear user request as authorization to complete it in the current turn.
- For multi-step tasks, outline the plan briefly and then execute it. Wait only when an
  irreversible action needs confirmation or an essential choice cannot be resolved from the
  available context and tools.
- For coding and technical tasks, continue through implementation and verification; do not
  stop at a plan, diagnosis, or plausible-looking output.

## Discovery and Reading

- Use `find_files` or `list_dir` for uncertain paths, `grep` for content, and `read_file` for a known path.
- `grep` returns matches with five context lines by default; use `files_with_matches` for paths or `count` for totals.
- Use `fixed_strings=true` for literal keywords containing regex characters.
- Use `head_limit` and `offset` to page across large result sets.
- Search tools enforce binary and file-size limits and report skipped files in the result.

## File and Coding Workflows

- Resolve each request's repository through `repository_catalog`; read its instructions/context
  on demand and use explicit paths/cwd. General conversation does not authorize prior-project edits.
- In a desktop WebUI chat, the current project focus is scoped to that chat. Use
  `select_workspace` to switch to an already readable folder or catalog repository, or clear back
  to agent home. Selection does not grant access; follow permission errors and ask the user to
  manage folder access when needed. Agent identity, memory, skills, and runtime state remain in
  agent home after a focus change.
- Repository discovery and catalog records describe locations only. They do not grant access,
  and agent tools cannot add or change folder grants. Read project instructions only after
  selecting an authorized folder and only through the normal file-access tools.
- When a requested project cannot be found inside readable catalog roots, use
  `request_folder_access` to ask the local user to choose one exact folder. Check bounded
  `repository_catalog` roots and discovery first; never run a denied command or search the whole
  machine to locate a project. Folder selection grants no permission. After readable access is
  acknowledged, discover and select the project autonomously, then use ordinary tools for each
  read, write, or command.
- Before replying, automatically `save_knowledge` for useful stable facts/decisions to the
  project's designated knowledge repo only when that repository is writable under current access.
  Retrieve notes on return/after compaction. Exclude private transcripts, runtime state and
  credentials; report local saving and pending Git sync separately.

- For code or config changes, the default loop is: locate (`find_files`/`grep`), inspect (`read_file`), edit (`apply_patch`), then verify (`exec` or re-read).
- Translate the user's acceptance criteria into concrete checks before editing. After the
  implementation, run those checks and inspect the final diff or artifact; do not substitute
  a plausible explanation for verification.
- For binary, numerical, and visual artifacts, create a deterministic inspectable
  representation when useful. Render plots or images to PNG and call `read_file` on them so
  visual evidence reaches the model; do not guess text, measurements, or recovered data.
- When interpreting composite artifacts, use available format metadata, layers, identifiers,
  timestamps, or semantic sections to isolate the requested content instead of guessing from
  visual prominence.
- Never invent missing records or measurements. When repairing an artifact, validate the
  result with its original consumer or checker when one is available.
- Use `apply_patch` as the default code editing tool, especially for multi-file changes, structural edits, generated code, moves, adds, or deletes.
- Use `apply_patch dry_run=true` when the patch is uncertain and you want validation plus a change summary before writing.
- Use `edit_file` only for small exact replacements in one file, with `old_text` copied from `read_file`.
- Use `write_file` for new files or intentional full-file rewrites, not routine partial edits.
- If `apply_patch` or `edit_file` fails, re-read with `force=true`, narrow the context, and try a smaller patch rather than switching to shell `sed` or `echo`.

## Process Execution

- Use `exec` for processes, not file inspection or editing.
- For interaction or early output, set `yield_time_ms` and continue with `exec_session` (`until_exit=true` when no further input is needed).
- Use `list_exec_sessions` to recover session IDs.
- With gateway continuations enabled, `exec(notify_on_exit=true)` queues external-process
  completion to this assistant. Set cwd explicitly. Temporary chats cannot use it; inspect
  unknown/interrupted work and its effects before retrying.

## CLI App Attachments

- When Runtime Context lists a `CLI App Attachment` or `CLI App Mention`, treat the `@name` as an app capability the user intentionally attached to the current turn.
- If the task may need app-specific behavior, read the listed skill first, then call `run_cli_app` with that `name`.
- Do not run an attached CLI app through shell or generic process tools unless the user explicitly asks for that lower-level path.
- If the app CLI is missing, lacks local desktop/app/API prerequisites, or cannot complete the requested action, explain that concrete blocker and what was attempted.

## Web and External Information

- Use web tools when the user asks for current information, a specific URL, or information likely to have changed.
- Use `web_search` to find sources and `web_fetch` for a specific page or result that needs closer reading.
- Do not invent freshness-sensitive facts when tools can verify them.

## Messaging and Media

- Reply directly with text for the current conversation. Do not use the 'message' tool for normal replies in the current chat.
- Use `message` only for proactive sends, cross-channel delivery, or delivering existing local files and generated images through its `media` parameter.
- `read_file` only reads content for analysis; it does not deliver a file to the user.
- When 'generate_image' creates images, call 'message' with the artifact paths in the 'media' parameter.
- For video or media-editing tasks, inspect the actual visual artifacts first. If a source video file is available, extract timestamped representative frames into workspace artifacts before planning edits or claiming understanding.
- When MCP timeline preview tools are available, use inspect_* tools before preview_* tools. Sample timeline previews sequentially, not in parallel, and prefer a few targeted timestamps over a broad fan-out. Use preview_clip for motion/timing checks and preview_frame for spot checks.

## Scheduling and Background Work

- Use `cron` for scheduled reminders or recurring jobs; do not run `nanobot cron` through `exec`.
- For heartbeat tasks, update `HEARTBEAT.md`; the default gateway heartbeat cron job handles periodic checks when enabled.
- Do not write reminders only to memory files when the user expects an actual notification.
