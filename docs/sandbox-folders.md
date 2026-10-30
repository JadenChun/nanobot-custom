# Sandbox and folders

Local WebUI chats start in **Sandbox**. It permits ordinary reads under your OS
account and runs normal Bash commands and their children. Writes are limited to
the agent home, approved writable folders, and private temporary/cache storage.
An explicit read-only child overrides a writable parent. The map is a navigation
index, not an exclusive list of accessible files.

**Full access** uses ordinary OS-account access without the Sandbox write boundary.
It remains a chat-specific, explicitly selected mode. Returning to Sandbox uses
the existing process retirement flow: running commands keep their launch access
until they finish. Sandbox never silently falls back to Full access. The write
boundary applies as soon as a chat is in Sandbox, before any folder has been
saved. A working Linux Bubblewrap installation is required for Sandbox commands;
failures identify that requirement.

Private Nanobot authority, sessions and runtime credentials remain protected by
file tools and hidden inside Sandbox commands. Host process memory/environment
trees are also unavailable to Sandbox file tools; confined commands inspect
their own PID namespace. Configured context/target
repositories retain their separate protected-file, managed-store and proposal
policies. Shell commands cannot access those configured repositories; use the
file tools for their permitted files. Ordinary indexed project folders do not
acquire these special policies. Network policy is unchanged: a remote
(non-loopback) WebUI client can preview only files inside the focused folder.

## Folders Settings

Use **Settings → Folders → Add folder** for any existing directory. Git is
optional. Give it a friendly name and purpose, choose its **Sandbox access**, and
choose whether to **Include in Agent map**:

- **Inherit/default** removes the folder's own explicit rule. The nearest live
  ancestor rule applies; otherwise the Sandbox default is read-only. The agent
  home and configured context policies remain separate built-in capabilities.
- **Read-only** keeps programs and file tools from writing there.
- **Read and write** permits writes. An existing ancestor grant is reused when it
  already provides that access.

The list joins saved grants and directory metadata by canonical path. It nests
locations beneath their nearest saved ancestor and keeps unrelated roots
independent. It does not scan intermediate folders. Effective access and its
source are shown on each row. Missing or replaced locations remain visible for
review and removal.

Map visibility does not change permissions. A broad Documents grant still covers
unlisted descendants. A mapped folder with default read-only access cannot
become writable by changing its description or appearing in model context.
Removing a location previews the metadata and explicit rule being removed, plus
the resulting inherited/default access. Parent grants remain in place.

Knowledge folder associations and explicitly selected Git synchronization paths
remain under **Knowledge and Git sync** in the same form. Existing IDs remain
stable when a friendly display name is edited.

## Agent proposals

`repository_catalog(action="remember", ...)` accepts ordinary folders and
proposes new entries in persistent WebUI chats. Its legacy tool name is retained
for compatibility. A task-linked card shows the path, purpose, current access,
requested Sandbox access and map choice. Only the user's confirmation saves the
metadata and selected access. Declining changes neither store. Temporary chats
cannot save references through this tool.

Combined saves use a durable operation ID. If metadata saves but permission
persistence fails, the UI reports that partial result and can retry the same
operation. Both stores must be confirmed before success is reported. Stale task,
source and revision checks remain mandatory for approval and continuation.

## Context and compatibility

The standing prompt includes bounded navigation metadata for mapped folders and
an independent capability summary for saved access roots. It does not preload
all folder contents. The currently focused folder's `AGENTS.md` is read when that
folder is used; unrelated folder instructions are not loaded. Absolute paths
allow work across directories. **Working in** supplies the default for relative
paths. Use the catalog's `list` action for the complete bounded directory map.

Catalog version 1 entries migrate on the next catalog write. Existing entries
are mapped by default; canonical aliases are merged and knowledge associations
are preserved. `catalog.pre-v2.json` retains the original bytes. Version 2 adds
map visibility, a friendly label and a catalog revision. Access remains in the
separate trusted grant store. Existing read-only rules remain unchanged.

The internal mode identifier `restricted` and the stored grant `commands` field
remain wire-compatible. The UI calls the mode Sandbox; that legacy command bit
does not restrict Sandbox execution and is preserved during metadata-only edits.
New folder grants include commands implicitly. Existing CLI/channel workspace
restriction settings and the optional legacy `tools.exec.sandbox` setting remain
separate. Do not enable the legacy sandbox globally to implement these modes.

Sandbox inherits the installed toolchain PATH without copying credential
environment variables. Package caches use private temporary storage. Cached
Corepack runtimes remain available read-only, and resolver configuration remains
available when `/run` is hidden. Other toolchains with custom writable caches may
need an approved cache directory or explicit cache configuration.
