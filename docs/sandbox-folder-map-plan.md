# Sandbox mode and unified folder map

Status: implemented, 30 October 2026. This document records the agreed behavior of
the shipped mode and folder-map work; verification evidence lives in
`.assets/validation/sandbox-folder-map-2026-10-30/`. Earlier DeepSeek
response/error repairs remain a separate change set.

## Agreed behavior

Nanobot supports work across several directories in one conversation. A default
working folder resolves relative paths; it does not restrict the conversation to
that one folder.

There are two visible modes:

| Mode | Files and commands |
| --- | --- |
| Sandbox | DSH-style write confinement: ordinary reads under OS permissions, writes within approved writable folders and controlled temporary/cache locations. Explicit read-only child rules override writable parents. |
| Full access | Ordinary OS-account access without the Sandbox folder-write boundary. Existing mode ownership and process supervision continue to apply. |

Sandbox is a write boundary, not a promise that all unlisted host files are
unreadable. This is the deliberate change from the earlier stricter design.
Nanobot's internal authority/state protections require an explicit audit when
applying the new policy; navigation metadata must never authorize edits to the
permission store. Network access and non-filesystem tool capabilities retain
their existing policies.

## One Folders interface

Replace the separately maintained Folder access and Known repositories sections
with one Folders list and one Add folder flow. Any existing directory is eligible;
Git is optional. A location may provide access, navigation metadata, or both.

Each displayed entry has:

- A canonical absolute path, optional friendly name, and short purpose.
- A Sandbox access choice: Inherit/default, Read-only, or Read and write.
- Effective access and its source: this folder, a named ancestor, or the default
  read-only policy.
- An Agent map choice controlling whether its navigation metadata is included in
  the model's directory index.
- Optional detected Git information; existing knowledge associations and opted-in
  Git synchronization settings remain supported where applicable.

There is no separate command-permission checkbox in the normal flow. Commands
and their child processes follow the effective filesystem write policy.

### Independent locations and automatic grouping

The list combines permission roots and saved map locations, once per canonical
path. Roots can be anywhere on the machine; Documents is not a required parent.

The backend resolves and validates real directory identity. The UI groups each
entry beneath the nearest listed ancestor using path components, not matching
folder names or raw string prefixes. Unrelated entries remain independent roots.
Adding or removing a listed ancestor recomputes the presentation. Intermediate
folders do not need to be scanned or registered merely to display a saved child.

For example, a Documents grant can cover all its descendants while T&C records
and a website folder appear as named shortcuts. An unrelated project under
`/mnt/work` can have its own grant and map entry. A reference folder under `/srv`
can be mapped with read-only access.

Tree position does not create authority. Effective access is always computed
from trusted rules, including rules not chosen for the Agent map.

### Editing and removal

- Reuse an ancestor grant when it already supplies the chosen access.
- An explicit read-only choice under a writable ancestor creates a specific
  read-only override. An explicit writable choice under a read-only ancestor
  requires an approved specific grant.
- Removing a navigation shortcut changes map visibility, not inherited access.
  Revoking an explicit grant must show the resulting inherited/default access.
- Deleting a combined entry must identify which metadata and explicit access
  would be removed; parent grants must not be silently revoked.
- While Full access is active, label these as stored Sandbox rules rather than
  implying that a read-only row restricts Full access.

## Agent request and user approval

When the agent proposes a new location, present a task-linked approval card with
the name, full path, purpose, current/inherited access, requested access, and Agent
map choice. The user can select read-only or read/write and confirm or decline.

Only trusted user actions may create or widen permissions. Agent-generated
catalog records, directory descriptions, discovered Git repositories, and prompt
text cannot grant access. Already authorized normal work does not require a new
approval each time the agent uses another folder under the allowed root.

Confirmation applies metadata and access as one user action. Retain durable
request identity, revisions, and ownership checks. If one persistence operation
fails, report the actual state and support an idempotent retry; do not report
success until both intended results are confirmed. Declining makes no changes.
Reuse the existing task-linked folder-access recovery mechanism where practical.

## Directory map in model context

Generalize the current repository catalog into a directory catalog. Navigation
and permission state can retain separate trusted backing stores; the API and UI
join them by canonical path into a single entry. Do not duplicate effective
permissions into agent-editable map metadata.

The model receives a compact index of selected location names, absolute paths,
short purposes, and effective access. Reads of files, local instructions, and
Git state happen on demand. Do not preload every directory's contents or combine
unrelated project instructions into the standing prompt.

Keep broad writable roots visible in the capability summary even when they are
not selected as navigation shortcuts. A Documents grant therefore still permits
work throughout Documents; the directory map highlights useful destinations.

For larger maps, provide a bounded index and a listing/lookup tool. Keep saved
navigation choices stable; do not inject unbounded filesystem discovery into
every request. Working in remains a default for relative paths, and absolute
paths allow one task to operate across multiple mapped or otherwise permitted
locations.

## Implementation packages and order

1. **Define the policy and common access resolution.** Specify Sandbox read/write
   behavior, nested overrides, temporary/cache writes, and existing private-state
   exceptions. Align read/write/edit/list/apply-patch tools and command admission.
   Keep legacy CLI/channel workspace restrictions separate from this local
   WebUI/desktop mode change. Retain path identity and symlink checks.

   Owners: `nanobot/context_repo/policy.py`, `nanobot/security/folder_grants.py`,
   `nanobot/security/workspace_access.py`, filesystem/path helpers, and
   `nanobot/session/workspace_focus.py`.

2. **Implement the Sandbox command environment.** Expose ordinary installed
   runtimes and libraries read-only, mount effective writable roots, apply child
   read-only overrides after broader mounts, and provide controlled temporary and
   package-cache storage. Preserve usable PATH/config discovery without copying
   credential environment variables indiscriminately. Keep process supervision,
   launch-time access checks, revocation behavior, and clear failure reporting.
   A sandbox failure must not silently select Full access. Do not implement this
   by globally enabling the legacy exec sandbox option, which can prevent Full
   access from being selected.

   Owners: `nanobot/agent/tools/sandbox.py`, `nanobot/agent/tools/shell.py`, and
   existing process/admission integrations.

3. **Generalize the catalog and migrate existing data.** Accept ordinary folders,
   introduce map visibility, join entries with trusted effective access, and
   preserve existing repository identities, purposes, knowledge links, and
   explicit read-only rules. Existing grants become access entries; existing
   repositories become map entries. Merge identical canonical paths. Version
   persistence changes where required and retain a recoverable pre-migration
   copy. Reconcile the legacy commands flag explicitly with the new mode policy;
   navigation records must not create new write grants.

   Owners: `nanobot/context_repo/catalog.py`, `nanobot/webui/workspaces.py`,
   `nanobot/agent/tools/repositories.py`, and request/response types.

4. **Build the unified Folders UI and approval flow.** Add the independent-root
   tree, inherited/default access labels, map selection, and combined Add folder
   form. Integrate agent proposals with user approval and resumable recovery.
   Rename visible Restricted labels to Sandbox and retain Full access. Preserve
   wire/session identifiers unless a behavior change requires a versioned
   migration; a display rename alone does not require rewriting stored chats.
   Update all supported locales and stale capability/help text together.

   Owners: folder settings, `WorkspaceControls`, folder-request cards, WebUI
   API/types, locale resources, and the owning session/recovery services. Keep
   product flow out of the core runner where existing edge services suffice.

5. **Update context construction and model tools.** Advertise the general folder
   map, accurate capabilities, and multi-directory usage. Make agent-originated
   additions proposals requiring user confirmation, while retaining compatibility
   with existing catalog operations where necessary. Read folder-specific
   instructions when the task actually uses that location.

   Owners: `nanobot/agent/context.py`, navigation/workspace tools, and prompt
   templates.

6. **Validate the integrated behavior, then roll out together.** Verify the API,
   file tools, commands, saved permissions, approvals, context, and UI agree.
   Publish new mode wording only with the matching backend behavior. Retain the
   earlier DeepSeek/error repairs and unrelated user work.

## Acceptance evidence

- Real Bash, Git, Node, Bun, npm, pnpm, and search commands can use the installed
  tools; local builds, tests, development servers, and representative dependency
  installs run with controlled writable caches.
- An approved project write succeeds; a write outside all approved roots fails.
  A read-only child remains unwritable beneath a writable parent. A separately
  approved root works independently. File tools and shell subprocesses agree.
- Unlisted Documents subfolders remain writable through the Documents grant.
  Map-only locations do not acquire write access merely by being listed.
- A normal non-Git folder can be registered, mapped, selected as default, and
  used alongside another folder in the same task. Existing Git/knowledge entries
  and relationships survive migration.
- User confirmation creates the intended map/access result; refusal does not.
  Stale revisions, partial persistence failures, cancellation, duplicate requests,
  and deleted/replaced source tasks cannot bypass approval or misreport success.
- Tree grouping handles unrelated roots, nearest ancestors, misleading string
  prefixes, symlink aliases, missing/moved directories, and adding a parent after
  its child. Map toggles do not change effective permissions.
- Switching Full access back to Sandbox preserves stored choices and handles
  already running processes through the existing retirement/supervision model.
- A bounded prompt includes mapped locations and truthful access information
  without loading all their files or unrelated instructions.
- Run owning Python and WebUI tests, a production WebUI build, lint, and the
  required frozen before/after full-source type-diagnostic comparison. Existing
  type debt is not a waiver for new diagnostic signatures, and this feature does
  not imply completion of the separate P6 migration stage.

## Evidence already collected

The installed DSH `0.2.1-rc.2` uses a read-only host filesystem view plus a writable
workspace/private temporary area in its Bubblewrap profile. Its filesystem mode
primarily restricts writes rather than reads.

A disposable copy of Nanobot's WebUI built successfully under that profile with
the host toolchain path available. Node, Bun, npm, pnpm, and ripgrep ran; an
outside-project write was denied; the copied build produced fresh output; the
temporary copy was removed; the real source diff was unchanged. This validates
one real build, not every project or package-install workflow. Reports live under
`.assets/validation/nanobot-command-modes-2026-10-30/`.

The independent-location and grouped-tree mockups are design evidence only. Live
permission settings and production mode behavior remain unchanged by this plan.
