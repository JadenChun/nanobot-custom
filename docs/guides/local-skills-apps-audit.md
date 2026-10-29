# Local Skills and Apps Audit

**Audited:** 2026-10-29 (Kuala Lumpur)

**Scope:** Read-only review of the configured nanobot Skills and Apps inventory, the shared skills library, and online skill sources referenced by this checkout.

## Current inventory

At the initial audit, the configured nanobot instance reported 16 bundled skills, no workspace skills, no installed CLI apps, and no enabled Agent Plugin skills. `disabledSkills` was empty. No skills downloaded from an online marketplace were found.

The external shared-skill audit found eight folders under `/home/jaden/.agents/skills`: `dashboard-pwa-builder`, `freelance-proposal`, `invoiceshelf`, `local-llm-benchmark`, `media-generation`, `orchestrate-development`, `project-kickoff`, and `ui-wireframe-visual-verify`. These folders are not nanobot workspace installs. A folder can provide instructions without creating a native app or tool row.

## Shared-library access

Before instance activation, the config used `contextPaths: ["/home/jaden/.agents"]` and had no `contextRepos` entry. The canonical shared manifest identifies `AGENTS.md` as context, `skills` as the skills entrypoint, and protects `AGENTS.md`, `nanobot.context.json`, and `skills/**`. That protection prevents the management catalog from enumerating the eight folders or reading their `SKILL.md` files. With this original attachment, the Skills page shows an incomplete catalog without listing hidden names.

Use a private per-instance manifest snapshot for the shared library. Copy the canonical manifest to `~/.nanobot-staging/context-views/shared-agents.json`; do not edit the canonical manifest. In that copy, remove only the `skills/**` item from `protected`, remove `entrypoints.context` if present, and remove the top-level `context` field if present. Keep every other field, including the `skills` entrypoint, `AGENTS.md`, the canonical `nanobot.context.json` protection, and the manifest's other protections. The default secret protections remain active as well.

The shared root contains `AGENTS.md` but no `CONTEXT.md`. With both context fields removed from the private snapshot, nanobot will not load `AGENTS.md` as shared context, and there is no `CONTEXT.md` fallback. The source root and canonical manifest remain untouched.

After the private snapshot is reviewed, the instance config can be changed separately to use it:

```json
{
  "contextPaths": [],
  "contextRepos": [
    {
      "path": "/home/jaden/.agents",
      "readOnly": true,
      "allowExec": false,
      "autoSync": false,
      "manifest": "/home/jaden/.nanobot-staging/context-views/shared-agents.json"
    }
  ]
}
```

`readOnly` blocks filesystem writes through nanobot's resource policy. The private snapshot preserves explicit protections for sensitive paths. `allowExec: false` adds an application-level deny-first shell rule for paths within that context repo, including command working directories and referenced paths. This policy does not create an operating-system process sandbox. Existing `contextPaths`, string entries, and context-repo entries that omit `allowExec` retain their prior execution behavior. A configured custom manifest must exist as a file; a missing or invalid override is reported instead of silently becoming an unmanaged repo.

These are staging instructions only. This source package change and audit did not edit the live config, create the staging snapshot, change the canonical manifest, or alter legacy config.

## Catalog meaning and compatibility

Catalog metadata and source verification show that nanobot can safely find and read a skill entry. They do not verify that the skill's full workflow is compatible with nanobot or that external tools are configured.

The eight shared folders do not declare nanobot `requires` metadata, so Skills cannot infer their system dependencies from skill metadata. Compatibility checks found these limits:

- `orchestrate-development` depends on Codex-only agent roles and cannot use nanobot's current `plan_task`, `delegate_task`, `review_work`, and `explore` tools as drop-in replacements.
- `freelance-proposal` requires the separate `documents` skill, which is not present in the shared catalog. `project-kickoff` inherits that dependency.
- `media-generation` references six configured ComfyUI tools, but nanobot's `mcp_comfy_` tool naming and media-delivery behavior differ from the skill's assumptions.
- Shared scripts remain blocked when the private context repo is configured with `allowExec: false`.

The custom ComfyUI MCP configuration exposes six allowed tools with a 900-second timeout. The tools were not invoked as part of this read-only audit, so runtime connectivity and generation were not verified.

No deletion is justified by this audit. If a user confirms they do not use them, possible *disable-only* candidates to review are `mobile-app-testing`, `mobile-dogfood`, and `summarize` (their referenced `xcrun`, `adb`, or `summarize` commands were not found); `desktop-use` and `image-generation` (their corresponding native tool configurations are disabled); and `clawhub` if online skill installation is not wanted. Keep their files unless separately requested.

## Catalog sources

The Skills marketplace code searches `skills.sh` and SkillHub (`api.skillhub.cn`). ClawHub is referenced by bundled instructions rather than treated as an installed local skill source. The Apps catalog also references CLI-Anything and Re-bin/nanobot-extension registries; their remote logos are fetched separately. These online catalog entries are discovery results, not installed apps or skills until the user chooses an install action.

## Source-package validation

Temporary shared-library fixtures in this checkout exercise both legacy permission mode and managed folder-grant mode. With a private manifest snapshot that exposes `skills` while retaining its other protections, the fixtures report 24 verified entries (eight shared and 16 bundled), a complete catalog, readable details, and runtime-loadable shared skills. Shared-library rows remain non-deletable, disabled skills remain listed, and the private snapshot does not add `AGENTS.md` to the prompt context.

Those results describe source-package behavior against temporary fixture paths. Instance activation is a separate step: until the private snapshot and config fragment are applied, the Skills page remains subject to the canonical `skills/**` protection. The initial audit made no live config changes, shared-file writes, skill installs, dependency installs, service changes, or deletions.
