# Local Skills and Apps Audit

**Audited:** 2026-10-29 (Kuala Lumpur)

**Scope:** Read-only review of the configured nanobot Skills and Apps inventory, its shared skills access, and the online skill sources referenced by this checkout.

## Current inventory

The configured nanobot instance reports 16 bundled skills, no workspace skills, no installed CLI apps, and no enabled Agent Plugin skills. `disabledSkills` is empty. No skills downloaded from an online marketplace were found.

The external shared-skill audit found eight folders under `/home/jaden/.agents/skills`: `dashboard-pwa-builder`, `freelance-proposal`, `invoiceshelf`, `local-llm-benchmark`, `media-generation`, `orchestrate-development`, `project-kickoff`, and `ui-wireframe-visual-verify`. These are not nanobot workspace installs. The live product should not expose their names while its access policy prevents listing or reading that library.

The Skills page should distinguish workspace skills, configured shared libraries, bundled skills, and enabled Agent Plugin skills. A skill folder supplies instructions; it does not create a native app or tool row. “Enabled” describes whether nanobot may load those instructions. “Needs setup” only reflects declared CLI or environment requirements, and neither status verifies that a full workflow works.

## Shared-library access

The current config uses `contextPaths: ["/home/jaden/.agents"]` and has no `contextRepos` entries. The shared repository manifest declares `AGENTS.md` as its context entrypoint, `skills` as its skills entrypoint, and protects `AGENTS.md`, `nanobot.context.json`, and `skills/**`. The managed folder policy treats `skills/**` as protected for listing and reading, including the skills root itself. As a result, nanobot cannot safely enumerate the eight folders or read their `SKILL.md` files. The product should show a generic incomplete-catalog notice and no placeholder rows or hidden names.

`ContextRepoManager.skill_roots()` can expand glob patterns before policy checks. Management inventory must therefore resolve only supported literal roots and mark unsupported patterns incomplete without globbing. Runtime skill loading already applies the resource policy; the management fix must keep using that same policy and must not change runtime loader behavior.

One proposed, unapplied remediation is to move `/home/jaden/.agents` from `contextPaths` into `contextRepos` as read-only, then remove `AGENTS.md` and `skills/**` from the manifest's custom protected list while keeping `nanobot.context.json` and nanobot's default secret protections protected. The sanitized config fragment would be:

```json
{
  "contextPaths": [],
  "contextRepos": [
    { "path": "/home/jaden/.agents", "readOnly": true }
  ]
}
```

The corresponding manifest fragment would be:

```json
{
  "entrypoints": {
    "context": "AGENTS.md",
    "skills": ["skills"]
  },
  "protected": ["nanobot.context.json"]
}
```

`readOnly: true` blocks writes but does not limit reads to the declared entrypoints; the shared repository's other non-protected paths would remain readable under the default read policy. The manifest keeps nanobot's default secret protections in force. Both changes matter: merely unprotecting `skills/**` while keeping the repository writable could expose the managed repository's default writable-skills rule. Shared scripts would still be subject to nanobot's separate execution policy. These are review fragments only; no config or manifest changes were made during this audit.

## Compatibility and cleanup

The eight shared folders do not declare nanobot `requires` metadata, so Skills cannot infer their system dependencies from skill metadata. Compatibility checks found these limits:

- `orchestrate-development` depends on Codex-only agent roles and cannot use nanobot's current `plan_task`, `delegate_task`, `review_work`, and `explore` tools as drop-in replacements.
- `freelance-proposal` requires the separate `documents` skill, which is not present in the shared catalog. `project-kickoff` inherits that dependency.
- `media-generation` references six configured ComfyUI tools, but nanobot's `mcp_comfy_` tool naming and media-delivery behavior differ from the skill's assumptions.
- Shared scripts remain blocked by the current execution policy.

The custom ComfyUI MCP configuration exposes six allowed tools with a 900-second timeout. The tools were not invoked as part of this read-only audit, so runtime connectivity and generation were not verified.

No deletion is justified by this audit. If a user confirms they do not use them, possible *disable-only* candidates to review are `mobile-app-testing`, `mobile-dogfood`, and `summarize` (their referenced `xcrun`, `adb`, or `summarize` commands were not found); `desktop-use` and `image-generation` (their corresponding native tool configurations are disabled); and `clawhub` if online skill installation is not wanted. Keep their files unless separately requested.

## Catalog sources

The Skills marketplace code searches `skills.sh` and SkillHub (`api.skillhub.cn`). ClawHub is referenced by bundled instructions rather than treated as an installed local skill source. The Apps catalog also references CLI-Anything and Re-bin/nanobot-extension registries; their remote logos are fetched separately. These online catalog entries are discovery results, not installed apps or skills until the user chooses an install action.

This audit made no live config changes, shared-file writes, skill installs, dependency installs, or service changes.

## Implementation and validation

The Skills catalog now labels workspace, shared-library, bundled, and plugin sources; includes disabled entries; and distinguishes verified, setup-required, and unverified skills. It uses the gateway's configured context manager and resource policy for listing, details, mutations, and marketplace annotations. When any configured source is opaque or restricted, the UI shows a generic incomplete-catalog notice, avoids claiming unknown skills are installed or enabled, and blocks unconfirmed marketplace installs. Runtime skill loading was not changed. The Settings page loads online marketplace results only when the user opens **Browse online**.

Focused checks passed: 43 Python Skills API/marketplace tests, 28 agent skill-loader tests, 28 marketplace and HTTP-route tests, 64 app-layout tests, and 5 Skills marketplace UI tests. Ruff passed for the changed Python files. The integrated WebUI suite passed 1,293 tests, gateway and context regressions passed 112 tests, the production WebUI build passed, and the sanitized preview showed the restricted catalog notice, unverified bundled rows, and disabled online installs with unknown local status.

Strict typing remains debt: BasedPyright 1.40.2 analyzed 499 files with 1,659 errors and 2 warnings both before and after this change. The all-file diagnostic comparator reported `PASS_NO_NEW_DIAGNOSTICS` (zero introduced signatures); this is not a strict-clean result or P6 completion. Earlier temporary reports had disappeared, so the pristine comparison was reconstructed from tracked commit `db75ea1839a5789e10d76a89b322438c1a2b64d0` in a separate archive and rerun against the same interpreter, package environment, checker version, and strict configuration as the edited worktree. Commands, environment details, source hashes, and reports are retained at `/home/jaden/.local/state/nanobot-local-skills-apps-validation-20261029/`.
