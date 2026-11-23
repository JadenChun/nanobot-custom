# Contributing to Kohaku

Kohaku accepts focused improvements to code and documentation. This guide explains how to prepare a change from a clone of this repository.

## Contribution Flow

### What Should I Open a PR For?

PRs are welcome for:

- New features or functionality
- Bug fixes with no behavior changes
- Documentation improvements
- Minor tweaks that don't affect functionality
- Refactoring that is clearly scoped and easy to review
- Changes to APIs or configuration, when the impact is documented

For riskier or larger changes, please open an issue or draft PR early so the
shape of the work can be discussed before the implementation grows too large.

### Starting Work

Before you make changes, update your checkout and create a topic branch.

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c your-topic-branch
```

Use the `origin` remote for this repository. Open an issue before a change that can break a public interface.

Keep unrelated local changes out of the topic branch. If your checkout already has
work in progress, use a separate worktree or finish that work before starting a
new branch.

## Development Setup

Use this setup from the root of a clone of this repository:

```bash
# Clone this repository
git clone https://github.com/JadenChun/nanobot-custom.git kohaku
cd kohaku

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint code
ruff check kohaku/

# Format code — optional. The existing tree predates `ruff format`,
# so running it broadly produces large unrelated diffs.
# Do not mix mechanical formatting churn into a functional PR.
# Use formatting only for the exact code your change intentionally touches.
ruff format <files-you-changed>
```

### Strict Type Checking

Strict type checking covers optional providers and channels. Reproduce the CI environment
with the same dependency sources and commands:

```bash
uv sync --all-extras --dev
uv run --no-sync python -m scripts.install_channel_dependencies --all-channels
uv run --no-sync basedpyright
```

Keep `--no-sync` on the final commands: channel dependencies come from their package
manifests and are installed explicitly by the setup step.

## Contribution License

By submitting a contribution, you confirm that you have the right to submit it
and agree that it will be licensed under the project's MIT License.

## Code Style

Keep Kohaku small, calm, and readable.

When contributing, please aim for code that feels:

- Simple: prefer the smallest change that solves the real problem
- Clear: optimize for the next reader, not for cleverness
- Decoupled: keep boundaries clean and avoid unnecessary new abstractions
- Honest: do not hide complexity, but do not create extra complexity either
- Durable: choose solutions that are easy to maintain, test, and extend

In practice:

- Line length: 100 characters (`ruff`)
- Target: Python 3.11+
- Linting: `ruff` with rules E, F, I, N, W (E501 ignored)
- Async: uses `asyncio` throughout; pytest with `asyncio_mode = "auto"`
- Prefer readable code over magical code
- Prefer focused patches over broad rewrites
- Do not mix mechanical formatting, line wrapping, import sorting, or quote churn
  into a feature or bugfix PR. If formatting cleanup is needed, make it a
  separate formatting-only PR.
- If a new abstraction is introduced, it should clearly reduce complexity rather than move it around

## Modifying CI Workflows

If your PR touches `.github/workflows/`, please keep the CI within
GitHub Actions' free tier:

- Use only standard GitHub-hosted runners (`ubuntu-latest`, `windows-latest`)
- Avoid macOS runners, larger runners (`*-cores`, `*-xlarge`, `*-gpu`),
  and self-hosted runners
- Avoid uploading large artifacts or using long retention
- Avoid paid Marketplace actions

If your change genuinely needs to step outside this, please call it out
explicitly in the PR description so it can be discussed before merge.

## Release Packaging Contract

A stable install must never combine Python from one version with a TUI from another. Publish in
this order:

1. Set the package version and publish the matching GitHub release tag (`vX.Y.Z`).
2. Review the pinned Bun/OpenTUI licenses, source offer, and relinking materials for that tag.
3. Manually run **Publish Terminal UI** for the exact tag and confirm the compliance review input.
4. Wait for every platform archive and checksum to appear on the release, then publish the matching
   `kohaku` wheel for that release.

The wheel contains the built WebUI. The native TUI stays a platform-specific release sidecar so
users download only the archive for their machine. Each archive must contain the executable,
target-specific third-party notices, project and runtime licenses, corresponding application
source, a written source offer, relinking instructions, and a checksum manifest. Never upload a
naked TUI executable. Source checkouts use an editable Python install, run `tui/` with Bun, and
rebuild stale `webui/` assets locally.

The confirmation is an operational commitment, not a cosmetic checkbox. Before accepting it,
verify that the exact Bun/WebKit revisions remain retrievable and that the project can honor the
archive's corresponding-source offer for its full stated period. Preserve published archives and
their source materials.

## Questions?

Open an [issue in this repository](https://github.com/JadenChun/nanobot-custom/issues) when you need to discuss a change. Include the command and output that show the problem.
