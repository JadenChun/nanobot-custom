# Upstream divergence audit — 2026-11-04

Status: informational baseline for a future code-quality audit. This is **not** a
code-quality verdict, acceptance record, or performance measurement. No test suite,
type checker, profiler, or runtime behavior was exercised for this report.

## Measurement refs

| Item | Ref |
|---|---|
| Subject | `staging` at `ea4d6a6599307819edd802401706a22b08cd8dd5` (2026-11-04) |
| Upstream | `HKUDS/nanobot` `main` at `8632940a996b15054ca875c7d2a32bc10f20a199` (v0.4.0, 2026-11-04) |
| Merge base | `63d646f731a505731e97909d114ce257c6de36bc` (v0.1.4.post6, 2026-03-31) |
| Upstream commits since base | 3,644 |

The `staging` worktree was clean at measurement time. Counts are line counts
(`wc -l`-equivalent), including blank lines, comments and docstrings; they are not
SLOC. Package counts include channel-local tests under
`nanobot/channels/*/tests`. `webui/` TypeScript, `mobile/`, `desktop/` Rust, docs and
generated assets are excluded unless stated. Commit counts in this fork are
unreliable because the history contains rebased upstream commits; tree comparisons
are the source of truth.

## Headline

| | Staging | Upstream | Delta |
|---|---:|---:|---:|
| `nanobot/` package | 268,327 | 188,987 | **+79,340 (+42%)** |
| `nanobot/` files | 591 | 551 | +40 |
| `tests/` | 276,761 | 165,706 | **+111,055 (+67%)** |
| Test files | 758 | 462 | +296 |
| `def test_` functions | 10,073 | 6,522 | +3,551 (+54%) |
| `scripts/` | 1,439 | 3,838 | −2,399 |

Context: at the merge base the package was only **21,555 lines**. Upstream grew
~8.8x since then; staging grew ~12.5x. This is not a small fork that added a feature
on top of a static upstream — both trees expanded heavily in opposite directions.

The release labels also diverge: staging reports `0.3.0`, upstream reports `0.4.0`.
Staging carries more code while being behind upstream's release line.

## How different the tree is

### Package (`nanobot/`)

| Category | Files | Lines |
|---|---:|---:|
| Byte-identical to upstream | 19 | 1,767 |
| Exist only in staging | 204 | 92,691 |
| Exist only upstream | 164 | 31,502 |
| Common but modified | 368 | 173,869 vs 155,718 upstream |

Only **19 of 591** package files are byte-identical. Byte-identical is sensitive to
formatting, but the direction is unambiguous: this is a heavily diverged fork, not
"upstream plus a patch set".

### Tests

| Category | Files | Lines |
|---|---:|---:|
| Byte-identical to upstream | 35 | — |
| Exist only in staging | 438 | 152,237 |
| Exist only upstream | 142 | 37,123 |
| Common but modified | 285 | — |

The test suite is a genuine asset: 10,073 test functions and a test-to-package line
ratio of 1.03, versus 0.88 upstream. It is also heavily fork-specific — 438 test
files do not exist upstream.

## Where the extra code is

| Subsystem | Staging | Upstream | Ratio |
|---|---:|---:|---:|
| `agent/` | 62,921 | 25,044 | **2.5x** |
| `session/` | 25,696 | 5,138 | **5.0x** |
| `cron/` | 17,306 | 1,470 | **11.8x** |
| `security/` | 4,645 | 903 | **5.1x** |
| `providers/` | 23,864 | 17,234 | 1.4x |
| `channels/` | 65,910 | 80,108 | **0.82x — behind upstream** |
| `webui/` | 24,024 | 23,658 | 1.0x |
| `cli/` | 9,745 | 9,853 | 1.0x |
| `config/` | 2,093 | 1,534 | 1.4x |
| `gateway/` | 1,975 | 1,279 | 1.5x |
| `utils/` | 5,981 | 4,853 | 1.2x |
| `skills/` | 1,437 | 738 | 1.9x |

Top-level paths absent upstream (some have functional analogues under upstream
`webui/` or `cli/`):

| Path | Lines |
|---|---:|
| `remote/` | 6,677 |
| `context_repo/` | 1,776 |
| `crawler/` | 944 |
| `desktop/` | 795 |
| `extensions/` | 165 |

`agent` + `session` + `cron` account for **+74,271 lines, ~94% of the net package
difference**. That is the redesigned harness and its durability layers, not random
spread across the project.

## Quality signals

| Signal | Staging | Upstream |
|---|---:|---:|
| Test-to-package line ratio | 1.03 | 0.88 |
| `def test_` functions | 10,073 | 6,522 |
| `@pytest.mark.skip` / `pytest.skip` | 159 | 126 |
| `type: ignore` | 160 | 150 |
| `# noqa` | 51 | 53 |
| Median package file | 211 lines | 141 lines |
| Files >500 lines | 154 | 107 |
| Files >1,000 lines | 68 | 44 |

Largest staging files versus upstream:

| File | Staging | Upstream |
|---|---:|---:|
| `agent/loop.py` | 4,876 | 2,730 |
| `session/manager.py` | 3,428 | 639 (plus `sqlite_store.py`/`state.py`) |
| `cron/delivery_store.py` | 3,066 | — |
| `agent/memory_recovery_storage.py` | 3,232 | — |

The single-file comparison is illustrative, not like-for-like: staging splits
functionality into more files than upstream in some areas and fewer in others. The
subsystem totals above are the fair comparison.

### Complexity hotspot: memory recovery

`agent/memory_recovery_*` is **24 files and 15,569 lines** in staging and has **no
upstream file of that name**. It is 24.7% of `agent/` and 5.8% of the whole package.
This is the first place to ask whether each state-transition module is required by
a real failure mode or whether the transaction can be collapsed.

Cron shows the same pattern at a smaller scale: separate delivery, action,
occurrence, scheduler, callback and definition stores total 17,306 lines, against
1,470 for all of upstream cron.

## Interpretation

### What the data does not show

Line count alone is not evidence of "bad code". The test suite is substantial and
roughly one-to-one with package code; type-ignore and noqa markers are close to
upstream's; only 159 skip markers are present. Nothing in these macros says the code
is sloppy or incorrect.

### What the data does show

1. **Complexity is concentrated, not diffuse.** Agent, session and cron are the
   audit targets. They contain ~94% of the net difference and the largest files.
   Memory recovery alone is a 24-file subsystem with no upstream equivalent.
2. **This is a two-way fork.** Staging is larger in the harness, session and
   security, but **smaller in channels** (65,910 vs 80,108). Upstream has work
   staging does not: `goal/`, Linear, WhatsApp agent, SQLite session storage,
   updater modules, remote-instance/SSH support. Staging has its own remote,
   desktop, context-repo and crawler stacks; the remote stack in particular overlaps
   upstream functionality. Divergence is not simply "we added more"; both sides
   built overlapping features independently.
3. **Upstream sync is now expensive.** Only 19 package files are byte-identical and
   3,644 upstream commits have landed since the merge base. Future upstream fixes
   will not merge cleanly by default. The fork needs an explicit sync policy
   (periodic merge, cherry-pick, or formally standalone) rather than implicit
   drift.
4. **The "lightweight" label no longer describes either tree.** Upstream is 188,987
   package lines; staging is 268,327. The question is not whether the project is
   still small, but whether the extra 79k lines buy required behavior.

## Recommended audit plan

Do this as a separate, evidence-producing pass; do not treat this report as the
audit result.

1. **Feature-overlap inventory**
   - Map staging `remote/` against upstream `webui/remote_*` and `cli/remote.py`.
   - Map staging session stores against upstream `session/sqlite_store.py` and
     `session/state.py`.
   - Map staging cron stores against upstream `cron/service.py`.
   - Identify staging modules with no requirement, test, or caller.
2. **Complexity**
   - `ruff check nanobot --select C901 --statistics`
   - `radon cc -s -a nanobot/agent nanobot/session nanobot/cron`
   - `radon mi -s nanobot/agent nanobot/session nanobot/cron`
3. **Dead and duplicate code**
   - `vulture nanobot --min-confidence 80`
   - `pylint --disable=all --enable=duplicate-code nanobot/agent nanobot/session nanobot/cron`
     (or `jscpd` if pylint is unavailable).
4. **Type debt**
   - `uv run --no-sync basedpyright --outputjson` and group errors by module.
   - Follow `docs/migration/p6-error-attribution.md`; do not refresh a baseline to
     absorb new signatures.
5. **Test quality**
   - Coverage for `agent/memory_recovery_*`, `cron/*_store.py`, and `session/*`.
   - Review the 159 skipped tests; separate platform skips from abandoned coverage.
   - Check whether the 438 staging-only test files assert public behavior or
     implementation details.
6. **Fork policy**
   - Choose and document one: periodic upstream merge, cherry-pick-only, or
     standalone fork with no upstream tracking.
   - Set a review trigger for new `*_recovery*.py` and `*_store.py` layers: require a
     named failure mode and a test before adding another persistence layer.

## Reproduction

Example commands; substitute the paths and refs for the audit run. The upstream
clone needs to be fetched into the same repository as the subject ref to compute a
merge base.

```bash
# Subject and upstream refs
git rev-parse staging
git fetch https://github.com/HKUDS/nanobot.git main
git rev-parse FETCH_HEAD
git merge-base staging FETCH_HEAD
git rev-list --count "$(git merge-base staging FETCH_HEAD)"..FETCH_HEAD

# Package and test line counts
find nanobot -name '*.py' -print0 | xargs -0 cat | wc -l
find tests   -name '*.py' -print0 | xargs -0 cat | wc -l

# File-level divergence
git diff --name-only staging FETCH_HEAD -- 'nanobot/**/*.py' | sort
git diff --numstat staging FETCH_HEAD -- 'nanobot/**/*.py'

# Per-subsystem totals (repeat for each directory)
for d in agent session cron security providers channels webui; do
  printf '%-12s %8s\n' "$d" "$(find "nanobot/$d" -name '*.py' -print0 | xargs -0 cat | wc -l)"
done
```

For byte-identical comparison, hash each relative path in both worktrees with
SHA-256 and compare hashes. Do not infer behavior from identical bytes or from
line counts; this report is a sizing baseline only.

## Out of scope

- No correctness, security, performance or maintainability conclusion.
- No full test-suite run, type-check run, coverage run or profiling.
- No comparison of `webui/`, `mobile/`, `desktop/` or docs.
- No per-commit attribution; rebased history makes commit counts misleading.
- Upstream `main` is a moving target; pin the SHA above for future comparison.
