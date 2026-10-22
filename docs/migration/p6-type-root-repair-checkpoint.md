# P6 source repair wave — bounded acceptance

2026-10-02: **bounded source repair ACCEPTED / root PASS / fresh GPT-6.1 Sol
Medium review PASS, no actionable findings**. Full strict gate FAIL; full P6
NOT_ACCEPTED. Runtime and D9 remain NOT_AUTHORIZED. This supersedes the earlier
"no source repair implemented" status only for the exact changes below.

## Delivered and verified

- Session journal dictionaries now retain their existing keyword shapes through a
  `TypedDict`; the existing lock decorator preserves method signatures through
  `ParamSpec`/`Concatenate`. Nine equivalent dictionary-literal replacements keep
  keys, values and evaluation order. No journal schema, lock order or operation
  flow changes. The638 repeated journal argument errors are eliminated.
- Eight unconditional-raise recovery helpers truthfully declare `NoReturn`.
  Only imports/annotations change. Root normalized-AST comparison matches all
  eight modules and the session module against the frozen pre-edit source.
- Legacy fallback accepts positional messages as its parent does. The credential
  pool accepts all six additional normal provider parameters and three stream
  callbacks positionally, mapping only supplied arguments. Omitted defaults and
  existing keyword/context/custom arguments remain unchanged. Duplicate arguments
  and overarity fail before routing, admission or provider calls. Four incompatible
  override diagnostics are eliminated; fallback/retry/state policy is unchanged.
- One unchanged recovery test failed both post-edit and on independent pre-edit
  replay because it expected a lexical symlink alias despite the accepted constructor
  resolving the workspace for admission. It now checks the canonical target plus
  alias samefile identity and intact symlink; existing migration/backup assertions
  remain. No runtime correction was made to manufacture that test's PASS.

Root independently repeated409 session/recovery cases and207 provider cases under
OS network denial, home-read denial and disposable-only writes. The disjoint root
selections pass; worker/root reruns are not summed. The new provider matrix has55
collected cases, including three inapplicable combinations that return without
assertions; those three are not meaningful independently exercised branches.
Provider regression red evidence19failed/36passed is retained, followed by207
passing cases including the existing policy/state/budget/stream tests. Scoped
Ruff and diff checks pass. Fresh review confirms all13 final scoped hashes match
both root test inventories and the final type-check inventory.

## Whole strict gate and scope

The unchanged full checker analyzes463 files and now reports **4,815 errors**,
down from5,970. Static multiset comparison removes1,218 signature instances and
introduces63 within owned files (net1,155 fewer); **zero new signatures outside
the owned files**. These counts are not independent runtime-bug counts. No checker
strictness, ignores, guard or dependency profile changed.

Exactly12 existing files and one new provider regression file changed before
publication;1,858 other candidate baseline files matched. Source scope is11
modules: one session module, eight recovery modules and two provider wrappers.
The only existing test change is the named symlink assertion. After acceptance,
only this checkpoint/evidence and three linked status documents change.

[Machine evidence](p6-type-root-repair-evidence.json) binds34 durable0400 raw records/
helpers/review summary/diff and13 exact scoped source snapshots under private bundle
`/Users/jaden/Documents/github/nanobot-harness-candidate/.assets/releases/p6-repair.9nRouv`.
Initial runner failures remain red: the disposable `/tmp` canonical alias and the
actual frozen Ruff executable were subsequently added as exact path allowances;
network/home/protected-original denial was retained. Pre-edit/post-edit lexical
test failures and provider red evidence are not retroactively green.

## What this does not accept

This is source-only repair validation, not installation of a repaired artifact.
Earlier P6-A/host acceptances remain immutable historical evidence; the current
corrected full source needs a new freeze/build/install after remaining strict
repairs pass. The remaining4,815 errors are still a source gate, not missing imports.
Native Python3.11/3.14/Windows/Linux/production-target evidence, owner-selected
production/dependency/channel/plugin/crawler profile, approved immutable old
rollback reader, truthful stopped copies/reference config/backups/artifact policy,
actual loader source policy and positive isolated rehearsal remain open.

No live state/config/credential read, channel/provider/listener/desktop/background
activation, guard removal, deployment, commit or external publication occurred.
Full P6 **NOT_ACCEPTED**; P7/H1/P8 remain separate and undelivered.
