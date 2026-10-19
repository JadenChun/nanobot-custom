# P6 host installation qualification — accepted

2026-10-02: **HOST_QUALIFICATION_ACCEPTED / root PASS / fresh review PASS**.
P6 remains IN_PROGRESS / full P6 NOT_ACCEPTED; this is not worker self-acceptance.
P5 DONE/ACCEPTED and P6-A ACCEPTED are unchanged. Runtime remains
NOT_READY/NOT_AUTHORIZED, D9 NOT_AUTHORIZED, P7/H1/P8 NOT_DELIVERED.
Exact review/root verdict and pre-final root scope/evidence checks are embedded
and retained0400 under `final_acceptance_evidence` in the machine evidence.
Final root publication recheck has not yet been performed or claimed.

## Bound artifact and host-only dependency closure

The accepted P6-A wheel SHA256 is
`3c72e14e4358f33368904c9bad138c4c301889a70b7acaf3d12665df17fb26ff`.
Frozen dirty-source archive SHA256 is
`fa582770bac7d9dd328558082f874ccefdf6eb1d5a069d51a8ea8194b85c4378`;
source-manifest SHA256 is
`2a7a25356f9f8ef6de52eaa40ce4e10ac2f04f773f5cc1c2d60b45b1ddadcdd2`.
These inputs and all earlier [P6-A](p6-artifact-checkpoint.md) history remain intact.

Fresh private bundle:
`/Users/jaden/Documents/github/nanobot-harness-candidate/.assets/releases/p6-installed.c7OniRNY`.
Host is macOS26.6.2 arm64/Python3.13.2; bootstrap pip24.3.1. Public PyPI
wheel-only acquisition was a separate network-enabled phase with isolated pip
configuration/private cache-free paths and sanitized environment, no private
index/credential inputs or dependency sdists/build hooks. The114-wheel freeze
includes core+api+dev only, not all extras or a selected production profile.
Fully pinned requirements SHA256 is
`c0232f91228f055db3c89d9318684a182a0ae9712e9a7e3cec8a18302daaf589`.
Every wheel name/version/METADATA/size/hash is bound in the
[machine evidence](p6-installed-evidence.json), with0400 retained wheelhouse and
requirements. The ignored existing uv.lock is not promoted into release authority.

## Actual installed checks

Actual install into a new empty environment used `--no-index --require-hashes`
under OS network denial and passed; `pip check` reports no broken requirements.
There are115 installed distributions including bootstrap pip. A `-I` check from
outside source proves nanobot/core/config/CLI/API/crawler modules originate in the
fresh site-packages, no editable package or source PYTHONPATH. All1274 package
member bytes match the accepted wheel, including guard, custom skills/scripts
and actual frontend assets. Root independently rechecked114 wheel hashes/pins,
installed bytes/origins and pip consistency; raw root records are separately bound.

Installed metadata discovers17 packaged channel manifests,10 declared extras and
16 built-in skills, including all five custom artifact-contract skills. This is
inventory, not channel enablement. Crawler entrypoint module imports; Crawl4AI
optional worker dependency/browser/runtime remains uninstalled/unqualified.

Installed `nanobot --help` and gateway/agent/webui/serve help each pass. Unmodified
foreground gateway/agent/serve/WebUI commands with explicit disposable nonexistent
config/workspace/log paths raise actual MigrationIncompleteError; the proposed
state/log root remains absent. This is guard-negative startup, not successful
service startup. No detached/service/update/deploy command or guard patch ran.

Ordinary application profiles deny all network, unauthorized execution and
home-state reads, permit only exact ancestor metadata needed to reach disposable
paths, and restrict writes to those paths. Strict typing has its separate finite
interpreter/packaged-node execution split. Literal argv/cwd/environment/profiles,
timeouts and full raw outputs are bound to private retained0400 records. HOME,
XDG and CODEX_HOME were not reassigned; no real config/credentials were loaded.

## Exact DOCX fixture and strict diagnostics

The root separately authorized only unchanged
`tests/test_api_attachment.py::test_docx_upload_passes_media_path` plus its frozen
root conftest, against the installed wheel. Supported public `set_config_path`
binds a new disposable instance parent before pytest, without Config/load_config
or config creation, fixture/app/assertion/guard edits, or HOME changes. A separate
localhost-only OS profile permits the fixture's test server; pure socket evidence
proves127.0.0.1 bind and PermissionError for1.1.1.1:443. Worker and independent
root exact runs each pass, separately recorded and never summed. Installed API
origins/no frozen-source sys.path are proved before/after pytest; copied test and
conftest hashes are retained. This is a mock-agent API upload fixture, not a live
channel/service or operator-loader qualification. Prior denied-loopback records
are unchanged.

Strict BasedPyright1.40.1 (pyright1.1.414) ran on the byte-identical frozen source
with unchanged strict config and isolated environment interpreter. **FAIL**:
463 files,7420 errors,0 warnings/information, exit1. All diagnostics are retained;
56 are direct unresolved optional/channel imports and7364 are other strict
diagnostics. Other reports include argument995, optional-member181, unknown
member1558, unknown argument1347 and private usage745; some unknown types may
cascade from missing optional imports. Agent2342/session1672/channels1608/cron851/
providers462 are package diagnostic counts. Missing imports are not used to
dismiss every error. No full all-extras CI regression verdict, import silencing,
config weakening or broad code fix is claimed.

## Provenance, unsuccessful attempts and remaining gates

Raw acquisition/install/probe/help/guard/type/DOCX records, contracts, gate analysis,
helpers and root independent records are durable0400 copies under this bundle's
`evidence/`, indexed by exact path/hash/size in the machine evidence. Separate
failed sandbox parsing/launcher attempts, incorrect initial helper skill-name
expectation and DOCX default-home denial are retained red, not relabeled green.
Narrow helper preparation/path binding corrected them; no candidate runtime fix
or relaxed product assertion occurred. Before publication, all1866 root-baseline
candidate files and1865 frozen source files matched. Guard hash remains
`5a653ef73328b4840d31a7737fc3d0a79fbf631bf648fd3ebf9793cbee012d74`.
Only these four authorized publication documents change afterward.

Capacity-only reuse of the completed P5 GPT-6.1 Sol Medium node reduced worker
isolation; no model substitution/recursive delegation occurred. Fresh integrated
GPT-6.1 Sol Medium review completed PASS/no findings; root accepted bounded AC1–AC5.
Independent final root publication recheck remains required.

Still open: unresolved strict diagnostics/full extras/channel typing environment;
native Windows/Linux/Python3.11/3.14/actual production target; selected production,
channel/crawler/browser closure/inventory; actual operator init-only loader policy;
truthful stopped copied-state/reference config/backups/artifact-read-root provenance;
genuine old immutable rollback artifact; positive guarded-service policy and
remaining release rehearsal. No synthetic state/operator rehearsal ran. Full P6
NOT_ACCEPTED and separate P7/D9/canary authority remain; background work stays off.

## 2026-10-02 — subsequent full host dependency diagnostic

The preceding accepted host subset and7420-error result remain history. The
[full environment checkpoint](p6-full-environment-checkpoint.md) now records all
ten extras/17 channel manifests,210 hash-pinned wheels, an OS-network-denied
offline installation, independently repeated pip check and input integrity PASS.
Unchanged strict typing still FAILS:5970 errors/463 files, zero missing imports;
pinned upstream has zero errors with the same tool/environment/policy.
The checkpoint identifies bounded root-first source repairs, not implemented fixes.
No runtime/guard/background change or full-P6 acceptance follows.
