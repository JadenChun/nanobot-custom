# P2 explicit factories and shell-policy checkpoint

The runtime loader now constructs only the explicit foreground manifest in `fork_factories.py`. Discovery and entry points remain available as inspection code but do not select runtime capabilities. Construction and collision checks precede registration; missing foreground manager/policy/locks fail without mutating the registry. Generic background/role scopes are rejected. Foreground role registries remain owned by their explicit manager builders. The public migration activation guard remains unchanged.

Factory dependencies carry the existing manager, resource policy and shared locks. Browser/device/desktop configuration reaches lazy constructors; crawler requires both existing switches; VPN retains exactly the existing `AGENT_VPN_BROKER_ENABLED` true predicate. Canonical image class selection is unique, but legacy image exposure/backend/config adaptation remains an open gate: this checkpoint must not be used as a runnable release.

Shell construction receives resource policy and desktop activation. Mandatory deny-first, allowlist, private-URL, desktop and managed-path checks run even in full-access mode. Managed execution roots must be declared context/target repositories and pass policy; sandbox mounts remain a separate requirement. The upstream process/session implementation is preserved.

Validation under network-denied sandbox:

- Factory, custom shell-policy, process reaping, environment and session tools: **99 passed**.
- Shell security, platform and allowlist tests: **147 passed, 6 skipped** (platform-specific).
- Scoped Ruff and whitespace checks pass.

Upstream tests that deliberately allowed full-access guard bypasses or allowlist override of denied commands were updated to the retained fork contract. These are intentional behavioral dispositions, not upstream parity claims. The reused High analyst narrowly rechecked the shell implementation against its intersecting-guards contract and found no concrete remaining defect; it did not rerun the reported tests. Full P2 acceptance and canonical loop construction remain pending.

Final combined clean-home validation: **246 passed, 6 skipped**, with explicit environment and network denied. Raw invocation/log/JUnit path and JUnit hash are recorded in p2-factory-validation.json; individual cases are in p2-factory-validation-cases.tsv.
