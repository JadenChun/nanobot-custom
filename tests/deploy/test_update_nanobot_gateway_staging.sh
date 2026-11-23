#!/usr/bin/env bash
# Focused transaction test for deploy/scripts/update_nanobot_gateway.sh (Q01-F05).
#
# Author cases C3 and C4 come from
# a3-evidence/a3-reconciliation-2026-11-14/req-a39b/CASES.md. Scenario, inputs
# and expected outcomes are preserved; only the fixture plumbing is added.
#
# Everything runs against a disposable fixture: a local origin, a live clone, a
# fake deploy user, fake systemctl/chown/python3 and a fake readiness probe. No
# systemd unit, live gateway, config or credential is touched.
#
# Usage: bash tests/deploy/test_update_nanobot_gateway_staging.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_UNDER_TEST="${REPO_ROOT}/deploy/scripts/update_nanobot_gateway.sh"
PYTHON_BIN="$(command -v python3)"

failures=0
checks=0
fixture_roots=()

fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

pass() {
  checks=$((checks + 1))
}

assert_eq() {
  local actual="$1" expected="$2" label="$3"
  if [[ "${actual}" == "${expected}" ]]; then
    pass
  else
    fail "${label}: expected '${expected}', got '${actual}'"
  fi
}

assert_ne() {
  local actual="$1" unexpected="$2" label="$3"
  if [[ "${actual}" != "${unexpected}" ]]; then
    pass
  else
    fail "${label}: expected a value other than '${unexpected}'"
  fi
}

assert_match() {
  local actual="$1" pattern="$2" label="$3"
  if [[ "${actual}" =~ ${pattern} ]]; then
    pass
  else
    fail "${label}: '${actual}' does not match '${pattern}'"
  fi
}

assert_empty() {
  local actual="$1" label="$2"
  if [[ -z "${actual}" ]]; then
    pass
  else
    fail "${label}: expected empty, got '${actual}'"
  fi
}

cleanup() {
  local root
  for root in "${fixture_roots[@]}"; do
    rm -rf -- "${root}"
  done
}
trap cleanup EXIT

tree_hash() {
  ( cd "$1" && find . -path ./.git -prune -o -type f -print0 \
      | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1 )
}

json_field() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
for part in sys.argv[2].split("."):
    value = value[part]
if value is None:
    print("")
elif isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
PY
}

# build_fixture [candidate]
# candidate=1 (default) pushes a second commit to origin; candidate=0 leaves the
# live clone current so the unchanged-code path can be exercised.
build_fixture() {
  local with_candidate="${1:-1}"
  ROOT="$(mktemp -d "${TMPDIR:-/tmp}/a39-deploy-fixture.XXXXXX")"
  fixture_roots+=("${ROOT}")
  BIN="${ROOT}/bin"
  STATE="${ROOT}/state"
  BASE="${ROOT}/base"
  LIVE="${ROOT}/live"
  ORIGIN="${ROOT}/origin.git"
  SEED="${ROOT}/seed"
  CONFIG="${ROOT}/config.json"
  mkdir -p "${BIN}" "${STATE}" "${BASE}"

  cat > "${BIN}/id" <<'FAKE'
#!/usr/bin/env bash
exit 0
FAKE

  cat > "${BIN}/runuser" <<'FAKE'
#!/usr/bin/env bash
shift 2
exec "$@"
FAKE

  cat > "${BIN}/systemctl" <<'FAKE'
#!/usr/bin/env bash
state="${FIXTURE_STATE:?}"
case "${1:-}" in
  is-active)
    [[ -f "${state}/service_active" ]] || exit 0
    [[ "$(cat "${state}/service_active")" == "1" ]] && exit 0
    exit 3
    ;;
  status)
    echo "fixture nanobot-gateway status"
    exit 0
    ;;
  *)
    exit 0
    ;;
esac
FAKE

  cat > "${BIN}/chown" <<'FAKE'
#!/usr/bin/env bash
exit 0
FAKE

  # Fake interpreter factory: "python3 -m venv DIR" creates a venv-shaped tree
  # whose commands record calls and honour per-check failure flags.
  cat > "${BIN}/python3" <<'FAKE'
#!/usr/bin/env bash
state="${FIXTURE_STATE:?}"
if [[ "${1:-}" == "-m" && "${2:-}" == "venv" ]]; then
  target="${3:?}"
  mkdir -p "${target}/bin"
  cat > "${target}/bin/python" <<'INNER'
#!/usr/bin/env bash
state="${FIXTURE_STATE:?}"
printf 'python %s\n' "$*" >> "${state}/python_calls.log"
for marker in default_webui_dist_dir load_config; do
  if [[ "$*" == *"${marker}"* && -f "${state}/${marker}_fail" ]]; then
    echo "fixture: ${marker} check failed" >&2
    exit 1
  fi
done
exit 0
INNER
  cat > "${target}/bin/pip" <<'INNER'
#!/usr/bin/env bash
state="${FIXTURE_STATE:?}"
printf 'pip %s\n' "$*" >> "${state}/pip_calls.log"
if [[ -f "${state}/pip_fail" ]]; then
  echo "fixture: dependency install failed" >&2
  exit 1
fi
exit 0
INNER
  cat > "${target}/bin/nanobot" <<'INNER'
#!/usr/bin/env bash
state="${FIXTURE_STATE:?}"
printf 'nanobot %s\n' "$*" >> "${state}/nanobot_calls.log"
if [[ -f "${state}/startup_fail" ]]; then
  echo "fixture: candidate start failed" >&2
  exit 1
fi
echo "nanobot 0.0.0-fixture"
exit 0
INNER
  chmod +x "${target}/bin/python" "${target}/bin/pip" "${target}/bin/nanobot"
  exit 0
fi
echo "fixture python3: unsupported invocation: $*" >&2
exit 2
FAKE

  chmod +x "${BIN}"/*

  # Fake readiness probe: fails while readiness_fail_remaining is above zero.
  cat > "${ROOT}/readiness.sh" <<'FAKE'
#!/usr/bin/env bash
state="${FIXTURE_STATE:?}"
printf 'readiness\n' >> "${state}/readiness_calls.log"
remaining="$(cat "${state}/readiness_fail_remaining" 2>/dev/null || echo 0)"
if [[ "${remaining}" -gt 0 ]]; then
  echo $((remaining - 1)) > "${state}/readiness_fail_remaining"
  echo "fixture: not ready" >&2
  exit 1
fi
exit 0
FAKE
  chmod +x "${ROOT}/readiness.sh"

  git init -q --bare "${ORIGIN}"
  git init -q "${SEED}"
  git -C "${SEED}" config user.email fixture@example.invalid
  git -C "${SEED}" config user.name Fixture
  mkdir -p "${SEED}/nanobot/cli" "${SEED}/deploy/scripts"
  printf '[project]\nname = "fixture"\n' > "${SEED}/pyproject.toml"
  printf 'VERSION = "1"\n' > "${SEED}/nanobot/cli/commands.py"
  cat > "${SEED}/deploy/scripts/deploy_nanobot_gateway.sh" <<'STUB'
#!/usr/bin/env bash
# Fixture deploy adapter: records the revision it was asked to deploy.
set -euo pipefail
state="${FIXTURE_STATE:?}"
source_dir="${1:?}"
revision="$(git -C "${source_dir}" rev-parse HEAD)"
printf '%s\n' "${revision}" > "${state}/serving_revision"
printf '%s\n' "${revision}" >> "${state}/deploy_calls.log"
if [[ -f "${state}/deploy_fail_remaining" ]]; then
  remaining="$(cat "${state}/deploy_fail_remaining")"
  if [[ "${remaining}" -gt 0 ]]; then
    echo $((remaining - 1)) > "${state}/deploy_fail_remaining"
    echo "fixture: deploy failed" >&2
    exit 1
  fi
fi
[[ -f "${state}/deploy_exit" ]] && exit "$(cat "${state}/deploy_exit")"
exit 0
STUB
  chmod +x "${SEED}/deploy/scripts/deploy_nanobot_gateway.sh"
  git -C "${SEED}" add -A
  git -C "${SEED}" commit -q -m "before"
  git -C "${SEED}" remote add origin "${ORIGIN}"
  git -C "${SEED}" push -q origin HEAD:refs/heads/main
  git -C "${ORIGIN}" symbolic-ref HEAD refs/heads/main

  git clone -q "${ORIGIN}" "${LIVE}"
  git -C "${LIVE}" config user.email fixture@example.invalid
  git -C "${LIVE}" config user.name Fixture

  printf '{"gateway": {"host": "127.0.0.1", "port": 18790}}\n' > "${CONFIG}"

  if [[ "${with_candidate}" == "1" ]]; then
    printf 'VERSION = "2"\n' > "${SEED}/nanobot/cli/commands.py"
    git -C "${SEED}" add -A
    git -C "${SEED}" commit -q -m "candidate"
    git -C "${SEED}" push -q origin HEAD:refs/heads/main
  fi

  printf '1\n' > "${STATE}/service_active"
  printf '0\n' > "${STATE}/readiness_fail_remaining"
  export FIXTURE_STATE="${STATE}"
}

run_update() {
  local output_file="$1"
  set +e
  env -i \
    PATH="${BIN}:/usr/bin:/bin" \
    HOME="${ROOT}" \
    FIXTURE_STATE="${STATE}" \
    MARKETING_AGENT_BASE_DIR="${BASE}" \
    MARKETING_AGENT_NANOBOT_DIR="${LIVE}" \
    MARKETING_AGENT_NANOBOT_BRANCH=main \
    MARKETING_AGENT_NANOBOT_CONFIG="${CONFIG}" \
    MARKETING_AGENT_DEPLOY_USER=marketing-agent \
    MARKETING_AGENT_READINESS_CMD="${ROOT}/readiness.sh" \
    unshare -r bash "${SCRIPT_UNDER_TEST}" > "${output_file}" 2>&1
  RUN_STATUS=$?
}

receipt() {
  printf '%s\n' "${BASE}/receipts/latest.json"
}

deploy_calls() {
  if [[ -f "${STATE}/deploy_calls.log" ]]; then
    wc -l < "${STATE}/deploy_calls.log"
  else
    echo 0
  fi
}

# --- C3: pre-switch verification failure leaves the live release intact ------
# Setup: verified LIVE, candidate fails exactly one pre-switch check class.
# Expect: nonzero exit, tree_hash(LIVE) unchanged, no switch, receipt states the
# pre-switch failure and neither success nor rollback.
c3_pre_switch_failure() {
  local check="$1" flag="$2"
  echo "C3 (${check} failure): pre-switch failure must leave the live release intact"
  build_fixture 1
  local before_head candidate_head before_tree
  before_head="$(git -C "${LIVE}" rev-parse HEAD)"
  git -C "${LIVE}" fetch -q origin
  candidate_head="$(git -C "${LIVE}" rev-parse origin/main)"
  before_tree="$(tree_hash "${LIVE}")"
  : > "${STATE}/${flag}"

  run_update "${ROOT}/run.out"

  assert_ne "${RUN_STATUS}" "0" "C3/${check} exit status"
  assert_eq "$(git -C "${LIVE}" rev-parse HEAD)" "${before_head}" \
    "C3/${check} live revision unchanged"
  assert_eq "$(git -C "${LIVE}" symbolic-ref -q --short HEAD)" "main" \
    "C3/${check} live branch unchanged"
  assert_eq "$(tree_hash "${LIVE}")" "${before_tree}" \
    "C3/${check} tree_hash(LIVE) unchanged"
  assert_eq "$(deploy_calls)" "0" "C3/${check} candidate never served"

  local file
  file="$(receipt)"
  assert_eq "$([[ -f "${file}" ]] && echo yes || echo no)" "yes" \
    "C3/${check} receipt exists"
  assert_eq "$(json_field "${file}" outcome)" "pre_switch_failed" \
    "C3/${check} receipt outcome"
  assert_eq "$(json_field "${file}" phase)" "pre_switch" \
    "C3/${check} receipt phase"
  assert_eq "$(json_field "${file}" failed_check)" "${check}" \
    "C3/${check} receipt failed check"
  assert_eq "$(json_field "${file}" source.revision)" "${candidate_head}" \
    "C3/${check} receipt source revision"
  assert_eq "$(json_field "${file}" source.previous_revision)" "${before_head}" \
    "C3/${check} receipt previous revision"
  assert_eq "$(json_field "${file}" previous_target)" "${before_head}" \
    "C3/${check} receipt previous target"
  assert_eq "$(json_field "${file}" current_target)" "${before_head}" \
    "C3/${check} receipt current target"
  assert_eq "$(json_field "${file}" rollback.attempted)" "false" \
    "C3/${check} receipt rollback not attempted"
  assert_match "$(json_field "${file}" artifact.content_sha256)" '^[0-9a-f]{64}$' \
    "C3/${check} artifact identity recorded"
  assert_eq "$(json_field "${file}" artifact.revision)" "${candidate_head}" \
    "C3/${check} artifact revision"
  assert_ne "$(json_field "${file}" outcome)" "switched" "C3/${check} no success outcome"
  assert_ne "$(json_field "${file}" outcome)" "rolled_back" "C3/${check} no rollback outcome"
}

# --- C4: post-switch failure restores the previous verified target -----------
# Setup: verified LIVE, candidate passes every pre-switch check, switches, then
# fails a post-switch readiness check.
# Expect: nonzero exit, active target serves pre-update content and version, the
# failed candidate is not serving, receipt states the rollback.
c4_post_switch_rollback() {
  echo "C4: post-switch readiness failure must restore the previous verified target"
  build_fixture 1
  local before_head candidate_head before_tree
  before_head="$(git -C "${LIVE}" rev-parse HEAD)"
  git -C "${LIVE}" fetch -q origin
  candidate_head="$(git -C "${LIVE}" rev-parse origin/main)"
  before_tree="$(tree_hash "${LIVE}")"
  # Fail exactly the first post-switch readiness probe; the rollback recheck
  # then passes.
  printf '1\n' > "${STATE}/readiness_fail_remaining"

  run_update "${ROOT}/run.out"

  assert_ne "${RUN_STATUS}" "0" "C4 exit status"
  assert_eq "$(git -C "${LIVE}" rev-parse HEAD)" "${before_head}" \
    "C4 checkout restored to previous revision"
  assert_empty "$(git -C "${LIVE}" symbolic-ref -q --short HEAD)" \
    "C4 restored target is a detached previous revision"
  assert_eq "$(tree_hash "${LIVE}")" "${before_tree}" \
    "C4 active tree_hash equals pre-update tree_hash"
  assert_eq "$(cat "${STATE}/serving_revision")" "${before_head}" \
    "C4 active target serves pre-update content"
  assert_eq "$(deploy_calls)" "2" "C4 candidate deployed once, previous target redeployed"
  assert_eq "$(grep -c "${candidate_head}" "${STATE}/deploy_calls.log")" "1" \
    "C4 failed candidate was not left serving"

  local file
  file="$(receipt)"
  assert_eq "$([[ -f "${file}" ]] && echo yes || echo no)" "yes" "C4 receipt exists"
  assert_eq "$(json_field "${file}" outcome)" "rolled_back" "C4 receipt outcome"
  assert_eq "$(json_field "${file}" phase)" "post_switch" "C4 receipt phase"
  assert_eq "$(json_field "${file}" failed_check)" "readiness" "C4 receipt failed check"
  assert_eq "$(json_field "${file}" source.revision)" "${candidate_head}" \
    "C4 receipt source revision"
  assert_eq "$(json_field "${file}" previous_target)" "${before_head}" \
    "C4 receipt previous target"
  assert_eq "$(json_field "${file}" current_target)" "${before_head}" \
    "C4 receipt current target"
  assert_eq "$(json_field "${file}" rollback.attempted)" "true" \
    "C4 receipt rollback attempted"
  assert_eq "$(json_field "${file}" rollback.restored)" "true" \
    "C4 receipt rollback restored"
  assert_eq "$(json_field "${file}" rollback.target)" "${before_head}" \
    "C4 receipt rollback target"
  assert_ne "$(json_field "${file}" outcome)" "switched" "C4 no success outcome"
}

# --- A failing deploy command is not masked by an active old unit ------------
# The pre-fix script ran the deploy adapter without checking its exit status and
# then treated "is-active" as success, so a failed deploy reported success. The
# same check class (deploy) must now restore the previous target.
deploy_command_failed() {
  echo "C4c: a failing deploy command must not be reported as success"
  build_fixture 1
  local before_head
  before_head="$(git -C "${LIVE}" rev-parse HEAD)"
  printf '1\n' > "${STATE}/deploy_fail_remaining"

  run_update "${ROOT}/run.out"

  assert_ne "${RUN_STATUS}" "0" "C4c exit status"
  assert_eq "$(git -C "${LIVE}" rev-parse HEAD)" "${before_head}" \
    "C4c checkout restored"
  assert_eq "$(cat "${STATE}/serving_revision")" "${before_head}" \
    "C4c previous target serving"
  assert_eq "$(json_field "$(receipt)" outcome)" "rolled_back" "C4c receipt outcome"
  assert_eq "$(json_field "$(receipt)" failed_check)" "deploy" \
    "C4c receipt failed check"
  assert_ne "$(json_field "$(receipt)" outcome)" "switched" "C4c no success outcome"
}

# --- Failed rollback is a reported failure that needs operator action ---------
c4_rollback_failed() {
  echo "C4b: a failed rollback must exit nonzero and require operator action"
  build_fixture 1
  printf '99\n' > "${STATE}/readiness_fail_remaining"

  run_update "${ROOT}/run.out"

  assert_ne "${RUN_STATUS}" "0" "C4b exit status"
  assert_eq "$(grep -c 'ROLLBACK FAILED' "${ROOT}/run.out")" "1" \
    "C4b operator message printed"
  assert_eq "$(json_field "$(receipt)" outcome)" "rollback_failed" "C4b receipt outcome"
  assert_eq "$(json_field "$(receipt)" rollback.restored)" "false" \
    "C4b receipt rollback not restored"
  assert_ne "$(json_field "$(receipt)" outcome)" "switched" "C4b no success outcome"
}

# --- Control: a verified candidate switches and records the outcome -----------
switched_outcome() {
  echo "Control: verified candidate switches and records the switched outcome"
  build_fixture 1
  local before_head candidate_head
  before_head="$(git -C "${LIVE}" rev-parse HEAD)"
  git -C "${LIVE}" fetch -q origin
  candidate_head="$(git -C "${LIVE}" rev-parse origin/main)"

  run_update "${ROOT}/run.out"

  assert_eq "${RUN_STATUS}" "0" "control exit status"
  assert_eq "$(git -C "${LIVE}" rev-parse HEAD)" "${candidate_head}" \
    "control live revision switched"
  assert_eq "$(cat "${STATE}/serving_revision")" "${candidate_head}" \
    "control candidate serving"
  assert_eq "$(json_field "$(receipt)" outcome)" "switched" "control receipt outcome"
  assert_eq "$(json_field "$(receipt)" phase)" "" "control receipt phase null"
  assert_eq "$(json_field "$(receipt)" previous_target)" "${before_head}" \
    "control receipt previous target"
  assert_eq "$(json_field "$(receipt)" current_target)" "${candidate_head}" \
    "control receipt current target"
  assert_eq "$(json_field "$(receipt)" rollback.attempted)" "false" \
    "control receipt rollback not attempted"
}

# --- Control: unchanged code and an active service skip the update -----------
unchanged_skip() {
  echo "Control: unchanged code and an active service skip the update"
  build_fixture 0
  local before_head
  before_head="$(git -C "${LIVE}" rev-parse HEAD)"

  run_update "${ROOT}/run.out"

  assert_eq "${RUN_STATUS}" "0" "skip exit status"
  assert_eq "$(grep -c 'skipping Nanobot redeploy' "${ROOT}/run.out")" "1" \
    "skip message"
  assert_eq "$(git -C "${LIVE}" rev-parse HEAD)" "${before_head}" "skip live revision"
  assert_eq "$(deploy_calls)" "0" "skip did not deploy"
}

if ! unshare -r true >/dev/null 2>&1; then
  echo "SKIP: unshare -r is unavailable; the root guard cannot be exercised safely" >&2
  exit 0
fi
if [[ ! -f "${SCRIPT_UNDER_TEST}" ]]; then
  echo "FAIL: missing ${SCRIPT_UNDER_TEST}" >&2
  exit 1
fi

c3_pre_switch_failure dependencies pip_fail
c3_pre_switch_failure bundle default_webui_dist_dir_fail
c3_pre_switch_failure config load_config_fail
c4_post_switch_rollback
deploy_command_failed
c4_rollback_failed
switched_outcome
unchanged_skip

echo "${checks} checks passed, ${failures} failed"
[[ "${failures}" -eq 0 ]] || exit 1
