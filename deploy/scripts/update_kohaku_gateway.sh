#!/usr/bin/env bash
# Staged Kohaku gateway update.
#
# Transaction:
#   1. fetch the configured branch and resolve the candidate revision;
#   2. stage the candidate from Git into a separate release directory, never
#      from the live tree (<base>/releases/<first 12 revision characters>);
#   3. verify dependencies, packaged WebUI bundle, config compatibility and
#      candidate startup inside the staged release;
#   4. only then advance the live checkout, deploy it and probe readiness;
#   5. on a post-switch failure restore the previous verified revision,
#      redeploy it and recheck readiness;
#   6. write a receipt under <base>/receipts with the source, artifact,
#      previous/current target and outcome.
#
# Contract used by tests/deploy/test_update_kohaku_gateway_staging.sh:
#   live target      MARKETING_AGENT_KOHAKU_DIR (git checkout plus .venv, the
#                    directory the systemd unit runs)
#   candidate        MARKETING_AGENT_RELEASES_DIR/<first 12 revision characters>
#   receipt          MARKETING_AGENT_RECEIPT_DIR/update-<utc>-<short>-<pid>.json
#                    plus latest.json
#   outcomes         pre_switch_failed | switched | rolled_back | rollback_failed
#   readiness        systemctl is-active kohaku-gateway plus the gateway /health
#                    contract (kohaku.gateway.runtime.gateway_health_ready)
#   rehearsal probe  MARKETING_AGENT_READINESS_CMD replaces the /health probe
set -euo pipefail

BASE_DIR="${MARKETING_AGENT_BASE_DIR:-/opt/marketing-agent}"
DEPLOY_USER="${MARKETING_AGENT_DEPLOY_USER:-marketing-agent}"
KOHAKU_DIR="${MARKETING_AGENT_KOHAKU_DIR:-${BASE_DIR}/kohaku}"
KOHAKU_BRANCH="${MARKETING_AGENT_KOHAKU_BRANCH:-main}"
RELEASES_DIR="${MARKETING_AGENT_RELEASES_DIR:-${BASE_DIR}/releases}"
RECEIPT_DIR="${MARKETING_AGENT_RECEIPT_DIR:-${BASE_DIR}/receipts}"
CONFIG_FILE="${MARKETING_AGENT_KOHAKU_CONFIG:-${BASE_DIR}/.kohaku/config.json}"
GATEWAY_HOST="${MARKETING_AGENT_GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${MARKETING_AGENT_GATEWAY_PORT:-18790}"
READINESS_CMD="${MARKETING_AGENT_READINESS_CMD:-}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script as root or with sudo." >&2
  exit 1
fi
if ! id "${DEPLOY_USER}" >/dev/null 2>&1; then
  echo "Deploy user does not exist: ${DEPLOY_USER}" >&2
  exit 1
fi
if [[ ! -d "${KOHAKU_DIR}/.git" ]]; then
  echo "Kohaku repo is missing: ${KOHAKU_DIR}" >&2
  exit 1
fi

run_as_deploy_user() {
  runuser -u "${DEPLOY_USER}" -- "$@"
}

json_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '%s' "${value}"
}

json_string_or_null() {
  if [[ -z "$1" ]]; then
    printf 'null'
  else
    printf '"%s"' "$(json_escape "$1")"
  fi
}

service_active() {
  systemctl is-active --quiet kohaku-gateway
}

current_head() {
  run_as_deploy_user git -C "${KOHAKU_DIR}" rev-parse HEAD
}

release_digest() {
  ( cd "$1" && find . -type f -not -path './.venv/*' -print0 \
      | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1 )
}

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
candidate_head=""
release_dir=""
artifact_sha256=""
failed_check=""

write_receipt() {
  local outcome="$1"
  local check="$2"
  local phase="$3"
  local current_target="$4"
  local rollback_attempted="$5"
  local rollback_restored="$6"
  local rollback_target="$7"
  local stamp file temporary
  install -d -m 0750 "${RECEIPT_DIR}"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  file="${RECEIPT_DIR}/update-${stamp}-${candidate_head:0:12}-$$.json"
  temporary="${file}.tmp"
  {
    printf '{\n'
    printf '  "schema_version": 1,\n'
    printf '  "started_at": "%s",\n' "$(json_escape "${started_at}")"
    printf '  "finished_at": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "source": {"branch": "%s", "revision": "%s", "previous_revision": "%s"},\n' \
      "$(json_escape "${KOHAKU_BRANCH}")" "$(json_escape "${candidate_head}")" \
      "$(json_escape "${before_head}")"
    printf '  "artifact": {"release_dir": "%s", "content_sha256": "%s", "revision": "%s"},\n' \
      "$(json_escape "${release_dir}")" "$(json_escape "${artifact_sha256}")" \
      "$(json_escape "${candidate_head}")"
    printf '  "live_dir": "%s",\n' "$(json_escape "${KOHAKU_DIR}")"
    printf '  "previous_target": "%s",\n' "$(json_escape "${before_head}")"
    printf '  "current_target": "%s",\n' "$(json_escape "${current_target}")"
    printf '  "outcome": "%s",\n' "${outcome}"
    printf '  "failed_check": %s,\n' "$(json_string_or_null "${check}")"
    printf '  "phase": %s,\n' "$(json_string_or_null "${phase}")"
    printf '  "rollback": {"attempted": %s, "restored": %s, "target": %s}\n' \
      "${rollback_attempted}" "${rollback_restored}" \
      "$(json_string_or_null "${rollback_target}")"
    printf '}\n'
  } > "${temporary}"
  mv "${temporary}" "${file}"
  cp "${file}" "${RECEIPT_DIR}/latest.json"
  echo "Outcome receipt: ${file} (${outcome})"
}

stage_candidate() {
  release_dir="${RELEASES_DIR}/${candidate_head:0:12}"
  case "${release_dir}" in
    "${RELEASES_DIR}"/?*) ;;
    *)
      echo "Refusing to stage outside the release root: ${release_dir}" >&2
      return 1
      ;;
  esac
  if [[ -e "${release_dir}" ]] && ! rm -rf -- "${release_dir}"; then
    return 1
  fi
  if ! install -d -m 0750 "${release_dir}"; then
    return 1
  fi
  # Immutable source: the committed tree of the verified revision, not the
  # working tree of the live checkout.
  if ! run_as_deploy_user git -C "${KOHAKU_DIR}" archive --format=tar \
      "${candidate_head}" | tar -xf - -C "${release_dir}"; then
    return 1
  fi
  if ! chown -R "${DEPLOY_USER}:${DEPLOY_USER}" "${release_dir}"; then
    return 1
  fi
  artifact_sha256="$(release_digest "${release_dir}")"
}

verify_candidate() {
  local venv="${release_dir}/.venv"
  if ! run_as_deploy_user python3 -m venv "${venv}"; then
    failed_check=dependencies
    return 1
  fi
  if ! run_as_deploy_user "${venv}/bin/pip" install --quiet "${release_dir}"; then
    failed_check=dependencies
    return 1
  fi
  if ! run_as_deploy_user "${venv}/bin/python" -c 'import sys
from pathlib import Path
from kohaku.webui.build import default_webui_dist_dir
raise SystemExit(0 if (Path(default_webui_dist_dir()) / "index.html").is_file() else 1)'; then
    failed_check=bundle
    return 1
  fi
  if [[ -f "${CONFIG_FILE}" ]]; then
    if ! run_as_deploy_user "${venv}/bin/python" -c 'import sys
from pathlib import Path
from kohaku.config.loader import load_config
load_config(Path(sys.argv[1]))' "${CONFIG_FILE}"; then
      failed_check=config
      return 1
    fi
  fi
  if ! run_as_deploy_user "${venv}/bin/kohaku" --version >/dev/null; then
    failed_check=readiness
    return 1
  fi
}

gateway_ready() {
  if ! systemctl is-active --quiet kohaku-gateway; then
    return 1
  fi
  if [[ -n "${READINESS_CMD}" ]]; then
    if ! bash -c "${READINESS_CMD}"; then
      return 1
    fi
    return 0
  fi
  run_as_deploy_user "${KOHAKU_DIR}/.venv/bin/python" -c 'import sys
from kohaku.gateway.runtime import gateway_health_ready
raise SystemExit(0 if gateway_health_ready(sys.argv[1], int(sys.argv[2])) else 1)' \
    "${GATEWAY_HOST}" "${GATEWAY_PORT}"
}

advance_source() {
  if run_as_deploy_user git -C "${KOHAKU_DIR}" rev-parse --verify --quiet \
      "${KOHAKU_BRANCH}" >/dev/null; then
    if ! run_as_deploy_user git -C "${KOHAKU_DIR}" switch "${KOHAKU_BRANCH}"; then
      return 1
    fi
  else
    if ! run_as_deploy_user git -C "${KOHAKU_DIR}" switch -c "${KOHAKU_BRANCH}" \
        --track "origin/${KOHAKU_BRANCH}"; then
      return 1
    fi
  fi
  # Fast-forward to exactly the revision that was staged and verified. No
  # second fetch can swap the artifact between verification and the switch.
  if ! run_as_deploy_user git -C "${KOHAKU_DIR}" merge --ff-only "${candidate_head}"; then
    return 1
  fi
  after_head="$(run_as_deploy_user git -C "${KOHAKU_DIR}" rev-parse HEAD)"
  if [[ "${after_head}" != "${candidate_head}" ]]; then
    echo "Live checkout did not advance to the verified revision." >&2
    return 1
  fi
}

deploy_candidate() {
  if ! bash "${KOHAKU_DIR}/deploy/scripts/deploy_kohaku_gateway.sh" "${KOHAKU_DIR}"; then
    failed_check=deploy
    return 1
  fi
  if ! gateway_ready; then
    failed_check=readiness
    return 1
  fi
}

rollback_candidate() {
  echo "Rolling back Kohaku to ${before_head}..." >&2
  if ! run_as_deploy_user git -C "${KOHAKU_DIR}" switch --detach "${before_head}"; then
    echo "ROLLBACK FAILED: restore ${before_head} manually and inspect kohaku-gateway." >&2
    return 1
  fi
  if deploy_candidate; then
    echo "Rollback restored ${before_head} and the gateway is active." >&2
    return 0
  fi
  echo "ROLLBACK FAILED: restore ${before_head} manually and inspect kohaku-gateway." >&2
  return 1
}

restore_previous_target() {
  local check="$1"
  if ! rollback_candidate; then
    write_receipt rollback_failed "${check}" post_switch "$(current_head)" true false ""
    echo "The previous verified target could not be restored; operator action is required." >&2
    exit 1
  fi
  write_receipt rolled_back "${check}" post_switch "$(current_head)" true true "${before_head}"
  echo "Update failed after the switch during ${check}; restored ${before_head}." >&2
  exit 1
}

before_head="$(run_as_deploy_user git -C "${KOHAKU_DIR}" rev-parse HEAD)"
echo "Updating Kohaku (${KOHAKU_BRANCH})..."
run_as_deploy_user git -C "${KOHAKU_DIR}" fetch origin
if ! run_as_deploy_user git -C "${KOHAKU_DIR}" rev-parse --verify --quiet \
    "origin/${KOHAKU_BRANCH}" >/dev/null; then
  echo "Remote branch is missing: origin/${KOHAKU_BRANCH}" >&2
  exit 1
fi
candidate_head="$(run_as_deploy_user git -C "${KOHAKU_DIR}" rev-parse "origin/${KOHAKU_BRANCH}")"

if [[ "${before_head}" == "${candidate_head}" ]] && service_active; then
  echo "Kohaku code unchanged and service is active; skipping Kohaku redeploy."
  systemctl status kohaku-gateway --no-pager || true
  exit 0
fi

if [[ "${before_head}" == "${candidate_head}" ]]; then
  echo "Kohaku code unchanged, but service is not active; deploying gateway service..."
else
  echo "Kohaku updated: ${before_head} -> ${candidate_head}; staging candidate..."
fi

if ! stage_candidate; then
  write_receipt pre_switch_failed stage pre_switch "${before_head}" false false ""
  echo "Update failed while staging the candidate; live release unchanged." >&2
  exit 1
fi

if ! verify_candidate; then
  write_receipt pre_switch_failed "${failed_check}" pre_switch "${before_head}" false false ""
  echo "Update failed before the switch during ${failed_check} verification; live release unchanged." >&2
  echo "Staged candidate retained for inspection: ${release_dir}" >&2
  exit 1
fi

echo "Candidate verified in ${release_dir}; switching the live release..."
if ! advance_source; then
  if [[ "$(current_head)" == "${before_head}" ]]; then
    write_receipt pre_switch_failed source pre_switch "${before_head}" false false ""
    echo "Update failed while advancing the live checkout; live release unchanged." >&2
    exit 1
  fi
  restore_previous_target source
fi

if ! deploy_candidate; then
  echo "Deploy or readiness failed after the switch; restoring the previous target..." >&2
  restore_previous_target "${failed_check}"
fi

write_receipt switched "" "" "${after_head}" false false ""
echo "Kohaku gateway deployed and ready: ${before_head} -> ${after_head}."
systemctl status kohaku-gateway --no-pager || true
