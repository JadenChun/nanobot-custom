#!/bin/sh
# Reproduce the P0 staging baseline without reading mutable worktree source,
# HOME, or XDG state. Run from the repository root.

set -eu

base=63d646f731a505731e97909d114ce257c6de36bc
staging=4aa562d92828189b3fcefad079773efbab1d715a
python_bin=${P0_PYTHON:-python3.11}
uv_bin=${P0_UV:-uv}
repo_root=$(git rev-parse --show-toplevel)
freeze_file="$repo_root/docs/migration/p0-baseline-requirements.txt"
verify_helper="$repo_root/docs/migration/verify-p0-archive.py"

command -v git >/dev/null
command -v "$python_bin" >/dev/null
command -v "$uv_bin" >/dev/null
command -v sandbox-exec >/dev/null
test -f "$freeze_file"
test -f "$verify_helper"

"$python_bin" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("P0_PYTHON must be Python 3.11 or newer")
PY

if test -n "${P0_ARCHIVE_DIR:-}"; then
  p0_root=$P0_ARCHIVE_DIR
  archive_mode=reused
  test -d "$p0_root" || {
    printf '%s\n' "P0_ARCHIVE_DIR is not a directory: $p0_root" >&2
    exit 1
  }
else
  p0_root=$(mktemp -d /tmp/nanobot-p0-staging-4aa562d.XXXXXX)
  archive_mode=fresh
fi

mkdir -p "$p0_root/evidence"
run_dir=$(mktemp -d "$p0_root/evidence/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
setup_log="$run_dir/setup.log"
verification_log="$run_dir/source-verification.txt"
probe_log="$run_dir/environment-probes.log"
versions_log="$run_dir/versions.txt"
package_freeze="$run_dir/package-freeze.txt"
status_file="$run_dir/phase-status.txt"

report_failure() {
  phase=$1
  code=$2
  printf '%s\n' "phase=$phase" "exit=$code" >> "$status_file"
  printf '%s\n' "P0 archive: $p0_root" "Evidence: $run_dir" "failed phase: $phase" >&2
  exit "$code"
}

printf '%s\n' \
  "archive_mode=$archive_mode" \
  "archive=$p0_root" \
  "staging=$staging" \
  "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"

if test "$archive_mode" = fresh; then
  set +e
  git -C "$repo_root" archive --format=tar "$staging" | tar -x -C "$p0_root" \
    > "$run_dir/archive-setup.log" 2>&1
  archive_status=$?
  set -e
  if test "$archive_status" -ne 0; then
    report_failure archive-setup "$archive_status"
  fi
else
  printf '%s\n' 'reused archive; source verification runs before environment reuse' \
    > "$run_dir/archive-setup.log"
fi

set +e
"$python_bin" "$verify_helper" \
  --git git \
  --commit "$staging" \
  --archive "$p0_root" \
  --report "$verification_log"
verification_status=$?
set -e
if test "$verification_status" -ne 0; then
  report_failure source-verification "$verification_status"
fi
printf '%s\n' 'phase=source-verification' 'exit=0' >> "$status_file"

venv_python="$p0_root/.venv/bin/python"
if test "$archive_mode" = fresh; then
  set +e
  (
    set -e
    "$python_bin" -m venv "$p0_root/.venv"
    cd "$p0_root"
    "$uv_bin" pip install --python "$venv_python" -r "$freeze_file"
    "$uv_bin" pip install --python "$venv_python" --no-deps -e .
  ) > "$setup_log" 2>&1
  setup_status=$?
  set -e
else
  set +e
  test -x "$venv_python"
  setup_status=$?
  set -e
  printf '%s\n' 'reused archive; installation skipped after pinned-source verification' > "$setup_log"
fi
if test "$setup_status" -ne 0; then
  report_failure environment-setup "$setup_status"
fi
printf '%s\n' 'phase=environment-setup' 'exit=0' >> "$status_file"

freeze_hash=$(shasum -a 256 "$freeze_file" | awk '{print $1}')
set +e
{
  printf '%s\n' "freeze_file=$freeze_file" "freeze_sha256=$freeze_hash"
  "$venv_python" --version
  "$venv_python" -m pytest --version
  "$uv_bin" --version
  git --version
  "$uv_bin" pip freeze --python "$venv_python"
} > "$versions_log" 2>&1
versions_status=$?
if test "$versions_status" -eq 0; then
  "$uv_bin" pip freeze --python "$venv_python" > "$package_freeze" 2>> "$versions_log"
  package_status=$?
else
  package_status=$versions_status
fi
set -e
if test "$package_status" -ne 0; then
  report_failure environment-evidence "$package_status"
fi
package_hash=$(shasum -a 256 "$package_freeze" | awk '{print $1}')
printf '%s\n' "package_freeze_sha256=$package_hash" >> "$versions_log"
printf '%s\n' 'phase=environment-evidence' 'exit=0' >> "$status_file"

if test -n "${P0_TEST_PATHS_FILE:-}"; then
  test -f "$P0_TEST_PATHS_FILE" || report_failure test-selection 1
  cp "$P0_TEST_PATHS_FILE" "$run_dir/requested-tests.txt"
else
  {
    git -C "$repo_root" diff --name-only "$base" "$staging" -- tests
    printf '%s\n' \
      tests/agent/test_consolidate_offset.py \
      tests/agent/test_loop_save_turn.py \
      tests/agent/test_session_manager_history.py \
      tests/providers/test_azure_openai_provider.py
  } | sort -u > "$run_dir/requested-tests.txt"
fi

: > "$run_dir/selected-tests.txt"
while IFS= read -r test_file; do
  case "$test_file" in
    tests/*.py) ;;
    *)
      printf '%s\n' "selector must be a Python file below archive tests/: $test_file" >&2
      report_failure test-selection 1
      ;;
  esac
  case "/$test_file/" in
    *'/../'*|*'//'*|*'/./'*)
      printf '%s\n' "selector must stay below archive tests/: $test_file" >&2
      report_failure test-selection 1
      ;;
  esac
  test -f "$p0_root/$test_file" || {
    printf '%s\n' "selected test is absent from verified archive: $test_file" >&2
    report_failure test-selection 1
  }
  printf '%s\n' "$test_file" >> "$run_dir/selected-tests.txt"
done < "$run_dir/requested-tests.txt"

test_count=$(wc -l < "$run_dir/selected-tests.txt" | tr -d ' ')
test "$test_count" -gt 0 || report_failure test-selection 1
if test -z "${P0_TEST_PATHS_FILE:-}"; then
  test "$test_count" = 61 || {
    printf '%s\n' "expected 61 existing mapped files, found $test_count" >&2
    report_failure test-selection 1
  }
fi
printf '%s\n' 'phase=test-selection' 'exit=0' >> "$status_file"

run_home="$run_dir/runtime-home"
run_xdg_config="$run_dir/runtime-xdg-config"
run_xdg_cache="$run_dir/runtime-xdg-cache"
run_xdg_data="$run_dir/runtime-xdg-data"
mkdir -p "$run_home" "$run_xdg_config" "$run_xdg_cache" "$run_xdg_data"
profile='(version 1) (allow default) (deny network*) (allow network-inbound (local ip "localhost:*")) (allow network-outbound (remote ip "localhost:*"))'

set +e
(
  cd "$p0_root"
  env -i \
  PATH="$p0_root/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  HOME="$run_home" \
  XDG_CONFIG_HOME="$run_xdg_config" \
  XDG_CACHE_HOME="$run_xdg_cache" \
  XDG_DATA_HOME="$run_xdg_data" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH= \
  sandbox-exec -p "$profile" "$venv_python" - "$run_home" <<'PY'
from pathlib import Path
import socket
import sys

assert Path.home() == Path(sys.argv[1]), (Path.home(), sys.argv[1])
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.close()
probe = socket.socket()
try:
    probe.connect(("1.1.1.1", 443))
except PermissionError:
    print("home=run-local")
    print("loopback-bind=allowed")
    print("external-network=denied")
else:
    raise SystemExit("external network connection was allowed")
finally:
    probe.close()
PY
) > "$probe_log" 2>&1
probe_status=$?
set -e
if test "$probe_status" -ne 0; then
  report_failure environment-probe "$probe_status"
fi
printf '%s\n' 'phase=environment-probe' 'exit=0' >> "$status_file"

set --
while IFS= read -r test_file; do
  set -- "$@" "$test_file"
done < "$run_dir/selected-tests.txt"

set +e
(
  cd "$p0_root"
  env -i \
  PATH="$p0_root/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  HOME="$run_home" \
  XDG_CONFIG_HOME="$run_xdg_config" \
  XDG_CACHE_HOME="$run_xdg_cache" \
  XDG_DATA_HOME="$run_xdg_data" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH= \
  sandbox-exec -p "$profile" "$venv_python" -m pytest -ra \
    --junitxml="$run_dir/junit.xml" \
    "$@"
) > "$run_dir/pytest.log" 2>&1
test_status=$?
set -e
printf '%s\n' 'phase=pytest' "exit=$test_status" >> "$status_file"

printf '%s\n' "P0 archive: $p0_root"
printf '%s\n' "Evidence: $run_dir"
printf '%s\n' "Source verification: $verification_log"
printf '%s\n' "Setup: $setup_log"
printf '%s\n' "JUnit: $run_dir/junit.xml"
printf '%s\n' "Log: $run_dir/pytest.log"
printf '%s\n' "pytest exit: $test_status"
exit "$test_status"
