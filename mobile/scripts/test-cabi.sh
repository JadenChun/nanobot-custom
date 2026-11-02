#!/usr/bin/env bash
# Host-only proof that the typed C ABI links and fails closed. This does not
# claim Android execution or a live tailnet path.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=mobile-env.sh
source "${script_dir}/mobile-env.sh"
mobile_require_toolchain

work_dir="${1:-${MOBILE_ROOT}/evidence/cabi-host}"
mkdir -p "${work_dir}"
cd "${MOBILE_ROOT}/native-network"

CGO_ENABLED=1 "${MOBILE_TOOLCHAIN}/go/bin/go" build -buildmode=c-shared -o "${work_dir}/libnanobot_network.so" ./cmd/lib
cc -std=c11 -Wall -Wextra -Werror cabi/smoke.c \
  -I include \
  -L "${work_dir}" -lnanobot_network \
  -Wl,-rpath,"${work_dir}" \
  -o "${work_dir}/cabi-smoke"
"${work_dir}/cabi-smoke"
sha256sum "${work_dir}/libnanobot_network.so" "${work_dir}/cabi-smoke"
