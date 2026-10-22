#!/usr/bin/env bash
# The application is Rust; this launcher performs no assessment logic.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -eq 0 ]]; then
  set -- explore
fi
case "${1:-}" in
  explore|select-teachers|hardware|compare-models|train|infer|serve|study-plan|study-run|study-select|study-evaluate|study-regrade|study-diagnose-training|distillation-arm-worker|collect-teacher-evidence|collect-teacher-content|collect-student-rollouts|evaluate)
    printf '%s\n' 'Preparing native hardware detection; download, cache and selection progress will follow.' >&2
    exec ./run-gpu.sh "$@"
    ;;
esac
# Cargo dependency acquisition is independent of application cache-only policy.
# Explicit application offline always takes precedence over online preparation.
cargo_network=(--offline)
if [[ ${LEC_CARGO_ONLINE:-0} == 1 || ${1:-} == bootstrap-online ]]; then
  cargo_network=()
fi
for argument in "$@"; do
  if [[ $argument == --offline ]]; then cargo_network=(--offline); fi
done
build_and_run() {
  local result=0
  cargo run --locked "${cargo_network[@]}" --release -- "$@" || result=$?
  if [[ $result != 0 && ${#cargo_network[@]} != 0 ]]; then
    printf '%s\n' 'Offline build failed. If Cargo reports a missing cached dependency, run cargo fetch --locked during online preparation (LEC_CARGO_ONLINE=1 enables online builds). Explicit --offline never fetches.' >&2
  fi
  exit "$result"
}
# Source checkouts must execute their current Rust sources. The bundled binary
# belongs to a separately inventoried delivery and may predate local changes.
if [[ -e .git || ${LEC_BUILD_FROM_SOURCE:-0} == 1 ]]; then
  printf '%s\n' 'Source checkout detected; Cargo will verify and reuse the current CPU build.' >&2
  build_and_run "$@"
fi
if [[ -x offline_bundle/bin/lec-distil ]] && offline_bundle/bin/lec-distil --version >/dev/null 2>&1; then
  exec offline_bundle/bin/lec-distil "$@"
fi
printf '%s\n' 'No compatible bundled executable is available; building the native CPU application.' >&2
build_and_run "$@"
