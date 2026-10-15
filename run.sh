#!/usr/bin/env bash
# The application is Rust; this launcher performs no assessment logic.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -eq 0 ]]; then
  set -- explore
fi
case "${1:-}" in
  explore|select-teachers|hardware|compare-models|train|infer|serve|study-plan|study-run|study-select|study-evaluate|distillation-arm-worker|collect-teacher-evidence|collect-teacher-content|collect-student-rollouts|evaluate)
    printf '%s\n' 'Preparing native hardware detection; download, cache and selection progress will follow.' >&2
    exec ./run-gpu.sh "$@"
    ;;
esac
if [[ -x offline_bundle/bin/lec-distil ]] && offline_bundle/bin/lec-distil --version >/dev/null 2>&1; then
  exec offline_bundle/bin/lec-distil "$@"
fi
printf '%s\n' 'No compatible bundled executable is available; building the native CPU application.' >&2
exec cargo run --locked --offline --release -- "$@"
