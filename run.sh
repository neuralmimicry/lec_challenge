#!/usr/bin/env bash
# The application is Rust; this launcher performs no assessment logic.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -x offline_bundle/bin/lec-distil ]]; then
  exec offline_bundle/bin/lec-distil "$@"
fi
exec cargo run --locked --offline --release -- "$@"
