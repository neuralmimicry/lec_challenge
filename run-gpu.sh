#!/bin/sh
# Rust detects resources, selects a compatible backend and runs the engine.
set -eu
cd "$(dirname "$0")"
if [ "${LEC_CARGO_ONLINE:-0}" = 1 ]; then
  case " $* " in
    *' --offline '*) ;;
    *) exec cargo run --locked --example platform_launch -- "$@" ;;
  esac
fi
cargo run --locked --offline --example platform_launch -- "$@" || {
  result=$?
  printf '%s\n' 'Offline launcher exited unsuccessfully. If Cargo reported a missing cached dependency, prepare with cargo fetch --locked online; LEC_CARGO_ONLINE=1 permits online builds. Explicit --offline never fetches.' >&2
  exit "$result"
}
