#!/bin/sh
# Rust detects resources, selects a compatible backend and runs the engine.
set -eu
cd "$(dirname "$0")"
case " $* " in
  *' --offline '*) exec cargo run --locked --offline --example platform_launch -- "$@" ;;
  *) exec cargo run --locked --example platform_launch -- "$@" ;;
esac
