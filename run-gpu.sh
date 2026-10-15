#!/bin/sh
# Native Rust preparation and execution; no credential file or Python runtime.
set -eu
cd "$(dirname "$0")"
lec_cuda_toolkit="${LEC_CUDA_TOOLKIT:-/usr/local/cuda-12.9}"
lec_cuda_overlay="$(pwd -P)/state/cuda-compat"
CARGO_TARGET_DIR=target cargo run --locked --offline --example cuda_compat -- "$lec_cuda_toolkit" "$lec_cuda_overlay"
export CUDA_ROOT="$lec_cuda_overlay"
export CUDA_PATH="$lec_cuda_overlay"
export CUDA_TOOLKIT_ROOT_DIR="$lec_cuda_overlay"
export PATH="$lec_cuda_toolkit/bin:$PATH"
export CUDARC_CUDA_VERSION=12090
export CUDA_COMPUTE_CAP="${LEC_CUDA_COMPUTE_CAP:-75}"
export NVCC_CCBIN="${LEC_CUDA_HOST_COMPILER:-/usr/bin/g++-14}"
export NVCC_PREPEND_FLAGS="-I$lec_cuda_overlay/include${NVCC_PREPEND_FLAGS:+ $NVCC_PREPEND_FLAGS}"
export LD_LIBRARY_PATH="$lec_cuda_toolkit/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CARGO_TARGET_DIR=target/cuda
exec cargo run --locked --offline --release --features cuda -- "$@"
