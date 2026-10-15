# Retained diagnostic source

`bitnet_movement-a805ea92ab02552c7a1a2a1641359f784ab5751397be7c544cf70b5c8dc6d475.rs`
retains the exact source bytes of the measured 48-step movement diagnostic.
Its SHA-256 matches the original build receipt's `source.sha256`. The original
receipt, pinned executable and measured report remain unchanged. Its original
source path was `examples/bitnet_movement.rs`, which is an editable development
path; use this archived copy to inspect those historical bytes.

The current example replaces `chunks_exact(4)` with `as_chunks::<4>()` to satisfy
the current strict Clippy lint. The preceding complete-F32-length check remains
in place. This source maintenance does not rerun or alter the original measured
diagnostic.
