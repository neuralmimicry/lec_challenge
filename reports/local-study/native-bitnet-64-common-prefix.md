# Observed common 48-update prefix

The fresh 64-step protocol independently recreated the preserved 48-step
round's adapter and Adam state at update 48. Both pairs of original safetensor
files were compared byte for byte with the native `cmp --silent` command;
both comparisons returned exit zero. SHA-256 observations were:

| Original and fresh file pair | Identical SHA-256 |
| --- | --- |
| `native-bitnet-48-operator/step-000048/adapter.safetensors` and `native-bitnet-64-operator/step-000048/adapter.safetensors` | `2ccde5653e6a0099e8b92c5faa466b08154570c4ea27296e5d058758ef34e5aa` |
| `native-bitnet-48-operator/step-000048/optimizer.safetensors` and `native-bitnet-64-operator/step-000048/optimizer.safetensors` | `c0b26845951d95fa13622cccb36b7aa09ce20f084442ac28d64610f7e4013ce4` |

Both fresh and original protocols retain their separate requests, freezes and
step journals. The new run starts from the original base; it does not load the
failed round's adapter or optimiser. This observation verifies reproduced
saved state at the common update count. It does not itself establish a changed
ternary lattice, a successful 64-step checkpoint, native numerical equivalence
or coding improvement.
