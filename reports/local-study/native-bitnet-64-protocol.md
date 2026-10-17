# Fresh 64-step BitNet operator protocol

This is a newly authorised numerical operator experiment, separate from the
preserved three-step and 48-step failures. It starts from the original frozen
base and does not import their latent adapters or optimiser state. Its output
directory must be unused on first invocation.

The request is `native-bitnet-64-request.json`. The matched worker is the
preserved executable
`state/engines/af675af8e54beb9ddc672d13bb860aec4175d7071b43fe9345af8c4ba9bbf182/lec-distil`.
It uses four Rayon threads and an external 3,600-second wall-clock bound.
The request keeps seed 1701, rank 2, alpha 4, learning rate 0.005, a 32-token
assistant window and the original verified local Qwen 7B training bundle.
Only the declared optimiser step budget changes from 48 to 64.

The read-only saved-state diagnostic measured a maximum update of
0.3269575376427052 original scale units against the 0.5 switch threshold.
The first switch would require at least a 1.5292505675351438 multiplier of the
saved 48-step update. This motivates a bounded longer experiment; it does not
predict that 64 genuine updates will succeed. Counterfactual scaling is not a
trained checkpoint.

The success criteria are finite genuine optimiser updates, changed latent
tensors, at least one changed effective ternary decision, a valid exported
checkpoint and an independent fresh-process reload. A manifest alone is
insufficient. An unchanged ternary lattice remains a failed adapted-checkpoint
experiment, with all journals and its blocking report retained. The short
assistant window and two-token reload probe establish operator behaviour only;
they cannot establish coding improvement, native numerical equivalence,
three-provider provenance or assessment promotion.

Launch waits for the current pinned native numerical measurement to release
its RAM reservation. The supervisor retains the worker's actual exit status
and complete log. The external bound may terminate the worker; such a run is
incomplete rather than successful.
