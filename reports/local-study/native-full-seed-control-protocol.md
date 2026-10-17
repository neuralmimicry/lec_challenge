# Second predeclared conventional training seed

This is a fresh exploratory protocol. Both seeds, 1729 and 2718, were declared
in `config/distillation-study.toml` before the original observations. This round
executes seed 2718 without searching seeds or changing the original seed 1729
checkpoint. It has no access to the assessment held-out set and cannot satisfy
the assessment's provider-trained seed-control drill.

The SFT request is `native-full-sft-seed2718-request.json`: the same Qwen 0.5B base,
verified Qwen 7B evidence bundle, complete-example 512-token rejection policy,
48 updates, rank 8, alpha 16 and learning rate 0.0002 as the original SFT round.
The complete accepted sample set must agree; its shuffled order and initial
adapter and the last partial cycle's exposure counts are allowed to change
with the declared seed. This is a whole-training seed control, including
initialisation and sample shuffling. Missing or corrupt
evidence is an error, never an instruction to regenerate a teacher response.

The exact original numerical helper is retained:
`state/engines/5db09c632f6ffccc636bdfec2f693af102bf5327ce7510bc593ebe7089c9c126/lec-distil`.
Launch variables remain RAYON=4, LEC=5, OMP=4 and MKL=4. The separate worker
uses affinity 4–11 so that it can coexist with the original CUDA evaluator on
0–3. This placement differs from the first round's 0–15 and is recorded rather
than hidden. Effective numerical pool sizes remain four. Reserve 16 GiB RAM
and five available CPU threads; pre-launch hardware detection observed over
46 billion usable RAM bytes. SFT requires no teacher tensor targets or GPU
allocation. The application does not make provider or Hub calls.

The companion `native-full-composed-seed2718-request.json` repeats the original
48-step SFT + temperature-two KL + features + relations arm with seed 2718 and
unit loss weights. It uses the same original numerical helper and already
captured complete-example teacher targets. Its `CacheOnly` policy must reject a
cache miss before optimisation; no teacher acquisition or GPU fallback is
permitted. This worker uses affinity 8–15 with the same numerical thread settings
and five-thread allocation. The affinity ranges overlap, but the three active
jobs' declared CPU allocations sum to 14 of the detected 16 logical threads.
Reserve another 16 GiB RAM; actual available memory must still cover both
reservations and the operating system before dispatch. These two CPU rounds
can run together while the original matched evaluation retains the GPU.

Preserve elapsed time, peak RSS, helper-launch contracts and every durable Adam
step. An interrupted job resumes the same request, executable and environment.
After completion, a separate process must reload the checkpoint and verify
finite logits. A subsequent matched development comparison must include both
original seeds and all failures. Training loss or changed tensors cannot
establish a coding-quality improvement or a seed difference.
