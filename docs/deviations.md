# Design decisions and departures

## 15 October 2026 — user-directed Rust implementation

The assessment email allows any stack. The subsequent user instruction explicitly requires Rust throughout unless absolutely necessary and justified. This takes precedence over the prebuild pack's Python recommendation. The supervisor, provider clients, ledger, task language, trusted grader, curator, tensor training, statistics and HTTP serving are therefore Rust. Shell is limited to launching the compiled command and operating-system/container build instructions; TOML, JSON, Markdown and OCI recipes are data/documentation formats.

Rust coding tasks replace the proposed Python functions. Qwen2.5-Coder remains the proposed base, with native Candle differentiable LoRA and CPU support in place of PyTorch/PEFT. These changes increase implementation and verification effort, but avoid a hidden Python training dependency. Model revisions, changed weights and real gradients must still be verified.

The original dated design and PDF remain in assessment-prebuild. The assessment design is frozen at the current real time, without backdating. Existing repository history is preserved.

## Evidence constraints at initial preflight

No DeepSeek, Together or Mistral credentials were present among environment-variable names, and no paid-provider authority or account-specific training permission was supplied. Live collection and actual provider-trained assessment checkpoints remain blocked until these inputs exist. Local engineering and tests proceed; no synthetic recordings replace provider evidence.

The Docker command currently reaches a rootless Podman service; Docker compatibility output reports several resource controls unsupported, while native cgroup information needs direct enforcement tests. Merely finding the Docker executable does not pass sandbox preflight. The NVIDIA GPU is an 8 GiB RTX 2080, rather than the hypothetical 12 GiB device. CPU support is required and CUDA is not assumed.

## Native tooling and final verification

The video evidence selection, captions, timing and manifest are produced by Rust. Installed ffmpeg/ffprobe encode and inspect the standard H.264 MP4 container: implementing a video codec would not improve the assessment's Rust application or its evidence. Native tar/gzip/ZIP utilities encode delivery archives. The minimal launcher, OCI recipes and CI commands orchestrate the Rust executable; none replaces training, grading or statistics with another language.

Direct native Podman probes subsequently proved CPU/memory/swap/PID limits despite the Docker compatibility flags. A stricter timing probe separated active workload termination from runtime cleanup and led to an inspected per-container cgroup-v2 kill path. Actual kernel acknowledgement was tens of milliseconds; cleanup took several seconds and is reported separately. The trusted ptrace supervisor runs inside the container outside the candidate process, while expected answers and case selection stay on the host. Its structured audit is verified by the host rather than accepted from generated code.

Corpus-domain review before any teacher collection corrected an out-of-contract private money-format boundary and clarified shared integer/size constraints. These are pre-collection correctness changes, not post-hoc tuning of teacher or student outcomes. The statistical simulation also has an invariant checking that its source sharing achieves the stated pairwise correlation, and an optimised decision path checked against the complete public gate.
