# Limitations and unfinished assessment evidence

## External inputs still required

No DeepSeek, Together or Mistral API credentials, authorised spending ceiling or account-specific output-training decision was supplied during the build. Public terms and licence documents were fetched and hashed, but that does not resolve the selected account/model permissions or current pricing. The roster therefore remains disabled and unpriced.

Consequently the repository contains no real teacher acquisition recordings, no provider-derived training dataset, no assessment-trained primary/seed/degraded/overfit adapter and no actual paired student held-out improvement result. The full-size gradient smoke is genuine training computation on synthetic tensor QA; it is not a replacement for the required provider-trained checkpoint. Required actual controls and complete offline replay remain incomplete. Evaluation reports retain INVALID EVALUATION until real complete evidence exists. No model promotion is justified.

## Corpus and statistical limits

The 96 local Rust contracts are grouped into 24 distinct practical source packs and split 48/16/32 by whole source. They share AI authorship, task-writing tooling and common algorithmic knowledge. Those facts weaken an independence claim; source groups are declared design units, not independently audited human authors or proof of pretraining non-exposure. Public expected examples are written separately from oracle code; private values use the oracle. Independent interval/fraction properties give extra checks, but a second independent oracle for every contract remains review work.

Lineage, exact specification hashes, lexical similarity and canonical Rust syntax reduce contamination and duplication. They do not prove semantic novelty. Reworded tasks with falsified lineage and sufficiently different wording can evade lexical screening. Similarity hits are conservative review blocks; no automatic embedding system establishes equivalence.

The primary held-out set has only eight source groups. Both the t interval and bootstrap can have fragile coverage under source dependence or skewed effects. Shared authorship may worsen dependence. The power sensitivity grid is a simulation under stated discordance/correlation assumptions; no real development-run calibration exists yet. More assertions and decoding seeds do not create more independent tasks or sources. A seed-only comparison may occasionally be significant and must be disclosed.

## Runtime and operational limits

The sandbox uses shared-kernel containers and a Linux x86-64 syscall decoder. It does not establish resistance to every kernel exploit or all future syscalls. The candidate ABI deliberately supports pure Rust functions using std/serde_json; process spawning, networking and arbitrary file access are prohibited. Trusted syscall traces prove the exercised probes, not every possible attack. Resource limits are verified by actual kernel probes on the tested host; another machine must pass those probes itself.

CPU generation cancellation occurs at token boundaries. A matrix multiplication cannot be interrupted from inside the model worker; a hard model-compute deadline requires process supervision. Candidate execution has a separate host deadline that removes the entire container. CPU/queue/timeout QA demonstrates bounded serving mechanics at low volume, not production-scale latency or availability.

No 12 GiB GPU assumption is used: the observed card has 8 GiB, and CPU is the verified model path. Training/inference timings are workload measurements on one laptop; some QA jobs overlapped and were not isolated throughput benchmarks. The 512-token synthetic gradient process peaked at 7,967,212 KiB RSS in `reports/native-gradient-512-time.txt`. Full provider-round and sustained-serving peak memory, throughput and compute/energy allocations remain unmeasured.

## Submission gaps

The repository, design, executable QA, reports, model assets, reference examples and a captioned evidence video can be inspected locally. The video honestly describes missing assessment evidence. Publication/email submission is not authorised and has not been attempted. A full credential-free disconnected rehearsal with all real recordings/adapters cannot be completed until those artifacts exist. Once credentials, rights and spending are authorised, the next required work is three real provider collections, audited curation, four predeclared adapter runs, one frozen evaluation, real checkpoint control interpretation and the final disconnected rehearsal.
