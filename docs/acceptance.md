# Assessment acceptance and evidence register

Frozen before assessment implementation on 15 October 2026. An unchecked item is not a claim of completion. The original repository already had an initial README commit; the design freeze is the first assessment commit, rather than a rewritten history.

| Requirement | Owner | Required acceptance evidence |
| --- | --- | --- |
| Three-page design and ceiling | Integration | Frozen Markdown/PDF hashes; deviations retained |
| Machine and runtime preflight | Sandbox | Real controls, resource probes, CPU timings |
| 96 tasks / 24 source groups | Sandbox | 48/16/32 source-disjoint tasks, executable oracles, public and private cases |
| Three independent real providers | Fleet | Dated model/account rights, real response/usage/request records |
| Durable best-of-N runner | Fleet | Transactional reservations, restart/cache, limits/failover and billing uncertainty |
| Secure candidate execution | Sandbox | Fresh containers, non-root/read-only/no-network, resource and trusted syscall evidence |
| Curation | Sandbox | Traceable passing rows, deduplication, held-out lineage/similarity rejection |
| Real base adaptation | Training | Assistant-only gradients, optimiser state, changed/reloaded tensors, second seed |
| Actual evaluation and gate | Integration | Complete checkpoint-bound outcomes, source intervals, clustered sensitivity and simulated power |
| Local inference/canary/rollback | Training | Actual generations, health, atomic routes and rollback |
| Crash recovery | Fleet/Integration | SIGKILL child, finished work not purchased again, UNKNOWN window retained |
| Error then rate limit | Fleet/Integration | Labelled injection through production retry/failover and real replay |
| Grader gaming | Sandbox | Executed skip/fake/hardcode programs fail and quarantine |
| Escape attempts | Sandbox | Network/filesystem/hidden-test probes fail with trusted traces |
| Planted held-out data | Sandbox | Exact and reworded/lineage rejection before optimiser |
| Degraded checkpoint | Training/Integration | Actual measured negative held-out delta blocked; fixture separate |
| Train gain / held-out loss | Training/Integration | Actual observed control pattern blocked; fixture separate |
| Seed-only comparison | Training/Integration | Predeclared two-seed real result; separate deterministic null fixture |
| Budget exceeded | Fleet/Integration | No attempted transport, durable progress, explicit stop |
| Infrastructure failure | Integration | Nonzero exit, failure artifact, incumbent retained |
| Offline full command | Integration | No-network fresh-directory rehearsal with bundled dependencies/images/weights/recordings |
| Costs | Fleet/Integration | Ledger-derived historical/new spend and real compute, no invented gains |
| Delivery | Integration | README, manifests, drill/eval/cost reports, limitations, video and zip |

All nine drills require observed assertions. A blocked dependency or absent real artifact is an incomplete case and a failing overall assessment command. Fixtures test implementation only.
