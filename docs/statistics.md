# Evaluation, promotion and detectable effects

The primary question is whether the chosen student checkpoint improves verified coding success on the sealed source distribution. A changed tensor or decreasing training loss does not answer that question. The normal command consumes complete cached evaluations bound to actual tensor hashes; an arbitrary accuracy summary cannot be promoted.

```sh
./run.sh evaluate --base offline_bundle/base --output reports/base-evaluation
./run.sh evaluate --base offline_bundle/base --checkpoint checkpoints/candidate --output reports/candidate-evaluation
./run.sh gate --incumbent reports/base-evaluation/evaluation.json --candidate reports/candidate-evaluation/evaluation.json
```

These are commands to execute after real acquisition, curation, training and effective sandbox checks. They do not imply that those missing evidence stages have already happened. The gate leaves the incumbent in place on every outcome except SHIP. An evaluation failure exits unsuccessfully; a missing generation is never removed from the denominator to improve a score.

Each evaluation freezes an immutable, checksum-bound preregistration before requesting any held-out generation. It records the exact checkpoint/base/tokenizer assets, corpus, gate configuration, decoding seeds and caps, sampling parameters, timeout, sandbox policy, immutable image identity and successful enforcement preflight. Reusing an existing preregistration output directory is refused. Each generation includes its exact task prompt, seed and decoding request hash. The verifier checks those against the preregistration, reloads tensor/manifest evidence, verifies all pinned assets, and revalidates controller reports against raw trusted audit responses. Two checkpoint evaluations must use identical decoding settings and sandbox policy/image. Local hashes and timestamps provide auditability under the trusted-host assumption; they are not an independent notarisation or proof that an operator never inspected earlier test runs.

A model generation error, empty response, truncation or inference timeout is a failed observation with a durable request/error or generation artifact. No invented sandbox success report is created for output rejected before execution. Model-load failures, a lost execution worker, missing audit artifacts and unenforced controls invalidate the evaluation. Ordinary incorrect answers are quality failures; prohibited operation attempts or incomplete execution tracing additionally veto promotion. The same failure handling applies to both checkpoints.

The planned held-out set has 32 tasks grouped into eight sources, with three common decoding seeds per task. For source s, first average the three paired candidate-minus-incumbent pass indicators within each task, then average those task differences within the source. The primary effect is the equally weighted mean of the eight source effects. A task with many assertions remains one task; decoding repeats do not create independent sources. Results also show source-macro incumbent and candidate success rates, unique task count and paired observation count.

The predeclared gate in `config/gate.toml` requires both of these lower bounds to exceed a practical margin of 0.02, meaning two percentage points:

- A two-sided 95% Student t interval over the eight source effects, with seven degrees of freedom.
- A percentile bootstrap interval resampling whole sources, with the fixed configured seed and trial count.

The source t interval assumes approximately independent source effects and reasonable small-sample t behaviour. The bootstrap is a sensitivity calculation using only eight observed clusters; it does not remove dependence between sources or repair a narrow source distribution. The locally authored packs share AI authorship and may share algorithms. These are material weaknesses, and the report states them.

Every prescribed task/seed must be present exactly once. Invalid identities, duplicate/missing outcomes, changed settings and failed security/operational checks block promotion. A source regression exceeding the predeclared slice ceiling also blocks promotion. Decisions distinguish SHIP, NO SHIP / REGRESSION, NO SHIP / INCONCLUSIVE and INVALID EVALUATION. A negative observed effect is reported as a regression rather than concealed behind a wide interval; it need not establish statistically significant harm. An interval containing zero is insufficient evidence of superiority, not proof of equality. Better training scores cannot rescue a held-out loss.

The gate must be frozen before sealed-set access. Do not change a margin, seed, source definition, selected checkpoint or schedule after reading the held-out result. A further confirmatory promotion needs fresh independent source packs or a predeclared sequential/multiple-comparison policy. Refusing to overwrite one output directory helps preserve evidence; it cannot enforce this scientific discipline against an operator using another directory.

The power calculation runs the actual configured t/bootstrap/slice decision rather than labelling a normal approximation as the gate's detectable effect:

```sh
./run.sh power --quick
./run.sh power
```

The quick calculation reduces Monte Carlo/bootstrap trials for software checks. The full command uses the frozen configuration. Both produce an explicitly labelled simulation report rather than evidence of learning. Sensitivity axes include discordance and within-source correlation. Shared source shocks and shared task outcomes preserve clustering; all decoding repeats share task outcomes in the current conservative simulation. The implementation does not claim to have estimated the seed-variance structure from absent development observations.

The reported minimum detectable effect is the smallest tested effect reaching 80% simulated shipping probability under that assumption set, not an interpolated precise threshold. Each point includes Monte Carlo standard error sqrt(p(1-p)/trials). If the tested range does not reach 80%, MDE is unavailable/above the tested range; no scalar is invented. Discordance bounds feasible effects, so some values cannot be simulated under a given assumption. Development calibration and a richer seed-variance sensitivity remain required before interpreting this as a measured study power analysis. The approximate 27.1-point result from 32 independent pairs and discordance 0.30 is only a rough sanity check and omits source clustering and the practical shipping margin.

The implementation tests regression, identical/null scores, missing/duplicate outcomes, security vetoes, a large consistent gain, deterministic bootstrap output and boundary conditions with clearly labelled score fixtures. Those fixtures establish decision logic only. The required degraded checkpoint, train-overfit control and two training seeds must be actual separately trained adapters, evaluated once with the frozen settings. Their observed patterns are not guaranteed by the training intervention. If the degraded control does not actually regress, the overfit control does not show a train gain/held-out loss, or the seed pair is significant, retain the real result and mark the corresponding assessment drill incomplete. Never search held-out seeds or control variants until the desired pattern appears.

There is presently no provider-trained candidate or verified held-out improvement. Therefore no measured real promotion decision, calibrated MDE or proven cost per point of improvement is claimed by this document.

Training-distribution diagnostics use their own explicitly bound split, and can never authorise promotion:

```sh
./run.sh evaluate --split train --output reports/base-train-evaluation
./run.sh evaluate --split train --checkpoint checkpoints/overfit --output reports/overfit-train-evaluation
```

`config/drill-controls.toml` predeclares the evaluation/configuration paths used by the actual control drills. The drill re-verifies checkpoint-bound held-out evaluations through the promotion verifier, verifies complete training evaluations separately, and confirms that train/held-out observations reference the same weights. The seed comparison additionally requires identical data, base, tokenizer and targets, the fixed distinct seeds and otherwise identical hyperparameters. Missing artefacts are listed; an actual control that fails to exhibit its required pattern stays incomplete. A summary claiming a control passed is never accepted as substitute evidence.
