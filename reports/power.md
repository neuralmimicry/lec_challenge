# Gate detectability sensitivity

Simulation only; no learning result. 8 source clusters, 4 tasks per source and 3 shared decoding repeats. 1000 trials per point; 10000 bootstrap resamples; simulation seed 851053. The source sharing construction has the stated pairwise intraclass correlation.

| Assumed discordance / source correlation | Smallest tested effect at estimated 80% power |
| --- | ---: |
| discordance=0.10, correlation=0.00 | MDE above tested range |
| discordance=0.10, correlation=0.25 | MDE above tested range |
| discordance=0.10, correlation=0.50 | MDE above tested range |
| discordance=0.30, correlation=0.00 | 30.0 percentage points |
| discordance=0.30, correlation=0.25 | MDE above tested range |
| discordance=0.30, correlation=0.50 | MDE above tested range |
| discordance=0.50, correlation=0.00 | 40.0 percentage points |
| discordance=0.50, correlation=0.25 | 50.0 percentage points |
| discordance=0.50, correlation=0.50 | 50.0 percentage points |

MDE is the first tested effect reaching 80% power, not an interpolated precision claim; null means MDE above tested range or unachievable at the specified discordance. Includes the exact reported t/bootstrap/slice gate configuration, with maximally shared seed outcomes; reduced settings are diagnostic, not the final predeclared MDE. Task/source dependence assumptions require development calibration and sensitivity review.

Each point in power.json also reports Monte Carlo standard error and an exact binomial 95% interval, including zero/one estimated power. A reduced diagnostic run is identified by its complete embedded configuration; only the frozen full configuration is the assessment gate's power calculation. Development calibration and shared-authorship limitations still apply.
