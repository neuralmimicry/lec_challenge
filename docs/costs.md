# Acquisition, cash and compute accounting

`./run.sh costs --round round-001` writes attempt CSV, Markdown and JSON summaries from the actual SQLite ledger. Without `--round`, it provides aggregate observations across rounds and deliberately does not claim a complete individual round. No provider output has yet been acquired: current reports show missing evidence rather than a completed zero-cost learning round.

All stored money uses integer USD nanodollars. Checked additions cover attempt, token and uncertainty aggregates; overflow is an error rather than a release-build wraparound. Price multiplication uses checked 128-bit intermediates with upward rounding. CSV strings are quoted and embedded quotes doubled. Provider/model aggregate keys are JSON tuples, so a slash in either identifier cannot merge different models. Missing usage contributes only to explicitly labelled known totals and increments a missing-usage counter; it is not asserted to be zero consumption.

Collection declares an immutable complete `RoundPlan` before its first attempt. The plan hashes the frozen configuration and lists every intended task/sample, its source/specification identity and exact permitted request keys for the preferred provider and configured failover. Restart accepts an identical plan; changing the plan requires a new explicitly declared round. An undeclared key cannot obtain a reservation once a plan exists. The cost verifier checks every planned job against a checksum-verified real response and independently recomputes the canonical request identity. A few completed rows cannot establish an unrecorded denominator. Failed HTTP retries remain charged without making an otherwise complete plan incomplete; missing/UNKNOWN/unplanned work and ambiguous completions block completeness.

The report keeps these distinct:

| Quantity | Evidence and interpretation |
| --- | --- |
| Historical API accrual interval | Real reported usage and the retained price schedule, or a conservative unresolved reservation interval. This is an accrual estimate, not an invoice cash payment. |
| New API accrual interval | The same interval for live attempts; zero for replay. Replaying an old paid response does not create another provider bill. |
| Historical cash, credits and discounts | Per-attempt operator invoice/credit reconciliation with retained artefact checksums. Unknown until that evidence exists. |
| Measured compute observations | Labelled real workload wall times and device details. Concurrent wall times cannot be summed into CPU core-hours or machine rental duration. |
| Complete attributable cash round | The selected round's reconciled acquisition cash plus its explicitly allocated measured compute cash, with all required phases represented. |

The optional `reports/api-settlements.json` contains `ApiSettlement` records: round/attempt IDs, attributed historical nanodollars, cash nanodollars, credit nanodollars, discount nanodollars, a textual basis and a checksum-linked invoice/export artefact. All parts must be nonnegative, sum to the attributed historical amount and fall inside the original ledger interval. Every selected attempt must be covered once; extra or duplicate attempt settlements fail. This report-level reconciliation does not silently release the ledger's conservative budget reservation. A provider reconciliation action would need its own audited evidence.

The optional `reports/compute-charges.json` contains `ComputeCharge` records: round, phase, device, elapsed seconds, cash nanodollars, allocation basis and checksum-linked metering/invoice evidence. Required phase labels are `training`, `evaluation`, `grading`, `storage` and `serving`. A zero-cash free resource still needs actual measured device time and a disclosed basis. Identical duplicate charge entries fail. Owned hardware electricity, depreciation and storage costs remain unmeasured unless supplied: the illustrative design tariff is never substituted for a measurement.

These optional files are not generated with invented values. Missing settlements or metering leave the corresponding figures unavailable. A round using already acquired cached responses can complete its logical work, but a new cash total requires an explicit historical allocation rather than charging that data again. The evaluation checkpoint's validated original dataset must also trace to the selected round's completed attempts before its measured gain is combined with the round cost.

Cost per point uses the actual verified gate effect in percentage points: `round_cash_usd / (100 * rate_gain)`. A positive but inconclusive effect permits a descriptive ratio only; a nonpositive effect is N/A. The arithmetic unit fixture checks that a hypothetical $10 round and a 0.05 rate gain mean five percentage points and $2 per point, not $200. Those fixture values are not a learning result. Proven cost per point additionally requires SHIP.

Break-even volume requires SHIP quality evidence and measured teacher/student marginal costs per controller-verified successful task on the same workload contract. It is `ceil(round_cost / (teacher_cost_per_success - student_cost_per_success))`, and is unavailable without positive measured savings. No break-even figure is inferred from an inexpensive base inference smoke test.
