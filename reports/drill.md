# Required drill evidence

Overall: **INCOMPLETE**. Missing evidence is never successful handling.

| Case | Status | Evidence |
| --- | --- | --- |
| kill_orchestrator_mid_round | INCOMPLETE | missing_required_real_evidence |
| provider_error_then_rate_limit | INCOMPLETE | missing_required_real_evidence |
| grader_gaming | PASSED | executed_adversarial_candidates_with_trusted_exact_denial_assertions |
| escape_attempts | PASSED | executed_adversarial_candidates_with_trusted_exact_denial_assertions |
| planted_heldout_task | PASSED | temporary_planted_datasets_rejected_by_actual_training_entrypoint |
| deliberately_worse_checkpoint | INCOMPLETE | missing_required_real_evidence |
| train_gain_heldout_loss | INCOMPLETE | missing_required_real_evidence |
| seed_only_checkpoints | INCOMPLETE | missing_required_real_evidence |
| round_budget_exceeded | PASSED | deterministic_budget_fault_through_production_reservation_and_no_dispatch_assertions |

The JSON report retains exact assertions, measured effects, paths and missing artifacts. All nine cases are checked dynamically against the configured real evidence paths. Local transport/score fixtures are separate implementation QA. A missing or unexpected real control pattern remains incomplete; no held-out seed/control search is performed.
