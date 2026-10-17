# Original assessment evidence audit

Original request satisfied: **false**. Read-only inspection; no provider calls, model execution, training or sealed evaluation.

| Requirement | Status | Finding | Next action |
| --- | --- | --- | --- |
| At least 40 executable source-separated tasks | Verified | 96 executable Rust contracts; train/development/held-out source separation verified | Repair corpus/source contracts before collecting outputs. |
| Design written first, at most three pages; eight requested topics | Observed | Frozen design artefacts are inspected separately from executable evidence; page count, chronology and prose quality require the retained design review. | Review docs/assessment-audit.md and the frozen design/PDF; do not rewrite the original freeze. |
| Three real independent providers with reusable recordings | Incomplete | 0 real training-eligible provider businesses; 0 usable checksum-verified completions; three businesses required | Supply locally configured API credentials, applicable training rights, exact models/prices and an explicit dollar ceiling; collect all three businesses. |
| Real deduplicated/decontaminated run-traceable training set | Incomplete | No such file or directory (os error 2) | Curate actual eligible passing provider runs; local-teacher bundles remain separate. |
| One genuine provider-derived training round and checkpoint | Incomplete | No such file or directory (os error 2) | Run the real trainer on the verified provider dataset; retain optimiser, assistant-mask, changed-tensor and reload evidence. |
| Two real checkpoints, held-out decision and interval | Incomplete | No such file or directory (os error 2) | Freeze one primary candidate/decoding plan; evaluate complete source-held-out pairs once. NO SHIP is an acceptable valid result. |
| Smallest detectable effect under the actual gate | Observed | Power report present; evidence_kind="simulation_assumptions_not_measured_learning"; development calibration remains a separate requirement | Bind the simulation assumptions to complete development measurements and retain sensitivity/Monte Carlo uncertainty; no normal-approximation scalar substitution. |
| kill_orchestrator_mid_round | Incomplete | Recorded status="INCOMPLETE", evidence_kind=missing_required_real_evidence; Original real recording replay is incomplete: three independently owned, enabled and rights-cleared providers are required; found 0. Labelled local SIGKILL/UNKNOWN/retry assertions remain in invariant_qa, without substituting for real acquisition evidence. | Retain actual assertions and full failure denominators. Missing real controls/recordings cannot be supplied by score/transport fixtures. |
| provider_error_then_rate_limit | Incomplete | Recorded status="INCOMPLETE", evidence_kind=missing_required_real_evidence; Original real recording replay is incomplete: three independently owned, enabled and rights-cleared providers are required; found 0. Labelled local SIGKILL/UNKNOWN/retry assertions remain in invariant_qa, without substituting for real acquisition evidence. | Retain actual assertions and full failure denominators. Missing real controls/recordings cannot be supplied by score/transport fixtures. |
| grader_gaming | Observed | Recorded status="PASSED", evidence_kind=executed_adversarial_candidates_with_trusted_exact_denial_assertions; historical assertion requires its original artefact chain | Retain actual assertions and full failure denominators. Missing real controls/recordings cannot be supplied by score/transport fixtures. |
| escape_attempts | Observed | Recorded status="PASSED", evidence_kind=executed_adversarial_candidates_with_trusted_exact_denial_assertions; historical assertion requires its original artefact chain | Retain actual assertions and full failure denominators. Missing real controls/recordings cannot be supplied by score/transport fixtures. |
| planted_heldout_task | Observed | Recorded status="PASSED", evidence_kind=temporary_planted_datasets_rejected_by_actual_training_entrypoint; historical assertion requires its original artefact chain | Retain actual assertions and full failure denominators. Missing real controls/recordings cannot be supplied by score/transport fixtures. |
| deliberately_worse_checkpoint | Incomplete | Recorded status="INCOMPLETE", evidence_kind=missing_required_real_evidence; missing required real artifacts: reports/incumbent-evaluation/evaluation.json, reports/degraded-control-evaluation/evaluation.json | Retain actual assertions and full failure denominators. Missing real controls/recordings cannot be supplied by score/transport fixtures. |
| train_gain_heldout_loss | Incomplete | Recorded status="INCOMPLETE", evidence_kind=missing_required_real_evidence; missing required real artifacts: reports/incumbent-evaluation/evaluation.json, reports/overfit-control-evaluation/evaluation.json, reports/base-train-evaluation/evaluation.json, reports/overfit-train-evaluation/evaluation.json | Retain actual assertions and full failure denominators. Missing real controls/recordings cannot be supplied by score/transport fixtures. |
| seed_only_checkpoints | Incomplete | Recorded status="INCOMPLETE", evidence_kind=missing_required_real_evidence; missing required real artifacts: reports/candidate-evaluation/evaluation.json, reports/seed-control-evaluation/evaluation.json | Retain actual assertions and full failure denominators. Missing real controls/recordings cannot be supplied by score/transport fixtures. |
| round_budget_exceeded | Observed | Recorded status="PASSED", evidence_kind=deterministic_budget_fault_through_production_reservation_and_no_dispatch_assertions; historical assertion requires its original artefact chain | Retain actual assertions and full failure denominators. Missing real controls/recordings cannot be supplied by score/transport fixtures. |
| Real cost per task, model, round and improvement point | Incomplete | Real businesses=0, acquisition complete=false, total USD=null, proven cost/point=null; nonpositive/inconclusive gains must remain N/A or descriptive | Reconstruct ledger-derived task/model costs, cash/credit settlements and measured compute. Never invent a price, gain or scalar cost per point. |
| Adapted checkpoint served locally; canary/rollback evidence | Incomplete | missing reports/candidate-endpoint.json: No such file or directory (os error 2) | Serve the actual assessment adapter and retain matching health/generation identity and rollback evidence. |
| One clean-machine command replays the real loop offline | Incomplete | Latest recorded assessment demo status: "INCOMPLETE" | Complete upstream recordings/training/evaluation/drills, package exact dependencies and run the disconnected extracted archive. |
| Fresh credential-free delivery rehearsal | Incomplete | Local QA passed=true; full assessment complete=false. A correctly failing rehearsal is not a complete assessment. | Rebuild and independently verify the final exact archive; old inventories/binaries cannot certify current changes. |
| Repository/zip, README/design/reports and video up to five minutes | Observed | Video duration=120 seconds; checksum verified=true; final snapshot currency remains unverified | Refresh delivery/video after final evidence stabilises. Do not publish or send without explicit instructions. |

Latest local exploration (reported observations):

```json
{
  "held_out_used": false,
  "interpretation": "Development observations only. One pass versus zero in sixteen development tasks is not a significant held-out gain. Stage report verification is not successful generation/training.",
  "original_three_provider_assessment_satisfied": false,
  "reported_development_comparisons": [
    {
      "arm_id": "arm-00886e19edc356c899a5d992",
      "assessment_promotion": false,
      "baseline_passed": 0,
      "candidate_passed": 0,
      "held_out_used": false,
      "planned_tasks": 16,
      "profile": "quick",
      "reported_difference_percentage_points": 0.0,
      "reported_quality_comparison_valid": true,
      "seed": 1729
    },
    {
      "arm_id": "arm-10219d27497b8da52d90f257",
      "assessment_promotion": false,
      "baseline_passed": 0,
      "candidate_passed": 0,
      "held_out_used": false,
      "planned_tasks": 16,
      "profile": "quick",
      "reported_difference_percentage_points": 0.0,
      "reported_quality_comparison_valid": true,
      "seed": 2718
    },
    {
      "arm_id": "arm-00886e19edc356c899a5d992",
      "assessment_promotion": false,
      "baseline_passed": 0,
      "candidate_passed": 1,
      "held_out_used": false,
      "planned_tasks": 16,
      "profile": "full",
      "reported_difference_percentage_points": 6.25,
      "reported_quality_comparison_valid": true,
      "seed": 1729
    },
    {
      "arm_id": "arm-10219d27497b8da52d90f257",
      "assessment_promotion": false,
      "baseline_passed": 0,
      "candidate_passed": 0,
      "held_out_used": false,
      "planned_tasks": 16,
      "profile": "full",
      "reported_difference_percentage_points": 0.0,
      "reported_quality_comparison_valid": true,
      "seed": 2718
    },
    {
      "arm_id": "arm-d81f55ca66908cb5884d507c",
      "assessment_promotion": false,
      "baseline_passed": 0,
      "candidate_passed": 0,
      "held_out_used": false,
      "planned_tasks": 16,
      "profile": "full",
      "reported_difference_percentage_points": 0.0,
      "reported_quality_comparison_valid": true,
      "seed": 1729
    }
  ],
  "stages": [
    {
      "healthy": true,
      "name": "baseline-conventional_qwen_lora",
      "status": "verified_complete"
    },
    {
      "healthy": false,
      "name": "baseline-ternary_bitnet",
      "status": "verified_complete"
    },
    {
      "healthy": false,
      "name": "cohort-76d0dbc197b33edc7d626d379df22cd2e09fd3a4b350d1fd1c6cd2812277b26c",
      "status": "bounded_evidence_gaps"
    },
    {
      "healthy": false,
      "name": "cohort-catalogue",
      "status": "verified_complete"
    },
    {
      "healthy": false,
      "name": "cohort-df1d0a2dee2c004484c773945f35cf3cc29c774a2c75981cfb5b53c45608da26",
      "status": "bounded_evidence_gaps"
    },
    {
      "healthy": false,
      "name": "content-evidence",
      "status": "bounded_evidence_gaps"
    },
    {
      "healthy": null,
      "name": "development-selection",
      "status": "verified_complete"
    },
    {
      "healthy": null,
      "name": "full-baseline-comparisons",
      "status": "verified_complete"
    },
    {
      "healthy": null,
      "name": "full-study",
      "status": "bounded_incomplete"
    },
    {
      "healthy": true,
      "name": "offline-family",
      "status": "verified_complete"
    },
    {
      "healthy": null,
      "name": "quick-baseline-comparisons",
      "status": "verified_complete"
    },
    {
      "healthy": null,
      "name": "quick-study",
      "status": "bounded_incomplete"
    },
    {
      "healthy": null,
      "name": "study-plan",
      "status": "verified_complete"
    },
    {
      "healthy": null,
      "name": "teacher-search",
      "status": "verified_complete"
    },
    {
      "healthy": false,
      "name": "training-evidence",
      "status": "verified_complete"
    }
  ],
  "status": "completed_with_teacher_target_shortfall",
  "success": false,
  "verified_failed_child_attempts": [
    {
      "attempt": {
        "bytes": 1149,
        "path": "reports/local-study/explore/round-001/cohorts/76d0dbc197b33edc7d626d379df22cd2e09fd3a4b350d1fd1c6cd2812277b26c/attempts/rollout-evidence/a80767d8-07ce-46a2-9cc2-f6c1c9ed5218/result.json",
        "sha256": "b4330a4e8ac976cd384ab48c8ee84d96290e3c56fd0204c3eb87401f42d21801"
      },
      "cohort": "cohort-76d0dbc197b33edc7d626d379df22cd2e09fd3a4b350d1fd1c6cd2812277b26c",
      "exit_code": 1,
      "reason": "required report /home/pbisaacs/Developer/neuralmimicry/lec_challenge/reports/local-study/explore/round-001/cohorts/76d0dbc197b33edc7d626d379df22cd2e09fd3a4b350d1fd1c6cd2812277b26c/rollouts/rollout-report.json is absent; process_success=false; exit_code=1; child failure: host deadline terminated probe pids; attempt=reports/local-study/explore/round-001/cohorts/76d0dbc197b33edc7d626d379df22cd2e09fd3a4b350d1fd1c6cd2812277b26c/attempts/rollout-evidence/a80767d8-07ce-46a2-9cc2-f6c1c9ed5218/result.json; stderr=reports/local-study/explore/round-001/cohorts/76d0dbc197b33edc7d626d379df22cd2e09fd3a4b350d1fd1c6cd2812277b26c/attempts/rollout-evidence/a80767d8-07ce-46a2-9cc2-f6c1c9ed5218/stderr.log",
      "recorded_process_success": false
    },
    {
      "attempt": {
        "bytes": 1147,
        "path": "reports/local-study/explore/round-001/cohorts/df1d0a2dee2c004484c773945f35cf3cc29c774a2c75981cfb5b53c45608da26/attempts/rollout-evidence/2d745bcc-e6d1-4094-af90-57a564d594a9/result.json",
        "sha256": "f9eedeed15c092fbb4b5b07567e1b70949ac9cc78db106aaee04e323ace39749"
      },
      "cohort": "cohort-df1d0a2dee2c004484c773945f35cf3cc29c774a2c75981cfb5b53c45608da26",
      "exit_code": 1,
      "reason": "Failed child supplied a report; inspect its current-batch failure counts.",
      "recorded_process_success": false
    }
  ]
}
```

Observation is distinct from independent verification. Missing, malformed, fixture-only and failed observations cannot establish completion. This report assigns no speculative assessment score.
