//! Translate independently verified local evidence into executable study arms.
//! This boundary cannot turn local weights into independent API businesses or
//! exploratory development results into assessment promotion evidence.
use crate::{
    artefact::{self, Artefact},
    distillation::{self, DistillationWorkerSpec, LossKind, WeightedLoss},
    experiments::{self, Capability, ContentMethod, SignalMethod, StudentArchitecture},
    model::{self, RuntimeOptions},
    platform,
    teacher_evidence::{self, ContentKind, SignalAvailability},
};
use anyhow::{Context, Result, ensure};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
use tokenizers::Tokenizer;

mod evaluation;
mod planner;
mod selection;
mod worker;
pub use evaluation::{CheckpointEvalOptions, evaluate_checkpoint};
use planner::capture;
pub use planner::{PlanOptions, create_plan};
pub use selection::{select, verify_selection};
pub use worker::{evaluate_worker, run_worker, worker_spec};
