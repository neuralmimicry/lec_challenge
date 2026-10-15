//! Configurable, bounded teacher acquisition and exact offline replay.
//!
//! The HTTP adapter intentionally supports only ordinary chat completions: tools,
//! search and uncapped reasoning cannot silently introduce another billable feature.

use crate::ledger::{Acquisition, BudgetLimits, Claim, Ledger, sha256, unix_millis};
use anyhow::{Context, Result, bail, ensure};
use async_trait::async_trait;
use futures_util::{StreamExt, stream};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime},
};
use tokio::sync::{Mutex, Semaphore};

mod config;
mod local_drills;
mod runner;
#[cfg(test)]
mod tests;
mod transport;

pub use config::*;
pub use local_drills::{drill_child, run_local_drills, run_recorded_drills};
pub use runner::*;
pub use transport::*;

fn valid_hash(hash: &str) -> bool {
    hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit())
}
