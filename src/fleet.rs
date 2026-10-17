//! Configurable, bounded teacher acquisition and exact offline replay.
//!
//! Native text-only API adapters reject tools, search and uncapped reasoning.

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

mod access;
mod config;
mod local_drills;
mod protocol;
#[cfg(test)]
mod provider_tests;
mod runner;
mod setup;
#[cfg(test)]
mod setup_tests;
#[cfg(test)]
mod tests;
mod transport;

pub use access::*;
pub use config::*;
pub use local_drills::{drill_child, run_local_drills, run_recorded_drills};
pub use protocol::*;
pub use runner::*;
pub use setup::*;
pub use transport::*;

fn valid_hash(hash: &str) -> bool {
    hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit())
}
