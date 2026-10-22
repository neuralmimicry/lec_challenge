//! Observe the active container store without migrating, resetting or pruning it.
use crate::{
    sandbox::{self, SandboxConfig, policy_sha256},
    sandbox_archive::verify_image_bundle,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use tokio::{
    process::Command,
    time::{Duration, timeout},
};

fn storage_warnings(info: &Value) -> Vec<String> {
    let mut warnings = Vec::new();
    let graph = info
        .pointer("/store/graphRoot")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if graph.is_empty() {
        warnings.push("Active graph root could not be established".into());
    }
    if graph.starts_with("/tmp/") || graph.starts_with("/var/tmp/") || graph.starts_with("/run/") {
        warnings.push(format!(
            "Temporary graph root {graph}; images in another store are not available here"
        ));
    }
    if let Ok(metadata) = std::fs::metadata(graph) {
        // SAFETY: geteuid has no pointers or process-state side effects.
        let uid = unsafe { libc::geteuid() };
        if metadata.uid() != uid {
            warnings.push(format!(
                "Foreign graph root owner UID {}; current UID {uid}",
                metadata.uid()
            ));
        }
    }
    if info.pointer("/host/security/rootless") != Some(&Value::Bool(true)) {
        warnings.push("Runtime did not prove rootless identity".into());
    }
    warnings
}

async fn metadata(runtime: &str, arguments: &[&str]) -> Result<std::process::Output> {
    Ok(timeout(
        Duration::from_secs(20),
        Command::new(runtime)
            .args(arguments)
            .kill_on_drop(true)
            .output(),
    )
    .await
    .context("sandbox doctor metadata deadline exceeded")??)
}

pub async fn inspect(cfg: &SandboxConfig, archive: Option<&Path>, restore: bool) -> Result<Value> {
    ensure!(
        !restore || archive.is_some(),
        "explicit restoration requires --archive-directory"
    );
    let info = metadata(&cfg.runtime, &["info", "--format=json"]).await;
    let (info, error) = match info {
        Ok(result) if result.status.success() => {
            (serde_json::from_slice::<Value>(&result.stdout)?, None)
        }
        Ok(result) => (
            Value::Null,
            Some(format!(
                "runtime info failed: {}",
                String::from_utf8_lossy(&result.stderr)
            )),
        ),
        Err(error) => (Value::Null, Some(format!("{error:#}"))),
    };
    let identity = metadata(
        &cfg.runtime,
        &["image", "inspect", "--format", "{{.Id}}", &cfg.image],
    )
    .await;
    let image = identity
        .ok()
        .filter(|r| r.status.success())
        .map(|r| String::from_utf8_lossy(&r.stdout).trim().to_owned());
    let archive_result = if let Some(directory) = archive {
        Some(match verify_image_bundle(directory, cfg).await {
            Ok((path, id)) => json!({"verified":true,"path":path,"exact_image_id":id}),
            Err(error) => json!({"verified":false,"error":format!("{error:#}")}),
        })
    } else {
        None
    };
    let restored = if restore {
        Some(sandbox::restore_image_bundle(archive.context("archive absent")?, cfg).await?)
    } else {
        None
    };
    let configuration_paths = ["CONTAINERS_CONF", "CONTAINERS_STORAGE_CONF"]
        .into_iter()
        .map(|name| (name, std::env::var_os(name).map(PathBuf::from)))
        .collect::<std::collections::BTreeMap<_, _>>();
    let warnings = storage_warnings(&info);
    Ok(
        json!({"schema_version":1,"runtime":cfg.runtime,"effective_uid":unsafe {libc::geteuid()},
        "rootless":info.pointer("/host/security/rootless"),"runtime_error":error,
        "graph_root":info.pointer("/store/graphRoot"),"run_root":info.pointer("/store/runRoot"),
        "effective_storage_config":info.pointer("/store/configFile"),"configuration_environment_paths":configuration_paths,
        "sandbox_tag":cfg.image,"active_image_id":image,"expected_image_id":cfg.expected_image_id,
        "policy_version":cfg.policy_version,"controller_version":cfg.controller_version,
        "policy_sha256":policy_sha256(cfg)?,"archive":archive_result,"restored_exact_image_id":restored,"warnings":warnings,
        "grading_authorised_by_doctor":false,
        "next_step":"Full sandbox-preflight remains mandatory before grading. Historical plans require their original exact image IDs. A rebuild creates a new identity and needs new study bindings."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn temporary_and_foreign_stores_are_visible_without_any_mutation() {
        let info = json!({"store":{"graphRoot":"/tmp/foreign-store"},"host":{"security":{"rootless":false}}});
        let warnings = storage_warnings(&info);
        assert_eq!(warnings.len(), 2);
        assert!(warnings[0].contains("/tmp/foreign-store"));
        assert!(storage_warnings(&json!({"store":{"graphRoot":"/home/operator/.local/share/containers/storage"},"host":{"security":{"rootless":true}}})).is_empty());
    }
}
