//! Read-only inventory of a bounded exploratory evidence closure.
//! Large existing model assets are described, not copied or rehashed.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::{BTreeMap, BTreeSet, VecDeque}, fs, io::Read, os::unix::fs::PermissionsExt, path::{Path, PathBuf}, time::SystemTime};

fn hash(path: &Path) -> std::io::Result<String> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut b = [0; 65536];
    loop { let n=f.read(&mut b)?; if n==0 {break} h.update(&b[..n]); }
    Ok(format!("{:x}",h.finalize()))
}
fn files(path: &Path, set: &mut BTreeSet<PathBuf>, latest_only: bool) -> std::io::Result<()> {
    if !path.exists() {return Ok(())}
    let m=fs::symlink_metadata(path)?;
    if m.is_dir() {
        for e in fs::read_dir(path)? {
            let p=e?.path();
            if latest_only && p.file_name().is_some_and(|n|n.to_string_lossy().starts_with("step-")) {continue}
            files(&p,set,false)?;
        }
    } else {set.insert(path.to_owned());}
    Ok(())
}
fn refs(value: &Value, out: &mut Vec<(String,Option<String>,Option<u64>)>) {
    match value {
        Value::Object(o)=> {
            if let Some(p)=o.get("path").and_then(Value::as_str) {
                if o.contains_key("sha256") || o.contains_key("bytes") {
                    out.push((p.to_owned(),o.get("sha256").and_then(Value::as_str).map(str::to_owned),o.get("bytes").and_then(Value::as_u64)));
                }
            }
            for v in o.values(){refs(v,out)}
        }
        Value::Array(a)=>for v in a{refs(v,out)},
        _=>{}
    }
}
fn package_status(path: &Path)-> &'static str {
    if path.starts_with("state") || path.starts_with("target") {"excluded_from_archive"}
    else if path.starts_with("reports") || path.starts_with("src") || path.starts_with("docs") || path.starts_with("examples") || path.starts_with("agents") || path.starts_with(".agents") {"archived_outside_immutable_asset_inventory"}
    else {"archived_check_asset_root_membership"}
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let root=std::env::current_dir()?.canonicalize()?;
    let output=Path::new("reports/local-study/delivery-closure");
    let mut selected=BTreeSet::new();
    let training=["native-full-sft-seed1729","native-full-composed-seed1729","native-full-sft-seed2718","native-full-composed-seed2718","native-bitnet-64-operator"];
    let mut resumptions=Vec::new();
    for name in training {
        let p=Path::new("reports/local-study").join(name);
        files(&p,&mut selected,true)?;
        let v:Value=serde_json::from_slice(&fs::read(p.join("progress.json"))?)?;
        let steps=v["steps"].as_u64().ok_or("missing committed steps")?;
        files(&p.join(format!("step-{steps:06}")),&mut selected,false)?;
        resumptions.push(json!({"directory":p,"committed_steps":steps,"adapter":v["adapter"],"optimizer":v["optimizer"],"final_manifest_present":p.join("manifest.json").exists()}));
    }
    for entry in fs::read_dir("reports/local-study")? {
        let p=entry?.path();
        if p.is_file() && p.file_name().is_some_and(|n| {let s=n.to_string_lossy();s.starts_with("native-full-") || s.starts_with("native-bitnet-64-")}) {selected.insert(p);}
    }
    let directories=[
        "reports/local-study/native-full-quality-001",
        "reports/local-study/qwen7b-evidence",
        "reports/local-study/qwen7b-train",
        "state/engines/5db09c632f6ffccc636bdfec2f693af102bf5327ce7510bc593ebe7089c9c126",
        "state/engines/f5d2f2f520f0c7a69186163c33bd28207c37414fb3cdae0499c169957a846944",
        "state/diagnostics/bitnet-layer-probe/round-001",
        "state/diagnostics/bitnet-layer-probe/alias-v2",
        "state/diagnostics/bitnet-layer-probe/oracle-source/versions/c85e38c6eeb079921cff42235bf415f506fb9f6c294e0c69e815bd0d95143872",
        "state/diagnostics/bitnet-rms-prototype",
        "state/diagnostics/cgroup-gated-001",
        "state/diagnostics/cgroup-gated-002",
        "state/diagnostics/cgroup-readonly-001",
        "docs/bitnet/provenance",
    ];
    for p in directories{files(Path::new(p),&mut selected,false)?;}
    for p in ["state/engines/bitnet-quant-v2-build/engine-manifest.json","docs/bitnet/engine-quantisation-v2.patch","run.sh","src/package.rs","Cargo.toml","Cargo.lock","build.rs","offline_bundle/bin/lec-distil","target/release/lec-distil","src/bitnet/native_rms.rs"] {files(Path::new(p),&mut selected,false)?;}
    for e in fs::read_dir("reports/comparative/bitnet")? {
        let p=e?.path();
        if p.is_file() && p.file_name().is_some_and(|n|n.to_string_lossy().starts_with("parity-quant-v2")) {selected.insert(p);}
    }
    let mut queue=VecDeque::from_iter(selected.iter().cloned());
    let mut expectations:BTreeMap<PathBuf,Vec<Value>>=BTreeMap::new();
    let mut invalid_references=Vec::new();
    while let Some(p)=queue.pop_front() {
        if p.extension().is_none_or(|x|x!="json") || !p.is_file() || fs::metadata(&p)?.len()>16*1024*1024 {continue}
        let Ok(v)=serde_json::from_slice::<Value>(&fs::read(&p)?) else{continue};
        let mut links=Vec::new();refs(&v,&mut links);
        for (name,sha,bytes) in links {
            let raw=PathBuf::from(&name);
            let rel=if raw.is_absolute(){match raw.strip_prefix(&root){Ok(p)=>p.to_owned(),Err(_)=>{invalid_references.push(json!({"from":p,"path":raw,"reason":"absolute external reference"}));continue}}}else{raw};
            if !rel.components().all(|c|matches!(c,std::path::Component::Normal(_))) {invalid_references.push(json!({"from":p,"path":rel,"reason":"unsafe or empty relative reference"}));continue}
            expectations.entry(rel.clone()).or_default().push(json!({"from":p,"sha256":sha,"bytes":bytes}));
            if selected.insert(rel.clone()){queue.push_back(rel);}
        }
        if selected.len()>20000{return Err("closure exceeded bounded file count".into())}
    }
    let mut inventory=Vec::new();let mut bytes_hashed=0u64;let mut unique=BTreeMap::new();let mut missing=0;let mut mismatches=0;
    for p in selected {
        let recorded=expectations.remove(&p).unwrap_or_default();
        let m=match fs::symlink_metadata(&p){Ok(m)=>m,Err(e)=>{missing+=1;inventory.push(json!({"path":p,"exists":false,"error":e.to_string(),"referenced_by":recorded,"package_status":package_status(&p)}));continue}};
        let target=fs::metadata(&p)?;
        if !target.is_file(){continue}
        let before=target.modified()?;
        let should_hash=target.len()<=128*1024*1024 && !p.starts_with("target") && bytes_hashed.saturating_add(target.len())<=768*1024*1024;
        let observed=if should_hash{bytes_hashed+=target.len();Some(hash(&p)?)}else{None};
        let after=fs::metadata(&p)?;
        let stable=before==after.modified()? && target.len()==after.len();
        let mismatch=recorded.iter().any(|r|r["bytes"].as_u64().is_some_and(|n|n!=target.len()) || observed.as_ref().zip(r["sha256"].as_str()).is_some_and(|(a,b)|a!=b));
        if mismatch{mismatches+=1}
        if let Some(h)=&observed{unique.entry(h.clone()).or_insert(target.len());}
        inventory.push(json!({"path":p,"exists":true,"bytes":target.len(),"sha256":observed,"hash_status":if should_hash{"measured_from_existing_bytes"}else{"not_rehashed_bounded_read_only_review"},"stable_during_read":stable,"reference_mismatch":mismatch,"mode_octal":format!("{:o}",target.permissions().mode()&0o7777),"symlink_target":if m.file_type().is_symlink(){Some(fs::read_link(&p)?)}else{None},"canonical_path":p.canonicalize()?,"package_status":package_status(&p),"referenced_by":recorded}));
    }
    let mut external=Vec::new();
    for p in ["/usr/lib/x86_64-linux-gnu/libcuda.so.1","/usr/local/cuda-13/targets/x86_64-linux/lib/libcurand.so.10","/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublas.so.12","/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublasLt.so.12","/usr/local/cuda-12.9/EULA.txt","/usr/local/cuda-13/EULA.txt","/usr/share/doc/libnvidia-compute-595/copyright"] {
        let p=Path::new(p);let m=fs::metadata(p)?;
        external.push(json!({"path":p,"canonical_path":p.canonicalize()?,"bytes":m.len(),"sha256":if m.len()<1024*1024{Some(hash(p)?)}else{None},"redistribution_review":"not established by this inventory; retain as explicit external runtime prerequisite unless a separate licence review permits bundling"}));
    }
    let result=json!({"schema_version":1,"evidence_kind":"bounded_read_only_exploratory_delivery_closure_inventory","observed_unix_seconds":SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs(),"original_absolute_root":root,"all_writers_quiesced":false,"active_regrade_excluded":"reports/local-study/regraded-v3-001","resumptions":resumptions,"files":inventory,"bytes_hashed_including_aliases":bytes_hashed,"unique_measured_payload_bytes":unique.values().sum::<u64>(),"missing_reference_count":missing,"reference_mismatch_count":mismatches,"external_runtime":external,"external_or_unsafe_references":invalid_references,"complete_delivery":false,"limitation":"This is a non-atomic inventory of selected current immutable evidence, not a package, a restore operation, a fresh-machine rehearsal, full model rehashing, or a claim that every historical compiler dependency must be bundled for execution."});
    fs::write(output.join("inventory.json"),serde_json::to_vec_pretty(&result)?)?;
    println!("inventoried {} files; hashed {} bytes; {} missing references; {} measured mismatches",result["files"].as_array().unwrap().len(),bytes_hashed,missing,mismatches);
    Ok(())
}
