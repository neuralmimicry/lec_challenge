//! Selected quantised weights with separately pinned original model data.
//!
//! GGUF and native safetensors manifests use separate validators. A quantisation
//! repository cannot silently supply a guessed tokeniser or chat template.

use super::*;
use std::collections::BTreeSet;

const ASSETS: &str = "gguf-assets.json";
const PROVENANCE: &str = "gguf-provenance.json";
const PLAN: &str = "gguf-plan.json";
const METADATA: &str = "gguf-hub-metadata.json";
const SOURCE_DIR: &str = "source";
const MAX_AUXILIARY_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HubGgufSource {
    pub model_id: String,
    pub revision: String,
}

#[derive(Clone, Debug)]
pub struct HubGgufDownloadOptions {
    pub model_id: String,
    pub revision: String,
    pub filename: String,
    pub destination: PathBuf,
    pub token_stdin: bool,
    pub source: HubGgufSource,
    pub expected_sha256: Option<String>,
    pub expected_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HubGgufSourceAssets {
    /// Relative to the composite bundle; always the literal `source` directory.
    pub directory: String,
    pub manifest: AssetManifest,
    pub provenance: HubProvenance,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HubGgufProvenance {
    pub schema_version: u32,
    pub model_id: String,
    pub revision: String,
    pub licence: String,
    pub filename: String,
    pub file_sha256: String,
    pub file_bytes: u64,
    pub assets: AssetManifest,
    pub quantisation: HubProvenance,
    pub source_assets: HubGgufSourceAssets,
    pub source_binding: String,
}

const SOURCE_BINDING: &str = "Explicit original-source repository and immutable commit; original config/tokeniser/template/licence independently hash-verified. This records provenance, not proof that quantisation preserves every floating-point tensor.";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct PrunedWeight {
    pub filename: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct PrunedWeightsProof {
    pub schema_version: u32,
    pub candidate_id: String,
    pub model_id: String,
    pub revision: String,
    pub directory: PathBuf,
    pub manifest_filename: String,
    pub manifest_sha256: String,
    pub created_unix_ms: i64,
    pub removal_state: String,
    pub removed_weights: Vec<PrunedWeight>,
}

fn selected_filename(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 240
        && name.ends_with(".gguf")
        && name != ".gguf"
        && !name.contains("..")
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}

fn planned_file(file: &ApiFile) -> Result<PlannedFile> {
    ensure!(
        file.size > 0 && file.size <= MAX_FILE_BYTES,
        "unsupported selected asset size"
    );
    let publisher_identity = if let Some(lfs) = &file.lfs {
        ensure!(
            valid_hex(&lfs.sha256, 64) && lfs.size == file.size,
            "invalid selected LFS metadata"
        );
        PublisherIdentity::LfsSha256(lfs.sha256.clone())
    } else {
        ensure!(
            valid_hex(&file.blob_id, 40),
            "invalid selected Git blob metadata"
        );
        PublisherIdentity::GitBlobSha1(file.blob_id.clone())
    };
    Ok(PlannedFile {
        filename: file.rfilename.clone(),
        bytes: file.size,
        publisher_identity,
    })
}

fn plan(model_id: &str, metadata: &ApiModel, selected: Option<&str>) -> Result<DownloadPlan> {
    validate_model_id(model_id)?;
    ensure!(
        metadata.id == model_id && valid_hex(&metadata.sha, 40),
        "selected repository metadata identity mismatch"
    );
    if let Some(selected) = selected {
        ensure!(
            selected_filename(selected),
            "unsafe or unsupported selected GGUF filename"
        );
    }
    let mut files = BTreeMap::new();
    let mut total = 0_u64;
    for file in &metadata.siblings {
        let weight = selected == Some(file.rfilename.as_str());
        let auxiliary = if selected.is_some() {
            file.rfilename == "README.md"
                || licence_filename(&file.rfilename)
                || matches!(file.rfilename.as_str(), "NOTICE" | "NOTICE.txt")
        } else {
            allowed_filename(&file.rfilename)
                && !file.rfilename.ends_with(".safetensors")
                && file.rfilename != "model.safetensors.index.json"
        };
        if !weight && !auxiliary {
            continue;
        }
        let value = planned_file(file)?;
        ensure!(
            weight || value.bytes <= MAX_AUXILIARY_BYTES,
            "original model metadata asset exceeds the bounded size"
        );
        total = total
            .checked_add(value.bytes)
            .context("selected asset size overflow")?;
        ensure!(
            total <= MAX_TOTAL_BYTES,
            "selected assets exceed the bounded total size"
        );
        ensure!(
            files.insert(value.filename.clone(), value).is_none(),
            "duplicate selected asset metadata"
        );
    }
    if let Some(selected) = selected {
        ensure!(
            files.contains_key(selected),
            "selected GGUF absent at the pinned commit"
        );
        ensure!(
            files.contains_key("README.md"),
            "original quantisation model card missing"
        );
    } else {
        ensure!(
            ["config.json", "tokenizer.json", "tokenizer_config.json"]
                .iter()
                .all(|name| files.contains_key(*name)),
            "original config/tokeniser/chat template missing; guessing is forbidden"
        );
        ensure!(
            files.keys().any(|name| licence_filename(name))
                || (files.contains_key("README.md")
                    && recognised_spdx_licence(&licence_from_metadata(metadata))),
            "original source licence evidence missing"
        );
    }
    Ok(DownloadPlan {
        model_id: model_id.to_owned(),
        revision: metadata.sha.clone(),
        files,
    })
}

async fn resolve(
    client: &Client,
    model_id: &str,
    revision: &str,
    bearer: Option<&Bearer>,
) -> Result<(ApiModel, Vec<u8>)> {
    validate_model_id(model_id)?;
    validate_ref(revision)?;
    eprintln!("Hub resolving immutable metadata for {model_id}");
    let (resolved, _) = metadata(client, model_id, revision, bearer).await?;
    ensure!(
        valid_hex(&resolved.sha, 40),
        "selected ref did not resolve to a commit"
    );
    let (pinned, bytes) = metadata(client, model_id, &resolved.sha, bearer).await?;
    ensure!(
        pinned.sha == resolved.sha,
        "selected pinned metadata changed"
    );
    Ok((pinned, bytes))
}

fn persist_plan(dir: &Path, filename: &str, plan: &DownloadPlan) -> Result<()> {
    ensure_safe_path(dir)?;
    fs::create_dir_all(dir)?;
    let path = dir.join(filename);
    ensure_safe_path(&path)?;
    if path.exists() {
        ensure!(
            serde_json::from_slice::<DownloadPlan>(&fs::read(path)?)? == *plan,
            "selected model destination has a different immutable plan"
        );
    } else {
        atomic_publish(&path, &serde_json::to_vec_pretty(plan)?)?;
    }
    Ok(())
}

fn make_provenance(
    plan: &DownloadPlan,
    requested: &str,
    metadata: &[u8],
    files: BTreeMap<String, HubAsset>,
    inherited_licence: Option<&str>,
) -> Result<HubProvenance> {
    let api: ApiModel = serde_json::from_slice(metadata)?;
    let licence_files: Vec<String> = files
        .keys()
        .filter(|name| licence_filename(name))
        .cloned()
        .collect();
    Ok(HubProvenance {
        schema_version: 1,
        model_id: plan.model_id.clone(),
        requested_revision: requested.into(),
        revision: plan.revision.clone(),
        licence: inherited_licence
            .map(str::to_owned)
            .unwrap_or_else(|| licence_from_metadata(&api)),
        licence_evidence: if inherited_licence.is_some() {
            "Original source licence retained in the independently pinned source bundle; quantisation repository model-card declarations retained separately".into()
        } else {
            licence_evidence(!licence_files.is_empty())
        },
        licence_files,
        recorded_unix_seconds: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        metadata_url: format!(
            "https://huggingface.co/api/models/{}/revision/{}?blobs=true",
            plan.model_id, plan.revision
        ),
        metadata_sha256: hex::encode(Sha256::digest(metadata)),
        metadata: serde_json::from_slice(metadata)?,
        files,
        authentication_policy: AUTHENTICATION_POLICY.into(),
    })
}

fn manifest(proof: &HubProvenance) -> AssetManifest {
    AssetManifest {
        model_id: proof.model_id.clone(),
        revision: proof.revision.clone(),
        licence: proof.licence.clone(),
        files: proof
            .files
            .iter()
            .map(|(name, file)| (name.clone(), file.sha256.clone()))
            .collect(),
    }
}

/// Download one explicitly selected GGUF and independently pinned original data.
/// The source plan excludes every original floating-point weight file.
pub async fn download_gguf(options: HubGgufDownloadOptions) -> Result<HubGgufProvenance> {
    ensure!(
        selected_filename(&options.filename),
        "unsafe or unsupported selected GGUF filename"
    );
    if let Some(expected) = &options.expected_sha256 {
        ensure!(valid_hex(expected, 64), "invalid expected GGUF SHA-256");
    }
    // A complete published cache is checked before reading credentials or making
    // even a metadata request. A corrupt published cache fails rather than being
    // silently replaced or treated as a connectivity event.
    if options.destination.join(ASSETS).exists() && options.destination.join(PROVENANCE).exists() {
        eprintln!(
            "Hub validating the published GGUF cache for {}",
            options.model_id
        );
        let cached_dir = options.destination.clone();
        let (cached, intentionally_missing) = tokio::task::spawn_blocking(move || {
            let missing = authorised_missing_weights(&cached_dir)?;
            Ok::<_, anyhow::Error>((
                validate_gguf_assets_inner(&cached_dir, &missing)?,
                !missing.is_empty(),
            ))
        })
        .await??;
        ensure!(
            cached.model_id == options.model_id
                && cached.filename == options.filename
                && (cached.revision == options.revision
                    || cached.quantisation.requested_revision == options.revision)
                && cached.source_assets.manifest.model_id == options.source.model_id
                && (cached.source_assets.manifest.revision == options.source.revision
                    || cached.source_assets.provenance.requested_revision
                        == options.source.revision),
            "published GGUF cache differs from the requested immutable candidate/source"
        );
        if let Some(expected) = &options.expected_sha256 {
            ensure!(
                &cached.file_sha256 == expected,
                "published GGUF cache differs from authorised SHA-256"
            );
        }
        if let Some(expected) = options.expected_bytes {
            ensure!(
                cached.file_bytes == expected,
                "published GGUF cache differs from authorised size"
            );
        }
        if !intentionally_missing {
            return Ok(cached);
        }
        eprintln!(
            "Hub rehydrating only explicitly pruned weights after verifying preserved metadata and ownership"
        );
    }
    let token_stdin = options.token_stdin;
    let bearer = tokio::task::spawn_blocking(move || load_bearer(token_stdin)).await??;
    let client = client()?;
    let (quant_api, quant_bytes) = resolve(
        &client,
        &options.model_id,
        &options.revision,
        bearer.as_ref(),
    )
    .await?;
    let (source_api, source_bytes) = resolve(
        &client,
        &options.source.model_id,
        &options.source.revision,
        bearer.as_ref(),
    )
    .await?;
    let quant_plan = plan(&options.model_id, &quant_api, Some(&options.filename))?;
    let source_plan = plan(&options.source.model_id, &source_api, None)?;
    let weight = &quant_plan.files[&options.filename];
    if let Some(expected) = options.expected_bytes {
        ensure!(
            weight.bytes == expected,
            "selected GGUF publisher size differs from the authorised candidate"
        );
    }
    if let Some(expected) = &options.expected_sha256 {
        ensure!(
            matches!(&weight.publisher_identity, PublisherIdentity::LfsSha256(hash) if hash == expected),
            "selected GGUF publisher hash differs from the authorised candidate"
        );
    }
    let directory = options.destination.clone();
    let _lock = tokio::task::spawn_blocking(move || DestinationLock::acquire(&directory)).await??;
    let source_dir = options.destination.join(SOURCE_DIR);
    let prepare_dir = options.destination.clone();
    let prepare_source = source_dir.clone();
    let persist_quant = quant_plan.clone();
    let persist_source = source_plan.clone();
    tokio::task::spawn_blocking(move || {
        persist_plan(&prepare_dir, PLAN, &persist_quant)?;
        persist_plan(&prepare_source, PLAN_FILE, &persist_source)
    })
    .await??;
    let mut source_files = BTreeMap::new();
    for file in source_plan.files.values() {
        let asset = download_file(
            &client,
            bearer.as_ref(),
            &source_dir,
            &source_plan.model_id,
            &source_plan.revision,
            file.clone(),
        )
        .await?;
        source_files.insert(asset.filename.clone(), asset);
    }
    let source_proof = make_provenance(
        &source_plan,
        &options.source.revision,
        &source_bytes,
        source_files,
        None,
    )?;
    let source_manifest = manifest(&source_proof);
    let mut quant_files = BTreeMap::new();
    for file in quant_plan.files.values() {
        let asset = download_file(
            &client,
            bearer.as_ref(),
            &options.destination,
            &quant_plan.model_id,
            &quant_plan.revision,
            file.clone(),
        )
        .await?;
        quant_files.insert(asset.filename.clone(), asset);
    }
    let quant_proof = make_provenance(
        &quant_plan,
        &options.revision,
        &quant_bytes,
        quant_files,
        Some(&source_proof.licence),
    )?;
    let mut assets = manifest(&quant_proof);
    for (name, hash) in &source_manifest.files {
        assets
            .files
            .insert(format!("{SOURCE_DIR}/{name}"), hash.clone());
    }
    let selected = &quant_proof.files[&options.filename];
    if let Some(expected) = options.expected_sha256 {
        ensure!(
            selected.sha256 == expected,
            "complete selected GGUF SHA-256 differs from authorised candidate"
        );
    }
    let proof = HubGgufProvenance {
        schema_version: 1,
        model_id: quant_plan.model_id,
        revision: quant_plan.revision,
        licence: source_proof.licence.clone(),
        filename: options.filename,
        file_sha256: selected.sha256.clone(),
        file_bytes: selected.bytes,
        assets,
        quantisation: quant_proof,
        source_assets: HubGgufSourceAssets {
            directory: SOURCE_DIR.into(),
            manifest: source_manifest,
            provenance: source_proof,
        },
        source_binding: SOURCE_BINDING.into(),
    };
    let directory = options.destination;
    let publish = proof.clone();
    tokio::task::spawn_blocking(move || {
        validate_config(&source_dir)?;
        validate_gguf_header(&directory.join(&publish.filename))?;
        atomic_publish(&source_dir.join("hub-metadata.json"), &source_bytes)?;
        atomic_publish(
            &source_dir.join(PROVENANCE_FILE),
            &serde_json::to_vec_pretty(&publish.source_assets.provenance)?,
        )?;
        atomic_publish(
            &source_dir.join("assets.json"),
            &serde_json::to_vec_pretty(&publish.source_assets.manifest)?,
        )?;
        atomic_publish(&directory.join(METADATA), &quant_bytes)?;
        atomic_publish(
            &directory.join(PROVENANCE),
            &serde_json::to_vec_pretty(&publish)?,
        )?;
        atomic_publish(
            &directory.join(ASSETS),
            &serde_json::to_vec_pretty(&publish.assets)?,
        )?;
        validate_gguf_assets(&directory)
    })
    .await??;
    Ok(proof)
}

fn validate_gguf_header(path: &Path) -> Result<()> {
    ensure_safe_path(path)?;
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags_no_follow()
        .open(path)?;
    let mut header = [0_u8; 8];
    file.read_exact(&mut header)?;
    ensure!(
        &header[..4] == b"GGUF" && matches!(u32::from_le_bytes(header[4..8].try_into()?), 2 | 3),
        "selected file is not supported GGUF version 2 or 3"
    );
    Ok(())
}

#[derive(Clone, Copy)]
enum WeightVerification<'a> {
    Full(&'a BTreeSet<String>),
    HistoricalMetadata,
}

fn verify_provenance(
    dir: &Path,
    metadata_path: &str,
    plan_path: &str,
    proof: &HubProvenance,
    selected: Option<&str>,
    inherited: Option<&str>,
    verification: WeightVerification<'_>,
) -> Result<AssetManifest> {
    ensure_safe_path(dir)?;
    for filename in [metadata_path, plan_path] {
        ensure_safe_path(&dir.join(filename))?;
    }
    let bytes = fs::read(dir.join(metadata_path))?;
    ensure!(
        bytes.len() <= MAX_METADATA_BYTES
            && hex::encode(Sha256::digest(&bytes)) == proof.metadata_sha256,
        "selected original metadata hash mismatch"
    );
    let api: ApiModel = serde_json::from_slice(&bytes)?;
    let plan = plan(&proof.model_id, &api, selected)?;
    ensure!(
        serde_json::from_slice::<DownloadPlan>(&fs::read(dir.join(plan_path))?)? == plan,
        "selected plan differs from original pinned metadata"
    );
    let expected = make_provenance(
        &plan,
        &proof.requested_revision,
        &bytes,
        proof.files.clone(),
        inherited,
    )?;
    ensure!(
        proof.schema_version == 1
            && proof.revision == plan.revision
            && proof.licence == expected.licence
            && proof.licence_files == expected.licence_files
            && proof.licence_evidence == expected.licence_evidence
            && proof.metadata_url == expected.metadata_url
            && proof.metadata == expected.metadata
            && proof.authentication_policy == AUTHENTICATION_POLICY,
        "selected provenance differs from original metadata"
    );
    validate_ref(&proof.requested_revision)?;
    ensure!(
        proof.files.len() == plan.files.len(),
        "selected provenance assets missing or unexpected"
    );
    for (name, file) in &plan.files {
        let asset = proof
            .files
            .get(name)
            .context("selected provenance asset absent")?;
        ensure!(
            asset.filename == *name
                && asset.bytes == file.bytes
                && asset.publisher_identity == file.publisher_identity
                && asset.canonical_url == canonical_file_url(&plan.model_id, &plan.revision, name),
            "selected publisher identity changed"
        );
        let path = dir.join(name);
        ensure_safe_path(&path)?;
        if matches!(verification, WeightVerification::HistoricalMetadata) && name.ends_with(".gguf")
        {
            super::historical::verify_weight_metadata(dir, ASSETS, file, &asset.sha256)?;
        } else if !matches!(verification, WeightVerification::Full(missing) if missing.contains(name) && !path.exists())
        {
            ensure!(
                verify_file(&path, file)? == asset.sha256,
                "selected original asset checksum mismatch"
            );
        }
    }
    Ok(manifest(proof))
}

fn same_manifest(a: &AssetManifest, b: &AssetManifest) -> bool {
    a.model_id == b.model_id
        && a.revision == b.revision
        && a.licence == b.licence
        && a.files == b.files
}

/// Verify both independently pinned repositories without any network access.
pub fn validate_gguf_assets(directory: &Path) -> Result<HubGgufProvenance> {
    validate_gguf_assets_inner(directory, &BTreeSet::new())
}

fn validate_gguf_assets_inner(
    directory: &Path,
    missing: &BTreeSet<String>,
) -> Result<HubGgufProvenance> {
    validate_gguf_assets_with_policy(directory, missing, false)
}

pub(super) fn validate_gguf_assets_with_policy(
    directory: &Path,
    missing: &BTreeSet<String>,
    historical: bool,
) -> Result<HubGgufProvenance> {
    ensure_safe_path(directory)?;
    for filename in [ASSETS, PROVENANCE] {
        ensure_safe_path(&directory.join(filename))?;
    }
    let proof: HubGgufProvenance = serde_json::from_slice(&fs::read(directory.join(PROVENANCE))?)?;
    ensure!(
        proof.schema_version == 1
            && selected_filename(&proof.filename)
            && proof.source_assets.directory == SOURCE_DIR
            && proof.source_binding == SOURCE_BINDING,
        "unsupported or unsafe composite GGUF provenance"
    );
    let source_dir = directory.join(SOURCE_DIR);
    let source = verify_provenance(
        &source_dir,
        "hub-metadata.json",
        PLAN_FILE,
        &proof.source_assets.provenance,
        None,
        None,
        WeightVerification::Full(&BTreeSet::new()),
    )?;
    ensure_safe_path(&source_dir.join("assets.json"))?;
    ensure_safe_path(&source_dir.join(PROVENANCE_FILE))?;
    let source_disk: AssetManifest =
        serde_json::from_slice(&fs::read(source_dir.join("assets.json"))?)?;
    let source_proof_disk: HubProvenance =
        serde_json::from_slice(&fs::read(source_dir.join(PROVENANCE_FILE))?)?;
    ensure!(
        same_manifest(&source, &source_disk)
            && same_manifest(&source, &proof.source_assets.manifest)
            && serde_json::to_value(source_proof_disk)?
                == serde_json::to_value(&proof.source_assets.provenance)?,
        "original source manifest/provenance differs from composite binding"
    );
    validate_config(&source_dir)?;
    let mut composite = verify_provenance(
        directory,
        METADATA,
        PLAN,
        &proof.quantisation,
        Some(&proof.filename),
        Some(&source.licence),
        if historical {
            WeightVerification::HistoricalMetadata
        } else {
            WeightVerification::Full(missing)
        },
    )?;
    for (name, hash) in source.files {
        composite.files.insert(format!("{SOURCE_DIR}/{name}"), hash);
    }
    let disk: AssetManifest = serde_json::from_slice(&fs::read(directory.join(ASSETS))?)?;
    let weight = proof
        .quantisation
        .files
        .get(&proof.filename)
        .context("selected quantised weight proof missing")?;
    ensure!(
        proof.model_id == composite.model_id
            && proof.revision == composite.revision
            && proof.licence == composite.licence
            && proof.file_sha256 == weight.sha256
            && proof.file_bytes == weight.bytes
            && same_manifest(&proof.assets, &composite)
            && same_manifest(&disk, &composite),
        "composite GGUF asset identity changed"
    );
    if !historical
        && !(missing.contains(&proof.filename) && !directory.join(&proof.filename).exists())
    {
        validate_gguf_header(&directory.join(&proof.filename))?;
    }
    Ok(proof)
}

fn authorised_missing_weights(directory: &Path) -> Result<BTreeSet<String>> {
    let marker = directory.join(".pruned-weights.json");
    ensure_safe_path(&marker)?;
    if !marker.exists() {
        return Ok(BTreeSet::new());
    }
    ensure!(
        fs::metadata(&marker)?.len() <= 1024 * 1024,
        "prune authority oversized"
    );
    let pruned: PrunedWeightsProof = serde_json::from_slice(&fs::read(marker)?)?;
    let owner_path = directory.join(".teacher-search-owned.json");
    ensure_safe_path(&owner_path)?;
    ensure!(
        fs::metadata(&owner_path)?.len() <= 1024 * 1024,
        "cache ownership oversized"
    );
    let owner: serde_json::Value = serde_json::from_slice(&fs::read(owner_path)?)?;
    let workspace = std::env::current_dir()?.canonicalize()?;
    let relative = if directory.is_absolute() {
        directory
            .strip_prefix(&workspace)
            .context("pruned cache lies outside its workspace")?
    } else {
        directory
    };
    ensure!(
        pruned.directory == relative
            && owner["directory"] == serde_json::to_value(&pruned.directory)?
            && owner["candidate_id"] == pruned.candidate_id
            && owner["model_id"] == pruned.model_id
            && owner["revision"] == pruned.revision
            && owner["schema_version"] == 1
            && pruned.schema_version == 1
            && matches!(pruned.removal_state.as_str(), "planned" | "completed")
            && pruned.manifest_filename == ASSETS,
        "invalid or unrelated prune authority"
    );
    let manifest_path = directory.join(ASSETS);
    ensure_safe_path(&manifest_path)?;
    let bytes = fs::read(&manifest_path)?;
    ensure!(
        hex::encode(Sha256::digest(&bytes)) == pruned.manifest_sha256,
        "preserved GGUF manifest changed after prune"
    );
    let manifest: AssetManifest = serde_json::from_slice(&bytes)?;
    ensure!(
        manifest.model_id == pruned.model_id && manifest.revision == pruned.revision,
        "pruned model identity differs from preserved manifest"
    );
    let proof: HubGgufProvenance = serde_json::from_slice(&fs::read(directory.join(PROVENANCE))?)?;
    let mut missing = BTreeSet::new();
    let mut declared = BTreeSet::new();
    for file in pruned.removed_weights {
        ensure!(
            selected_filename(&file.filename)
                && file.filename == proof.filename
                && declared.insert(file.filename.clone()),
            "prune authority names an unexpected or duplicate asset"
        );
        ensure!(
            manifest.files.get(&file.filename) == Some(&file.sha256)
                && proof.file_sha256 == file.sha256
                && proof.file_bytes == file.bytes,
            "pruned weight differs from original publisher-bound identity"
        );
        let path = directory.join(&file.filename);
        ensure_safe_path(&path)?;
        if !path.exists() {
            missing.insert(file.filename);
        }
    }
    Ok(missing)
}

/// True only for absent model weights covered by durable, preserved prune proof.
/// All remaining original bytes and both repository identities are revalidated.
pub fn is_intentionally_pruned_gguf(directory: &Path) -> Result<bool> {
    let missing = authorised_missing_weights(directory)?;
    if missing.is_empty() {
        return Ok(false);
    }
    validate_gguf_assets_inner(directory, &missing)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata_fixture(id: &str, selected: bool) -> ApiModel {
        let names = if selected {
            vec![
                "README.md",
                "model-q4_k_m.gguf",
                "another-q8.gguf",
                "model.safetensors",
            ]
        } else {
            vec![
                "README.md",
                "LICENSE",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors",
                "model.py",
            ]
        };
        ApiModel {
            id: id.into(),
            sha: "a".repeat(40),
            card_data: serde_json::json!({"license":"apache-2.0"}),
            siblings: names
                .into_iter()
                .map(|name| ApiFile {
                    rfilename: name.into(),
                    blob_id: "b".repeat(40),
                    size: 32,
                    lfs: None,
                })
                .collect(),
        }
    }

    #[test]
    fn selection_never_downloads_other_quantisations_or_original_weights() -> Result<()> {
        let quant = plan(
            "organisation/quant",
            &metadata_fixture("organisation/quant", true),
            Some("model-q4_k_m.gguf"),
        )?;
        assert_eq!(
            quant.files.keys().cloned().collect::<Vec<_>>(),
            ["README.md", "model-q4_k_m.gguf"]
        );
        let source = plan(
            "organisation/base",
            &metadata_fixture("organisation/base", false),
            None,
        )?;
        assert!(!source.files.contains_key("model.safetensors"));
        assert!(!source.files.contains_key("model.py"));
        for name in [
            "../model.gguf",
            "sub/model.gguf",
            "/model.gguf",
            "model.gguf.py",
            "model.gguf?token=anything",
        ] {
            assert!(!selected_filename(name));
        }
        Ok(())
    }

    #[test]
    fn source_requires_original_tokeniser_and_gguf_header_is_checked() -> Result<()> {
        let mut source = metadata_fixture("organisation/base", false);
        source
            .siblings
            .retain(|file| file.rfilename != "tokenizer_config.json");
        assert!(plan("organisation/base", &source, None).is_err());
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("test.gguf");
        fs::write(&path, b"GGUF\x03\x00\x00\x00")?;
        validate_gguf_header(&path)?;
        fs::write(&path, b"GGUF\x04\x00\x00\x00")?;
        assert!(validate_gguf_header(&path).is_err());
        fs::write(&path, b"not-gguf")?;
        assert!(validate_gguf_header(&path).is_err());
        Ok(())
    }

    fn fixture_composite(directory: &Path) -> Result<()> {
        fs::create_dir_all(directory.join(SOURCE_DIR))?;
        let source_contents: BTreeMap<&str, &[u8]> = BTreeMap::from([
            ("README.md", b"original card".as_slice()),
            ("LICENSE", b"original test licence".as_slice()),
            ("config.json", b"{}".as_slice()),
            ("tokenizer.json", b"{}".as_slice()),
            ("tokenizer_config.json", b"{}".as_slice()),
        ]);
        let quant_contents: BTreeMap<&str, &[u8]> = BTreeMap::from([
            ("README.md", b"quantisation card".as_slice()),
            ("model-q4_k_m.gguf", b"GGUF\x03\x00\x00\x00".as_slice()),
        ]);
        let build = |id: &str,
                     contents: &BTreeMap<&str, &[u8]>,
                     selected: Option<&str>,
                     folder: &Path,
                     inherited: Option<&str>|
         -> Result<(HubProvenance, Vec<u8>)> {
            let mut siblings = Vec::new();
            for (name, bytes) in contents {
                let mut git = Sha1::new();
                git.update(format!("blob {}\0", bytes.len()));
                git.update(bytes);
                siblings.push(ApiFile {
                    rfilename: (*name).into(),
                    blob_id: hex::encode(git.finalize()),
                    size: bytes.len() as u64,
                    lfs: name.ends_with(".gguf").then(|| ApiLfs {
                        sha256: hex::encode(Sha256::digest(bytes)),
                        size: bytes.len() as u64,
                    }),
                });
            }
            let metadata = ApiModel {
                id: id.into(),
                sha: "a".repeat(40),
                card_data: serde_json::json!({"license":"apache-2.0"}),
                siblings,
            };
            let bytes = serde_json::to_vec(&metadata)?;
            let plan = plan(id, &metadata, selected)?;
            persist_plan(
                folder,
                if selected.is_some() { PLAN } else { PLAN_FILE },
                &plan,
            )?;
            let mut files = BTreeMap::new();
            for (name, contents) in contents {
                fs::write(folder.join(name), contents)?;
                let planned = &plan.files[*name];
                files.insert(
                    (*name).into(),
                    HubAsset {
                        filename: (*name).into(),
                        bytes: contents.len() as u64,
                        publisher_identity: planned.publisher_identity.clone(),
                        canonical_url: canonical_file_url(id, &plan.revision, name),
                        sha256: hex::encode(Sha256::digest(contents)),
                    },
                );
            }
            Ok((
                make_provenance(&plan, &plan.revision, &bytes, files, inherited)?,
                bytes,
            ))
        };
        let source_dir = directory.join(SOURCE_DIR);
        let (source, source_bytes) = build(
            "organisation/base",
            &source_contents,
            None,
            &source_dir,
            None,
        )?;
        let source_manifest = manifest(&source);
        let (quant, quant_bytes) = build(
            "organisation/quant",
            &quant_contents,
            Some("model-q4_k_m.gguf"),
            directory,
            Some(&source.licence),
        )?;
        let mut composite = manifest(&quant);
        for (name, hash) in &source_manifest.files {
            composite
                .files
                .insert(format!("{SOURCE_DIR}/{name}"), hash.clone());
        }
        let file = &quant.files["model-q4_k_m.gguf"];
        let proof = HubGgufProvenance {
            schema_version: 1,
            model_id: quant.model_id.clone(),
            revision: quant.revision.clone(),
            licence: quant.licence.clone(),
            filename: file.filename.clone(),
            file_sha256: file.sha256.clone(),
            file_bytes: file.bytes,
            assets: composite.clone(),
            quantisation: quant,
            source_assets: HubGgufSourceAssets {
                directory: SOURCE_DIR.into(),
                manifest: source_manifest.clone(),
                provenance: source.clone(),
            },
            source_binding: SOURCE_BINDING.into(),
        };
        fs::write(source_dir.join("hub-metadata.json"), source_bytes)?;
        fs::write(
            source_dir.join(PROVENANCE_FILE),
            serde_json::to_vec(&source)?,
        )?;
        fs::write(
            source_dir.join("assets.json"),
            serde_json::to_vec(&source_manifest)?,
        )?;
        fs::write(directory.join(METADATA), quant_bytes)?;
        fs::write(directory.join(PROVENANCE), serde_json::to_vec(&proof)?)?;
        fs::write(directory.join(ASSETS), serde_json::to_vec(&composite)?)?;
        Ok(())
    }

    #[test]
    fn composite_binds_both_repositories_and_rejects_source_weight_or_pin_changes() -> Result<()> {
        let dir = tempfile::tempdir()?;
        fixture_composite(dir.path())?;
        let proof = validate_gguf_assets(dir.path())?;
        assert_eq!(proof.source_assets.manifest.model_id, "organisation/base");
        assert!(
            proof
                .assets
                .files
                .contains_key("source/tokenizer_config.json")
        );
        assert!(
            validate_downloaded_assets(dir.path()).is_err(),
            "dense validator must remain separate"
        );
        fs::write(dir.path().join("source/tokenizer_config.json"), b"[]")?;
        assert!(validate_gguf_assets(dir.path()).is_err());
        fixture_composite(dir.path())?;
        fs::write(
            dir.path().join("model-q4_k_m.gguf"),
            b"GGUF\x02\x00\x00\x00",
        )?;
        assert!(validate_gguf_assets(dir.path()).is_err());
        fixture_composite(dir.path())?;
        let path = dir.path().join(PROVENANCE);
        let mut proof: HubGgufProvenance = serde_json::from_slice(&fs::read(&path)?)?;
        proof.source_assets.provenance.revision = "b".repeat(40);
        fs::write(&path, serde_json::to_vec(&proof)?)?;
        assert!(validate_gguf_assets(dir.path()).is_err());
        fixture_composite(dir.path())?;
        let mut proof: HubGgufProvenance = serde_json::from_slice(&fs::read(&path)?)?;
        proof.source_assets.directory = "../base".into();
        fs::write(path, serde_json::to_vec(&proof)?)?;
        assert!(validate_gguf_assets(dir.path()).is_err());
        Ok(())
    }

    #[tokio::test]
    async fn published_cache_requires_no_hub_request_and_checks_authorised_identity() -> Result<()>
    {
        let dir = tempfile::tempdir()?;
        fixture_composite(dir.path())?;
        let options = HubGgufDownloadOptions {
            model_id: "organisation/quant".into(),
            revision: "a".repeat(40),
            filename: "model-q4_k_m.gguf".into(),
            destination: dir.path().into(),
            token_stdin: false,
            source: HubGgufSource {
                model_id: "organisation/base".into(),
                revision: "a".repeat(40),
            },
            expected_sha256: Some(hex::encode(Sha256::digest(b"GGUF\x03\x00\x00\x00"))),
            expected_bytes: Some(8),
        };
        // These deliberately fictitious repositories cannot supply the fixture.
        // Success therefore requires the complete local publisher-bound cache.
        let proof =
            tokio::time::timeout(Duration::from_secs(2), download_gguf(options.clone())).await??;
        assert_eq!(proof.file_bytes, 8);
        let mut wrong = options;
        wrong.expected_sha256 = Some("b".repeat(64));
        let error = download_gguf(wrong).await.unwrap_err();
        assert!(!is_connectivity_error(&error));
        Ok(())
    }

    #[test]
    fn only_durable_owned_prune_allows_absent_weights_and_never_source_corruption() -> Result<()> {
        let dir = tempfile::tempdir_in("target")?;
        fixture_composite(dir.path())?;
        let original = validate_gguf_assets(dir.path())?;
        let relative = if dir.path().is_absolute() {
            dir.path()
                .strip_prefix(std::env::current_dir()?)?
                .to_owned()
        } else {
            dir.path().to_owned()
        };
        fs::write(
            dir.path().join(".teacher-search-owned.json"),
            serde_json::to_vec(
                &serde_json::json!({"schema_version":1,"directory":relative,"candidate_id":"fixture","model_id":original.model_id,"revision":original.revision}),
            )?,
        )?;
        let proof = PrunedWeightsProof {
            schema_version: 1,
            candidate_id: "fixture".into(),
            model_id: original.model_id.clone(),
            revision: original.revision.clone(),
            directory: relative,
            manifest_filename: ASSETS.into(),
            manifest_sha256: hex::encode(Sha256::digest(fs::read(dir.path().join(ASSETS))?)),
            created_unix_ms: 1,
            removal_state: "completed".into(),
            removed_weights: vec![PrunedWeight {
                filename: original.filename.clone(),
                sha256: original.file_sha256.clone(),
                bytes: original.file_bytes,
            }],
        };
        fs::write(
            dir.path().join(".pruned-weights.json"),
            serde_json::to_vec(&proof)?,
        )?;
        fs::remove_file(dir.path().join(&original.filename))?;
        assert!(is_intentionally_pruned_gguf(dir.path())?);
        validate_gguf_asset_metadata(dir.path())?;
        assert!(
            validate_gguf_assets(dir.path()).is_err(),
            "serving still requires actual weights"
        );
        fs::write(dir.path().join("source/tokenizer_config.json"), b"[]")?;
        assert!(is_intentionally_pruned_gguf(dir.path()).is_err());
        assert!(validate_gguf_asset_metadata(dir.path()).is_err());
        fixture_composite(dir.path())?;
        fs::remove_file(dir.path().join(&original.filename))?;
        let mut forged = proof.clone();
        forged.manifest_sha256 = "f".repeat(64);
        fs::write(
            dir.path().join(".pruned-weights.json"),
            serde_json::to_vec(&forged)?,
        )?;
        assert!(is_intentionally_pruned_gguf(dir.path()).is_err());
        assert!(validate_gguf_asset_metadata(dir.path()).is_err());
        fs::write(
            dir.path().join(".pruned-weights.json"),
            serde_json::to_vec(&proof)?,
        )?;
        fs::write(dir.path().join(&original.filename), b"GGUF\x02\x00\x00\x00")?;
        assert!(
            validate_gguf_assets_inner(dir.path(), &authorised_missing_weights(dir.path())?)
                .is_err(),
            "present corruption cannot gain rehydration authority"
        );
        Ok(())
    }
}
