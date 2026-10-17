//! Explicit, separately built numerical protocol. Original engine artifacts
//! remain available for replay of the earlier failed parity observations.
use super::*;

pub async fn bootstrap_quantisation_v2(offline: bool, jobs: usize) -> Result<EngineIdentity> {
    // This acquires its own bootstrap lease before returning. Do not nest the
    // second lease or allow a partial new build to mutate the original source.
    bootstrap_engine(offline, jobs).await?;
    let source = PathBuf::from("state/engines-src/bitnet-quant-v2");
    let build = PathBuf::from("state/engines/bitnet-quant-v2-build");
    let receipt = build.join("engine-manifest.json");
    let _lease =
        build_lease::BuildLease::acquire(Path::new("state/engines/.bitnet-bootstrap.lock")).await?;
    if receipt.exists() {
        return tokio::task::spawn_blocking(move || verify_engine(&receipt)).await?;
    }
    fs::create_dir_all(&build)?;
    let log = build.join("bootstrap.log");
    let source_text = source.to_str().context("invalid native source path")?;
    let build_text = build.to_str().context("invalid native build path")?;
    if !source.join(".git").exists() {
        run_build(
            &[
                "git",
                "clone",
                "--no-hardlinks",
                "--no-checkout",
                "state/engines-src/bitnet",
                source_text,
            ],
            &log,
        )
        .await?;
    }
    run_build(
        &[
            "git",
            "-C",
            source_text,
            "checkout",
            "--detach",
            ENGINE_COMMIT,
        ],
        &log,
    )
    .await?;
    let submodule = source.join("3rdparty/llama.cpp");
    let submodule_text = submodule
        .to_str()
        .context("invalid native submodule path")?;
    if !submodule.join(".git").exists() {
        run_build(
            &[
                "git",
                "clone",
                "--no-hardlinks",
                "--no-checkout",
                "state/engines-src/bitnet/3rdparty/llama.cpp",
                submodule_text,
            ],
            &log,
        )
        .await?;
    }
    run_build(
        &[
            "git",
            "-C",
            submodule_text,
            "checkout",
            "--detach",
            LLAMA_COMMIT,
        ],
        &log,
    )
    .await?;
    let diff = git(&submodule, &["diff", "HEAD", "--"])?;
    if diff.is_empty() {
        let patch = std::env::current_dir()?.join("docs/bitnet/engine-quantisation-v2.patch");
        run_build(
            &[
                "git",
                "-C",
                submodule_text,
                "apply",
                patch.to_str().context("invalid numerical patch path")?,
            ],
            &log,
        )
        .await?;
    } else {
        ensure!(
            reviewed_patch(diff.as_bytes())?.0 == 2,
            "numerical-v2 build has another engine patch"
        );
    }
    run_build(
        &[
            "cmake",
            "-S",
            source_text,
            "-B",
            build_text,
            "-DGGML_CUDA=OFF",
            "-DLLAMA_BUILD_COMMON=ON",
            "-DLLAMA_BUILD_TOOLS=ON",
            "-DLLAMA_BUILD_EXAMPLES=ON",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_UI=OFF",
            "-DLLAMA_USE_PREBUILT_UI=OFF",
            "-DLLAMA_OPENSSL=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_NATIVE=OFF",
        ],
        &log,
    )
    .await?;
    run_build(
        &[
            "cmake",
            "--build",
            build_text,
            "--target",
            "llama-cli",
            "llama-server",
            "-j",
            &jobs.to_string(),
        ],
        &log,
    )
    .await?;
    tokio::task::spawn_blocking(move || record_engine(&source, &build, &receipt)).await?
}
