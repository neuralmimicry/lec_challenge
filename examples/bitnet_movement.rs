//! Read only saved low-rank tensors and selected packed q/v matrices/scales.
//! This diagnoses a retained failed round without allocating the full model.
use anyhow::{Context, Result, ensure};
use lec_distil::artefact::{self, Artefact};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::Command,
};

struct Tensors {
    file: File,
    header: Value,
    data: u64,
    header_sha256: String,
}
impl Tensors {
    fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut length = [0; 8];
        file.read_exact(&mut length)?;
        let length = u64::from_le_bytes(length);
        ensure!(length <= 8 << 20, "oversized safetensors header");
        let mut bytes = vec![0; length as usize];
        file.read_exact(&mut bytes)?;
        Ok(Self {
            file,
            header: serde_json::from_slice(&bytes)?,
            data: 8 + length,
            header_sha256: artefact::hash(&bytes),
        })
    }
    fn bytes(&mut self, name: &str, dtype: &str) -> Result<Vec<u8>> {
        let item = &self.header[name];
        ensure!(item["dtype"] == dtype, "wrong tensor dtype for {name}");
        let first = item["data_offsets"][0]
            .as_u64()
            .context("missing tensor start")?;
        let end = item["data_offsets"][1]
            .as_u64()
            .context("missing tensor end")?;
        ensure!(
            end >= first
                && end - first <= 4 << 20
                && self.data + end <= self.file.metadata()?.len(),
            "tensor read exceeds diagnostic bound"
        );
        let mut bytes = vec![0; (end - first) as usize];
        self.file.seek(SeekFrom::Start(self.data + first))?;
        self.file.read_exact(&mut bytes)?;
        Ok(bytes)
    }
    fn floats(&mut self, name: &str) -> Result<Vec<f32>> {
        let bytes = self.bytes(name, "F32")?;
        ensure!(bytes.len().is_multiple_of(4), "incomplete F32 tensor");
        let values = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "non-finite saved tensor"
        );
        Ok(values)
    }
}
fn capture(path: &Path) -> Result<Artefact> {
    Artefact::capture(&std::env::current_dir()?, path)
}
fn change(before: &[f32], after: &[f32]) -> Result<Value> {
    ensure!(
        before.len() == after.len(),
        "saved adapter inventory changed"
    );
    Ok(
        json!({"values":after.len(),"changed_values":before.iter().zip(after).filter(|(a,b)|a!=b).count(),
        "maximum_absolute_value":after.iter().map(|x|f64::from(x.abs())).fold(0.,f64::max),
        "maximum_absolute_change":before.iter().zip(after).map(|(a,b)|f64::from((a-b).abs())).fold(0.,f64::max),
        "changed_l2":before.iter().zip(after).map(|(a,b)|(f64::from(*a)-f64::from(*b)).powi(2)).sum::<f64>().sqrt()}),
    )
}
fn main() -> Result<()> {
    let executable = std::env::current_exe()?.canonicalize()?;
    let pinned =
        lec_distil::engine::pin_executable(&executable, Path::new("state/engines/diagnostics"))?;
    if executable != pinned.path {
        return Err(Command::new(&pinned.path)
            .args(std::env::args_os().skip(1))
            .exec())
        .context("exec pinned movement diagnostic");
    }
    let arguments = std::env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let build_receipt = Path::new("state/diagnostics/bitnet-movement/build/receipt.json");
    if arguments
        .first()
        .is_some_and(|arg| arg == Path::new("--record-build"))
    {
        ensure!(
            !build_receipt.exists(),
            "retain original movement build receipt"
        );
        let source = Path::new("examples/bitnet_movement.rs");
        ensure!(
            artefact::file_hash(source)? == artefact::hash(include_bytes!("bitnet_movement.rs")),
            "diagnostic source changed after compilation"
        );
        let dependencies = [
            "target/release/deps/liblec_distil-a275046bc9fb4f1b.rlib",
            "target/release/deps/libanyhow-30a66fa3a3794c63.rlib",
            "target/release/deps/libserde_json-a2d822203098752a.rlib",
        ]
        .into_iter()
        .map(|path| capture(Path::new(path)))
        .collect::<Result<Vec<_>>>()?;
        let rustc = Command::new("rustc").arg("--version").output()?;
        ensure!(
            rustc.status.success(),
            "cannot record Rust compiler identity"
        );
        artefact::write_json(
            build_receipt,
            &json!({"schema_version":1,"evidence_kind":"standalone_rust_read_only_diagnostic_build",
            "source":capture(source)?,"dependencies":dependencies,"pinned_executable":pinned,"rustc":String::from_utf8(rustc.stdout)?,
            "compiler_arguments":["--edition=2024","-O","-C","lto=thin","-L","dependency=target/release/deps",
                "--extern","lec_distil=target/release/deps/liblec_distil-a275046bc9fb4f1b.rlib",
                "--extern","anyhow=target/release/deps/libanyhow-30a66fa3a3794c63.rlib",
                "--extern","serde_json=target/release/deps/libserde_json-a2d822203098752a.rlib",
                "examples/bitnet_movement.rs","-o","state/diagnostics/bitnet-movement/build/bitnet-movement"],
            "cargo_invoked":false,"training_performed":false}),
        )?;
        println!("{}", build_receipt.display());
        return Ok(());
    }
    ensure!(
        build_receipt.exists(),
        "record standalone diagnostic build before observation"
    );
    let directory = arguments
        .first()
        .cloned()
        .unwrap_or_else(|| "reports/local-study/native-bitnet-48-operator".into());
    let output = arguments
        .get(1)
        .cloned()
        .unwrap_or_else(|| "reports/local-study/native-bitnet-48-movement.json".into());
    ensure!(
        !output.exists(),
        "retain existing movement diagnostic and choose a new output"
    );
    let freeze: Value = serde_json::from_slice(&fs::read(directory.join("freeze.json"))?)?;
    ensure!(
        freeze["spec"]["initial_checkpoint"].is_null(),
        "this diagnostic currently measures base-initial rounds only"
    );
    let base = PathBuf::from(
        freeze["spec"]["student_base"]
            .as_str()
            .context("frozen base missing")?,
    );
    let rank = freeze["spec"]["rank"]
        .as_u64()
        .context("frozen rank missing")? as usize;
    let factor = freeze["spec"]["alpha"]
        .as_f64()
        .context("frozen alpha missing")?
        / rank as f64;
    let mut original = Tensors::open(&base.join("model.safetensors"))?;
    let mut initial = Tensors::open(&directory.join("initial-adapter.safetensors"))?;
    let mut final_adapter = Tensors::open(&directory.join("adapter.safetensors"))?;
    let mut optimiser = Tensors::open(&directory.join("optimizer.safetensors"))?;
    let names = final_adapter
        .header
        .as_object()
        .context("adapter header is not an object")?
        .keys()
        .filter_map(|name| name.strip_suffix(".qat_a").map(str::to_owned))
        .collect::<Vec<_>>();
    ensure!(!names.is_empty(), "no saved ternary latent adapters");
    let gains = [1., 2., 4., 8., 16., 32., 64., 128.];
    let mut gain_changes = [0u64; 8];
    let mut rows = Vec::new();
    let mut all_values = 0u64;
    let mut changed_latents = 0u64;
    let mut active_moments = 0u64;
    let mut maximum_normalised = 0f64;
    let mut closest = f64::INFINITY;
    for prefix in names {
        let aname = format!("{prefix}.qat_a");
        let bname = format!("{prefix}.qat_b");
        let columns = final_adapter.header[&aname]["shape"][1]
            .as_u64()
            .context("adapter columns missing")? as usize;
        let outputs = final_adapter.header[&bname]["shape"][0]
            .as_u64()
            .context("adapter outputs missing")? as usize;
        ensure!(outputs.is_multiple_of(4), "packed output shape differs");
        let a = final_adapter.floats(&aname)?;
        let b = final_adapter.floats(&bname)?;
        ensure!(
            a.len() == rank * columns && b.len() == outputs * rank,
            "low-rank dimensions changed"
        );
        let a_change = change(&initial.floats(&aname)?, &a)?;
        let b_change = change(&initial.floats(&bname)?, &b)?;
        changed_latents += a_change["changed_values"].as_u64().unwrap()
            + b_change["changed_values"].as_u64().unwrap();
        let mut moments = BTreeMap::new();
        for name in [&aname, &bname] {
            for suffix in ["m", "v"] {
                let values = optimiser.floats(&format!("{name}.{suffix}"))?;
                let active = values.iter().filter(|x| **x != 0.).count();
                active_moments += active as u64;
                moments.insert(format!("{name}.{suffix}"),json!({"values":values.len(),"nonzero_values":active,
                "maximum_absolute_value":values.iter().map(|x|f64::from(x.abs())).fold(0.,f64::max)}));
            }
        }
        let scale = original.bytes(&format!("{prefix}.weight_scale"), "BF16")?;
        ensure!(scale.len() == 2, "original scale is not scalar BF16");
        let scale = f64::from(f32::from_bits(
            u32::from(u16::from_le_bytes(scale.try_into().unwrap())) << 16,
        ));
        ensure!(scale.is_finite() && scale > 0., "original scale is invalid");
        let packed = original.bytes(&format!("{prefix}.weight"), "U8")?;
        ensure!(
            packed.len() == outputs * columns / 4,
            "original packed matrix dimensions changed"
        );
        let mut max_delta = 0f64;
        let mut min_margin = f64::INFINITY;
        let mut min_gain = f64::INFINITY;
        let mut moving = 0u64;
        for row in 0..outputs {
            for column in 0..columns {
                let delta = factor
                    * (0..rank)
                        .map(|k| f64::from(b[row * rank + k]) * f64::from(a[k * columns + column]))
                        .sum::<f64>();
                max_delta = max_delta.max(delta.abs());
                let packed_index = (row % (outputs / 4)) * columns + column;
                let trit = i32::from((packed[packed_index] >> ((row / (outputs / 4)) * 2)) & 3) - 1;
                ensure!((-1..=1).contains(&trit), "invalid original packed trit");
                let direction = if trit == 0 {
                    delta.abs()
                } else {
                    -f64::from(trit) * delta
                };
                if direction > 0. {
                    moving += 1;
                    let margin = 0.5 - direction / scale;
                    min_margin = min_margin.min(margin);
                    let gain = 0.5 * scale / direction;
                    min_gain = min_gain.min(gain);
                    for (index, g) in gains.iter().enumerate() {
                        if direction * g > 0.5 * scale
                            || (trit != 0 && direction * g == 0.5 * scale)
                        {
                            gain_changes[index] += 1;
                        }
                    }
                }
            }
        }
        all_values += (outputs * columns) as u64;
        maximum_normalised = maximum_normalised.max(max_delta / scale);
        closest = closest.min(min_gain);
        rows.push(json!({"projection":prefix,"shape":[outputs,columns],"official_scale":scale,"half_bin_width":scale*0.5,
            "adapter_a":a_change,"adapter_b":b_change,"adam_moments":moments,"maximum_absolute_low_rank_update":max_delta,
            "maximum_absolute_update_over_scale":max_delta/scale,"minimum_signed_threshold_margin_in_scale_units":if min_margin.is_finite(){Some(min_margin)}else{None},
            "minimum_saved_update_gain_to_first_switch":if min_gain.is_finite(){Some(min_gain)}else{None},"weights_moving_towards_switch":moving}));
    }
    let counterfactual = gains
        .iter()
        .zip(gain_changes)
        .map(|(gain, changes)| json!({"saved_update_gain":gain,"values_that_would_change":changes}))
        .collect::<Vec<_>>();
    let blocked: Value = serde_json::from_slice(&fs::read(
        directory.join("blocked-no-discrete-update.json"),
    )?)?;
    ensure!(
        blocked["effective_trits"]["final"]["values"] == all_values
            && blocked["effective_trits"]["final"]["changed_from_frozen_base"] == gain_changes[0],
        "reconstructed current update disagrees with the actual worker's effective lattice observation"
    );
    let report = json!({"schema_version":1,"evidence_kind":"actual_saved_bitnet_latent_threshold_movement_diagnostic",
        "frozen_protocol":capture(&directory.join("freeze.json"))?,"initial_adapter":capture(&directory.join("initial-adapter.safetensors"))?,
        "final_adapter":capture(&directory.join("adapter.safetensors"))?,"optimiser":capture(&directory.join("optimizer.safetensors"))?,
        "base_assets":capture(&base.join("assets.json"))?,"selected_base_header_sha256":original.header_sha256,
        "diagnostic_executable_sha256":artefact::file_hash(&std::env::current_exe()?)?,"build_receipt":capture(build_receipt)?,"rank":rank,"alpha_over_rank":factor,
        "adapted_values_examined":all_values,"changed_latent_values":changed_latents,"nonzero_adam_moment_values":active_moments,
        "maximum_absolute_update_over_scale":maximum_normalised,"minimum_saved_update_gain_to_first_switch":if closest.is_finite(){Some(closest)}else{None},
        "counterfactual_scaled_saved_update_only":counterfactual,"projections":rows,"production_eligible":false,
        "limitations":["Reads selected original packed q/v matrices and scales, not a full model allocation. Base publisher integrity is bound by the original frozen worker and retained assets manifest; this diagnostic does not rehash the full base.",
            "Low-rank products are reconstructed in f64 from saved f32 tensors; the actual worker's discrete lattice digest remains authoritative.",
            "Counterfactual gains scale the saved update for diagnosis only. They are not trained checkpoints, measured learning, or permission to alter a frozen protocol."]});
    artefact::write_json(&output, &report)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"output":output,"adapted_values":all_values,
        "maximum_absolute_update_over_scale":maximum_normalised,"minimum_saved_update_gain_to_first_switch":closest,"counterfactual_gains":counterfactual}))?
    );
    Ok(())
}
