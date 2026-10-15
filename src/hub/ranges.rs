//! Four bounded conditional HTTP ranges, retaining a pre-existing prefix.
//!
//! Segment files remain separate until complete. Interrupted assembly resumes
//! from its exact prefix; the publisher's whole-file hash still gates publication.

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncReadExt, AsyncSeekExt};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Segment {
    start: u64,
    end: u64,
    filename: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct RangePlan {
    schema_version: u32,
    canonical_url: String,
    asset: PlannedFile,
    prefix_bytes: u64,
    etag: String,
    segments: Vec<Segment>,
}

pub(super) struct RangeDownload<'a> {
    pub client: &'a Client,
    pub bearer: Option<&'a Bearer>,
    pub destination: &'a Path,
    pub url: &'a str,
    pub partial: &'a Path,
    pub planned: &'a PlannedFile,
    pub test_http: bool,
}

fn range_directory(destination: &Path, name: &str) -> PathBuf {
    destination.join(format!(".{name}.ranges"))
}

fn regular_unlinked(path: &Path) -> Result<Option<fs::Metadata>> {
    ensure_safe_path(path)?;
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(metadata.is_file(), "range artefact is not a regular file");
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                ensure!(metadata.nlink() == 1, "hard-linked range artefact rejected");
            }
            Ok(Some(metadata))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).context("inspect resumable range artefact"),
    }
}

fn validate_response(
    response: &Response,
    start: u64,
    end: u64,
    total: u64,
    expected_etag: Option<&str>,
) -> Result<String> {
    if response.status() != StatusCode::PARTIAL_CONTENT {
        if !response.status().is_success() {
            return Err(HubRequestFailure::Http {
                status: response.status().as_u16(),
            }
            .into());
        }
        bail!("range request requires 206; full or unexpected response rejected");
    }
    let range = response
        .headers()
        .get(header::CONTENT_RANGE)
        .context("range response lacks Content-Range")?
        .to_str()?;
    ensure!(
        range == format!("bytes {start}-{end}/{total}"),
        "range response boundaries or pinned size changed"
    );
    if let Some(length) = response.content_length() {
        ensure!(
            length == end - start + 1,
            "range Content-Length differs from exact requested bytes"
        );
    }
    let etag = response
        .headers()
        .get(header::ETAG)
        .context("range response lacks a stable ETag")?
        .to_str()?;
    ensure!(
        !etag.starts_with("W/")
            && etag.len() <= 256
            && etag.starts_with('"')
            && etag.ends_with('"'),
        "range response requires a bounded strong ETag"
    );
    if let Some(expected) = expected_etag {
        ensure!(
            etag == expected,
            "range ETag changed within immutable download"
        );
    }
    Ok(etag.to_owned())
}

fn segments(prefix: u64, total: u64) -> Vec<Segment> {
    let chunk = (total - prefix).div_ceil(4);
    let mut output = Vec::new();
    let mut start = prefix;
    while start < total {
        let end = (start + chunk - 1).min(total - 1);
        output.push(Segment {
            start,
            end,
            filename: format!("segment-{}.part", output.len()),
        });
        start = end + 1;
    }
    output
}

fn load_plan(options: &RangeDownload<'_>, folder: &Path) -> Result<Option<RangePlan>> {
    ensure_safe_path(folder)?;
    let path = folder.join("plan.json");
    if regular_unlinked(&path)?.is_none() {
        ensure!(
            !folder.exists() || fs::read_dir(folder)?.next().is_none(),
            "untracked range directory rejected"
        );
        return Ok(None);
    }
    ensure!(
        fs::metadata(&path)?.len() <= 64 * 1024,
        "range plan oversized"
    );
    let plan: RangePlan = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(
        plan.schema_version == 1
            && plan.asset == *options.planned
            && plan.canonical_url == options.url
            && plan.prefix_bytes < plan.asset.bytes
            && plan.segments == segments(plan.prefix_bytes, plan.asset.bytes),
        "range resume differs from immutable publisher plan"
    );
    ensure!(
        !plan.etag.starts_with("W/")
            && plan.etag.len() <= 256
            && plan.etag.starts_with('"')
            && plan.etag.ends_with('"'),
        "invalid saved range ETag"
    );
    Ok(Some(plan))
}

async fn probe(options: &RangeDownload<'_>, etag: Option<&str>) -> Result<String> {
    for attempt in 0..3 {
        let result = async {
            let response = request_with_range(
                options.client,
                Url::parse(options.url)?,
                options.bearer,
                Some(HttpRange {
                    start: 0,
                    end: Some(0),
                    if_range: etag.map(str::to_owned),
                }),
                options.test_http,
            )
            .await?;
            let observed = validate_response(&response, 0, 0, options.planned.bytes, etag)?;
            ensure!(
                bounded_body(response, 1).await?.len() == 1,
                "range identity probe returned no byte"
            );
            Ok::<_, anyhow::Error>(observed)
        }
        .await;
        match result {
            Ok(observed) => return Ok(observed),
            Err(error) if attempt < 2 && is_connectivity_error(&error) => {
                tokio::time::sleep(Duration::from_secs(1 << attempt)).await
            }
            Err(error) => return Err(error).context("conditional range identity probe failed"),
        }
    }
    bail!("range identity probe exhausted its bounded attempts")
}

async fn segment_download(
    options: &RangeDownload<'_>,
    folder: &Path,
    segment: Segment,
    etag: &str,
    progress: Arc<AtomicU64>,
) -> Result<()> {
    let path = folder.join(&segment.filename);
    let size = segment.end - segment.start + 1;
    for attempt in 0..3 {
        let have = regular_unlinked(&path)?.map(|m| m.len()).unwrap_or(0);
        ensure!(have <= size, "range partial exceeds its pinned segment");
        progress.store(have, Ordering::Relaxed);
        if have == size {
            return Ok(());
        }
        let result = async {
            let first = segment.start + have;
            let mut response = request_with_range(
                options.client,
                Url::parse(options.url)?,
                options.bearer,
                Some(HttpRange {
                    start: first,
                    end: Some(segment.end),
                    if_range: Some(etag.into()),
                }),
                options.test_http,
            )
            .await?;
            validate_response(
                &response,
                first,
                segment.end,
                options.planned.bytes,
                Some(etag),
            )?;
            let mut open = tokio::fs::OpenOptions::new();
            open.write(true).append(true).create(true);
            #[cfg(unix)]
            open.custom_flags(libc::O_NOFOLLOW);
            let mut file = open.open(&path).await?;
            let mut received = have;
            while let Some(chunk) =
                response
                    .chunk()
                    .await
                    .map_err(|_| HubRequestFailure::Connectivity {
                        stage: "range body",
                    })?
            {
                received = received
                    .checked_add(chunk.len() as u64)
                    .context("range byte counter overflow")?;
                ensure!(received <= size, "range body exceeds pinned segment");
                file.write_all(&chunk).await?;
                progress.store(received, Ordering::Relaxed);
            }
            file.sync_all().await?;
            ensure!(received == size, "range body incomplete");
            Ok::<_, anyhow::Error>(())
        }
        .await;
        match result {
            Ok(()) => return Ok(()),
            Err(error) if attempt < 2 && is_connectivity_error(&error) => {
                tokio::time::sleep(Duration::from_secs(1 << attempt)).await
            }
            Err(error) => return Err(error).context("bounded range transfer failed"),
        }
    }
    bail!("range transfer exhausted its bounded attempts")
}

async fn assemble(options: &RangeDownload<'_>, folder: &Path, plan: &RangePlan) -> Result<()> {
    let have = regular_unlinked(options.partial)?
        .map(|m| m.len())
        .unwrap_or(0);
    ensure!(
        (plan.prefix_bytes..=plan.asset.bytes).contains(&have),
        "original range prefix was truncated or replaced"
    );
    let meter = FileProgress::new(&plan.asset.filename, "assembling", plan.asset.bytes, have);
    meter.report(0);
    let mut last_report = Instant::now();
    let mut open = tokio::fs::OpenOptions::new();
    open.append(true).write(true).create(true);
    #[cfg(unix)]
    open.custom_flags(libc::O_NOFOLLOW);
    let mut output = open.open(options.partial).await?;
    let mut present = have;
    let mut buffer = vec![0_u8; 1024 * 1024];
    for segment in &plan.segments {
        if present > segment.end {
            continue;
        }
        ensure!(present >= segment.start, "gap in range prefix assembly");
        let path = folder.join(&segment.filename);
        ensure!(
            regular_unlinked(&path)?
                .context("range segment missing during assembly")?
                .len()
                == segment.end - segment.start + 1,
            "range segment incomplete during assembly"
        );
        let mut input = tokio::fs::File::open(path).await?;
        input
            .seek(std::io::SeekFrom::Start(present - segment.start))
            .await?;
        while present <= segment.end {
            let maximum = buffer.len().min((segment.end - present + 1) as usize);
            let count = input.read(&mut buffer[..maximum]).await?;
            ensure!(count > 0, "range segment ended during assembly");
            output.write_all(&buffer[..count]).await?;
            present += count as u64;
            if last_report.elapsed() >= Duration::from_secs(15) {
                meter.report(present - have);
                last_report = Instant::now();
            }
        }
    }
    output.sync_all().await?;
    ensure!(
        present == plan.asset.bytes,
        "range assembly did not fill the pinned file"
    );
    meter.report(present - have);
    Ok(())
}

pub(super) async fn download(options: RangeDownload<'_>) -> Result<()> {
    let folder = range_directory(options.destination, &options.planned.filename);
    let partial_size = regular_unlinked(options.partial)?
        .map(|m| m.len())
        .unwrap_or(0);
    ensure!(
        partial_size <= options.planned.bytes,
        "original prefix exceeds pinned object"
    );
    if partial_size == options.planned.bytes {
        return Ok(());
    }
    let existing = load_plan(&options, &folder)?;
    let etag = probe(&options, existing.as_ref().map(|p| p.etag.as_str())).await?;
    let plan = match existing {
        Some(plan) => {
            ensure!(
                partial_size >= plan.prefix_bytes,
                "immutable original prefix has been truncated"
            );
            plan
        }
        None => {
            let plan = RangePlan {
                schema_version: 1,
                canonical_url: options.url.into(),
                asset: options.planned.clone(),
                prefix_bytes: partial_size,
                etag,
                segments: segments(partial_size, options.planned.bytes),
            };
            fs::create_dir_all(&folder)?;
            atomic_publish(
                &folder.join("plan.json"),
                &serde_json::to_vec_pretty(&plan)?,
            )?;
            plan
        }
    };
    let progress: Vec<_> = plan
        .segments
        .iter()
        .map(|segment| {
            regular_unlinked(&folder.join(&segment.filename))
                .map(|m| Arc::new(AtomicU64::new(m.map(|m| m.len()).unwrap_or(0))))
        })
        .collect::<Result<_>>()?;
    let initial_received: u64 = progress.iter().map(|p| p.load(Ordering::Relaxed)).sum();
    let meter = FileProgress::new(
        &options.planned.filename,
        "receiving four ranges",
        options.planned.bytes,
        plan.prefix_bytes + initial_received,
    );
    meter.report(0);
    let mut jobs = stream::iter(plan.segments.iter().cloned().zip(progress.iter().cloned()))
        .map(|(segment, progress)| {
            segment_download(&options, &folder, segment, &plan.etag, progress)
        })
        .buffer_unordered(4);
    let mut heartbeat = tokio::time::interval_at(
        tokio::time::Instant::now() + Duration::from_secs(15),
        Duration::from_secs(15),
    );
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            result = jobs.next() => match result {Some(result) => result?, None => break},
            _ = heartbeat.tick() => meter.report(progress.iter().map(|p|p.load(Ordering::Relaxed)).sum::<u64>().saturating_sub(initial_received)),
        }
    }
    drop(jobs);
    meter.report(
        progress
            .iter()
            .map(|p| p.load(Ordering::Relaxed))
            .sum::<u64>()
            .saturating_sub(initial_received),
    );
    assemble(&options, &folder, &plan).await
}

/// Called only after a complete whole-file publisher hash has been verified.
pub(super) fn cleanup(destination: &Path, asset: &PlannedFile) -> Result<()> {
    let folder = range_directory(destination, &asset.filename);
    ensure_safe_path(&folder)?;
    if !folder.exists() {
        return Ok(());
    }
    let plan: RangePlan = serde_json::from_slice(&fs::read(folder.join("plan.json"))?)?;
    ensure!(
        plan.schema_version == 1
            && plan.asset == *asset
            && plan.prefix_bytes < asset.bytes
            && plan.segments == segments(plan.prefix_bytes, asset.bytes),
        "cannot clean a mismatched range plan"
    );
    let allowed: Vec<_> = std::iter::once("plan.json".to_owned())
        .chain(plan.segments.iter().map(|s| s.filename.clone()))
        .collect();
    for entry in fs::read_dir(&folder)? {
        let entry = entry?;
        ensure!(
            allowed
                .iter()
                .any(|name| entry.file_name() == name.as_str()),
            "unknown file in range directory; preserving it"
        );
        regular_unlinked(&entry.path())?.context("range cleanup file missing")?;
    }
    for entry in fs::read_dir(&folder)? {
        fs::remove_file(entry?.path())?;
    }
    fs::remove_dir(folder)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncReadExt;

    async fn server(
        bytes: &'static [u8],
        requests: usize,
        faulty: bool,
    ) -> Result<(String, tokio::task::JoinHandle<Result<()>>)> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let task = tokio::spawn(async move {
            for _ in 0..requests {
                let (mut socket, _) = listener.accept().await?;
                let mut request = Vec::new();
                while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                    let mut buffer = [0_u8; 1024];
                    let count = socket.read(&mut buffer).await?;
                    ensure!(
                        count > 0 && request.len() < 8192,
                        "invalid range test request"
                    );
                    request.extend_from_slice(&buffer[..count]);
                }
                let request = String::from_utf8(request)?.to_ascii_lowercase();
                let header = request
                    .lines()
                    .find_map(|line| line.strip_prefix("range: bytes="))
                    .context("range header missing")?;
                let (start, end) = header.split_once('-').context("range bounds missing")?;
                let start: usize = start.parse()?;
                let end: usize = end.parse()?;
                if faulty {
                    socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", bytes.len()).as_bytes()).await?;
                    socket.write_all(bytes).await?;
                } else {
                    if start != 0 || end != 0 {
                        ensure!(
                            request.contains("if-range: \"stable-object\""),
                            "segment lacks conditional object identity"
                        );
                    }
                    socket.write_all(format!("HTTP/1.1 206 Partial Content\r\nETag: \"stable-object\"\r\nContent-Range: bytes {start}-{end}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", bytes.len(), end-start+1).as_bytes()).await?;
                    socket.write_all(&bytes[start..=end]).await?;
                }
            }
            Ok(())
        });
        Ok((format!("http://{address}/model.gguf"), task))
    }

    #[tokio::test]
    async fn four_ranges_preserve_existing_prefix_and_assembly_resume() -> Result<()> {
        static BYTES: &[u8] = b"an immutable model object with an existing prefix";
        let dir = tempfile::tempdir()?;
        let partial = dir.path().join(".model.gguf.partial");
        fs::write(&partial, &BYTES[..7])?;
        let planned = PlannedFile {
            filename: "model.gguf".into(),
            bytes: BYTES.len() as u64,
            publisher_identity: PublisherIdentity::LfsSha256(hex::encode(Sha256::digest(BYTES))),
        };
        let (url, task) = server(BYTES, 5, false).await?;
        let client = client()?;
        download(RangeDownload {
            client: &client,
            bearer: None,
            destination: dir.path(),
            url: &url,
            partial: &partial,
            planned: &planned,
            test_http: true,
        })
        .await?;
        task.await??;
        assert_eq!(fs::read(&partial)?, BYTES);
        verify_file(&partial, &planned)?;
        let folder = range_directory(dir.path(), &planned.filename);
        let plan: RangePlan = serde_json::from_slice(&fs::read(folder.join("plan.json"))?)?;
        fs::write(&partial, &BYTES[..13])?;
        let options = RangeDownload {
            client: &client,
            bearer: None,
            destination: dir.path(),
            url: &url,
            partial: &partial,
            planned: &planned,
            test_http: true,
        };
        assemble(&options, &folder, &plan).await?;
        assert_eq!(fs::read(&partial)?, BYTES);
        cleanup(dir.path(), &planned)?;
        assert!(!folder.exists());
        Ok(())
    }

    #[tokio::test]
    async fn unexpected_full_response_never_appends_or_discards_prefix() -> Result<()> {
        static BYTES: &[u8] = b"the immutable source object";
        let dir = tempfile::tempdir()?;
        let partial = dir.path().join(".model.gguf.partial");
        fs::write(&partial, b"the ")?;
        let planned = PlannedFile {
            filename: "model.gguf".into(),
            bytes: BYTES.len() as u64,
            publisher_identity: PublisherIdentity::LfsSha256(hex::encode(Sha256::digest(BYTES))),
        };
        let (url, task) = server(BYTES, 1, true).await?;
        assert!(
            download(RangeDownload {
                client: &client()?,
                bearer: None,
                destination: dir.path(),
                url: &url,
                partial: &partial,
                planned: &planned,
                test_http: true
            })
            .await
            .is_err()
        );
        task.await??;
        assert_eq!(fs::read(partial)?, b"the ");
        Ok(())
    }

    #[test]
    fn saved_identity_and_link_hazards_are_rejected() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let planned = PlannedFile {
            filename: "model.gguf".into(),
            bytes: 100,
            publisher_identity: PublisherIdentity::LfsSha256("a".repeat(64)),
        };
        let folder = range_directory(dir.path(), &planned.filename);
        fs::create_dir(&folder)?;
        let plan = RangePlan {
            schema_version: 1,
            canonical_url: "https://huggingface.co/organisation/model/resolve/commit/model.gguf"
                .into(),
            asset: planned.clone(),
            prefix_bytes: 10,
            etag: "\"object\"".into(),
            segments: segments(10, 100),
        };
        fs::write(folder.join("plan.json"), serde_json::to_vec(&plan)?)?;
        let options = RangeDownload {
            client: &client()?,
            bearer: None,
            destination: dir.path(),
            url: "https://huggingface.co/different/model",
            partial: &dir.path().join(".partial"),
            planned: &planned,
            test_http: false,
        };
        assert!(load_plan(&options, &folder).is_err());
        #[cfg(unix)]
        {
            let real = dir.path().join("real");
            fs::write(&real, b"bytes")?;
            let link = dir.path().join("link");
            std::os::unix::fs::symlink(&real, &link)?;
            assert!(regular_unlinked(&link).is_err());
            fs::remove_file(&link)?;
            fs::hard_link(&real, &link)?;
            assert!(regular_unlinked(&link).is_err());
        }
        Ok(())
    }

    #[tokio::test]
    async fn saved_partial_segments_resume_and_changed_etag_cannot_mix_objects() -> Result<()> {
        static BYTES: &[u8] = b"immutable range test bytes with previously received pieces";
        let dir = tempfile::tempdir()?;
        let partial = dir.path().join(".model.gguf.partial");
        fs::write(&partial, &BYTES[..7])?;
        let planned = PlannedFile {
            filename: "model.gguf".into(),
            bytes: BYTES.len() as u64,
            publisher_identity: PublisherIdentity::LfsSha256(hex::encode(Sha256::digest(BYTES))),
        };
        let (url, task) = server(BYTES, 5, false).await?;
        let folder = range_directory(dir.path(), &planned.filename);
        fs::create_dir(&folder)?;
        let plan = RangePlan {
            schema_version: 1,
            canonical_url: url.clone(),
            asset: planned.clone(),
            prefix_bytes: 7,
            etag: "\"stable-object\"".into(),
            segments: segments(7, planned.bytes),
        };
        fs::write(folder.join("plan.json"), serde_json::to_vec(&plan)?)?;
        let first = &plan.segments[0];
        fs::write(
            folder.join(&first.filename),
            &BYTES[first.start as usize..first.start as usize + 3],
        )?;
        download(RangeDownload {
            client: &client()?,
            bearer: None,
            destination: dir.path(),
            url: &url,
            partial: &partial,
            planned: &planned,
            test_http: true,
        })
        .await?;
        task.await??;
        assert_eq!(fs::read(&partial)?, BYTES);
        verify_file(&partial, &planned)?;
        let dir = tempfile::tempdir()?;
        let partial = dir.path().join(".model.gguf.partial");
        fs::write(&partial, &BYTES[..7])?;
        let (url, task) = server(BYTES, 1, false).await?;
        let folder = range_directory(dir.path(), &planned.filename);
        fs::create_dir(&folder)?;
        let plan = RangePlan {
            canonical_url: url.clone(),
            etag: "\"different-object\"".into(),
            ..plan
        };
        fs::write(folder.join("plan.json"), serde_json::to_vec(&plan)?)?;
        assert!(
            download(RangeDownload {
                client: &client()?,
                bearer: None,
                destination: dir.path(),
                url: &url,
                partial: &partial,
                planned: &planned,
                test_http: true
            })
            .await
            .is_err()
        );
        task.await??;
        assert_eq!(fs::read(partial)?, &BYTES[..7]);
        Ok(())
    }

    #[tokio::test]
    async fn response_requires_exact_boundary_total_and_strong_identity() -> Result<()> {
        static BYTES: &[u8] = b"pinned bytes";
        let (url, task) = server(BYTES, 1, false).await?;
        let response = request_with_range(
            &client()?,
            Url::parse(&url)?,
            None,
            Some(HttpRange {
                start: 0,
                end: Some(0),
                if_range: None,
            }),
            true,
        )
        .await?;
        validate_response(&response, 0, 0, BYTES.len() as u64, None)?;
        assert!(validate_response(&response, 1, 1, BYTES.len() as u64, None).is_err());
        assert!(validate_response(&response, 0, 0, BYTES.len() as u64 + 1, None).is_err());
        assert!(
            validate_response(
                &response,
                0,
                0,
                BYTES.len() as u64,
                Some("\"another-object\"")
            )
            .is_err()
        );
        task.await??;
        Ok(())
    }
}
