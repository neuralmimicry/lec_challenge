# Captioned evidence walkthrough

`reports/evidence-walkthrough.mp4` is a two-minute, captioned walkthrough generated from the actual design and local QA reports. It states the unfinished assessment evidence explicitly. It is a rendered evidence presentation, not a claimed screen recording, narration, trained-student demonstration or proof of learning.

The Rust example `examples/evidence_video.rs` reads the base inference, 512-token synthetic tensor QA, real endpoint, drill, evaluation and cost reports, wraps the captions, invokes the media encoder and checks the finished duration with `ffprobe`. The JSON manifest records input hashes, the exact slide text and the video SHA-256. Regenerate after the final reports change:

```sh
cargo run --release --example evidence_video
```

The installed `ffmpeg`/`ffprobe` executables and DejaVu Sans font are explicit media prerequisites. The media encoder is a justified non-Rust tool dependency; application logic, evidence selection, formatting and manifest generation remain Rust. Encoding a standard H.264 MP4 with a mature codec avoids introducing a separate codec implementation into the assessment.

The slides cover the design ceiling, source separation, durable budgets, sandbox/contamination evidence, genuine base inference and synthetic-gradient verification, the real bounded endpoint, gate statistics, outstanding evidence and reproduction. Actual provider recordings and a provider-trained candidate must exist before a later video can claim to show those results. A numerical NO SHIP is a valid assessment outcome; missing evidence is an incomplete assessment.
