//! A persistent I/O thread owns its Tokio runtime, so moving or dropping the
//! model on a different worker never drops a runtime inside an async executor.
use super::*;
use std::{sync::mpsc, thread::JoinHandle};
struct Job {
    url: String,
    body: Option<serde_json::Value>,
    deadline: Duration,
    reply: mpsc::SyncSender<std::result::Result<serde_json::Value, String>>,
}
pub(super) struct HttpDriver {
    sender: Option<mpsc::SyncSender<Job>>,
    thread: Option<JoinHandle<()>>,
}
impl HttpDriver {
    pub fn new() -> Result<Self> {
        let (sender, receiver) = mpsc::sync_channel::<Job>(1);
        let (ready, readiness) = mpsc::sync_channel(1);
        let thread = std::thread::Builder::new()
            .name("bitnet-native-http".into())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = ready.send(Err(e.to_string()));
                        return;
                    }
                };
                let client = match reqwest::Client::builder()
                    .no_proxy()
                    // This thread parks between requests instead of polling
                    // the runtime. A server can close a keep-alive connection
                    // during a long model load while its pooled socket still
                    // appears reusable. Fresh loopback connections avoid that
                    // stale POST failure without replaying generation requests.
                    .pool_max_idle_per_host(0)
                    .redirect(reqwest::redirect::Policy::none())
                    .connect_timeout(Duration::from_secs(3))
                    .build()
                {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = ready.send(Err(e.to_string()));
                        return;
                    }
                };
                let _ = ready.send(Ok(()));
                while let Ok(job) = receiver.recv() {
                    let result = runtime
                        .block_on(request(&client, job.url, job.body, job.deadline))
                        .map_err(|e| format!("{e:#}"));
                    let _ = job.reply.send(result);
                }
            })?;
        match readiness.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(())) => Ok(Self {
                sender: Some(sender),
                thread: Some(thread),
            }),
            Ok(Err(error)) => {
                let _ = thread.join();
                anyhow::bail!("native HTTP driver initialisation failed: {error}");
            }
            Err(error) => {
                drop(sender);
                let _ = thread.join();
                Err(error.into())
            }
        }
    }
    pub fn call(
        &self,
        url: String,
        body: Option<serde_json::Value>,
        deadline: Duration,
    ) -> Result<serde_json::Value> {
        let (reply, result) = mpsc::sync_channel(1);
        self.sender
            .as_ref()
            .context("native HTTP driver closed")?
            .send(Job {
                url,
                body,
                deadline,
                reply,
            })
            .context("native HTTP driver terminated")?;
        result
            .recv_timeout(deadline + Duration::from_secs(3))
            .context("native HTTP driver deadline exceeded")?
            .map_err(anyhow::Error::msg)
    }
}
impl Drop for HttpDriver {
    fn drop(&mut self) {
        drop(self.sender.take());
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            eprintln!("BitNet native HTTP driver panicked during cleanup");
        }
    }
}
async fn request(
    client: &reqwest::Client,
    url: String,
    body: Option<serde_json::Value>,
    deadline: Duration,
) -> Result<serde_json::Value> {
    let request = if let Some(body) = body {
        client.post(url).json(&body)
    } else {
        client.get(url)
    };
    let response = request
        .timeout(deadline)
        .send()
        .await
        .context("native loopback request failed")?;
    ensure!(
        response.status().is_success(),
        "native loopback HTTP{}",
        response.status()
    );
    ensure!(
        response.content_length().is_none_or(|n| n <= 4 << 20),
        "native response exceeds output bound"
    );
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        ensure!(
            bytes.len() + chunk.len() <= 4 << 20,
            "native response exceeds output bound"
        );
        bytes.extend_from_slice(&chunk);
    }
    Ok(serde_json::from_slice(&bytes)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader, Read, Write},
        net::TcpListener,
    };

    #[test]
    fn a_closed_idle_connection_cannot_break_the_next_generation_post() -> Result<()> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let url = format!("http://{}", listener.local_addr()?);
        let server = std::thread::spawn(move || -> Result<()> {
            for _ in 0..2 {
                let (stream, _) = listener.accept()?;
                stream.set_read_timeout(Some(Duration::from_secs(3)))?;
                stream.set_write_timeout(Some(Duration::from_secs(3)))?;
                let mut request = BufReader::new(stream);
                let mut length = 0;
                loop {
                    let mut line = String::new();
                    ensure!(
                        request.read_line(&mut line)? > 0,
                        "request headers ended early"
                    );
                    if line == "\r\n" {
                        break;
                    }
                    if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        length = value.trim().parse::<usize>()?;
                    }
                }
                ensure!(length <= 1024, "controlled test request exceeded bound");
                request.read_exact(&mut vec![0; length])?;
                request.get_mut().write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\n{}",
                )?;
                // Close a nominally persistent socket while the HTTP runtime
                // is parked, matching the observed native-server idle timeout.
                std::thread::sleep(Duration::from_millis(100));
            }
            Ok(())
        });
        let driver = HttpDriver::new()?;
        driver.call(format!("{url}/health"), None, Duration::from_secs(3))?;
        std::thread::sleep(Duration::from_millis(300));
        let response = driver.call(
            format!("{url}/completion"),
            Some(serde_json::json!({})),
            Duration::from_secs(3),
        )?;
        assert_eq!(response, serde_json::json!({}));
        server
            .join()
            .map_err(|_| anyhow::anyhow!("controlled HTTP server panicked"))??;
        Ok(())
    }
}
