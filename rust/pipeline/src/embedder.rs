//! llama.cpp / llama-swap OpenAI-compatible embedding client — parity with
//! processors/llama_cpp_embedder.py, with server-side batching upgrade.
//!
//! - base_url normalization: strip trailing /v1 (stored without it);
//!   embeddings at {base}/v1/embeddings, probe GET {base}/props -> n_ctx.
//! - new(): probe embedding_dim (single text embed), max_input_length via
//!   /props n_ctx (fallback 512), supports_batch_input via 2-string input
//!   array probe.
//! - embed(): prepend prefix to each text, return L2-normalized Vec<f32>.
//! - Retry 4 attempts total (backoff 2s/5s/10s) on >=500/transport errors;
//!   4xx fails immediately with model name + llama-swap hint.
//! - Batch mode: one request for the whole batch. Single mode: one request
//!   per text, rayon across self.concurrency, order preserved.

use serde_json::{json, Value};
use std::time::Duration;

pub struct RemoteEmbedder {
    pub model_name: String,
    pub base_url: String, // no trailing /v1
    pub api_key: Option<String>,
    pub concurrency: usize,
    pub embedding_dim: usize,
    pub max_input_length: usize,
    pub supports_batch_input: bool,
    client: reqwest::blocking::Client,
}

pub(crate) fn normalize_base(base_url: &str) -> String {
    let mut b = base_url.trim_end_matches('/').to_string();
    if b.ends_with("/v1") {
        b.truncate(b.len() - 3);
        b = b.trim_end_matches('/').to_string();
    }
    b
}

fn normalize_vector(v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.into_iter().map(|x| x / norm).collect()
    } else {
        v
    }
}

impl RemoteEmbedder {
    pub fn new(
        model_name: &str,
        base_url: &str,
        api_key: Option<String>,
        concurrency: usize,
    ) -> Result<RemoteEmbedder, String> {
        let mut e = RemoteEmbedder {
            model_name: model_name.to_string(),
            base_url: normalize_base(base_url),
            api_key,
            concurrency: concurrency.max(1),
            embedding_dim: 0,
            max_input_length: 512,
            supports_batch_input: false,
            client: reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .map_err(|e| e.to_string())?,
        };
        e.embedding_dim = match e.embed_raw(json!("dimension probe")) {
            Ok(v) if !v.is_empty() => v[0].len(),
            Ok(_) => return Err("embedding server returned no vectors".to_string()),
            Err(err) => return Err(err),
        };
        if let Ok(resp) = e.client.get(format!("{}/props", e.base_url)).send() {
            if let Ok(body) = resp.json::<Value>() {
                if let Some(n) = body.get("n_ctx").and_then(|v| v.as_u64()) {
                    if n > 0 {
                        e.max_input_length = n as usize;
                    }
                }
            }
        }
        e.supports_batch_input = matches!(
            e.embed_raw(json!(["probe a", "probe b"])),
            Ok(v) if v.len() == 2
        );
        Ok(e)
    }

    fn post_with_retry(&self, payload: Value) -> Result<Value, String> {
        let mut backoffs = [2u64, 5, 10].into_iter();
        let mut last_err = String::new();
        for _attempt in 0..4 {
            let mut req = self
                .client
                .post(format!("{}/v1/embeddings", self.base_url))
                .json(&payload);
            if let Some(key) = &self.api_key {
                req = req.header("Authorization", format!("Bearer {key}"));
            }
            match req.send() {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        return resp
                            .json()
                            .map_err(|e| format!("bad JSON from embeddings: {e}"));
                    }
                    let body = resp.text().unwrap_or_default();
                    if status.as_u16() >= 500 {
                        last_err = format!("status {status}: {}", &body[..body.len().min(200)]);
                    } else {
                        return Err(format!(
                            "embeddings request failed (status {status}) for model '{}' at {}: {} \
                             (with llama-swap, the model id must exactly match a config section name)",
                            self.model_name, self.base_url,
                            &body[..body.len().min(200)]
                        ));
                    }
                }
                Err(err) => last_err = format!("transport error: {err}"),
            }
            if let Some(secs) = backoffs.next() {
                std::thread::sleep(Duration::from_secs(secs));
            }
        }
        Err(format!(
            "embeddings request failed after 4 attempts for model '{}': {last_err}",
            self.model_name
        ))
    }

    fn embed_raw(&self, input: Value) -> Result<Vec<Vec<f32>>, String> {
        let body = self.post_with_retry(json!({
            "model": self.model_name,
            "input": input,
            "encoding_format": "float",
        }))?;
        let data = body
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| format!("no data array in embeddings response: {body}"))?;
        let mut out = Vec::with_capacity(data.len());
        for item in data {
            let emb = item
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| "no embedding array in response item".to_string())?;
            out.push(
                emb.iter()
                    .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                    .collect(),
            );
        }
        Ok(out)
    }

    /// L2-normalized f32 embeddings, same order as inputs.
    pub fn embed(&self, texts: &[String], prefix: &str) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let prefixed: Vec<String> = if prefix.is_empty() {
            texts.to_vec()
        } else {
            texts.iter().map(|t| format!("{prefix}{t}")).collect()
        };
        if self.supports_batch_input {
            let raw = self.embed_raw(json!(prefixed))?;
            if raw.len() != prefixed.len() {
                return Err(format!(
                    "batch embed returned {} vectors for {} inputs",
                    raw.len(),
                    prefixed.len()
                ));
            }
            Ok(raw.into_iter().map(normalize_vector).collect())
        } else {
            use rayon::prelude::*;
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(self.concurrency)
                .build()
                .map_err(|e| e.to_string())?;
            let results: Vec<Result<Vec<f32>, String>> = pool.install(|| {
                prefixed
                    .par_iter()
                    .map(|t| {
                        self.embed_raw(json!(t))
                            .map(|mut v| normalize_vector(v.remove(0)))
                    })
                    .collect()
            });
            results.into_iter().collect()
        }
    }
}

#[cfg(test)]
pub(crate) mod mock {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;

    pub struct MockServer {
        pub port: u16,
    }

    /// Handler receives (method, path, body, hit_count) -> (status, json_body).
    pub fn spawn(
        handler: impl Fn(&str, &str, &str, usize) -> (u16, String) + Send + Sync + 'static,
    ) -> MockServer {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let hits = Arc::new(AtomicUsize::new(0));
        let hits2 = hits.clone();
        thread::spawn(move || {
            let handler = std::sync::Arc::new(handler);
            for stream in listener.incoming() {
                let mut stream = match stream {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let handler = handler.clone();
                let hits = hits2.clone();
                thread::spawn(move || {
                    let mut buf = [0u8; 65536];
                    let mut data = Vec::new();
                    loop {
                        match stream.read(&mut buf) {
                            Ok(0) => break,
                            Ok(n) => {
                                data.extend_from_slice(&buf[..n]);
                                if let Some(pos) = find_headers_end(&data) {
                                    let header = String::from_utf8_lossy(&data[..pos]).to_string();
                                    let clen: usize = header
                                        .lines()
                                        .find(|l| l.to_lowercase().starts_with("content-length:"))
                                        .and_then(|l| l.split(':').nth(1))
                                        .and_then(|v| v.trim().parse().ok())
                                        .unwrap_or(0);
                                    while data.len() < pos + clen {
                                        if let Ok(m) = stream.read(&mut buf) {
                                            if m == 0 {
                                                break;
                                            }
                                            data.extend_from_slice(&buf[..m]);
                                        } else {
                                            break;
                                        }
                                    }
                                    let body = String::from_utf8_lossy(
                                        &data[pos..(pos + clen).min(data.len())],
                                    )
                                    .to_string();
                                    let head_line = header.lines().next().unwrap_or("").to_string();
                                    let mut it = head_line.split_whitespace();
                                    let method = it.next().unwrap_or("").to_string();
                                    let path = it.next().unwrap_or("").to_string();
                                    let n = hits.fetch_add(1, Ordering::SeqCst) + 1;
                                    let (status, json_body) = handler(&method, &path, &body, n);
                                    let resp = format!(
                                        "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                                        status,
                                        json_body.len(),
                                        json_body
                                    );
                                    let _ = stream.write_all(resp.as_bytes());
                                    break;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                });
            }
        });
        MockServer { port }
    }

    fn find_headers_end(data: &[u8]) -> Option<usize> {
        data.windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|p| p + 4)
    }

    pub fn embeddings_json(vectors: &[&[f64]]) -> String {
        let items: Vec<String> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                format!(
                    "{{\"index\":{i},\"embedding\":[{}]}}",
                    v.iter()
                        .map(|x| format!("{x}"))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .collect();
        format!("{{\"data\":[{}]}}", items.join(","))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim_props_batch_probes_and_batch_embed() {
        let srv = mock::spawn(|method, path, body, _n| {
            if method == "GET" && path == "/props" {
                return (200, r#"{"n_ctx": 2048}"#.to_string());
            }
            // probe 1: single "dimension probe"; probe 2: 2-array batch probe;
            // later embed calls: array of N
            let v: serde_json::Value = serde_json::from_str(body).unwrap();
            match v["input"].as_array().map(|a| a.len()).unwrap_or(1) {
                1 => (200, mock::embeddings_json(&[&[1.0, 2.0, 2.0, 2.0][..]])),
                2 => (
                    200,
                    mock::embeddings_json(&[&[3.0, 4.0, 0.0, 0.0][..], &[0.0, 0.0, 5.0, 12.0][..]]),
                ),
                n => (
                    200,
                    mock::embeddings_json(&vec![&[3.0, 4.0, 0.0, 0.0][..]; n]),
                ),
            }
        });
        let e = RemoteEmbedder::new(
            "QwenEmbedding",
            &format!("http://127.0.0.1:{}/v1", srv.port),
            None,
            2,
        )
        .unwrap();
        assert_eq!(e.embedding_dim, 4);
        assert_eq!(e.max_input_length, 2048);
        assert!(e.supports_batch_input);
        assert_eq!(e.base_url, format!("http://127.0.0.1:{}", srv.port));
        let out = e
            .embed(&["hello".to_string(), "world".to_string()], "P: ")
            .unwrap();
        assert_eq!(out.len(), 2);
        let norm = out[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        assert!((out[0][0] - 0.6).abs() < 1e-5); // 3/5
    }

    #[test]
    fn batch_probe_falls_back_to_single_mode() {
        let srv = mock::spawn(|method, path, body, _n| {
            if method == "GET" && path == "/props" {
                return (200, "{}".to_string());
            }
            let v: serde_json::Value = serde_json::from_str(body).unwrap();
            if v["input"].is_array() {
                (400, r#"{"error":"single input only"}"#.to_string())
            } else {
                (200, mock::embeddings_json(&[&[1.0, 0.0][..]]))
            }
        });
        let e =
            RemoteEmbedder::new("M", &format!("http://127.0.0.1:{}", srv.port), None, 2).unwrap();
        assert!(!e.supports_batch_input);
        assert_eq!(e.max_input_length, 512); // fallback
        let out = e
            .embed(&["a".to_string(), "b".to_string(), "c".to_string()], "")
            .unwrap();
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn prefix_applied_in_single_mode() {
        use std::sync::Mutex;
        static CAPTURED: std::sync::LazyLock<Mutex<Vec<String>>> =
            std::sync::LazyLock::new(|| Mutex::new(Vec::new()));
        let srv = mock::spawn(|method, path, body, _n| {
            if method == "GET" && path == "/props" {
                return (200, "{}".to_string());
            }
            let v: serde_json::Value = serde_json::from_str(body).unwrap();
            if v["input"].is_array() {
                (400, "{}".to_string())
            } else {
                CAPTURED
                    .lock()
                    .unwrap()
                    .push(v["input"].as_str().unwrap().to_string());
                (200, mock::embeddings_json(&[&[1.0, 0.0][..]]))
            }
        });
        let e =
            RemoteEmbedder::new("M", &format!("http://127.0.0.1:{}", srv.port), None, 1).unwrap();
        e.embed(&["body".to_string()], "PREFIX ").unwrap();
        let cap = CAPTURED.lock().unwrap();
        assert!(cap.iter().any(|t| t == "PREFIX body"));
    }

    #[test]
    fn retry_on_503_then_success() {
        let srv = mock::spawn(|method, path, _body, n| {
            if method == "GET" && path == "/props" {
                return (200, "{}".to_string());
            }
            // dim probe hits: fail first two, succeed third; everything else 200
            if n <= 2 {
                (503, r#"{"error":"loading"}"#.to_string())
            } else {
                (200, mock::embeddings_json(&[&[1.0, 1.0][..]]))
            }
        });
        let e =
            RemoteEmbedder::new("M", &format!("http://127.0.0.1:{}", srv.port), None, 1).unwrap();
        assert_eq!(e.embedding_dim, 2);
    }

    #[test]
    fn model_name_in_404_error() {
        let srv = mock::spawn(|_method, _path, _body, _n| {
            (404, r#"{"error":"model not found"}"#.to_string())
        });
        let e = RemoteEmbedder::new(
            "QwenEmbedding-TYPO",
            &format!("http://127.0.0.1:{}", srv.port),
            None,
            1,
        );
        let err = e.err().unwrap();
        assert!(err.contains("QwenEmbedding-TYPO"));
        assert!(err.contains("llama-swap"));
    }
}
