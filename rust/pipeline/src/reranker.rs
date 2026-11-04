//! llama-swap cross-encoder rerank client — parity with processors/reranker.py.
//!
//! - base_url: explicit arg, else env RERANK_BASE_URL, else EMBEDDING_BASE_URL;
//!   strip trailing /v1. api_key: arg else env EMBEDDING_API_KEY.
//! - rerank(): POST {base}/rerank. llama.cpp JSON schema varies by build:
//!   TEI style first, Cohere style on 4xx. Lenient response parsing.
//! - Timeout 120s, no retries (query path; fail fast with model name).

use crate::embedder::normalize_base;
use serde_json::{json, Value};
use std::time::Duration;

pub struct RemoteReranker {
    pub model_name: String,
    pub base_url: String, // no trailing /v1
    pub api_key: Option<String>,
    client: reqwest::blocking::Client,
}

impl RemoteReranker {
    pub fn new(
        model_name: &str,
        base_url: Option<&str>,
        api_key: Option<String>,
    ) -> RemoteReranker {
        let base_url = base_url
            .map(|s| s.to_string())
            .or_else(|| std::env::var("RERANK_BASE_URL").ok())
            .or_else(|| std::env::var("EMBEDDING_BASE_URL").ok())
            .unwrap_or_else(|| "http://127.0.0.1:8080".to_string());
        RemoteReranker {
            model_name: model_name.to_string(),
            base_url: normalize_base(&base_url),
            api_key: api_key.or_else(|| std::env::var("EMBEDDING_API_KEY").ok()),
            client: reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("reqwest client"),
        }
    }

    fn post(&self, payload: &Value) -> Result<(u16, String), String> {
        let mut req = self
            .client
            .post(format!("{}/rerank", self.base_url))
            .json(payload);
        if let Some(key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }
        let resp = req.send().map_err(|e| format!("transport error: {e}"))?;
        let status = resp.status().as_u16();
        let body = resp.text().unwrap_or_default();
        Ok((status, body))
    }

    fn parse_items(body: &str) -> Option<Vec<(usize, f32)>> {
        let v: Value = serde_json::from_str(body).ok()?;
        let items: &Vec<Value> = v
            .as_array()
            .or_else(|| v.get("results").and_then(|r| r.as_array()))
            .or_else(|| v.get("data").and_then(|r| r.as_array()))?;
        let mut out = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let idx = item
                .get("index")
                .and_then(|x| x.as_u64())
                .unwrap_or(i as u64) as usize;
            let score = item
                .get("relevance_score")
                .and_then(|s| s.as_f64())
                .or_else(|| item.get("score").and_then(|s| s.as_f64()))
                .or_else(|| {
                    item.get("embedding")
                        .and_then(|e| e.as_array())
                        .and_then(|a| a.first())
                        .and_then(|x| x.as_f64())
                });
            out.push((idx, score? as f32));
        }
        Some(out)
    }

    /// Returns (candidate_index, score) sorted by score desc.
    pub fn rerank(&self, query: &str, texts: &[String]) -> Result<Vec<(usize, f32)>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let attempts = [
            json!({
                "model": self.model_name,
                "input": { "query": query, "documents": texts },
            }),
            json!({
                "model": self.model_name,
                "query": query,
                "documents": texts,
            }),
        ];
        let mut last = String::new();
        for payload in attempts {
            let (status, body) = self.post(&payload)?;
            if status == 200 {
                if let Some(mut scored) = Self::parse_items(&body) {
                    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
                    return Ok(scored);
                }
                last = format!("unparseable 200 response: {}", &body[..body.len().min(200)]);
            } else {
                last = format!("status {status}: {}", &body[..body.len().min(200)]);
            }
        }
        Err(format!(
            "rerank failed for model '{}' at {}/rerank — {last} \
             (with llama-swap, the reranker model id must match a config section name)",
            self.model_name, self.base_url
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::mock;

    #[test]
    fn tei_happy_path() {
        let srv = mock::spawn(|_m, path, _b, _n| {
            assert_eq!(path, "/rerank");
            (
                200,
                r#"{"data":[{"index":0,"score":0.1},{"index":1,"score":0.9},{"index":2,"score":0.5}]}"#
                    .to_string(),
            )
        });
        let r = RemoteReranker::new(
            "QwenReranking",
            Some(&format!("http://127.0.0.1:{}", srv.port)),
            None,
        );
        let out = r
            .rerank("q", &["a".to_string(), "b".to_string(), "c".to_string()])
            .unwrap();
        assert_eq!(out[0], (1, 0.9));
        assert_eq!(out[2], (0, 0.1));
    }

    #[test]
    fn cohere_fallback_when_tei_rejected() {
        let srv = mock::spawn(|_m, _p, body, _n| {
            let v: serde_json::Value = serde_json::from_str(body).unwrap();
            if v.get("input").is_some() {
                (400, r#"{"error":"need query/documents"}"#.to_string())
            } else {
                (
                    200,
                    r#"{"results":[{"index":1,"relevance_score":0.8},{"index":0,"relevance_score":0.2}]}"#
                        .to_string(),
                )
            }
        });
        let r = RemoteReranker::new("M", Some(&format!("http://127.0.0.1:{}", srv.port)), None);
        let out = r.rerank("q", &["x".to_string(), "y".to_string()]).unwrap();
        assert_eq!(out[0].0, 1);
    }

    #[test]
    fn error_message_contains_model_name() {
        let srv = mock::spawn(|_m, _p, _b, _n| (404, r#"{"error":"model not found"}"#.to_string()));
        let r = RemoteReranker::new(
            "TypoReranker",
            Some(&format!("http://127.0.0.1:{}", srv.port)),
            None,
        );
        let err = r.rerank("q", &["a".to_string()]).err().unwrap();
        assert!(err.contains("TypoReranker"));
        assert!(err.contains("llama-swap"));
    }
}
