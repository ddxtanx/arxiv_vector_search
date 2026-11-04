//! llama-swap cross-encoder rerank client — parity with
//! processors/reranker.py.
//!
//! CONTRACT (do not change signatures):
//! - base_url: explicit arg, else env RERANK_BASE_URL, else EMBEDDING_BASE_URL;
//!   strip trailing /v1. api_key: arg else env EMBEDDING_API_KEY.
//! - rerank(): POST {base}/rerank. llama.cpp JSON schema varies by build:
//!   try TEI style {"model":M,"input":{"query":q,"documents":[...]}} first;
//!   on 4xx try Cohere style {"model":M,"query":q,"documents":[...]}.
//!   Parse leniently: top-level array, {"results":[...]}, or {"data":[...]};
//!   score from relevance_score|score|embedding[0]; index defaults to
//!   position. Return sorted desc by score (index, score).
//! - Timeout 120s. No retry (query path, fail fast with model name in msg).
//!
//! Tests (this file): mock /rerank in a thread (same tiny HTTP pattern):
//! TEI happy path, cohere fallback on 400-for-TEI, 404 error message
//! contains model name.

pub struct RemoteReranker {
    pub model_name: String,
    pub base_url: String, // no trailing /v1
    pub api_key: Option<String>,
}

impl RemoteReranker {
    pub fn new(
        model_name: &str,
        base_url: Option<&str>,
        api_key: Option<String>,
    ) -> RemoteReranker {
        todo!("implemented by reranker module")
    }

    /// Returns (candidate_index, score) sorted by score desc.
    pub fn rerank(
        &self,
        query: &str,
        texts: &[String],
    ) -> Result<Vec<(usize, f32)>, String> {
        todo!("implemented by reranker module")
    }
}
