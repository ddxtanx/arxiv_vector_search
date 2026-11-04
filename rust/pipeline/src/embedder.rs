//! llama.cpp / llama-swap OpenAI-compatible embedding client — parity with
//! processors/llama_cpp_embedder.py, with server-side batching upgrade.
//!
//! CONTRACT (do not change signatures):
//! - base_url normalization: strip trailing /v1 (stored without it);
//!   embeddings at {base}/v1/embeddings, probe GET {base}/props -> n_ctx.
//! - new(): construct client, probe embedding_dim (embed "dimension probe",
//!   read data[0].embedding length), probe max_input_length via /props n_ctx
//!   (fallback 512 on failure), probe supports_batch_input: POST
//!   {"model":M,"input":["a","b"],"encoding_format":"float"}; batch=true iff
//!   200 with data length 2, else single-input mode.
//! - embed_*: apply prefix (document_prefix/query_prefix already included
//!   by caller? NO — caller passes texts raw and prefix separately; prepend
//!   prefix to each text here). Return L2-normalized Vec<f32> per text.
//! - Transport: reqwest blocking, 120 s timeout. Retry 4 attempts total,
//!   backoff sleeps 2s/5s/10s, on >=500 status OR transport error. 4xx fails
//!   immediately with message including the model name (llama-swap 404 =
//!   wrong section id — mention that hint).
//! - Batch mode: send whole batch as input array in one request.
//!   Single mode: one request per text, run with rayon across
//!   self.concurrency threads (preserve input order in output).
//!
//! Tests (this file): spin a mock llama-server in a thread using
//! std::net::TcpListener (tiny hand-rolled HTTP/1.1: read headers/body,
//! respond JSON). Cover: dim probe, batch probe fallback (single on 400),
//! prefix application, normalization (norm 1.0), retry on 503 twice then
//! 200, 404 message contains model name.

pub struct RemoteEmbedder {
    pub model_name: String,
    pub base_url: String, // no trailing /v1
    pub api_key: Option<String>,
    pub concurrency: usize,
    pub embedding_dim: usize,
    pub max_input_length: usize,
    pub supports_batch_input: bool,
}

impl RemoteEmbedder {
    pub fn new(
        model_name: &str,
        base_url: &str,
        api_key: Option<String>,
        concurrency: usize,
    ) -> Result<RemoteEmbedder, String> {
        todo!("implemented by embedder module")
    }

    /// L2-normalized f32 embeddings, same order as inputs.
    pub fn embed(
        &self,
        texts: &[String],
        prefix: &str,
    ) -> Result<Vec<Vec<f32>>, String> {
        todo!("implemented by embedder module")
    }
}
