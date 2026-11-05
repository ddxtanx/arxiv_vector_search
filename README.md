# arxiv_vector_search

Embeds arXiv (and other) PDFs into PostgreSQL + pgvector and answers
semantic queries with cosine top-k search.

## Embedding backends

Two backends share the same storage/query pipeline. Which one runs per
model is recorded in the `models` table.

### 1. Remote server (default on this machine — recommended)

Embeds via any OpenAI-compatible `/v1/embeddings` endpoint. On
`ddxarchlinux` this is the llama-swap router at `http://127.0.0.1:8194`
serving GGUF embedding models (e.g. `QwenEmbedding` / `QwenEmbeddingSmall`
— Qwen3-Embedding), the same server Hindsight and LCM already use.
No torch install is needed at all.

```bash
export DATABASE_URL=postgresql://...
export EMBEDDING_BASE_URL=http://127.0.0.1:8194/v1   # or --embedding_url
python -m arxiv_vector_search --embed
# model name must EXACTLY match the llama-swap config section name
```

| env / flag | meaning |
|---|---|
| `EMBEDDING_BASE_URL` / `--embedding_url` | server base URL (activates remote mode) |
| `EMBEDDING_API_KEY` / `--embedding_api_key` | optional bearer token |
| `EMBED_CONCURRENCY` / `--concurrency` | parallel requests (default 4) — keep low if the server also answers interactive traffic, or that traffic gets starved |
| `EMBED_TOKENIZER` | optional HF tokenizer name for exact token-based chunking without torch |

Note on llama-swap: the served model id is the config section name; a wrong
id is a confusing 404 loop (the client retries with backoff then gives a
pointed error). The first request after idle pays the model reload; keep
`sleep-idle-seconds` high during bulk embedding runs.

Embedding dimensions are probed from the server at startup, so the
per-model `halfvec` table adapts automatically.

### 2. Local sentence-transformers (legacy path)

```bash
uv sync --extra local   # torch ROCm, flash-attn, bitsandbytes...
python -m arxiv_vector_search --embed
```

## Typical flow

```bash
python -m arxiv_vector_search --update_arxiv_metadata   # seed document list from Kaggle snapshot
python -m arxiv_vector_search --embed                   # download -> extract -> split -> embed
python -m arxiv_vector_search --query "your question"
python tools/add_hnsw_index.py                          # one-time HNSW index per model
```

`tools/add_hnsw_index.py` builds `hnsw (embedding vector_cosine_ops)`
indexes with `CREATE INDEX CONCURRENTLY`; without it every query is an
exact scan. Queries set `hnsw.ef_search` to at least `2 * top_k` when the
index exists.

## PDF extraction

PyMuPDF, with `get_text(sort=True)`. Measured on this repo's corpus mix
(2017-2023 papers + pre-2007 legacy math with custom-encoded Computer
Modern Type1 fonts):

- On legacy files the default block order can come out word-per-line with
  dropped characters; `sort=True` restores correct text at ~17 ms/paper
  cost — noise next to embedding time.
- Candidates benchmarked: pypdf is 20-40x slower at equal fidelity;
  pdfium (pypdfium2) is speed-parity but emits control characters on
  legacy files; pdf_oxide (Rust, MIT/Apache) is 2.4-3.7x slower than
  MuPDF with parity fidelity and a nice header/footer-artifact API —
  worth watching, and it is the extractor a Rust port would use.

## Rust pipeline (`rust/`) — full rewrite

`rust/pipeline/` is a complete Rust rewrite of the pipeline (ingest, embed,
query, rerank) in a single binary. `rust/fastsplit/` is its extract+chunk
library (also usable standalone).

Build:
```bash
cd rust && cargo build --release        # binary: rust/target/release/pipeline
```

Usage mirrors the Python CLI:
```bash
DATABASE_URL=... EMBEDDING_BASE_URL=http://127.0.0.1:8194 \
./rust/target/release/pipeline --update_arxiv_metadata   # seed doc list (needs Kaggle creds)
./rust/target/release/pipeline --embed --model QwenEmbedding --batch_size 200
RERANK_MODEL=QwenReranking ./rust/target/release/pipeline --query "motives in Shimura varieties"
```
Env vars: `DATABASE_URL`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`,
`EMBED_CONCURRENCY`, `RERANK_MODEL`, `RERANK_BASE_URL`, `KAGGLE_USERNAME`/
`KAGGLE_KEY` (or `~/.kaggle/access_token`).

### New schema (the old DB was wiped; no migration)
- `documents(id bigserial, identifier text unique, pdf_type text)`
- `models(id, name, embedding_dim, batch_size, chunk_size, chunk_overlap,
  document_prefix, query_prefix, base_url)`
- `doc_states(document_id, model_id, state, error_message, updated_at)` with
  **PRIMARY KEY (document_id, model_id)** — the Python layer allowed
  duplicate state rows; failures now keep their `error_message`.
- `embeddings_<model>(document_id, page_index, chunk_index,
  embedding halfvec)` — **no chunk text column**: text is roughly the
  same size as the vectors and does not belong in the DB. Reranking
  **refetches** it (see below). The table is created **with its HNSW
  cosine index** (`halfvec_cosine_ops`; no separate indexing tool); FKs
  use `ON DELETE CASCADE`.
- state values are lowercase (`missing`, `download_error`, `split_error`,
  `embedded`) instead of SQLAlchemy enum names.

### Optimizations vs the Python pipeline (each defensible, each testable)
- **Server-side embedding batches**: probes whether llama.cpp accepts
  `input` as an array; sends whole batches in one request when supported
  (Python: one request per chunk).
- **Prefetched downloads**: while batch N is being split/embedded, batch
  N+1's PDFs download on a background thread (skip-existing makes the
  sync pass instant). Downloads hit the public bucket over anonymous
  HTTPS — no google-cloud SDK, no credentials.
- **SET LOCAL hnsw.ef_search runs inside a transaction** — in the Python
  layer it was a silent no-op outside autocommit (real recall bug).
- **Refetch instead of store for reranking**: rerank candidates get their
  text by re-downloading (or reading the local `--dl_path` cache) the
  candidate PDFs and re-running the deterministic splitter with the
  model row's stored chunk params; `chunk_index` maps hit -> text. Cold
  (10 fresh PDFs) ~1.8s; warm cache ~0.9s.
- **Pluggable document scoring** (`--doc_score`, `--score_topk`): chunk
  scores aggregate to a document score via `topk` (default, mean of the
  best k=5 chunks), `mean`, `max` (old min-distance), or `noisy_or`
  (probabilistic union; sensible only on reranker-calibrated scores).
  Why topk: `mean` dilutes long docs (measured: the true match fell to
  4th on an 88-chunk corpus), `max` is gamed by any long doc with one
  lucky chunk; topk rewards multiple relevant passages and is
  length-neutral once len>=k.
- Multi-row transactional inserts (500/txn), rayon extract+split
  (byte-parity with langchunking proven by `tools/validate_fastsplit.py`),
  retry/backoff on GCS and embedding transport, stored error messages for
  failed docs (`--flush_errors` re-queues them).

### Verification status
- Unit/integration: **24 Rust tests** — chunker parity re-verified
  (**129 docs / 5,581 chunks / 0 mismatches**, incl. legacy pre-2007
  CM-font files), embedder/reranker/metadata against mock HTTP servers,
  downloader against live GCS.
- **Live end-to-end in the dev sandbox** against real Postgres 18.6 +
  pgvector 0.8.7 and a deterministic stand-in for llama-swap: 10 arXiv
  papers (legacy + modern) ingested (download -> split -> embed ->
  halfvec + HNSW), reranked query (vector top-50 -> refetch 369 chunk
  texts -> cross-encoder scores -> topk aggregation), scoring-method
  comparison, state transitions + idempotent re-runs. The one component
  not exercised for real is llama.cpp model inference — run
  `--embed/--query` against your llama-swap on :8194 for that.

## Notes

- Chunk sizing: `chunk_size` tokens for the model's tokenizer (or
  `3.5 * chunk_size` characters when no tokenizer is available, e.g.
  remote mode without `EMBED_TOKENIZER`). Stored per model and reused
  automatically on re-embed runs.
- Chunks shorter than 128 chars are dropped (`DEFAULT_MIN_CHUNK_CHARS`).
- Embedding inserts are `ON CONFLICT DO NOTHING`, so re-processing a
  document after a crash mid-batch is safe.
- A 6th cleanup pass scrubs control characters (legacy CM-font arXiv
  files emit them from every extractor).

## Experimental: Rust split engine (`--split_engine rust`)

`rust/fastsplit/` is a rayon-parallel extract+split binary (pdf_oxide + a
byte-for-byte port of this repo's exact langchain chunking semantics,
verified by `tools/validate_fastsplit.py`). Build with
`cargo build --release --manifest-path rust/fastsplit/Cargo.toml`, then:

```bash
python -m arxiv_vector_search --embed --split_engine rust
```

Measured on 66 mixed papers (this machine's sandbox, 24 threads vs the
Python forkserver pool): extract+split 2.4 s vs 5.3 s (w=10) / 3.1 s
(w=24) / 22 s sequential — ~2-3x faster end-to-end for the stage, with
scaling 14.6 s single-thread -> 1.4 s at 24 threads. Caveats: the
extractor differs from MuPDF (chunk text differs slightly at the same
chunk boundaries for identical input text — never mix engines within one
model's rows), and chunking is character-based only (no tokenizer path).

The full-pipeline context: extraction+splitting was measured at <1% of
stage cost for the *split* part itself and the whole stage remains dwarfed
by embedding throughput, so this is an optional accelerator, not a reason
to port the rest.
