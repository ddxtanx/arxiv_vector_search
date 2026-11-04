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
