//! arxiv_vector_search — full Rust pipeline CLI (embed/ingest/query/rerank).
//! Mirrors the Python CLI surface; see repo README for usage.

mod common;
mod db;
mod downloader;
mod embedder;
mod metadata;
mod reranker;

use clap::Parser;
use common::{Document, DocumentType, State};
use db::Db;
use downloader::Downloader;
use embedder::RemoteEmbedder;
use reranker::RemoteReranker;
use std::io::Write;
use std::thread;

const CHUNK_CHAR_FACTOR: f64 = 3.5;
const MIN_CHUNK_CHARS: usize = 128;

#[derive(Parser, Debug)]
#[command(about = "arXiv vector search pipeline (Rust)")]
struct Cli {
    #[arg(long)]
    query: Option<String>,
    #[arg(long)]
    embed: bool,
    #[arg(long)]
    model: Option<String>,
    #[arg(long = "batch_size", default_value_t = 100)]
    batch_size: usize,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long)]
    flush: bool,
    #[arg(long = "flush_errors")]
    flush_errors: bool,
    #[arg(long = "update_arxiv_metadata")]
    update_arxiv_metadata: bool,
    #[arg(long)]
    embedding_url: Option<String>,
    #[arg(long)]
    embedding_api_key: Option<String>,
    #[arg(long)]
    concurrency: Option<usize>,
    #[arg(long)]
    rerank_model: Option<String>,
    #[arg(long = "rerank_candidates", default_value_t = 50)]
    rerank_candidates: usize,
    #[arg(long = "chunk_size", default_value_t = 512)]
    chunk_size: i32,
    #[arg(long = "chunk_overlap")]
    chunk_overlap: Option<i32>,
    #[arg(long = "document_prefix", default_value = "")]
    document_prefix: String,
    #[arg(long = "query_prefix", default_value = "")]
    query_prefix: String,
    #[arg(long = "dl_path", default_value = "pdfs/arxiv")]
    dl_path: String,
    #[arg(long = "keep_pdfs")]
    keep_pdfs: bool,
}

fn prompt(msg: &str) -> String {
    print!("{msg}");
    let _ = std::io::stdout().flush();
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).unwrap_or(0);
    s.trim().to_string()
}

fn main() {
    let cli = Cli::parse();
    let db_url = match std::env::var("DATABASE_URL") {
        Ok(u) => u,
        Err(_) => {
            eprintln!("DATABASE_URL environment variable not set. Exiting.");
            std::process::exit(1);
        }
    };
    let base_url = cli
        .embedding_url
        .clone()
        .or_else(|| std::env::var("EMBEDDING_BASE_URL").ok());
    let base_url = match base_url {
        Some(u) => u,
        None => {
            eprintln!("No embedding server: pass --embedding_url or set EMBEDDING_BASE_URL.");
            std::process::exit(1);
        }
    };

    // Model selection mirrors the Python CLI (list + prompt when unspecified).
    let mut db = Db::connect(&db_url).expect("db");
    let model_name = match &cli.model {
        Some(m) => m.clone(),
        None => {
            let models = db.existing_model_names();
            if models.len() == 1 {
                models[0].clone()
            } else {
                for (i, name) in models.iter().enumerate() {
                    println!("{i}: {name}");
                }
                let answer = prompt("Enter the number corresponding to the model, or its name: ");
                if answer.parse::<usize>().is_ok()
                    && answer.parse::<usize>().unwrap() < models.len()
                {
                    models[answer.parse::<usize>().unwrap()].clone()
                } else {
                    answer
                }
            }
        }
    };

    let embedder = match RemoteEmbedder::new(
        &model_name,
        &base_url,
        cli.embedding_api_key
            .clone()
            .or_else(|| std::env::var("EMBEDDING_API_KEY").ok()),
        cli.concurrency
            .or_else(|| {
                std::env::var("EMBED_CONCURRENCY")
                    .ok()
                    .and_then(|v| v.parse().ok())
            })
            .unwrap_or(4),
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("embedding server unavailable: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "Model '{model_name}': dim={} ctx={} batch_input={}",
        embedder.embedding_dim, embedder.max_input_length, embedder.supports_batch_input
    );

    let chunk_overlap = cli
        .chunk_overlap
        .unwrap_or((cli.chunk_size as f64 * 0.15) as i32);
    db.get_or_create_model(
        &model_name,
        embedder.embedding_dim as i32,
        cli.batch_size as i32,
        cli.chunk_size,
        chunk_overlap,
        &cli.document_prefix,
        &cli.query_prefix,
        Some(&base_url),
    )
    .expect("model row");
    db.create_embedding_table().expect("embedding table");

    if cli.flush {
        let confirm = prompt(
            "Are you sure you want to flush this model's data and start from scratch? Type 'yes' to confirm: ",
        );
        if confirm.eq_ignore_ascii_case("yes") {
            db.flush_model().expect("flush");
            println!("Flushed model '{model_name}'.");
        } else {
            println!("Flush cancelled. Exiting.");
            std::process::exit(0);
        }
    }
    if cli.flush_errors {
        let confirm = prompt(
            "Reset errored documents to missing so they are reprocessed? Type 'yes' to confirm: ",
        );
        if confirm.eq_ignore_ascii_case("yes") {
            db.flush_errors().expect("flush_errors");
        } else {
            println!("Flush errors cancelled. Exiting.");
            std::process::exit(0);
        }
    }

    if cli.update_arxiv_metadata {
        println!("Seeding arXiv math document list (Kaggle snapshot)...");
        let cache_dir = std::path::Path::new(&cli.dl_path).join("metadata_cache");
        match metadata::seed_math_documents(&cache_dir) {
            Ok(docs) => {
                println!("Adding {} documents...", docs.len());
                db.add_documents(&docs).expect("add_documents");
            }
            Err(e) => {
                eprintln!("metadata seeding failed: {e}");
                std::process::exit(1);
            }
        }
    }

    if cli.embed {
        run_embed(&mut db, &embedder, &cli);
    }

    if let Some(q) = &cli.query {
        run_query(&mut db, &embedder, &cli, q);
    }
}

fn run_embed(db: &mut Db, embedder: &RemoteEmbedder, cli: &Cli) {
    let dl = Downloader::new(&cli.dl_path);
    db.add_missing_states().expect("add_missing_states");
    for (state, n) in db.state_counts().unwrap_or_default() {
        println!("state {state}: {n}");
    }
    let batch_size = db.model().batch_size.max(1) as usize;
    let chunk_size = db.model().chunk_size;
    let chunk_overlap_tokens = db.model().chunk_overlap;
    let params = fastsplit::Params {
        chunk_size: (chunk_size as f64 * CHUNK_CHAR_FACTOR) as usize,
        chunk_overlap: (chunk_overlap_tokens as f64 * CHUNK_CHAR_FACTOR) as usize,
        min_chunk_chars: MIN_CHUNK_CHARS,
    };
    loop {
        let batch = db.get_missing(batch_size as i64).expect("get_missing");
        if batch.is_empty() {
            break;
        }
        println!("Processing batch of {} documents.", batch.len());
        let idents: Vec<String> = batch.iter().map(|d| d.identifier.clone()).collect();

        // Phase 1: download (parallel; skips files already on disk).
        let t = std::time::Instant::now();
        let dl_results = dl.download_all(&batch, cli.threads * 4, true);
        let mut ok_paths: Vec<(String, std::path::PathBuf)> = Vec::new();
        let mut dl_errs: Vec<(String, String)> = Vec::new();
        for (ident, outcome) in dl_results {
            match outcome {
                downloader::DownloadOutcome::Downloaded(p) => ok_paths.push((ident, p)),
                downloader::DownloadOutcome::Error(e) => dl_errs.push((ident, e)),
            }
        }
        println!(
            "Downloaded {} docs in {:?}; {} errors.",
            ok_paths.len(),
            t.elapsed(),
            dl_errs.len()
        );
        if !dl_errs.is_empty() {
            let first_msg = dl_errs[0].1.clone();
            db.set_states(
                &dl_errs.iter().map(|(i, _)| i.clone()).collect::<Vec<_>>(),
                State::DownloadError,
                Some(&first_msg),
            )
            .expect("set_states");
        }

        // Optimization: prefetch the NEXT batch's downloads while we split +
        // embed this one (skip_existing makes the sync pass instant after).
        {
            let next = db.get_missing(batch_size as i64).unwrap_or_default();
            if !next.is_empty() {
                let dl2 = Downloader::new(&cli.dl_path);
                thread::spawn(move || {
                    let _ = dl2.download_all(&next, 4, true);
                });
            }
        }

        // Phase 2: extract + split (rayon).
        let t = std::time::Instant::now();
        let files: Vec<(String, String)> = ok_paths
            .iter()
            .map(|(id, p)| (id.clone(), p.to_string_lossy().to_string()))
            .collect();
        let split_results = fastsplit::extract_documents_parallel(&files, &params, cli.threads);
        let mut chunks: Vec<(String, i32, i32, String)> = Vec::new();
        let mut split_errs: Vec<(String, String)> = Vec::new();
        for (ident, res) in split_results {
            match res {
                Ok(cs) => {
                    if cs.is_empty() {
                        split_errs.push((ident.clone(), "no chunks produced".to_string()));
                    }
                    for c in cs {
                        chunks.push((ident.clone(), c.page_index, c.chunk_index as i32, c.text));
                    }
                }
                Err(e) => split_errs.push((ident, e)),
            }
        }
        println!(
            "Split into {} chunks in {:?}; {} split errors.",
            chunks.len(),
            t.elapsed(),
            split_errs.len()
        );
        if !split_errs.is_empty() {
            let first_msg = split_errs[0].1.clone();
            db.set_states(
                &split_errs
                    .iter()
                    .map(|(i, _)| i.clone())
                    .collect::<Vec<_>>(),
                State::SplitError,
                Some(&first_msg),
            )
            .expect("set_states");
        }

        // Phase 3: embed via llama-swap.
        let t = std::time::Instant::now();
        let texts: Vec<String> = chunks.iter().map(|(_, _, _, txt)| txt.clone()).collect();
        let vectors = match embedder.embed(&texts, &db.model().document_prefix) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("embedding failed for batch: {e}");
                std::process::exit(1);
            }
        };
        println!("Embedded {} chunks in {:?}.", vectors.len(), t.elapsed());

        // Phase 4: store + state transitions.
        let ident_map = db.warm_ids(&idents).expect("warm_ids");
        let rows: Vec<(String, i32, i32, Vec<f32>, String)> = chunks
            .into_iter()
            .zip(vectors)
            .map(|((ident, page, chunk_i, text), vec)| (ident, page, chunk_i, vec, text))
            .collect();
        db.add_embeddings(&ident_map, &rows)
            .expect("add_embeddings");
        let good_docs: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            rows.iter()
                .filter(|r| seen.insert(r.0.clone()))
                .map(|r| r.0.clone())
                .collect()
        };
        db.set_states(&good_docs, State::Embedded, None)
            .expect("set_states");

        if !cli.keep_pdfs {
            for (_, path) in &ok_paths {
                let _ = std::fs::remove_file(path);
            }
        }
        println!("Batch complete.");
    }
    println!("No missing documents remain for this model.");
    for (state, n) in db.state_counts().unwrap_or_default() {
        println!("  {state}: {n}");
    }
}

fn run_query(db: &mut Db, embedder: &RemoteEmbedder, cli: &Cli, q: &str) {
    let qvec = match embedder.embed(&[q.to_string()], &db.model().query_prefix) {
        Ok(mut v) => v.pop().unwrap(),
        Err(e) => {
            eprintln!("query embedding failed: {e}");
            std::process::exit(1);
        }
    };

    if let Some(rmodel) = cli
        .rerank_model
        .clone()
        .or_else(|| std::env::var("RERANK_MODEL").ok())
    {
        let reranker = RemoteReranker::new(
            &rmodel,
            cli.embedding_url.as_deref(),
            cli.embedding_api_key
                .clone()
                .or_else(|| std::env::var("EMBEDDING_API_KEY").ok()),
        );
        let candidates = db
            .query_embeddings(&qvec, cli.rerank_candidates as i64)
            .expect("query");
        let texts: Vec<String> = candidates.iter().map(|c| c.4.clone()).collect();
        match reranker.rerank(q, &texts) {
            Ok(order) => {
                println!(
                    "\nTop {} results after reranking {} candidates:",
                    order.len().min(10),
                    candidates.len()
                );
                for (rank, (ci, score)) in order.iter().take(10).enumerate() {
                    let (ident, _t, page, dist, _) = &candidates[*ci];
                    println!(
                        "{:>2}. {} p{} rerank={:.4} vec_dist={:.4}",
                        rank + 1,
                        Document {
                            identifier: ident.clone(),
                            pdf_type: DocumentType::Arxiv
                        }
                        .url(),
                        page,
                        score,
                        dist
                    );
                }
            }
            Err(e) => eprintln!("reranking failed: {e} (showing vector-only results)"),
        }
    }

    let results = db.query_embeddings(&qvec, 1000).expect("query");
    let mut by_ident: std::collections::HashMap<String, (f64, Vec<i32>)> =
        std::collections::HashMap::new();
    for (ident, _t, page, dist, _) in &results {
        let e = by_ident
            .entry(ident.clone())
            .or_insert_with(|| (*dist, Vec::new()));
        e.0 = e.0.min(*dist);
        e.1.push(*page);
    }
    let mut ranked: Vec<(&String, (f64, Vec<i32>))> =
        by_ident.iter().map(|(k, v)| (k, v.clone())).collect();
    ranked.sort_by(|a, b| a.1 .0.total_cmp(&b.1 .0));
    println!("\nTop 10 results sorted by minimum distance:");
    for (ident, (dist, pages)) in ranked.iter().take(10) {
        let mut pages = pages.clone();
        pages.sort_unstable();
        pages.dedup();
        println!(
            "URL: {}, Min Distance: {dist:.4}, Pages: {pages:?}",
            Document {
                identifier: ident.to_string(),
                pdf_type: DocumentType::Arxiv
            }
            .url()
        );
    }
    let avg = db.query_avg_by_doc(&qvec, 10).expect("avg query");
    println!("\nTop 10 results sorted by average distance:");
    for (ident, t, dist) in avg {
        println!(
            "URL: {}, Avg Distance: {dist:.4}",
            Document {
                identifier: ident,
                pdf_type: t
            }
            .url()
        );
    }
}
