//! fastsplit CLI: thin wrapper over the fastsplit library.
//! Extract mode:  fastsplit --mode extract [pdf paths...]
//! Split-only mode (parity testing): JSONL {"identifier","text"} on stdin.
//! NDJSON one object per chunk on stdout; timing on stderr.

use fastsplit::{Chunk, Params, chunk_text, extract_documents_parallel};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::process;
use std::time::Instant;

fn emit(identifier: &str, chunks: &[Chunk], out: &mut dyn Write) {
    for c in chunks {
        let v = json!({
            "identifier": identifier,
            "page_index": c.page_index,
            "chunk_index": c.chunk_index,
            "start_index": c.start_index,
            "text": c.text,
        });
        let line = serde_json::to_string(&v).unwrap();
        let _ = writeln!(out, "{line}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut mode = "extract";
    let mut p = Params {
        chunk_size: 1792,
        chunk_overlap: 266,
        min_chunk_chars: 128,
    };
    let mut positional: Vec<String> = Vec::new();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                mode = args.get(i + 1).map(|s| s.as_str()).unwrap_or("extract");
                i += 2;
            }
            "--chunk-chars" => {
                p.chunk_size = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--chunk-overlap" => {
                p.chunk_overlap = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--min-chars" => {
                p.min_chunk_chars = args[i + 1].parse().unwrap();
                i += 2;
            }
            a => {
                positional.push(a.to_string());
                i += 1;
            }
        }
    }

    let t0 = Instant::now();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    let mut nchunks = 0usize;

    if mode == "extract" {
        let files: Vec<(String, String)> = positional
            .iter()
            .map(|path| {
                let id = path.rsplit('/').next().unwrap_or(path).to_string();
                (id.trim_end_matches(".pdf").to_string(), path.clone())
            })
            .collect();
        let results = extract_documents_parallel(&files, &p, 0); // 0 => rayon default
        for (id, res) in &results {
            match res {
                Ok(chunks) => {
                    nchunks += chunks.len();
                    emit(id, chunks, &mut out);
                }
                Err(e) => eprintln!("error: {id}: {e}"),
            }
        }
    } else {
        let stdin = io::stdin();
        let docs: Vec<(String, String)> = stdin
            .lock()
            .lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| {
                let v: Value = serde_json::from_str(&l).ok()?;
                Some((
                    v["identifier"].as_str()?.to_string(),
                    v["text"].as_str()?.to_string(),
                ))
            })
            .collect();
        for (id, text) in &docs {
            let chunks = chunk_text(id, text, &p);
            nchunks += chunks.len();
            emit(id, &chunks, &mut out);
        }
    }
    let _ = out.flush();
    eprintln!("fastsplit: chunks={nchunks} elapsed={:?}", t0.elapsed());
    if nchunks == 0 && mode == "extract" && !positional.is_empty() {
        process::exit(2);
    }
}
