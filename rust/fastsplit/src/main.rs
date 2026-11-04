//! fastsplit: parallel PDF text extraction + chunking, mirroring the exact
//! semantics of this repo's Python pipeline:
//!   documents/document.py   (per-page text, 5 regex cleanup passes, page_lens)
//!   processors/splitter.py  (langchain RecursiveCharacterTextSplitter,
//!                            keep_separator=True, char length, add_start_index)
//! Extract mode:  fastsplit --mode extract [pdf paths...]
//! Split-only mode (parity testing): JSONL {"identifier","text"} on stdin.
//! Output: one NDJSON object per chunk, then a summary line.

use rayon::prelude::*;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::process;
use std::time::Instant;

const SEPARATORS: &[&str] = &[
    "\n\n\n", "\n\n", ". \n", ".\n", ". ", "? ", "! ", "; ", "\n", " ", "",
];

struct Params {
    chunk_size: usize,
    chunk_overlap: usize,
    min_chunk_chars: usize,
}

// ---------- char helpers (all offsets/lengths are char-based, like Python) ----------

fn split_once<'a>(text: &'a [char], sep: &[char]) -> Vec<&'a [char]> {
    // Python re.split(literal, text) semantics: keep empty parts.
    if sep.is_empty() {
        return text.iter().map(std::slice::from_ref).collect();
    }
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i + sep.len() <= text.len() {
        if &text[i..i + sep.len()] == sep {
            parts.push(&text[start..i]);
            i += sep.len();
            start = i;
        } else {
            i += 1;
        }
    }
    parts.push(&text[start..]);
    parts
}

fn contains(text: &[char], sep: &[char]) -> bool {
    !sep.is_empty() && split_once(text, sep).len() > 1
}

// ---------- RecursiveCharacterTextSplitter port (keep_separator == True) ----------

fn join_docs(parts: &[String], sep: &str) -> Option<String> {
    let text = parts.join(sep);
    let text = text.trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn merge_splits(splits: &[String], sep: &str, p: &Params) -> Vec<String> {
    // Faithful port of langchain_text_splitters TextSplitter._merge_splits.
    let sep_len = sep.chars().count();
    let mut docs: Vec<String> = Vec::new();
    let mut current_doc: Vec<String> = Vec::new();
    let mut total = 0usize;
    for d in splits {
        let len_ = d.chars().count();
        let joiner = if !current_doc.is_empty() { sep_len } else { 0 };
        if total + len_ + joiner > p.chunk_size {
            if total > p.chunk_size {
                eprintln!(
                    "warn: chunk of size {total} exceeds chunk_size {}",
                    p.chunk_size
                );
            }
            if !current_doc.is_empty() {
                if let Some(doc) = join_docs(&current_doc, sep) {
                    docs.push(doc);
                }
                while total > p.chunk_overlap
                    || (total + len_ + if current_doc.len() > 0 { sep_len } else { 0 }
                        > p.chunk_size
                        && total > 0)
                {
                    let first = current_doc[0].chars().count();
                    let sub_joiner = if current_doc.len() > 1 { sep_len } else { 0 };
                    total -= first + sub_joiner;
                    current_doc.remove(0);
                }
            }
        }
        current_doc.push(d.clone());
        total += len_ + if current_doc.len() > 1 { sep_len } else { 0 };
    }
    if let Some(doc) = join_docs(&current_doc, sep) {
        docs.push(doc);
    }
    docs
}

fn split_text(text: &[char], sep_from: usize, p: &Params) -> Vec<String> {
    // Port of RecursiveCharacterTextSplitter._split_text(text, separators[sep_from..])
    let seps: Vec<&str> = SEPARATORS[sep_from..].to_vec();
    let mut separator: Vec<char> = seps.last().unwrap().chars().collect();
    let mut chosen: Option<usize> = None;
    for (i, s_) in seps.iter().enumerate() {
        let sc: Vec<char> = s_.chars().collect();
        if sc.is_empty() {
            separator = sc;
            chosen = None; // empty separator: no new_separators afterwards
            break;
        }
        if contains(text, &sc) {
            separator = sc;
            chosen = Some(i);
            break;
        }
    }
    // _split_text_with_regex with keep_separator=True ("start"):
    // parts = split_once(text, sep) = [t0, t1, ..., tn] (separators excluded);
    // Python re.split WITH capture yields [t0, SEP, t1, SEP, t2, ...] and the
    // keep-start logic prepends t0 and emits SEP+ti for every following part.
    let parts = split_once(text, &separator);
    let mut splits: Vec<Vec<char>> = Vec::new();
    if separator.is_empty() {
        splits = text.iter().map(|c| vec![*c]).collect();
    } else if parts.len() > 1 {
        splits.push(parts[0].to_vec());
        for part in &parts[1..] {
            let mut piece = separator.clone();
            piece.extend_from_slice(part);
            splits.push(piece);
        }
        splits.retain(|s| !s.is_empty());
    } else {
        splits.push(text.to_vec());
    }

    let mut final_chunks: Vec<String> = Vec::new();
    let mut good: Vec<String> = Vec::new();
    // keep_separator=True => merge separator is ""
    let merge_sep = "";
    let new_sep_start = match chosen {
        Some(i) => sep_from + i + 1,
        None => SEPARATORS.len(), // empty separator reached: recursion stops
    };
    for s in splits {
        if s.len() < p.chunk_size {
            good.push(s.iter().collect());
        } else {
            if !good.is_empty() {
                final_chunks.extend(merge_splits(&good, merge_sep, p));
                good.clear();
            }
            if new_sep_start >= SEPARATORS.len() {
                final_chunks.push(s.iter().collect());
            } else {
                final_chunks.extend(split_text(&s, new_sep_start, p));
            }
        }
    }
    if !good.is_empty() {
        final_chunks.extend(merge_splits(&good, merge_sep, p));
    }
    final_chunks
}

// ---------- create_documents start_index scan + page mapping ----------

fn find_char(text: &[char], needle: &[char], from: usize) -> usize {
    if needle.is_empty() {
        return from;
    }
    let start = from.min(text.len());
    for i in start..text.len() {
        if text[i..].starts_with(needle) {
            return i;
        }
    }
    // Python would return -1; langchain only calls find on its own chunks.
    // Mirror the repo fallback by clamping.
    start
}

struct Chunk {
    page_index: i32,
    chunk_index: usize,
    start_index: usize,
    text: String,
}

fn map_page(start_index: usize, page_lens: &[usize]) -> i32 {
    let mut cum = 0usize;
    for (i, pl) in page_lens.iter().enumerate() {
        cum += pl + 2;
        if cum > start_index {
            return i as i32;
        }
    }
    (page_lens.len().saturating_sub(1)) as i32
}

fn make_chunks(full: Vec<char>, page_lens: &[usize], p: &Params) -> Vec<Chunk> {
    let texts = split_text(&full, 0, p);
    let mut chunks = Vec::with_capacity(texts.len());
    let mut index = 0usize;
    let mut previous_chunk_len = 0usize;
    for (ci, t) in texts.iter().enumerate() {
        let needle: Vec<char> = t.chars().collect();
        let raw = index as i64 + previous_chunk_len as i64 - p.chunk_overlap as i64;
        let found = find_char(&full, &needle, raw.max(0) as usize);
        index = found;
        previous_chunk_len = t.chars().count();
        if t.trim().chars().count() >= p.min_chunk_chars {
            chunks.push(Chunk {
                page_index: map_page(found, page_lens),
                chunk_index: ci,
                start_index: found,
                text: t.clone(),
            });
        }
    }
    chunks
}

// ---------- regex cleanup passes (port of document.py regex_pass) ----------

fn is_w(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn regex_passes(page: Vec<char>) -> Vec<char> {
    let mut out = page;
    // 1. DEHYPHENATE: -\n(\w) -> \1
    let mut r = Vec::with_capacity(out.len());
    let mut i = 0usize;
    while i < out.len() {
        if i + 2 < out.len() && out[i] == '-' && out[i + 1] == '\n' && out[i + 2].is_alphanumeric()
        {
            r.push(out[i + 2]);
            i += 3;
        } else {
            r.push(out[i]);
            i += 1;
        }
    }
    out = r;
    // 2. MID_WORD_BREAK: (?<=[a-z])\n(?=[a-z]) -> ' '
    for i in 0..out.len() {
        if out[i] == '\n'
            && i > 0
            && i + 1 < out.len()
            && out[i - 1].is_ascii_lowercase()
            && out[i + 1].is_ascii_lowercase()
        {
            out[i] = ' ';
        }
    }
    // 3. PAGE_NUMBER: \n\s*\d{1,3}\s*\n -> "\n"
    let mut r = Vec::with_capacity(out.len());
    let mut i = 0usize;
    while i < out.len() {
        if out[i] == '\n' {
            let mut j = i + 1;
            while j < out.len() && out[j].is_whitespace() {
                j += 1;
            }
            let digits_start = j;
            while j < out.len() && out[j].is_ascii_digit() {
                j += 1;
            }
            let ndigits = j - digits_start;
            if (1..=3).contains(&ndigits) {
                let mut after = j;
                while after < out.len() && out[after].is_whitespace() {
                    after += 1;
                }
                if after < out.len() && out[after] == '\n' {
                    r.push('\n');
                    i = after + 1;
                    continue;
                }
            }
        }
        r.push(out[i]);
        i += 1;
    }
    out = r;
    // 4. EXCESS_WHITESPACE: [ \t]{2,} -> " "
    let mut r = Vec::with_capacity(out.len());
    let mut i = 0usize;
    while i < out.len() {
        if out[i] == ' ' || out[i] == '\t' {
            let mut j = i + 1;
            while j < out.len() && (out[j] == ' ' || out[j] == '\t') {
                j += 1;
            }
            if j - i >= 2 {
                r.push(' ');
            } else {
                r.extend_from_slice(&out[i..j]);
            }
            i = j;
        } else {
            r.push(out[i]);
            i += 1;
        }
    }
    out = r;
    // 5. SYMBOL_LINE: \n[^\w\s]{3,}\n -> "\n"
    let mut r = Vec::with_capacity(out.len());
    let mut i = 0usize;
    while i < out.len() {
        if out[i] == '\n' {
            let mut j = i + 1;
            while j < out.len() && !is_w(out[j]) && !out[j].is_whitespace() {
                j += 1;
            }
            if j - i - 1 >= 3 && j < out.len() && out[j] == '\n' {
                r.push('\n');
                i = j + 1;
                continue;
            }
        }
        r.push(out[i]);
        i += 1;
    }
    out = r;
    // 6. CONTROL: [\x00-\x08\x0b\x0c\x0e-\x1f\x7f] -> " "
    // (legacy CM Type1 fonts make pdfium/pdf_oxide emit \r where 'c' belongs;
    // control chars are embedding noise). Mirrors _CONTROL in document.py.
    for c in out.iter_mut() {
        if (*c as u32) < 0x20 && *c != '\n' && *c != '\t' {
            *c = ' ';
        } else if *c == '\x7f' {
            *c = ' ';
        }
    }
    out
}

// ---------- document processing ----------

fn text_to_pages(full_text: String) -> (Vec<usize>, Vec<char>) {
    let chars: Vec<char> = full_text.chars().collect();
    (vec![chars.len()], chars)
}

fn process_extract(identifier: &str, path: &str, p: &Params) -> Vec<Chunk> {
    let doc = match pdf_oxide::document::PdfDocument::open(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {identifier}: open failed: {e}");
            return Vec::new();
        }
    };
    let n = match doc.page_count() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("error: {identifier}: page_count: {e}");
            return Vec::new();
        }
    };
    let mut cleaned: Vec<Vec<char>> = Vec::with_capacity(n);
    for i in 0..n {
        let txt = doc.extract_text(i).unwrap_or_default();
        let ch = regex_passes(txt.chars().collect());
        let start = ch
            .iter()
            .position(|c| !c.is_whitespace())
            .unwrap_or(ch.len());
        let end = ch
            .iter()
            .rposition(|c| !c.is_whitespace())
            .map(|x| x + 1)
            .unwrap_or(start);
        cleaned.push(ch[start..end].to_vec());
    }
    let page_lens: Vec<usize> = cleaned.iter().map(|c| c.len()).collect();
    let mut full: Vec<char> = Vec::new();
    for (i, page) in cleaned.iter().enumerate() {
        full.extend_from_slice(page);
        if i + 1 < cleaned.len() {
            full.push('\n');
            full.push('\n');
        }
    }
    let start = full
        .iter()
        .position(|c| !c.is_whitespace())
        .unwrap_or(full.len());
    let end = full
        .iter()
        .rposition(|c| !c.is_whitespace())
        .map(|x| x + 1)
        .unwrap_or(start);
    let full = full[start..end].to_vec();
    make_chunks(full, &page_lens, p)
}

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
        let results: Vec<(String, Vec<Chunk>)> = positional
            .par_iter()
            .map(|path| {
                let id = path.rsplit('/').next().unwrap_or(path).to_string();
                let id = id.trim_end_matches(".pdf").to_string();
                let chunks = process_extract(&id, path, &p);
                (id, chunks)
            })
            .collect();
        for (id, chunks) in &results {
            nchunks += chunks.len();
            emit(id, chunks, &mut out);
        }
    } else {
        // split-only: JSONL {"identifier","text"} on stdin (parity testing)
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
        let results: Vec<(String, Vec<Chunk>)> = docs
            .par_iter()
            .map(|(id, text)| {
                let (lens, chars) = text_to_pages(text.clone());
                (id.clone(), make_chunks(chars, &lens, &p))
            })
            .collect();
        for (id, chunks) in &results {
            nchunks += chunks.len();
            emit(id, chunks, &mut out);
        }
    }
    let _ = out.flush();
    eprintln!("fastsplit: chunks={nchunks} elapsed={:?}", t0.elapsed());
    if nchunks == 0 && mode == "extract" && !positional.is_empty() {
        process::exit(2);
    }
}
