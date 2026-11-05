//! GCS downloader for the public arxiv-dataset bucket — parity with
//! documents/arxiv/arxiv_downloader.py, optimized.
//!
//! CONTRACT (do not change signatures):
//! - Downloads via anonymous HTTPS:
//!   `https://storage.googleapis.com/arxiv-dataset/<url-encoded blob name>`
//!   (the bucket is public; no auth, no google-cloud crate needed).
//! - Writes to <dl_root>/<identifier>.pdf, creating parent folders for
//!   legacy identifiers ("math/0211052v1" -> dl_root/math/0211052v1.pdf).
//! - skip_existing: if the target file exists AND is non-empty AND starts
//!   with b"%PDF", treat as success without re-downloading (optimization;
//!   python re-downloaded unconditionally).
//! - Retry up to 3 times with 2s/5s backoff on transport errors and 5xx/429.
//! - 404 => immediate error result (missing blob is permanent).
//! - Parallel across documents with rayon (workers param; 0 = all cores).
//!
//! Tests to provide (this file): unit tests for the URL encoding + blob
//! naming using common::Document rules, and one real-network test marked
//! #[ignore]? NO — sandbox has network; include two live tests downloading
//! (a) legacy blob arxiv/math/pdf/0211/0211052v1.pdf and
//! (b) new-style arxiv/arxiv/pdf/2401/2401.00001v1.pdf, asserting %PDF magic.

use crate::common::Document;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use reqwest::blocking::Client;
use reqwest::StatusCode;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

const GCS_BASE: &str = "https://storage.googleapis.com/arxiv-dataset/";

/// Backoff waits (seconds) applied *between* attempts: attempt, 2s, attempt,
/// 5s, attempt => 3 total tries per the contract.
const BACKOFF_SECS: [u64; 2] = [2, 5];

/// Percent-encode one path segment (RFC 3986 unreserved chars pass through).
fn percent_encode_segment(seg: &str) -> String {
    let mut out = String::with_capacity(seg.len());
    for b in seg.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// URL-encode a GCS blob name, keeping '/' as the path separator.
pub fn encode_blob_path(blob: &str) -> String {
    blob.split('/')
        .map(percent_encode_segment)
        .collect::<Vec<_>>()
        .join("/")
}

/// Full anonymous-GCS URL for a document's blob (common::Document rules).
pub fn blob_url(doc: &Document) -> String {
    format!("{}{}", GCS_BASE, encode_blob_path(&doc.gcloud_blob_name()))
}

/// skip_existing predicate: file exists, is non-empty, starts with b"%PDF".
fn is_existing_pdf(path: &Path) -> bool {
    let mut f = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut magic = [0u8; 4];
    match f.read_exact(&mut magic) {
        Ok(()) => &magic == b"%PDF",
        Err(_) => false, // empty or shorter than 4 bytes => not a PDF
    }
}

/// Write bytes atomically (tmp file + rename) so a crash can't leave a
/// truncated PDF that skip_existing would trust.
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_file_name(format!(
        "{}.part",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("download")
    ));
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)
}

#[derive(Debug)]
pub enum DownloadOutcome {
    Downloaded(PathBuf),
    Error(String),
}

pub struct Downloader {
    pub dl_root: PathBuf,
}

impl Downloader {
    pub fn new(dl_root: impl Into<PathBuf>) -> Downloader {
        Downloader {
            dl_root: dl_root.into(),
        }
    }

    /// Download all documents; returns one outcome per input document,
    /// same order.
    pub fn download_all(
        &self,
        docs: &[Document],
        workers: usize,
        skip_existing: bool,
    ) -> Vec<(String, DownloadOutcome)> {
        // One shared client => connection pool reuse across all downloads.
        let client = match Client::builder().timeout(Duration::from_secs(300)).build() {
            Ok(c) => c,
            Err(e) => {
                return docs
                    .iter()
                    .map(|d| {
                        (
                            d.identifier.clone(),
                            DownloadOutcome::Error(format!("http client: {e}")),
                        )
                    })
                    .collect()
            }
        };

        let work = |doc: &Document| {
            (
                doc.identifier.clone(),
                self.download_one(&client, doc, skip_existing),
            )
        };

        if workers > 0 {
            match ThreadPoolBuilder::new().num_threads(workers).build() {
                Ok(pool) => pool.install(|| docs.par_iter().map(work).collect()),
                Err(_) => docs.par_iter().map(work).collect(),
            }
        } else {
            docs.par_iter().map(work).collect()
        }
    }

    fn download_one(
        &self,
        client: &Client,
        doc: &Document,
        skip_existing: bool,
    ) -> DownloadOutcome {
        let target = self.dl_root.join(doc.filename());
        if skip_existing && is_existing_pdf(&target) {
            return DownloadOutcome::Downloaded(target);
        }
        if let Some(parent) = target.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                return DownloadOutcome::Error(format!("cannot create {}: {e}", parent.display()));
            }
        }
        let url = blob_url(doc);
        match fetch_with_retry(client, &url, &target) {
            Ok(()) => DownloadOutcome::Downloaded(target),
            Err(e) => DownloadOutcome::Error(e),
        }
    }
}

/// GET the blob, retrying transport errors / 429 / 5xx with 2s/5s backoff.
/// 404 (and other non-retryable 4xx) returns immediately.
fn fetch_with_retry(client: &Client, url: &str, target: &Path) -> Result<(), String> {
    let attempts = BACKOFF_SECS.len() + 1;
    let mut last_err = String::new();
    for attempt in 0..attempts {
        if attempt > 0 {
            thread::sleep(Duration::from_secs(BACKOFF_SECS[attempt - 1]));
        }
        match client.get(url).send() {
            Err(e) => {
                last_err = format!("transport error: {e}");
                continue;
            }
            Ok(resp) => {
                let status = resp.status();
                if status == StatusCode::NOT_FOUND {
                    return Err(format!("HTTP 404 (blob missing, not retried): {url}"));
                }
                if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
                    last_err = format!("HTTP {status}");
                    continue;
                }
                if !status.is_success() {
                    return Err(format!("HTTP {status}: {url}"));
                }
                let mut bytes = Vec::new();
                let mut reader = resp;
                if let Err(e) = reader.read_to_end(&mut bytes) {
                    last_err = format!("body read error: {e}");
                    continue;
                }
                if !bytes.starts_with(b"%PDF") {
                    return Err(format!(
                        "response is not a PDF ({} bytes): {url}",
                        bytes.len()
                    ));
                }
                if let Err(e) = write_atomic(target, &bytes) {
                    return Err(format!("write {}: {e}", target.display()));
                }
                return Ok(());
            }
        }
    }
    Err(format!("gave up after {attempts} attempts: {}", last_err))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dl_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p =
            std::env::temp_dir().join(format!("arxivvs_dl_{tag}_{}_{nanos}", std::process::id()));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn legacy_identifier_blob_name_url_and_filename() {
        let d = Document::arxiv("math/0211052v1");
        assert_eq!(d.gcloud_blob_name(), "arxiv/math/pdf/0211/0211052v1.pdf");
        assert_eq!(d.filename(), "math/0211052v1.pdf");
        assert_eq!(
            blob_url(&d),
            "https://storage.googleapis.com/arxiv-dataset/arxiv/math/pdf/0211/0211052v1.pdf"
        );
        assert_eq!(
            encode_blob_path(&d.gcloud_blob_name()),
            "arxiv/math/pdf/0211/0211052v1.pdf"
        );
    }

    #[test]
    fn new_style_identifier_blob_name_url_and_filename() {
        let d = Document::arxiv("2401.00001v1");
        assert_eq!(
            d.gcloud_blob_name(),
            "arxiv/arxiv/pdf/2401/2401.00001v1.pdf"
        );
        assert_eq!(d.filename(), "2401.00001v1.pdf");
        assert_eq!(
            blob_url(&d),
            "https://storage.googleapis.com/arxiv-dataset/arxiv/arxiv/pdf/2401/2401.00001v1.pdf"
        );
    }

    #[test]
    fn url_encoding_keeps_slashes_escapes_other_bytes() {
        assert_eq!(
            encode_blob_path("arxiv/math/pdf/0211/a b%v1.pdf"),
            "arxiv/math/pdf/0211/a%20b%25v1.pdf"
        );
        // unreserved chars pass through untouched
        assert_eq!(encode_blob_path("a-b_c.d~e/1.2"), "a-b_c.d~e/1.2");
    }

    #[test]
    fn skip_existing_rejects_non_pdf_but_accepts_pdf_magic() {
        let dir = temp_dl_dir("skipmagic");
        let not_pdf = dir.join("junk.pdf");
        fs::write(&not_pdf, b"<!doctype html>").unwrap();
        assert!(!is_existing_pdf(&not_pdf));
        let empty = dir.join("empty.pdf");
        fs::write(&empty, b"").unwrap();
        assert!(!is_existing_pdf(&empty));
        let real = dir.join("real.pdf");
        fs::write(&real, b"%PDF-1.7\n...").unwrap();
        assert!(is_existing_pdf(&real));
        assert!(!is_existing_pdf(&dir.join("missing.pdf")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn live_download_legacy_math_blob() {
        let root = temp_dl_dir("legacy");
        let dl = Downloader::new(&root);
        let res = dl.download_all(&[Document::arxiv("math/0211052v1")], 1, false);
        assert_eq!(res.len(), 1);
        let path = match &res[0].1 {
            DownloadOutcome::Downloaded(p) => p.clone(),
            other => panic!(
                "expected Downloaded, got {other:?} (identifier {})",
                res[0].0
            ),
        };
        assert_eq!(path, root.join("math/0211052v1.pdf"));
        let bytes = fs::read(&path).unwrap();
        assert!(bytes.starts_with(b"%PDF"), "legacy blob is not a PDF");
        assert!(bytes.len() > 10_000);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn live_download_new_style_blob() {
        let root = temp_dl_dir("newstyle");
        let dl = Downloader::new(&root);
        let res = dl.download_all(&[Document::arxiv("2401.00001v1")], 2, true);
        assert_eq!(res.len(), 1);
        let path = match &res[0].1 {
            DownloadOutcome::Downloaded(p) => p.clone(),
            other => panic!(
                "expected Downloaded, got {other:?} (identifier {})",
                res[0].0
            ),
        };
        assert_eq!(path, root.join("2401.00001v1.pdf"));
        let bytes = fs::read(&path).unwrap();
        assert!(bytes.starts_with(b"%PDF"), "new-style blob is not a PDF");
        let _ = fs::remove_dir_all(&root);
    }
}
