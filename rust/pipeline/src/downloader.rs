//! GCS downloader for the public arxiv-dataset bucket — parity with
//! documents/arxiv/arxiv_downloader.py, optimized.
//!
//! CONTRACT (do not change signatures):
//! - Downloads via anonymous HTTPS:
//!     https://storage.googleapis.com/arxiv-dataset/<url-encoded blob name>
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
use std::path::PathBuf;

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
        todo!("implemented by downloader module")
    }

    /// Download all documents; returns one outcome per input document,
    /// same order.
    pub fn download_all(
        &self,
        docs: &[Document],
        workers: usize,
        skip_existing: bool,
    ) -> Vec<(String, DownloadOutcome)> {
        todo!("implemented by downloader module")
    }
}
