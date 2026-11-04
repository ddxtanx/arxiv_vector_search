//! arXiv metadata snapshot seeding — parity with arxiv_updater.py, optimized.
//!
//! CONTRACT (do not change signatures):
//! - Dataset: Cornell-University/arxiv, file arxiv-metadata-oai-snapshot.json
//!   (JSON lines). Download via Kaggle API:
//!     GET https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv
//!   with basic auth (username, key). The response is a ZIP; extract ONLY
//!   the arxiv-metadata-oai-snapshot.json entry, streaming to
//!   <cache_dir>/arxiv-metadata-oai-snapshot.json (skip download if that
//!   cache file already exists).
//!   Credentials: env KAGGLE_USERNAME/KAGGLE_KEY, else parse
//!   ~/.kaggle/access_token JSON {"username":..,"key":..}. Error if absent.
//! - Parse each line with serde_json; select identifier =
//!   line["id"] + versions[-1]["version"] (versions is an array of objects
//!   with "version"; if versions missing/empty, use id + "v1").
//! - Filter: categories field CONTAINS the substring "math" (python used
//!   `in` on the string — replicate exactly, NOT word-boundary), and
//!   comments (may be null/absent) does not contain "withdrawn"
//!   (case-insensitive).
//! - Returns Vec<Document> (pdf_type Arxiv) of ALL matching identifiers.
//! - Streaming parse: do NOT load whole file into memory (it is ~10 GB
//!   uncompressed; read line-by-line via BufReader).
//!
//! Tests (this file): PURE parser tests with synthetic JSON lines (no
//! network, no kaggle): withdrawn filtered out, categories substring math
//! (e.g. "math.GT cs.CG" matches, "physics.math" matches by substring too),
//! versions[-1] picks v3 of [v1,v2,v3], missing versions -> v1.

use crate::common::Document;
use std::path::Path;

pub fn kaggle_credentials() -> Result<(String, String), String> {
    todo!("implemented by metadata module")
}

pub fn seed_math_documents(cache_dir: &Path) -> Result<Vec<Document>, String> {
    todo!("implemented by metadata module")
}

/// Parse one JSON-lines snapshot record -> Some(identifier) or None (filtered).
pub fn parse_snapshot_line(line: &str) -> Option<String> {
    todo!("implemented by metadata module")
}
