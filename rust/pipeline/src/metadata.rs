//! arXiv metadata snapshot seeding — parity with arxiv_updater.py.
//!
//! Dataset: Cornell-University/arxiv, file arxiv-metadata-oai-snapshot.json
//! (JSON lines). Kaggle API basic-auth download endpoint returns a ZIP; we
//! stream-save the ZIP, extract ONLY that entry to
//! <cache_dir>/arxiv-metadata-oai-snapshot.json, then delete the ZIP. A
//! present cached JSON skips the download entirely.
//!
//! Parse semantics (EXACT parity with the Python updater):
//! - identifier = id + versions[-1]["version"] ("v1" if versions absent/empty)
//! - keep if categories CONTAINS the substring "math" (Python `in` on the
//!   string, NOT a word boundary)
//! - drop if comments (when present as a string) contains "withdrawn"
//!   (case-insensitive); missing/null comments pass.
//!
//! The file is ~10 GB uncompressed — always stream line by line.

use crate::common::Document;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

pub fn kaggle_credentials() -> Result<(String, String), String> {
    if let (Ok(u), Ok(k)) = (
        std::env::var("KAGGLE_USERNAME"),
        std::env::var("KAGGLE_KEY"),
    ) {
        if !u.is_empty() && !k.is_empty() {
            return Ok((u, k));
        }
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let path = Path::new(&home).join(".kaggle").join("access_token");
    if let Ok(s) = std::fs::read_to_string(&path) {
        if let Ok(v) = serde_json::from_str::<Value>(&s) {
            if let (Some(u), Some(k)) = (
                v.get("username").and_then(|x| x.as_str()),
                v.get("key").and_then(|x| x.as_str()),
            ) {
                return Ok((u.to_string(), k.to_string()));
            }
        }
    }
    Err(
        "no Kaggle credentials: set KAGGLE_USERNAME + KAGGLE_KEY or create ~/.kaggle/access_token"
            .to_string(),
    )
}

/// Parse one JSON-lines snapshot record -> Some(identifier) or None (filtered).
pub fn parse_snapshot_line(line: &str) -> Option<String> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let v: Value = serde_json::from_str(line).ok()?;
    let id = v.get("id")?.as_str()?;
    let categories = v.get("categories").and_then(|c| c.as_str()).unwrap_or("");
    if !categories.contains("math") {
        return None;
    }
    if let Some(comments) = v.get("comments").and_then(|c| c.as_str()) {
        if comments.to_lowercase().contains("withdrawn") {
            return None;
        }
    }
    let version = v
        .get("versions")
        .and_then(|vs| vs.as_array())
        .and_then(|vs| vs.last())
        .and_then(|last| last.get("version"))
        .and_then(|x| x.as_str())
        .unwrap_or("v1");
    Some(format!("{id}{version}"))
}

pub fn seed_math_documents(cache_dir: &Path) -> Result<Vec<Document>, String> {
    std::fs::create_dir_all(cache_dir).map_err(|e| format!("cache dir: {e}"))?;
    let cached_json = cache_dir.join("arxiv-metadata-oai-snapshot.json");
    if !cached_json.exists() {
        let (user, key) = kaggle_credentials()?;
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(3600))
            .build()
            .map_err(|e| e.to_string())?;
        println!(
            "Downloading Cornell-University/arxiv from Kaggle (multi-GB, one-time; \
             cached to {})...",
            cached_json.display()
        );
        let resp = client
            .get("https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv")
            .basic_auth(&user, Some(&key))
            .send()
            .map_err(|e| format!("kaggle download: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!(
                "kaggle download: status {} (check dataset consent at \
                 https://www.kaggle.com/datasets/Cornell-University/arxiv)",
                resp.status()
            ));
        }
        let zip_path = cache_dir.join("arxiv-dataset.zip");
        {
            let mut f = File::create(&zip_path).map_err(|e| e.to_string())?;
            let mut reader = resp;
            let mut buf = vec![0u8; 1 << 20];
            loop {
                let n = reader
                    .read(&mut buf)
                    .map_err(|e| format!("download: {e}"))?;
                if n == 0 {
                    break;
                }
                f.write_all(&buf[..n]).map_err(|e| e.to_string())?;
            }
        }
        // Extract only the metadata JSON from the ZIP.
        let zf = File::open(&zip_path).map_err(|e| e.to_string())?;
        let mut zip = zip::ZipArchive::new(BufReader::new(zf)).map_err(|e| e.to_string())?;
        let mut entry = zip
            .by_name("arxiv-metadata-oai-snapshot.json")
            .map_err(|e| format!("zip entry: {e}"))?;
        let mut out = File::create(&cached_json).map_err(|e| e.to_string())?;
        std::io::copy(&mut entry, &mut out).map_err(|e| e.to_string())?;
        drop(out);
        let _ = std::fs::remove_file(&zip_path);
    }
    let f = File::open(&cached_json).map_err(|e| format!("open snapshot: {e}"))?;
    let reader = BufReader::new(f);
    let mut docs = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        if let Some(ident) = parse_snapshot_line(&line) {
            docs.push(Document::arxiv(ident));
        }
    }
    Ok(docs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn math_substring_filter() {
        // categories substring semantics (python `in`)
        let a = r#"{"id":"2401.00001","categories":"math.GT cs.CG","versions":[{"version":"v1"},{"version":"v2"}]}"#;
        assert_eq!(parse_snapshot_line(a).as_deref(), Some("2401.00001v2"));
        let b = r#"{"id":"2401.00002","categories":"cs.LG","versions":[{"version":"v1"}]}"#;
        assert_eq!(parse_snapshot_line(b), None);
        // substring "math" inside another token still matches (parity)
        let c = r#"{"id":"x/1","categories":"foo.mathbar","versions":[{"version":"v1"}]}"#;
        assert!(parse_snapshot_line(c).is_some());
    }

    #[test]
    fn withdrawn_dropped_comments_maybe_null() {
        let a = r#"{"id":"1","categories":"math","comments":"Withdrawn by author","versions":[{"version":"v1"}]}"#;
        assert_eq!(parse_snapshot_line(a), None);
        let b = r#"{"id":"1","categories":"math","comments":null,"versions":[{"version":"v1"}]}"#;
        assert_eq!(parse_snapshot_line(b).as_deref(), Some("1v1"));
        let c = r#"{"id":"1","categories":"math","versions":[{"version":"v1"}]}"#;
        assert_eq!(parse_snapshot_line(c).as_deref(), Some("1v1"));
    }

    #[test]
    fn version_pick_last_and_fallback() {
        let a = r#"{"id":"math/0211052","categories":"math.AG","versions":[{"version":"v1"},{"version":"v2"},{"version":"v3"}]}"#;
        assert_eq!(parse_snapshot_line(a).as_deref(), Some("math/0211052v3"));
        let b = r#"{"id":"math/0211052","categories":"math.AG","versions":[]}"#;
        assert_eq!(parse_snapshot_line(b).as_deref(), Some("math/0211052v1"));
        let c = r#"{"id":"math/0211052","categories":"math.AG"}"#;
        assert_eq!(parse_snapshot_line(c).as_deref(), Some("math/0211052v1"));
    }

    #[test]
    fn blank_and_malformed_lines_skipped() {
        assert_eq!(parse_snapshot_line(""), None);
        assert_eq!(parse_snapshot_line("   "), None);
        assert_eq!(parse_snapshot_line("not json"), None);
        assert_eq!(parse_snapshot_line(r#"{"nope":1}"#), None);
    }
}
