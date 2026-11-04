//! Shared types and identifier semantics — parity with documents/document.py
//! and documents/arxiv/arxiv_document.py.

use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DocumentType {
    Arxiv,
    Url,
    Doi,
}

impl DocumentType {
    /// SQLAlchemy Enum stores the *member name* in the DB.
    pub fn db_name(&self) -> &'static str {
        match self {
            DocumentType::Arxiv => "ARXIV",
            DocumentType::Url => "URL",
            DocumentType::Doi => "DOI",
        }
    }
    pub fn from_db_name(s: &str) -> DocumentType {
        match s {
            "ARXIV" => DocumentType::Arxiv,
            "URL" => DocumentType::Url,
            _ => DocumentType::Doi,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Document {
    pub identifier: String,
    pub pdf_type: DocumentType,
}

impl Document {
    pub fn arxiv(identifier: impl Into<String>) -> Document {
        Document {
            identifier: identifier.into(),
            pdf_type: DocumentType::Arxiv,
        }
    }

    pub fn url(&self) -> String {
        match self.pdf_type {
            DocumentType::Arxiv => format!("https://arxiv.org/pdf/{}.pdf", self.identifier),
            DocumentType::Url => self.identifier.clone(),
            DocumentType::Doi => format!("https://doi.org/{}", self.identifier),
        }
    }

    /// Parity with ArxivDocument.get_gcloud_blob_name (bucket arxiv-dataset).
    pub fn gcloud_blob_name(&self) -> String {
        if let Some(pos) = self.identifier.rfind('/') {
            let (subj, id) = (&self.identifier[..pos], &self.identifier[pos + 1..]);
            let year_month = &id[..4];
            format!("arxiv/{}/pdf/{}/{}.pdf", subj, year_month, id)
        } else {
            let year_month = self.identifier.split('.').next().unwrap_or("");
            format!("arxiv/arxiv/pdf/{}/{}.pdf", year_month, self.identifier)
        }
    }

    pub fn filename(&self) -> String {
        format!("{}.pdf", self.identifier)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    /// Row-creation default is written in SQL, but kept for completeness.
    #[allow(dead_code)]
    Missing,
    DownloadError,
    SplitError,
    Embedded,
}

impl State {
    pub fn db_name(&self) -> &'static str {
        match self {
            State::Missing => "MISSING",
            State::DownloadError => "DOWNLOAD_ERROR",
            State::SplitError => "SPLIT_ERROR",
            State::Embedded => "EMBEDDED",
        }
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.db_name())
    }
}
