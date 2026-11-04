//! PostgreSQL layer for the Rust pipeline. Fresh schema design (the old DB
//! was wiped; no back-compat debt). Deliberate changes vs the Python layer,
//! each defensible:
//! - doc_states with PRIMARY KEY (document_id, model_id): the Python layer
//!   had no uniqueness, so concurrent runs could double-insert rows.
//! - state/error_message as plain TEXT + persisted error_message: the old
//!   layer lost why a document failed.
//! - ON DELETE CASCADE everywhere: dropping a model's rows can't strand
//!   state rows.
//! - embeddings table created WITH its HNSW cosine index (empty-table cost
//!   is ~free; the old flow needed a separate indexing tool).
//! - query sets hnsw.ef_search INSIDE a transaction (SET LOCAL outside a
//!   transaction is a silent no-op — a real bug in the Python layer).
//! - BIGINT document ids (multi-million-row arXiv corpus).

use crate::common::{Document, DocumentType, State};
use pgvector::HalfVector;
use postgres::{Client, NoTls};
use std::collections::HashMap;

/// (identifier, pdf_type, page_index, cosine_distance, chunk_text)
pub type QueryHit = (String, DocumentType, i32, f64, String);

pub struct ModelRow {
    pub id: i32,
    pub name: String,
    pub embedding_dim: i32,
    pub batch_size: i32,
    pub chunk_size: i32,
    pub chunk_overlap: i32,
    pub document_prefix: String,
    pub query_prefix: String,
}

pub struct Db {
    pub client: Client,
    pub model: Option<ModelRow>,
}

fn safe_table(model_name: &str) -> String {
    let t = model_name.replace('/', "-");
    assert!(
        t.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.'),
        "unsafe model name for table identifier: {model_name}"
    );
    t
}

pub fn embeddings_table(model_name: &str) -> String {
    format!("embeddings_{}", safe_table(model_name))
}

impl Db {
    pub fn connect(db_url: &str) -> Result<Db, String> {
        let client = Client::connect(db_url, NoTls).map_err(|e| format!("db connect: {e}"))?;
        let mut db = Db {
            client,
            model: None,
        };
        db.ensure_schema()?;
        Ok(db)
    }

    pub fn ensure_schema(&mut self) -> Result<(), String> {
        self.client
            .batch_execute(
                "CREATE TABLE IF NOT EXISTS documents (
                     id BIGSERIAL PRIMARY KEY,
                     identifier TEXT UNIQUE NOT NULL,
                     pdf_type TEXT NOT NULL
                 );
                 CREATE TABLE IF NOT EXISTS models (
                     id SERIAL PRIMARY KEY,
                     name TEXT UNIQUE NOT NULL,
                     embedding_dim INT NOT NULL,
                     batch_size INT NOT NULL,
                     chunk_size INT NOT NULL,
                     chunk_overlap INT NOT NULL,
                     document_prefix TEXT NOT NULL DEFAULT '',
                     query_prefix TEXT NOT NULL DEFAULT '',
                     base_url TEXT
                 );
                 CREATE TABLE IF NOT EXISTS doc_states (
                     document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                     model_id INT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
                     state TEXT NOT NULL,
                     error_message TEXT,
                     updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                     PRIMARY KEY (document_id, model_id)
                 );
                 CREATE INDEX IF NOT EXISTS idx_doc_states_model_state
                     ON doc_states (model_id, state);",
            )
            .map_err(|e| format!("schema: {e}"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// Insert-or-load the model row. Existing rows win on params (the
    /// stored settings define the vector space); mismatches warn loudly.
    pub fn existing_model_names(&mut self) -> Vec<String> {
        self.client
            .query("SELECT name FROM models ORDER BY name", &[])
            .unwrap_or_default()
            .iter()
            .map(|r| r.get("name"))
            .collect::<Vec<String>>()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn get_or_create_model(
        &mut self,
        name: &str,
        embedding_dim: i32,
        batch_size: i32,
        chunk_size: i32,
        chunk_overlap: i32,
        document_prefix: &str,
        query_prefix: &str,
        base_url: Option<&str>,
    ) -> Result<(), String> {
        self.client
            .execute(
                "INSERT INTO models (name, embedding_dim, batch_size, chunk_size, chunk_overlap,
                                     document_prefix, query_prefix, base_url)
                 VALUES ($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT (name) DO NOTHING",
                &[
                    &name,
                    &embedding_dim,
                    &batch_size,
                    &chunk_size,
                    &chunk_overlap,
                    &document_prefix,
                    &query_prefix,
                    &base_url,
                ],
            )
            .map_err(|e| format!("model insert: {e}"))?;
        let row = self
            .client
            .query_one(
                "SELECT id, name, embedding_dim, batch_size, chunk_size, chunk_overlap,
                        document_prefix, query_prefix FROM models WHERE name = $1",
                &[&name],
            )
            .map_err(|e| format!("model select: {e}"))?;
        let m = ModelRow {
            id: row.get("id"),
            name: row.get("name"),
            embedding_dim: row.get("embedding_dim"),
            batch_size: row.get("batch_size"),
            chunk_size: row.get("chunk_size"),
            chunk_overlap: row.get("chunk_overlap"),
            document_prefix: row.get("document_prefix"),
            query_prefix: row.get("query_prefix"),
        };
        if m.embedding_dim != embedding_dim {
            eprintln!(
                "warning: model '{name}' exists with dim {}, requested {} — using stored",
                m.embedding_dim, embedding_dim
            );
        }
        if m.document_prefix != document_prefix || m.query_prefix != query_prefix {
            eprintln!(
                "warning: model '{name}' stored prefixes ({:?}/{:?}) differ from requested ({:?}/{:?}) — using stored",
                m.document_prefix, m.query_prefix, document_prefix, query_prefix
            );
        }
        self.model = Some(m);
        Ok(())
    }

    pub fn model(&self) -> &ModelRow {
        self.model.as_ref().expect("model not initialized")
    }

    pub fn create_embedding_table(&mut self) -> Result<(), String> {
        let m = self.model();
        let table = embeddings_table(&m.name);
        self.client
            .batch_execute(&format!(
                "CREATE TABLE IF NOT EXISTS {table} (
                     document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                     page_index SMALLINT NOT NULL,
                     chunk_index SMALLINT NOT NULL,
                     embedding HALFVEC({}) NOT NULL,
                     text TEXT NOT NULL,
                     PRIMARY KEY (document_id, page_index, chunk_index)
                 );
                 CREATE INDEX IF NOT EXISTS ix_{table}_doc ON {table} (document_id);
                 CREATE INDEX IF NOT EXISTS ix_{table}_hnsw
                     ON {table} USING hnsw (embedding vector_cosine_ops);",
                m.embedding_dim
            ))
            .map_err(|e| format!("create embedding table: {e}"))?;
        Ok(())
    }

    /// Upsert documents; returns nothing but warms nothing (ids resolved
    /// on demand by identifier).
    pub fn add_documents(&mut self, docs: &[Document]) -> Result<(), String> {
        for chunk in docs.chunks(1000) {
            let mut tx = self.client.transaction().map_err(|e| e.to_string())?;
            for d in chunk {
                tx.execute(
                    "INSERT INTO documents (identifier, pdf_type) VALUES ($1,$2)
                     ON CONFLICT (identifier) DO NOTHING",
                    &[&d.identifier, &d.pdf_type.db_name()],
                )
                .map_err(|e| format!("add_documents: {e}"))?;
            }
            tx.commit().map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Create 'missing' state rows for documents of this model that have none.
    pub fn add_missing_states(&mut self) -> Result<(), String> {
        let m = self.model().id;
        self.client
            .execute(
                "INSERT INTO doc_states (document_id, model_id, state)
                 SELECT d.id, $1, 'missing' FROM documents d
                 WHERE NOT EXISTS (
                    SELECT 1 FROM doc_states s WHERE s.document_id = d.id AND s.model_id = $1)
                 ON CONFLICT DO NOTHING",
                &[&m],
            )
            .map_err(|e| format!("add_missing_states: {e}"))?;
        Ok(())
    }

    pub fn get_missing(&mut self, limit: i64) -> Result<Vec<Document>, String> {
        let m = self.model().id;
        let rows = self
            .client
            .query(
                "SELECT d.identifier, d.pdf_type FROM doc_states s
                 JOIN documents d ON d.id = s.document_id
                 WHERE s.model_id = $1 AND s.state = 'missing'
                 ORDER BY d.id LIMIT $2",
                &[&m, &limit],
            )
            .map_err(|e| e.to_string())?;
        Ok(rows
            .iter()
            .map(|r| Document {
                identifier: r.get("identifier"),
                pdf_type: DocumentType::from_db_name(r.get("pdf_type")),
            })
            .collect())
    }

    pub fn set_states(
        &mut self,
        identifiers: &[String],
        state: State,
        error: Option<&str>,
    ) -> Result<(), String> {
        if identifiers.is_empty() {
            return Ok(());
        }
        let m = self.model().id;
        for chunk in identifiers.chunks(5000) {
            self.client
                .execute(
                    "UPDATE doc_states SET state = $3, error_message = $4, updated_at = now()
                     WHERE model_id = $1 AND document_id IN
                       (SELECT id FROM documents WHERE identifier = ANY($2))",
                    &[&m, &chunk, &state.db_name(), &error],
                )
                .map_err(|e| format!("set_states: {e}"))?;
        }
        Ok(())
    }

    pub fn state_counts(&mut self) -> Result<Vec<(String, i64)>, String> {
        let m = self.model().id;
        let rows = self
            .client
            .query(
                "SELECT state, count(*) AS n FROM doc_states WHERE model_id = $1 GROUP BY state",
                &[&m],
            )
            .map_err(|e| e.to_string())?;
        Ok(rows.iter().map(|r| (r.get("state"), r.get("n"))).collect())
    }

    pub fn add_embeddings(
        &mut self,
        ident_map: &HashMap<String, i64>,
        rows: &[(String, i32, i32, Vec<f32>, String)],
    ) -> Result<(), String> {
        let table = embeddings_table(&self.model().name);
        for chunk in rows.chunks(500) {
            let mut tx = self.client.transaction().map_err(|e| e.to_string())?;
            for (ident, page, chunk_i, vec, text) in chunk {
                let doc_id = match ident_map.get(ident) {
                    Some(id) => *id,
                    None => continue,
                };
                let v = HalfVector::from_f32_slice(vec);
                tx.execute(
                    &format!(
                        "INSERT INTO {table} (document_id, page_index, chunk_index, embedding, text)
                         VALUES ($1,$2,$3,$4,$5) ON CONFLICT DO NOTHING"
                    ),
                    &[&doc_id, page, chunk_i, &v, text],
                )
                .map_err(|e| format!("add_embeddings: {e}"))?;
            }
            tx.commit().map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn query_embeddings(&mut self, query: &[f32], top_k: i64) -> Result<Vec<QueryHit>, String> {
        let table = embeddings_table(&self.model().name);
        let q = HalfVector::from_f32_slice(query);
        let ef = (top_k as usize * 2).max(100);
        let mut tx = self.client.transaction().map_err(|e| e.to_string())?;
        tx.execute(&format!("SET LOCAL hnsw.ef_search = {ef}"), &[])
            .map_err(|e| e.to_string())?;
        let rows = tx
            .query(
                &format!(
                    "SELECT d.identifier, d.pdf_type, e.page_index,
                            e.embedding <=> $1 AS distance, e.text
                     FROM {table} e JOIN documents d ON d.id = e.document_id
                     ORDER BY e.embedding <=> $1 LIMIT $2"
                ),
                &[&q, &top_k],
            )
            .map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(rows
            .iter()
            .map(|r| {
                (
                    r.get("identifier"),
                    DocumentType::from_db_name(r.get("pdf_type")),
                    r.get("page_index"),
                    r.get("distance"),
                    r.get("text"),
                )
            })
            .collect())
    }

    pub fn query_avg_by_doc(
        &mut self,
        query: &[f32],
        top_k: i64,
    ) -> Result<Vec<(String, DocumentType, f64)>, String> {
        let table = embeddings_table(&self.model().name);
        let q = HalfVector::from_f32_slice(query);
        let ef = (top_k as usize * 2).max(100);
        let mut tx = self.client.transaction().map_err(|e| e.to_string())?;
        tx.execute(&format!("SET LOCAL hnsw.ef_search = {ef}"), &[])
            .map_err(|e| e.to_string())?;
        let rows = tx
            .query(
                &format!(
                    "SELECT d.identifier, d.pdf_type,
                            avg(e.embedding <=> $1) AS distance
                     FROM {table} e JOIN documents d ON d.id = e.document_id
                     GROUP BY d.id, d.identifier, d.pdf_type
                     ORDER BY distance LIMIT $2"
                ),
                &[&q, &top_k],
            )
            .map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(rows
            .iter()
            .map(|r| {
                (
                    r.get("identifier"),
                    DocumentType::from_db_name(r.get("pdf_type")),
                    r.get("distance"),
                )
            })
            .collect())
    }

    /// Drop this model's embeddings + states (documents/models kept).
    pub fn flush_model(&mut self) -> Result<(), String> {
        let m = self.model().id;
        let table = embeddings_table(&self.model().name);
        self.client
            .batch_execute(&format!(
                "DROP TABLE IF EXISTS {table};
                 DELETE FROM doc_states WHERE model_id = {m};"
            ))
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn flush_errors(&mut self) -> Result<(), String> {
        let m = self.model().id;
        self.client
            .execute(
                "UPDATE doc_states SET state = 'missing', error_message = NULL, updated_at = now()
                 WHERE model_id = $1 AND state IN ('download_error','split_error')",
                &[&m],
            )
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// identifier -> doc id map for a set of identifiers (single query).
    pub fn warm_ids(&mut self, identifiers: &[String]) -> Result<HashMap<String, i64>, String> {
        let mut map = HashMap::new();
        for chunk in identifiers.chunks(10000) {
            let rows = self
                .client
                .query(
                    "SELECT id, identifier FROM documents WHERE identifier = ANY($1)",
                    &[&chunk],
                )
                .map_err(|e| e.to_string())?;
            for r in rows {
                let id: i64 = r.get("id");
                let ident: String = r.get("identifier");
                map.insert(ident, id);
            }
        }
        Ok(map)
    }
}
