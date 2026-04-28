from sqlalchemy.orm import aliased
from sqlalchemy import exists
from sqlalchemy import literal
from sqlalchemy import func
from sqlalchemy.orm import joinedload
import numpy as np
from .tables import (
    Model,
    Document,
    EmbeddingMetadata,
    create_embedding_table,
    Base,
    EmbeddingState,
)
from arxiv_vector_search.processors.embedder import Embedder, Embedding
from arxiv_vector_search.documents import Document as PdfDocument, DocumentType
from arxiv_vector_search.documents.arxiv import ArxivDocument
from arxiv_vector_search.documents.doi import DOIDocument
from arxiv_vector_search.documents.url import URLDocument
from sqlalchemy import create_engine, Engine, delete, update, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from sqlalchemy.pool import QueuePool
from pgvector.sqlalchemy import avg


class QueryResult:
    document: PdfDocument
    page_index: int
    distance: float

    def __init__(self, document: PdfDocument, page_index: int, distance: float):
        self.document = document
        self.page_index = page_index
        self.distance = distance

    def get_url(self) -> str:
        return self.document.get_url()


class Database:
    engine: Engine
    model_to_embedding_table: dict[str, type]
    ident_to_doc_id_cache: dict[str, int]

    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=10,
            connect_args={"prepare_threshold": None},
        )
        self.model_to_embedding_table = {}
        Base.metadata.create_all(self.engine)
        with Session(self.engine) as session:
            docs = session.execute(select(Document)).scalars().all()
            self.ident_to_doc_id_cache = {doc.identifier: doc.id for doc in docs}

    def add_model(self, embedder: Embedder):
        with Session(self.engine) as session:
            session.execute(
                insert(Model)
                .values(
                    name=embedder.get_model_name(),
                    batch_size=embedder.get_batch_size(),
                    embedding_dim=embedder.get_embedding_dim(),
                )
                .on_conflict_do_nothing(index_elements=["name"])
            )
            session.commit()

    def add_document(self, document: PdfDocument):
        with Session(self.engine) as session:
            doc_id = session.execute(
                insert(Document)
                .values(identifier=document.identifier, pdf_type=document.document_type)
                .on_conflict_do_nothing(index_elements=["identifier"])
                .returning(Document.id)
            ).scalar_one_or_none()
            if doc_id is not None:
                self.ident_to_doc_id_cache[document.identifier] = doc_id
            session.commit()

    def add_documents(self, documents: list[PdfDocument]):
        with Session(self.engine) as session:
            values = [
                {"identifier": doc.identifier, "pdf_type": doc.document_type}
                for doc in documents
            ]
            docs = session.execute(
                insert(Document)
                .on_conflict_do_nothing(index_elements=["identifier"])
                .returning(Document.identifier, Document.id),
                values,
            ).all()
            for identifier, doc_id in docs:
                self.ident_to_doc_id_cache[identifier] = int(doc_id)
            session.commit()

    def add_embedding_metadata(
        self, document: PdfDocument, embedder: Embedder, state: EmbeddingState
    ) -> EmbeddingMetadata:
        with Session(self.engine) as session:
            model = session.execute(
                select(Model).where(Model.name == embedder.get_model_name())
            ).scalar_one()
            doc = session.execute(
                select(Document).where(Document.identifier == document.identifier)
            ).scalar_one()
            embedding_metadata = EmbeddingMetadata(
                document_id=doc.id, model_id=model.id, state=state
            )
            session.add(embedding_metadata)
            session.commit()
        return embedding_metadata

    def get_documents(self) -> list[PdfDocument]:
        with Session(self.engine) as session:
            docs = session.execute(select(Document)).scalars().all()
            pdf_docs = []
            for doc in docs:
                if doc.pdf_type == DocumentType.ARXIV:
                    pdf_docs.append(ArxivDocument(doc.identifier))
                elif doc.pdf_type == DocumentType.URL:
                    pdf_docs.append(URLDocument(doc.identifier))
                elif doc.pdf_type == DocumentType.DOI:
                    pdf_docs.append(DOIDocument(doc.identifier))
            return pdf_docs

    def create_embedding_table_for_model(self, embedder: Embedder):
        table = create_embedding_table(
            embedder.get_model_name(), embedder.get_embedding_dim()
        )
        Base.metadata.create_all(self.engine)
        self.model_to_embedding_table[embedder.get_model_name()] = table

    def get_models(self) -> list[Model]:
        with Session(self.engine) as session:
            return session.execute(select(Model)).scalars().all()

    def get_embedding_metadata_by_model(
        self, model: Embedder
    ) -> list[EmbeddingMetadata]:
        with Session(self.engine) as session:
            model_record = session.execute(
                select(Model).where(Model.name == model.get_model_name())
            ).scalar_one()
            return model_record.embedding_metadatas

    def get_missing_embeddings_for_model(
        self, model: Embedder, limit: int, offset: int
    ) -> list[PdfDocument]:
        with Session(self.engine) as session:
            model_record = session.execute(
                select(Model).where(Model.name == model.get_model_name())
            ).scalar_one()
            missing_metadata = session.execute(
                select(EmbeddingMetadata, Document)
                .options(joinedload(EmbeddingMetadata.document))
                .join(Document, EmbeddingMetadata.document_id == Document.id)
                .where(
                    EmbeddingMetadata.model_id == model_record.id,
                    EmbeddingMetadata.state == EmbeddingState.MISSING,
                )
                .limit(limit)
                .offset(offset)
            ).scalars()
            missing_docs = []
            for metadata in missing_metadata:
                doc = metadata.document
                if doc.pdf_type == DocumentType.ARXIV:
                    missing_docs.append(ArxivDocument(doc.identifier))
                elif doc.pdf_type == DocumentType.URL:
                    missing_docs.append(URLDocument(doc.identifier))
                elif doc.pdf_type == DocumentType.DOI:
                    missing_docs.append(DOIDocument(doc.identifier))
            return missing_docs

    def add_embeddings(self, embeddings: list[Embedding], embedder: Embedder):
        if embedder.get_model_name() not in self.model_to_embedding_table:
            self.create_embedding_table_for_model(embedder)
        EmbeddingType = self.model_to_embedding_table.get(embedder.get_model_name())
        for embedding in embeddings:
            if embedding["document_id"] not in self.ident_to_doc_id_cache:
                raise ValueError(
                    f"Document identifier {embedding['document_id']} not found in database"
                )
            embedding["document_id"] = self.ident_to_doc_id_cache[
                embedding["document_id"]
            ]
        with Session(self.engine) as session:
            session.execute(
                insert(EmbeddingType),
                embeddings,
            )
            session.commit()

    def update_embedding_metadata_states_by_idents(
        self, embedder: Embedder, doc_idents: list[str], state: EmbeddingState
    ):
        with Session(self.engine) as session:
            model_record = session.execute(
                select(Model).where(Model.name == embedder.get_model_name())
            ).scalar_one()
            model_id = model_record.id
            doc_ids = set(self.ident_to_doc_id_cache.get(ident) for ident in doc_idents)
            session.execute(
                update(EmbeddingMetadata)
                .where(EmbeddingMetadata.model_id == model_id)
                .where(EmbeddingMetadata.document_id.in_(doc_ids))
                .values(state=state)
            )
            session.commit()

    def update_embedding_metadata_states(
        self, embedder: Embedder, documents: list[PdfDocument], state: EmbeddingState
    ):
        with Session(self.engine) as session:
            model_record = session.execute(
                select(Model).where(Model.name == embedder.get_model_name())
            ).scalar_one()
            model_id = model_record.id
            doc_ids = set(
                self.ident_to_doc_id_cache.get(doc.identifier) for doc in documents
            )
            session.execute(
                update(EmbeddingMetadata)
                .where(EmbeddingMetadata.model_id == model_id)
                .where(EmbeddingMetadata.document_id.in_(doc_ids))
                .values(state=state)
            )
            session.commit()

    def delete_embeddings_for_document(self, document: PdfDocument):
        doc_id = self.ident_to_doc_id_cache.get(document.identifier)
        with Session(self.engine) as session:
            session.execute(
                update(EmbeddingMetadata)
                .where(EmbeddingMetadata.document_id == doc_id)
                .values(state=EmbeddingState.MISSING)
            )
            for table in self.model_to_embedding_table.values():
                session.execute(delete(table).where(table.document_id == doc_id))
            session.commit()

    def delete_embeddings_for_model(self, embedder: Embedder):
        with Session(self.engine) as session:
            model_record = session.execute(
                select(Model).where(Model.name == embedder.get_model_name())
            ).scalar_one()
            model_id = model_record.id
            session.execute(
                update(EmbeddingMetadata)
                .where(EmbeddingMetadata.model_id == model_id)
                .values(state=EmbeddingState.MISSING)
            )
            embedding_table = self.model_to_embedding_table.get(
                embedder.get_model_name()
            )
            if embedding_table:
                session.execute(delete(embedding_table))
            session.commit()

    def flush_errors_for_model(self, embedder: Embedder):
        with Session(self.engine) as session:
            model_record = session.execute(
                select(Model).where(Model.name == embedder.get_model_name())
            ).scalar_one()
            model_id = model_record.id
            session.execute(
                update(EmbeddingMetadata)
                .where(EmbeddingMetadata.model_id == model_id)
                .where(
                    (EmbeddingMetadata.state == EmbeddingState.DOWNLOAD_ERROR)
                    | (EmbeddingMetadata.state == EmbeddingState.SPLIT_ERROR)
                )
                .values(state=EmbeddingState.MISSING)
            )
            session.commit()

    def add_missing_metadata(self, embedder: Embedder):
        with Session(self.engine) as session:
            model_record = session.execute(
                select(Model).where(Model.name == embedder.get_model_name())
            ).scalar_one()
            model_id = model_record.id
            # get documents that have no embedding metadata entries with document and model ids corresponding to given
            missing_docs = (
                session.execute(
                    select(Document).where(
                        select(func.count(EmbeddingMetadata.id))
                        .where(
                            EmbeddingMetadata.document_id == Document.id,
                            EmbeddingMetadata.model_id == model_id,
                        )
                        .scalar_subquery()
                        == 0
                    )
                )
                .scalars()
                .all()
            )
            new_embedding_metadata = [
                {
                    "document_id": doc.id,
                    "model_id": model_id,
                    "state": EmbeddingState.MISSING,
                }
                for doc in missing_docs
            ]
            if new_embedding_metadata:
                session.execute(insert(EmbeddingMetadata), new_embedding_metadata)
            session.commit()

    def query_embeddings(
        self,
        embedder: Embedder,
        query_embedding: np.ndarray[tuple[int], np.dtype[np.float16]],
        top_k: int,
    ) -> list[QueryResult]:
        if embedder.get_model_name() not in self.model_to_embedding_table:
            self.create_embedding_table_for_model(embedder)
        EmbeddingType = self.model_to_embedding_table.get(embedder.get_model_name())
        embedding_col = EmbeddingType.embedding.cosine_distance(query_embedding).label(
            "distance"
        )
        with Session(self.engine) as session:
            print("Querying database for similar embeddings...")
            results = session.execute(
                select(EmbeddingType, Document, embedding_col)
                .join(EmbeddingType.document)
                .where(EmbeddingType.chunk_index != -1)
                .order_by(embedding_col)
                .limit(top_k)
                .execution_options(readonly=True)
            )
            query_results = []
            for embedding, doc, distance in results:
                document = None
                if doc.pdf_type == DocumentType.ARXIV:
                    document = ArxivDocument(doc.identifier)
                elif doc.pdf_type == DocumentType.URL:
                    document = URLDocument(doc.identifier)
                elif doc.pdf_type == DocumentType.DOI:
                    document = DOIDocument(doc.identifier)
                query_results.append(
                    QueryResult(
                        document=document,
                        page_index=embedding.page_index,
                        distance=distance,
                    )
                )
            return query_results

    def add_missing_doc_avgs_for_model(self, embedder: Embedder):
        EmbeddingType = self.model_to_embedding_table.get(embedder.get_model_name())
        with Session(self.engine) as session:
            selected_alias = aliased(EmbeddingType, name="e1")
            avg_col = avg(selected_alias.embedding).label("avg_embedding")
            filter_alias = aliased(EmbeddingType, name="e2")
            select_stmt = (
                select(
                    selected_alias.document_id,
                    avg_col,
                    literal(-1).label("chunk_index"),
                    literal(0).label("page_index"),
                )
                .where(
                    ~exists().where(
                        (filter_alias.document_id == selected_alias.document_id)
                        & (filter_alias.chunk_index == -1)
                    )
                )
                .group_by(selected_alias.document_id)
            )

            insert_stmt = insert(EmbeddingType).from_select(
                ["document_id", "embedding", "chunk_index", "page_index"], select_stmt
            )
            session.execute(insert_stmt)
            session.commit()

    def query_embeddings_avg(
        self,
        embedder: Embedder,
        query_embedding: np.ndarray[tuple[int], np.dtype[np.float16]],
        top_k: int,
    ) -> list[QueryResult]:
        if embedder.get_model_name() not in self.model_to_embedding_table:
            self.create_embedding_table_for_model(embedder)
        self.add_missing_doc_avgs_for_model(embedder)
        EmbeddingType = self.model_to_embedding_table.get(embedder.get_model_name())
        embedding_col = EmbeddingType.embedding.cosine_distance(query_embedding).label(
            "distance"
        )
        with Session(self.engine) as session:
            print("Querying database for similar document average embeddings...")
            results = session.execute(
                select(EmbeddingType, Document, embedding_col)
                .join(EmbeddingType.document)
                .where(EmbeddingType.chunk_index == -1)
                .order_by(embedding_col)
                .limit(top_k)
                .execution_options(readonly=True)
            )
            query_results = []
            for embedding, doc, distance in results:
                document = None
                if doc.pdf_type == DocumentType.ARXIV:
                    document = ArxivDocument(doc.identifier)
                elif doc.pdf_type == DocumentType.URL:
                    document = URLDocument(doc.identifier)
                elif doc.pdf_type == DocumentType.DOI:
                    document = DOIDocument(doc.identifier)
                query_results.append(
                    QueryResult(
                        document=document,
                        page_index=embedding.page_index,
                        distance=distance,
                    )
                )
            return query_results
