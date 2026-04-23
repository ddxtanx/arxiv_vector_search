from sqlalchemy.orm import joinedload
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


class QueryResult:
    document: PdfDocument
    page_number: int
    distance: float

    def __init__(self, document: PdfDocument, page_number: int, distance: float):
        self.document = document
        self.page_number = page_number
        self.distance = distance

    def get_url(self) -> str:
        return self.document.get_url()


class Database:
    engine: Engine
    model_to_embedding_table: dict[str, type]

    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=10,
            connect_args={"prepare_threshold": None},
            insertmanyvalues_page_size=10000,
        )
        self.model_to_embedding_table = {}
        self.session = Session(self.engine)
        Base.metadata.create_all(self.engine)

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

    def get_identifier_to_doc_id(self, identifiers: set[str]) -> dict[str, int]:
        with Session(self.engine) as session:
            docs = (
                session.execute(
                    select(Document).where(Document.identifier.in_(identifiers))
                )
                .scalars()
                .all()
            )
            return {doc.identifier: doc.id for doc in docs}

    def add_document(self, document: PdfDocument):
        with Session(self.engine) as session:
            session.execute(
                insert(Document)
                .values(identifier=document.identifier, pdf_type=document.document_type)
                .on_conflict_do_nothing(index_elements=["identifier"])
            )
            session.commit()

    def add_documents(self, documents: list[PdfDocument]):
        with Session(self.engine) as session:
            values = [
                {"identifier": doc.identifier, "pdf_type": doc.document_type}
                for doc in documents
            ]
            session.execute(
                insert(Document).on_conflict_do_nothing(index_elements=["identifier"]),
                values,
            )
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
        document_identifiers = set(
            embedding.document_identifier for embedding in embeddings
        )
        identifier_to_doc_id = self.get_identifier_to_doc_id(document_identifiers)
        with Session(self.engine) as session:
            session.execute(
                insert(EmbeddingType),
                [
                    {
                        "document_id": identifier_to_doc_id.get(
                            embedding.document_identifier
                        ),
                        "page_number": embedding.page_index,
                        "chunk_number": embedding.chunk_index,
                        "embedding": embedding.embedding,
                    }
                    for embedding in embeddings
                ],
            )
            session.commit()

    def update_embedding_metadata_states(
        self, embedder: Embedder, documents: list[PdfDocument], state: EmbeddingState
    ):
        doc_identifiers = set([doc.identifier for doc in documents])
        identifier_to_doc_id = self.get_identifier_to_doc_id(doc_identifiers)
        with Session(self.engine) as session:
            model_record = session.execute(
                select(Model).where(Model.name == embedder.get_model_name())
            ).scalar_one()
            model_id = model_record.id
            doc_ids = set(
                identifier_to_doc_id.get(identifier) for identifier in doc_identifiers
            )
            session.execute(
                update(EmbeddingMetadata)
                .where(EmbeddingMetadata.model_id == model_id)
                .where(EmbeddingMetadata.document_id.in_(doc_ids))
                .values(state=state)
            )
            session.commit()

    def delete_embeddings_for_document(self, document: PdfDocument):
        doc_id = self.get_identifier_to_doc_id(set([document.identifier])).get(
            document.identifier
        )
        with Session(self.engine) as session:
            session.execute(
                delete(EmbeddingMetadata).where(EmbeddingMetadata.document_id == doc_id)
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
                delete(EmbeddingMetadata).where(EmbeddingMetadata.model_id == model_id)
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
            missing_docs = session.execute(
                select(Document).where(
                    ~Document.id.in_(
                        select(EmbeddingMetadata.document_id).where(
                            EmbeddingMetadata.model_id == model_id
                        )
                    )
                )
            ).scalars()
            new_embedding_metadata = [
                {
                    "document_id": doc.id,
                    "model_id": model_id,
                    "state": EmbeddingState.MISSING,
                }
                for doc in missing_docs
            ]
            session.execute(insert(EmbeddingMetadata), new_embedding_metadata)
            session.commit()

    def query_embeddings(
        self, embedder: Embedder, query_embedding: list[float], top_k: int
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
                        page_number=embedding.page_number,
                        distance=distance,
                    )
                )
            return query_results
