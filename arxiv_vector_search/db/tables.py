import enum
from typing import List

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import String, ForeignKey, SmallInteger, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from arxiv_vector_search.documents.document import DocumentType


class Base(DeclarativeBase):
    pass


class Model(Base):
    __tablename__ = "models"
    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    batch_size: Mapped[int] = mapped_column(nullable=False)
    embedding_dim: Mapped[int] = mapped_column(nullable=False)
    embedding_metadatas: Mapped[List["EmbeddingMetadata"]] = relationship(
        "EmbeddingMetadata", back_populates="model"
    )


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    identifier: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    pdf_type: Mapped[DocumentType] = mapped_column(nullable=False)
    __table_args__ = (Index("idx_identifier", "identifier"),)


class EmbeddingState(enum.Enum):
    MISSING = "missing"
    DOWNLOAD_ERROR = "download_error"
    SPLIT_ERROR = "split_error"
    EMBEDDED = "embedded"


class EmbeddingMetadata(Base):
    __tablename__ = "embedding_metadata"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    document: Mapped[Document] = relationship("Document")
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), index=True)
    model: Mapped[Model] = relationship("Model", back_populates="embedding_metadatas")
    state: Mapped[EmbeddingState] = mapped_column(nullable=False)
    __table_args__ = (
        Index("idx_document_model", "document_id", "model_id"),
        Index("idx_model_state", "model_id", "state"),
    )


def create_embedding_table(model_name: str, embedding_dim):
    safe_model_name = model_name.replace("/", "-")
    attrs = {
        "__tablename__": f"embeddings_{safe_model_name}",
        "document_id": mapped_column(
            ForeignKey("documents.id"), primary_key=True, index=True
        ),
        "document": relationship("Document"),
        "page_number": mapped_column(SmallInteger, nullable=False, primary_key=True),
        "chunk_number": mapped_column(SmallInteger, nullable=False, primary_key=True),
        "embedding": mapped_column(HALFVEC(embedding_dim), nullable=False),
    }
    return type(f"Embedding_{safe_model_name}", (Base,), attrs)
