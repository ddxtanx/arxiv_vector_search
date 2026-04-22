from sqlalchemy import String, create_engine, select, insert, update
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from pgvector.sqlalchemy import Vector
from arxiv_vector_search.db import (
    Model,
    Document,
    create_embedding_table,
    EmbeddingMetadata,
    EmbeddingState,
)
from arxiv_vector_search.processors import Embedder
import os
import sys


class Base(DeclarativeBase):
    pass


class PreviousEmbeddings(Base):
    __tablename__: str = f"embedded_pdfs_sentence-transformers-all-MiniLM-L6-v2"
    id: Mapped[int] = mapped_column(primary_key=True)
    pdf_name: Mapped[str] = mapped_column(String(255))
    page_index: Mapped[int] = mapped_column()
    chunk_index: Mapped[int] = mapped_column()
    embedding: Mapped[Vector] = mapped_column(Vector(384))


model_name = "sentence-transformers/all-MiniLM-L6-v2"
safe_model_name = model_name.replace("/", "-")
NewEmbeddings = create_embedding_table(model_name, embedding_dim=384)

db_url = os.getenv("DATABASE_URL")
if not db_url:
    print("DATABASE_URL environment variable not set")
    sys.exit(1)

embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")

engine = create_engine(db_url)

batch_size = 1000000
with Session(engine) as session:
    print("Fetching previous embeddings...")
    select_stmt = select(PreviousEmbeddings).execution_options(yield_per=batch_size)
    model = session.execute(
        select(Model).where(Model.name == "sentence-transformers/all-MiniLM-L6-v2")
    ).scalar_one_or_none()
    if not model:
        print(
            "Failed to find model document for sentence-transformers/all-MiniLM-L6-v2"
        )
        sys.exit(1)
    model_id = model.id
    print(f"Model document ID: {model_id}")
    print("Fetching documents...")
    documents = session.execute(select(Document)).scalars().all()
    print(f"Found {len(documents)} documents")
    ident_to_id = {doc.identifier: doc.id for doc in documents}
    print("Inserting new embeddings...")
    update_ids = set()
    new_values = []
    for prev in session.scalars(select_stmt):
        doc_id = ident_to_id.get(prev.pdf_name.replace(".pdf", ""))
        if not doc_id:
            print(f"Failed to find document ID for {prev.pdf_name}")
            continue
        update_ids.add(doc_id)
        new_values.append(
            {
                "document_id": doc_id,
                "model_id": model_id,
                "page_number": prev.page_index,
                "chunk_number": prev.chunk_index,
                "embedding": prev.embedding,
            }
        )
        if len(new_values) >= batch_size:
            session.execute(insert(NewEmbeddings), new_values)
            new_values = []
    session.execute(
        update(EmbeddingMetadata)
        .where(
            (EmbeddingMetadata.model_id == model_id)
            & (EmbeddingMetadata.document_id in update_ids)
        )
        .values(state=EmbeddingState.EMBEDDED)
    )
    session.commit()
