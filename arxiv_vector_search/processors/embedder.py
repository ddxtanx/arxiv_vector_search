from typing import Any
import torch
from sentence_transformers import SentenceTransformer
from arxiv_vector_search.documents import SplitDocument, SplitData, Document


def get_params(model_name: str) -> Any:
    base = {
        "device": "cuda",
        "model_kwargs": {"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
        "config_kwargs": {"_attn_implementation": "sdpa"},
    }
    if (
        "sentence-transformers" not in model_name
        and "UAE-Large" not in model_name
        and "Snowflake/snowflake-arctic-embed-l-v2.0" not in model_name
    ):
        base["model_kwargs"]["attn_implementation"] = "flash_attention_2"
        base["config_kwargs"]["_attn_implementation"] = "flash_attention_2"
    if "all-mpnet-base-v2" in model_name:
        del base["model_kwargs"]["attn_implementation"]
        del base["config_kwargs"]["_attn_implementation"]
    return base


def create_model(model_name: str, **kwargs) -> SentenceTransformer:
    params = get_params(model_name)
    params.update(kwargs)
    return SentenceTransformer(model_name, **params)


class Embedding:
    document_identifier: str
    page_index: int
    chunk_index: int
    embedding: torch.Tensor

    def __init__(
        self,
        document_identifier: str,
        page_index: int,
        chunk_index: int,
        embedding: torch.Tensor,
    ):
        self.document_identifier = document_identifier
        self.page_index = page_index
        self.chunk_index = chunk_index
        self.embedding = embedding


class Embedder:
    model_name: str
    model: SentenceTransformer
    batch_size: int

    def __init__(self, model_name: str, batch_size: int = 32, **kwargs):
        self.model_name = model_name
        self.model = create_model(model_name, **kwargs)
        self.batch_size = batch_size

    def embed_documents(self, documents: list[SplitDocument]) -> list[Embedding]:
        splits = [split for doc in documents for split in doc]
        texts = [split.text for split in splits]
        embeddings = self.model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=True
        )
        return [
            Embedding(
                document_identifier=split.identifier,
                page_index=split.page_index,
                chunk_index=split.chunk_index,
                embedding=embedding,
            )
            for split, embedding in zip(splits, embeddings)
        ]

    def get_model_name(self) -> str:
        return self.model_name

    def get_embedding_dim(self) -> int:
        return self.model.get_embedding_dimension()

    def get_batch_size(self) -> int:
        return self.batch_size
