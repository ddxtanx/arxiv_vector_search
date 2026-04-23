from typing import TypeAlias
import numpy as np
from typing import Any
import torch
from sentence_transformers import SentenceTransformer
from arxiv_vector_search.processors.splitter import Splits
from arxiv_vector_search.documents import DocumentSplitIterator

SentenceEmbedding: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float16]]


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


class Embeddings:
    embeddings_dict: dict[str, list[list[SentenceEmbedding]]]

    def __init__(self, embeddings_dict: dict[str, list[list[SentenceEmbedding]]]):
        self.embeddings_dict = embeddings_dict

    def __iter__(self):
        return DocumentSplitIterator[SentenceEmbedding](self.embeddings_dict)


class Embedder:
    model_name: str
    model: SentenceTransformer
    batch_size: int

    def __init__(self, model_name: str, batch_size: int = 32, **kwargs):
        self.model_name = model_name
        self.model = create_model(model_name, **kwargs)
        self.batch_size = batch_size

    def embed_documents(self, splits: Splits) -> Embeddings:
        doc_embeds: dict[str, list[list[SentenceEmbedding | None]]] = {}
        indices: list[tuple[str, int, int]] = []
        texts: list[str] = []
        for doc_id, page_index, chunk_index, text in splits:
            indices.append((doc_id, page_index, chunk_index))
            texts.append(text)

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        for indice, embedding in zip(indices, embeddings):
            doc_id, page_index, chunk_index = indice
            if doc_id not in doc_embeds:
                doc_embeds[doc_id] = []
            while len(doc_embeds[doc_id]) <= page_index:
                doc_embeds[doc_id].append([])
            while len(doc_embeds[doc_id][page_index]) <= chunk_index:
                doc_embeds[doc_id][page_index].append(None)
            doc_embeds[doc_id][page_index][chunk_index] = embedding.astype(np.float16)

        return Embeddings(doc_embeds)

    def get_model_name(self) -> str:
        return self.model_name

    def get_embedding_dim(self) -> int:
        return self.model.get_embedding_dimension()

    def get_batch_size(self) -> int:
        return self.batch_size
