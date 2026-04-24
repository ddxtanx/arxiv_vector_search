from typing import TypeAlias
import numpy as np
from typing import Any
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from sentence_transformers import SentenceTransformer
from arxiv_vector_search.processors.splitter import Splits
from arxiv_vector_search.documents import DocumentSplitIterator

SentenceEmbedding: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float16]]


def get_params() -> Any:
    base = {
        "device": "cuda",
        "model_kwargs": {"dtype": torch.float16, "attn_implementation": "sdpa"},
        "config_kwargs": {"_attn_implementation": "sdpa"},
    }
    return base


def create_model(model_name: str, **kwargs) -> SentenceTransformer:
    params = get_params()
    params.update(kwargs)
    model = SentenceTransformer(model_name, **params)
    model.eval()
    model[0].auto_model = torch.compile(
        model[0].auto_model,
        mode="reduce-overhead",
    )
    return model


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

        with (
            torch.inference_mode(),
            sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ],
                set_priority=True,
            ),
        ):
            embeddings = (
                self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=False,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
                .half()
                .cpu()
                .numpy()
            )
        torch.cuda.empty_cache()
        for indice, embedding in zip(indices, embeddings):
            doc_id, page_index, chunk_index = indice
            if doc_id not in doc_embeds:
                doc_embeds[doc_id] = []
            while len(doc_embeds[doc_id]) <= page_index:
                doc_embeds[doc_id].append([])
            while len(doc_embeds[doc_id][page_index]) <= chunk_index:
                doc_embeds[doc_id][page_index].append(None)
            doc_embeds[doc_id][page_index][chunk_index] = embedding

        return Embeddings(doc_embeds)

    def get_model_name(self) -> str:
        return self.model_name

    def get_embedding_dim(self) -> int:
        return self.model.get_embedding_dimension()

    def get_batch_size(self) -> int:
        return self.batch_size
