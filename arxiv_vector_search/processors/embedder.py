from arxiv_vector_search.documents.document import SplitDocument
from typing import TypeAlias
import numpy as np
from typing import Any, TypedDict
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from sentence_transformers import SentenceTransformer

SentenceEmbedding: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float16]]


def get_params() -> Any:
    base = {
        "device": "cuda",
        "model_kwargs": {"dtype": torch.float16},
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


class Embedding(TypedDict):
    document_id: str | int
    page_number: int
    chunk_number: int
    embedding: SentenceEmbedding


class Embedder:
    model_name: str
    model: SentenceTransformer
    batch_size: int

    def __init__(self, model_name: str, batch_size: int = 32, **kwargs):
        torch.backends.cuda.preferred_rocm_fa_library("aotriton")
        self.model_name = model_name
        self.model = create_model(model_name, **kwargs)
        self.batch_size = batch_size

    def encode_text(
        self, texts: list[str], batch_size: int, show_progress: bool = False
    ) -> list[SentenceEmbedding]:
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
                    batch_size=batch_size,
                    convert_to_numpy=False,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=show_progress,
                )
                .half()
                .cpu()
                .numpy()
            )
        torch.cuda.empty_cache()
        return embeddings

    def embed_documents(self, split_docs: list[SplitDocument]) -> list[Embedding]:
        splits = [split for doc in split_docs for split in doc]
        del split_docs
        texts = [split.text for split in splits]

        embeddings = self.encode_text(texts, self.batch_size)
        return [
            {
                "document_id": split.identifier,
                "page_number": split.page_index,
                "chunk_number": split.chunk_index,
                "embedding": embedding,
            }
            for split, embedding in zip(splits, embeddings)
        ]

    def get_model_name(self) -> str:
        return self.model_name

    def get_embedding_dim(self) -> int:
        return self.model.get_embedding_dimension()

    def get_batch_size(self) -> int:
        return self.batch_size
