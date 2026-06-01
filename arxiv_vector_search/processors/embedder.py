import traceback
import math
from transformers import PreTrainedTokenizerBase
from arxiv_vector_search.processors.splitter import SplitData
from typing import TypeAlias
import numpy as np
from typing import Any, TypedDict
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from sentence_transformers import SentenceTransformer
import os
import torch.cuda.tunable as tunable
from enum import Enum

SentenceEmbedding: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float16]]

TOKEN_CHUNKSIZE = 512
TOKEN_OVERHEAD_FACTOR = 0.925


class EmbeddingType(Enum):
    DOCUMENT = "document"
    QUERY = "query"
    GENERIC = "generic"


def get_params() -> Any:
    base = {
        "device": "cuda",
        "model_kwargs": {
            "dtype": torch.float16,
            "attn_implementation": "flash_attention_2",
        },
        "processor_kwargs": {
            "use_fast": True,
        },
        "config_kwargs": {
            "dtype": torch.float16,
            "use_memory_efficient_attention": True,
            "_attn_implementation": "flash_attention_2",
            "unpad_inputs": True,
        },
        "trust_remote_code": True,
    }
    return base


def create_model(model_name: str, chunk_size: int, **kwargs) -> SentenceTransformer:
    params = get_params()
    params.update(kwargs)
    try:
        model = SentenceTransformer(model_name, **params)
    except ValueError as e:
        traceback.print_exception(e)
        params["model_kwargs"]["attn_implementation"] = "sdpa"
        params["config_kwargs"]["_attn_implementation"] = "sdpa"
        del params["config_kwargs"]["unpad_inputs"]
        model = SentenceTransformer(model_name, **params)
    model.eval()
    model.to("cuda").half()
    # cur_seq_len = model.max_seq_length
    # resized = math.ceil(chunk_size / TOKEN_OVERHEAD_FACTOR)
    # if resized < cur_seq_len:
    #     model.max_seq_length = resized
    model.compile(mode="max-autotune", dynamic=False, fullgraph=True)
    return model


class Embedding(TypedDict):
    document_id: str | int
    page_index: int
    chunk_index: int
    embedding: SentenceEmbedding


class Embedder:
    model_name: str
    model: SentenceTransformer
    batch_size: int
    document_prefix: str
    query_prefix: str
    chunk_size: int

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        document_prefix: str = "",
        query_prefix: str = "",
        chunk_size: int = TOKEN_CHUNKSIZE,
        **kwargs,
    ):
        torch.backends.cuda.preferred_rocm_fa_library("aotriton")
        torch.cuda.set_per_process_memory_fraction(0.95)
        self.model_name = model_name
        self.batch_size = batch_size
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

        self.model = create_model(model_name, chunk_size, **kwargs)
        max_chunk_size = self.model.max_seq_length * TOKEN_OVERHEAD_FACTOR
        self.chunk_size = chunk_size
        if chunk_size > max_chunk_size:
            print(
                f"Warning: chunk_size {chunk_size} is greater than the maximum allowed {max_chunk_size} for model {model_name}. Setting chunk_size to {int(max_chunk_size)}."
            )
            self.chunk_size = int(max_chunk_size)

    def encode_text(
        self,
        texts: list[str],
        batch_size: int,
        show_progress: bool = False,
        embedding_type: EmbeddingType = EmbeddingType.GENERIC,
    ) -> list[SentenceEmbedding]:
        kwargs = {}
        if embedding_type == EmbeddingType.QUERY:
            if self.query_prefix:
                kwargs["prompt"] = self.query_prefix
            else:
                kwargs["prompt_name"] = "query"
        elif embedding_type == EmbeddingType.DOCUMENT:
            if self.document_prefix:
                kwargs["prompt"] = self.document_prefix
            else:
                kwargs["prompt_name"] = "document"
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
                    **kwargs,
                )
                .half()
                .cpu()
                .numpy()
            )
        torch.cuda.empty_cache()
        return embeddings

    def embed_documents(
        self, splits: list[SplitData], show_progress: bool = False
    ) -> list[Embedding]:
        texts = [split.text for split in splits]

        embeddings = self.encode_text(
            texts,
            self.batch_size,
            show_progress,
            embedding_type=EmbeddingType.DOCUMENT,
        )
        return [
            {
                "document_id": split.identifier,
                "page_index": split.page_index,
                "chunk_index": split.chunk_index,
                "embedding": embedding,
            }
            for split, embedding in zip(splits, embeddings)
        ]

    def embed_queries(
        self, queries: list[str], show_progress: bool = False
    ) -> list[SentenceEmbedding]:
        return self.encode_text(
            queries, self.batch_size, show_progress, embedding_type=EmbeddingType.QUERY
        )

    def get_model_name(self) -> str:
        return self.model_name

    def get_embedding_dim(self) -> int:
        return self.model.get_embedding_dimension()

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_max_input_length(self) -> int:
        return self.model.max_seq_length or 512

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.model.tokenizer

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size
