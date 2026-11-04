"""Structural interface shared by the local (torch) Embedder and the
remote llama.cpp RemoteEmbedder. The DB layer and __main__ code against
this protocol only."""

from typing import Any, Protocol


class EmbedderLike(Protocol):
    document_prefix: str
    query_prefix: str
    chunk_size: int

    def encode_text(
        self, texts: list[str], batch_size: int, show_progress: bool = False, embedding_type: Any = None
    ) -> Any: ...

    def embed_documents(self, splits: list[Any], show_progress: bool = False) -> Any: ...

    def embed_queries(self, queries: list[str], show_progress: bool = False) -> Any: ...

    def get_model_name(self) -> str: ...

    def get_embedding_dim(self) -> int: ...

    def get_batch_size(self) -> int: ...

    def set_batch_size(self, batch_size: int) -> None: ...

    def get_max_input_length(self) -> int: ...

    def get_tokenizer(self) -> Any: ...
