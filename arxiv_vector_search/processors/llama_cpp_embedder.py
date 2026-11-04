"""Embedding client for an OpenAI-compatible /v1/embeddings server.

Designed for a locally hosted llama.cpp llama-server (or llama-swap router)
serving GGUF embedding models, e.g. the llama-swap instance on this machine
at http://127.0.0.1:8194/v1. The model name registered in this database is
passed through as the `model` field of each request, so it must match the
served model id (for llama-swap, the config section name) EXACTLY.

Advantages over the local torch/sentence-transformers path:
- no torch/flash-attn/GPU dependency chain in this project;
- one shared embedder process for the whole machine (Hindsight, LCM, this);
- quantized GGUF models with configurable keep-alive.

Cost: embedding requests go over HTTP, so throughput is bounded by the
server's batching (--parallel) and GPU availability rather than by this
process. Keep EMBED_CONCURRENCY modest if the same server also answers
interactive traffic, or that traffic will be starved during bulk embedding.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import requests

from arxiv_vector_search.processors.embedder import (
    Embedding,
    EmbeddingType,
    SentenceEmbedding,
    TOKEN_CHUNKSIZE,
    TOKEN_OVERHEAD_FACTOR,
)
from arxiv_vector_search.processors.splitter import SplitData

DEFAULT_CONCURRENCY = 4
REQUEST_TIMEOUT_S = 120.0
RETRY_ATTEMPTS = 4
RETRY_BACKOFF_S = (2.0, 5.0, 10.0)


class EmbeddingServerError(Exception):
    pass


class RemoteEmbedder:
    """Duck-compatible with processors.embedder.Embedder for the DB layer."""

    model_name: str
    base_url: str
    batch_size: int
    document_prefix: str
    query_prefix: str
    chunk_size: int

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int = 32,
        document_prefix: str = "",
        query_prefix: str = "",
        chunk_size: int = TOKEN_CHUNKSIZE,
        concurrency: int | None = None,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ):
        base_url = base_url or os.getenv("EMBEDDING_BASE_URL")
        if not base_url:
            raise ValueError(
                "No embedding server URL: pass base_url or set EMBEDDING_BASE_URL."
            )
        api_key = api_key or os.getenv("EMBEDDING_API_KEY")
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self.timeout_s = timeout_s
        env_concurrency = os.getenv("EMBED_CONCURRENCY")
        self.concurrency = concurrency or (
            int(env_concurrency) if env_concurrency else DEFAULT_CONCURRENCY
        )

        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

        self._embedding_dim = len(self._embed_single("dimension probe"))
        self._max_input_length = self._probe_max_input_length()

        max_chunk_size = self._max_input_length * TOKEN_OVERHEAD_FACTOR
        self.chunk_size = chunk_size
        if chunk_size > max_chunk_size:
            print(
                f"Warning: chunk_size {chunk_size} is greater than the maximum allowed {max_chunk_size} for model {model_name}. Setting chunk_size to {int(max_chunk_size)}."
            )
            self.chunk_size = int(max_chunk_size)

    def _probe_max_input_length(self) -> int:
        """llama.cpp /props exposes the served context (n_ctx). Fall back to
        a conservative 512 so the splitter never exceeds an unknown server."""
        try:
            resp = self.session.get(f"{self.base_url}/props", timeout=10)
            if resp.status_code == 200:
                props = resp.json()
                n_ctx = props.get("n_ctx") or props.get("context_size") or props.get(
                    "max_context_size"
                )
                if n_ctx:
                    return int(n_ctx)
        except Exception:
            pass
        print(
            "Warning: could not read context size from /props; assuming 512 tokens max input."
        )
        return 512

    def _embed_single(self, text: str) -> np.ndarray:
        payload = {"model": self.model_name, "input": text}
        last_err: Exception | None = None
        for attempt in range(RETRY_ATTEMPTS):
            try:
                resp = self.session.post(
                    f"{self.base_url}/v1/embeddings",
                    json=payload,
                    timeout=self.timeout_s,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    embedding = np.asarray(
                        data["data"][0]["embedding"], dtype=np.float32
                    )
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    return embedding.astype(np.float16)
                # llama-swap/llama-server error bodies name the problem and
                # are far more actionable than a bare status code.
                last_err = EmbeddingServerError(
                    f"HTTP {resp.status_code} from {self.base_url}/v1/embeddings: {resp.text[:500]}"
                )
                # A cold llama-swap slot can briefly 4xx/5xx while loading;
                # an unknown model id never recovers, but we cannot tell them
                # apart reliably, so retry with backoff either way.
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_S[attempt])
        raise EmbeddingServerError(
            f"Embedding request failed after {RETRY_ATTEMPTS} attempts "
            f"(model '{self.model_name}' at {self.base_url}): {last_err}. "
            "Check that the server is up and that the model name matches the "
            "served model id exactly."
        )

    def encode_text(
        self,
        texts: list[str],
        batch_size: int,
        show_progress: bool = False,
        embedding_type: EmbeddingType = EmbeddingType.GENERIC,
    ) -> list[SentenceEmbedding]:
        prefix = ""
        if embedding_type == EmbeddingType.QUERY:
            prefix = self.query_prefix
        elif embedding_type == EmbeddingType.DOCUMENT:
            prefix = self.document_prefix
        prefixed = [prefix + text for text in texts]
        workers = max(1, min(self.concurrency, len(prefixed)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(self._embed_single, prefixed))
        if show_progress:
            print(f"Embedded {len(results)} texts via {self.base_url}")
        return results

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
        return self._embedding_dim

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_max_input_length(self) -> int:
        return self._max_input_length

    def get_tokenizer(self) -> Any:
        # No local tokenizer: the splitter falls back to its character-based
        # sizing (DEFAULT_CHUNK_FACTOR). Set EMBED_TOKENIZER to a HuggingFace
        # tokenizer name if exact token counting is wanted without torch.
        tokenizer_name = os.getenv("EMBED_TOKENIZER")
        if tokenizer_name:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(tokenizer_name)
        return None

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size
