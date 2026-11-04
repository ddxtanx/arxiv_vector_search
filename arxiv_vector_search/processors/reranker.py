"""Client for the llama.cpp llama-swap rerank endpoint (/rerank).

Pairs with the embedding setup in llama_cpp_embedder.py: same base URL
(e.g. http://127.0.0.1:8194), model name must match the llama-swap config
section id of the reranker (Qwen3-Reranker family on this machine).

Typical use is hybrid retrieval: pgvector returns ~50 candidate chunks,
the cross-encoder reranks query+chunk pairs. llama-swap reranker ctx on
this box is 8192 per pair — keep candidates <= ~50 and chunk length under
~1500 tokens so a pair fits (documented in llama-swap config.ini).

llama.cpp's rerank JSON schema has varied between builds, so the request
tries the TEI style first and falls back to the Cohere style on 4xx; the
response parser accepts both shapes ({"data":[{index,score}]} and
{"results":[{index,relevance_score}]}, or a bare list).
"""

import os
from typing import Any

import requests


class RerankError(Exception):
    pass


class RemoteReranker:
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
    ):
        base_url = base_url or os.getenv("RERANK_BASE_URL") or os.getenv("EMBEDDING_BASE_URL")
        if not base_url:
            raise ValueError(
                "No rerank server URL: pass base_url or set RERANK_BASE_URL / EMBEDDING_BASE_URL."
            )
        # embeddings config uses .../v1; rerank lives at the server root
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
        self.model_name = model_name
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.session = requests.Session()
        api_key = api_key or os.getenv("EMBEDDING_API_KEY")
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _post(self, payload: dict[str, Any]) -> requests.Response:
        return self.session.post(
            f"{self.base_url}/rerank", json=payload, timeout=self.timeout_s
        )

    def rerank(self, query: str, texts: list[str]) -> list[tuple[int, float]]:
        """Return (candidate_index, relevance_score) sorted by score desc."""
        if not texts:
            return []
        # llama.cpp / TEI style first, Cohere style fallback.
        attempts = [
            {
                "model": self.model_name,
                "input": {"query": query, "documents": texts},
            },
            {"model": self.model_name, "query": query, "documents": texts},
        ]
        resp = None
        for payload in attempts:
            resp = self._post(payload)
            if resp.status_code == 200:
                break
        if resp is None or resp.status_code != 200:
            raise RerankError(
                f"rerank failed (model '{self.model_name}' at {self.base_url}/rerank): "
                f"{resp.status_code if resp is not None else 'no response'} {resp.text[:400] if resp is not None else ''}"
            )
        body = resp.json()
        if isinstance(body, dict):
            items = body.get("results") or body.get("data") or []
        else:
            items = body
        scored: list[tuple[int, float]] = []
        for i, item in enumerate(items):
            idx = item.get("index", i)
            score = item.get("relevance_score", item.get("score"))
            if score is None:
                emb = item.get("embedding")
                score = emb[0] if isinstance(emb, list) and emb else None
            if score is None:
                raise RerankError(f"unparseable rerank response item: {str(item)[:200]}")
            scored.append((int(idx), float(score)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored
