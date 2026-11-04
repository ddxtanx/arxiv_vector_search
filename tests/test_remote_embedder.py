"""End-to-end-ish tests for RemoteEmbedder against a mock llama-server.

Runnable two ways:
    pytest tests/test_remote_embedder.py
    python tests/test_remote_embedder.py
"""

import http.server
import json
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arxiv_vector_search.processors.embedder import EmbeddingType
from arxiv_vector_search.processors.llama_cpp_embedder import (
    EmbeddingServerError,
    RemoteEmbedder,
)
from arxiv_vector_search.processors.splitter import SplitData

DIM = 8
RECEIVED: list[dict] = []
FLAKY_COUNT = {"n": 0}


class MockHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/props":
            self._json({"n_ctx": 16384, "model": "QwenEmbeddingSmall"})
        else:
            self._json({"error": "nope"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        RECEIVED.append(body)
        if self.path != "/v1/embeddings":
            self._json({"error": "unknown path"}, 404)
            return
        if not isinstance(body.get("input"), str):
            self._json(
                {"error": {"message": "only one input at a time supported"}}, 400
            )
            return
        if body["input"] == "flaky" and FLAKY_COUNT["n"] < 2:
            FLAKY_COUNT["n"] += 1
            self._json({"error": {"message": "model loading"}}, 503)
            return
        if body["input"] == "boom":
            self._json(
                {"error": {"message": "model not found", "code": 404}}, 404
            )
            return
        # deterministic pseudo-embedding, length encodes the text length
        base = [1.0] * (DIM - 1) + [float(len(body["input"]))]
        self._json(
            {
                "data": [
                    {"embedding": base, "index": 0, "object": "embedding"}
                ],
                "model": body.get("model"),
                "object": "list",
            }
        )


def make_embedder(**kw) -> RemoteEmbedder:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    kw.setdefault("concurrency", 4)
    emb = RemoteEmbedder("QwenEmbeddingSmall", base_url=f"http://127.0.0.1:{port}", **kw)
    return emb


def test_dim_probe_and_props():
    emb = make_embedder()
    assert emb.get_embedding_dim() == DIM
    assert emb.get_max_input_length() == 16384  # from /props
    assert emb.get_tokenizer() is None


def test_batch_shapes_dtype_norm():
    emb = make_embedder()
    out = emb.encode_text(["hello world", "second text", "third"], batch_size=3)
    assert len(out) == 3
    for vec in out:
        assert vec.dtype == np.float16
        assert vec.shape == (DIM,)
        assert abs(float(np.linalg.norm(vec.astype(np.float32))) - 1.0) < 1e-3


def test_prefixes_applied():
    RECEIVED.clear()
    emb = make_embedder(document_prefix="DOC ", query_prefix="QRY ")
    splits = [SplitData("2301.00001", 0, 0, "body text")]
    emb.embed_documents(splits)
    assert RECEIVED[-1]["input"] == "DOC body text"
    emb.embed_queries(["what is cohomology"])
    assert RECEIVED[-1]["input"] == "QRY what is cohomology"


def test_model_name_passed_through():
    RECEIVED.clear()
    emb = make_embedder()
    emb.embed_queries(["x"])
    assert RECEIVED[-1]["model"] == "QwenEmbeddingSmall"


def test_retry_on_transient_503():
    FLAKY_COUNT["n"] = 0
    emb = make_embedder()
    out = emb.encode_text(["flaky"], batch_size=1)
    assert len(out) == 1
    assert FLAKY_COUNT["n"] == 2  # two failures then success


def test_persistent_failure_raises_actionable():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    emb = RemoteEmbedder.__new__(RemoteEmbedder)  # skip __init__ probe
    import requests

    emb.model_name = "typo-model"
    emb.base_url = f"http://127.0.0.1:{port}"
    emb.timeout_s = 5
    emb.session = requests.Session()
    try:
        emb._embed_single("boom")
        raise AssertionError("expected EmbeddingServerError")
    except EmbeddingServerError as e:
        assert "typo-model" in str(e)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
