"""RemoteReranker tests against a mock llama-swap /rerank (both schemas).

Runnable via pytest or directly: python tests/test_reranker.py
"""

import http.server
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arxiv_vector_search.processors.reranker import RemoteReranker, RerankError

MODE = {"kind": "tei"}  # or "cohere"


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path == "/rerank", self.path
        if MODE["kind"] == "tei":
            if "input" not in body:
                self._send({"error": "expected input.query/input.documents"}, 400)
                return
            query, docs = body["input"]["query"], body["input"]["documents"]
        else:
            if "query" not in body or "documents" not in body:
                self._send({"error": "expected query/documents"}, 400)
                return
            query, docs = body["query"], body["documents"]
        # deterministic: score = length of query-relevant overlap heuristic
        scores = [float(len(d) % 7 + 1) for d in docs]
        if MODE["kind"] == "tei":
            resp = {"data": [{"index": i, "score": s} for i, s in enumerate(scores)]}
        else:
            resp = {
                "results": [
                    {"index": i, "relevance_score": s} for i, s in enumerate(scores)
                ]
            }
        self._send(resp)

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


def server(port_base=0):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port_base), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_tei_schema():
    MODE["kind"] = "tei"
    srv = server()
    r = RemoteReranker("QwenReranking", base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1")
    out = r.rerank("what is cohomology", ["short one", "a much longer chunk text here", "mid length text"])
    assert out == sorted(out, key=lambda t: -t[1])
    assert {i for i, _ in out} == {0, 1, 2}


def test_cohere_fallback():
    MODE["kind"] = "cohere"
    srv = server()
    r = RemoteReranker("QwenReranking", base_url=f"http://127.0.0.1:{srv.server_address[1]}")
    out = r.rerank("q", ["abc", "abcd", "abcde"])
    assert len(out) == 3


def test_error_raises():
    class F(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.send_response(404)
            b = b'{"error":"model not found"}'
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), F)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    r = RemoteReranker("typo", base_url=f"http://127.0.0.1:{srv.server_address[1]}")
    try:
        r.rerank("q", ["a"])
        raise AssertionError("expected RerankError")
    except RerankError as e:
        assert "typo" in str(e)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all reranker tests passed")
