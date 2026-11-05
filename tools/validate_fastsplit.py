#!/usr/bin/env python3
"""Verify rust/fastsplit's chunker matches langchain byte-for-byte.

Feeds documents/document.py + DocumentSplitter-cleaned text (split-only
mode) to the binary and compares (start_index, chunk_index, text) for
every chunk of every document. Chunker semantics must be identical; the
PDF *extractor* is allowed to differ between engines (different text).

Usage:
    DATABASE-independent; point FILES at PDFs on disk.
    python tools/validate_fastsplit.py [pdf_path ...]
"""

import glob
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arxiv_vector_search.documents.document import (
    DocumentType,
    DownloadedDocument,
    PagedDocument,
)
from arxiv_vector_search.processors.splitter import (
    DEFAULT_MIN_CHUNK_CHARS,
    DocumentSplitter,
)

BINARY = (
    Path(__file__).resolve().parents[1]
    / "rust"
    / "fastsplit"
    / "target"
    / "release"
    / "fastsplit"
)


def main(files: list[str]) -> int:
    if not BINARY.exists():
        print(f"Build the binary first: {BINARY} missing")
        return 2
    sp = DocumentSplitter(chunk_size=512, tokenizer=None)
    docs = []
    for f in files:
        dl = DownloadedDocument(Path(f).stem, DocumentType.ARXIV, Path(f))
        pages = dl.get_pages_text()
        if isinstance(pages, Exception):
            continue
        docs.append((Path(f).stem, PagedDocument(Path(f).stem, DocumentType.ARXIV, pages).get_text()))
    py = {}
    for ident, text in docs:
        chunks = sp.recursive_splitter.create_documents([text])
        py[ident] = [
            (c.metadata["start_index"], i, c.page_content)
            for i, c in enumerate(chunks)
            if len(c.page_content.strip()) >= DEFAULT_MIN_CHUNK_CHARS
        ]
    inp = "\n".join(json.dumps({"identifier": i, "text": t}) for i, t in docs)
    res = subprocess.run(
        [str(BINARY), "--mode", "text", "--chunk-chars", str(sp.chunk_size), "--chunk-overlap", str(sp.chunk_overlap)],
        input=inp.encode(),
        capture_output=True,
    )
    if res.returncode != 0:
        print("fastsplit failed:", res.stderr.decode()[:400])
        return 1
    rs = {}
    for line in res.stdout.split(b"\n"):
        if line.strip():
            o = json.loads(line.decode("utf-8", "surrogatepass"))
            rs.setdefault(o["identifier"], []).append(
                (o["start_index"], o["chunk_index"], o["text"])
            )
    mismatched = [i for i, v in py.items() if v != rs.get(i, [])]
    print(
        f"docs={len(py)} py_chunks={sum(map(len, py.values()))} "
        f"rust_chunks={sum(map(len, rs.values()))} mismatched={len(mismatched)}"
    )
    for i in mismatched[:3]:
        print(f"  mismatch: {i}: py={len(py[i])} rust={len(rs.get(i, []))}")
    return 1 if mismatched else 0


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob("pdfs/**/*.pdf", recursive=True))
    if not files:
        print("No PDFs given and pdfs/ is empty; pass paths as arguments.")
        sys.exit(2)
    sys.exit(main(files))
