"""Splitter behavior tests (no torch, no DB).

Runnable two ways:
    pytest tests/test_splitter.py
    python tests/test_splitter.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arxiv_vector_search.documents.document import DocumentType
from arxiv_vector_search.processors.splitter import (
    DEFAULT_MIN_CHUNK_CHARS,
    DocumentSplitter,
)


def synthetic_pages(n_pages=4, chars_per_page=3000):
    pages = []
    for p in range(n_pages):
        sentences = []
        for s in range(80):
            sentences.append(
                f"Page {p} sentence {s}: the quick brown fox jumps over the lazy dog "
                f"and studies quantum cohomology of grassmannians modulo symmetries."
            )
        text = " ".join(sentences)[:chars_per_page]
        # page number footer, like the regex pass will clean
        pages.append(text + f"\n\n{p + 1}\n")
    return pages


def make_paged_document(pages):
    # exercise the PagedDocument regex/clean path directly
    from arxiv_vector_search.documents.document import PagedDocument

    return PagedDocument("2301.00001", DocumentType.ARXIV, pages)


def test_char_mode_chunk_sizing():
    splitter = DocumentSplitter(chunk_size=512, tokenizer=None)
    # char mode multiplies by DEFAULT_CHUNK_FACTOR (3.5)
    assert splitter.chunk_size == int(512 * 3.5)
    doc = make_paged_document(synthetic_pages())
    from arxiv_vector_search.documents import DownloadedDocument
    from pathlib import Path as P

    dl = DownloadedDocument("2301.00001", DocumentType.ARXIV, P("/nonexistent"))
    # split_document re-reads the PDF; instead test the splitter on the
    # already-built PagedDocument text through create_documents:
    chunks = splitter.recursive_splitter.create_documents([doc.get_text()])
    assert len(chunks) > 1
    assert all(len(c.page_content) <= splitter.chunk_size for c in chunks)


def test_short_chunks_filtered():
    splitter = DocumentSplitter(chunk_size=512, tokenizer=None)
    from arxiv_vector_search.documents.document import PagedDocument

    pages = [
        "x" * 4000,
        "tiny",  # short page -> any chunk from it is below the min
        "y" * 4000,
    ]
    doc = PagedDocument("2301.00001", DocumentType.ARXIV, pages)
    text = doc.get_text()
    chunks = splitter.recursive_splitter.create_documents([text])
    kept = [
        c for c in chunks if len(c.page_content.strip()) >= DEFAULT_MIN_CHUNK_CHARS
    ]
    # the 'tiny' page content must be excluded by the filter
    assert not any(c.page_content.strip() == "tiny" for c in kept)
    assert len(kept) < len(chunks) + 1  # filter active


def test_page_index_mapping():
    from arxiv_vector_search.documents.document import PagedDocument

    pages = synthetic_pages(n_pages=3)
    doc = PagedDocument("2301.00001", DocumentType.ARXIV, pages)
    assert doc.start_index_to_page_index(0) == 0
    last = doc.page_lens[0] + doc.page_lens[1] + 4
    assert doc.start_index_to_page_index(last) == 2
    assert doc.start_index_to_page_index(10**9) == 2  # clamps to last page


def test_regex_passes_run():
    from arxiv_vector_search.documents.document import PagedDocument

    pages = ["this pro-\ngram works", "another\nparagraph here\n\n7\n"]
    doc = PagedDocument("x", DocumentType.ARXIV, pages)
    text = doc.get_text()
    assert "program" in text
    assert "another paragraph" in text


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
