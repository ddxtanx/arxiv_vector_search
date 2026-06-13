import traceback
from arxiv_vector_search.processors.splitter import SplitData
from arxiv_vector_search.db import Database
from arxiv_vector_search.processors import Embedder, DocumentSplitter
from arxiv_vector_search.documents import (
    DownloadedDocument,
    DocumentDownloader,
    DocumentType,
)
from arxiv_vector_search.documents.arxiv.arxiv_downloader import ArxivDownloader
import os
import random
import time

import torch
import logging
import gc
import codecs
import argparse

TIME_TOL = 0.1
iters = 3
ATTEMPT_BATCHES = 100


def unescaped_input(prompt: str) -> str:
    """Get user input without interpreting escape sequences."""
    return codecs.decode(input(prompt), "unicode_escape")


def time_encode(embedder, texts, batch_size):
    embedder.set_batch_size(batch_size)
    print(f"Testing batch size {batch_size}")
    total_time = 0
    attempt_size = batch_size * ATTEMPT_BATCHES
    texts_batch = random.sample(texts, min(attempt_size, len(texts)))
    attempt_size = len(texts_batch)
    try:
        print("Warming up...")
        for _ in range(iters):
            test_run = embedder.embed_documents(texts_batch, show_progress=True)
    except torch.OutOfMemoryError:
        traceback.print_stack()
        return float("inf")
    print("Running timed tests...")
    for _ in range(iters):
        start_time = time.time()
        try:
            encodings = embedder.embed_documents(texts_batch, show_progress=True)
        except torch.OutOfMemoryError:
            traceback.print_stack()
            return float("inf")
        end_time = time.time()
        total_time += end_time - start_time
    num_tokens = sum(
        len(
            embedder.get_tokenizer().encode(
                split.text, add_special_tokens=False, verbose=False
            )
        )
        for split in texts_batch
    )
    return num_tokens / (total_time / iters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search arXiv papers using vector search."
    )
    _ = parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The model to use for embedding.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    db_url = os.getenv("DATABASE_URL")
    db = Database(db_url)

    models = db.get_models()

    model = None
    if args.model:
        model = args.model
    else:
        for i, model in enumerate(models):
            print(f"{i}: {model.name}")

        model = input("Select a model: ")

    if model.isdigit() or model in set(model.name for model in models):
        model_obj = None
        if model.isdigit():
            model = models[int(model)]
        else:
            model_idx = [model.name for model in models].index(model)
            model = models[model_idx]
    else:
        print("Not a number, interpreting it as a model name")
        document_prefix = unescaped_input("Document prefix (default ''): ")
        query_prefix = unescaped_input("Query prefix (default ''): ")
        chunk_size_input = input("Chunk size (default 512): ")
        chunk_size = int(chunk_size_input) if chunk_size_input else 512
        model = {
            "name": model,
            "document_prefix": document_prefix,
            "query_prefix": query_prefix,
            "chunk_size": chunk_size,
            "batch_size": 32,
        }
    if isinstance(model, dict):
        default_model = Embedder(
            model["name"],
            model["batch_size"],
            model["document_prefix"],
            model["query_prefix"],
            model["chunk_size"],
        )
    else:
        default_model = Embedder(
            model.name,
            model.batch_size,
            model.document_prefix,
            model.query_prefix,
            model.chunk_size,
        )
    default_model.encode_text(["Test encoding to initialize model and tokenizer."], 1)

    docs = db.get_documents()
    random.shuffle(docs)

    num_docs = 5000
    docs = docs[:num_docs]

    ax_downloader = ArxivDownloader()
    doc_downloader = DocumentDownloader()
    doc_downloader.register_downloader(DocumentType.ARXIV, ax_downloader)
    doc_downloader.add_documents(docs)

    downloaded_docs = doc_downloader.batch_download(40)
    downloaded_docs = [
        doc for doc in downloaded_docs if isinstance(doc, DownloadedDocument)
    ]

    splitter = DocumentSplitter(
        default_model.chunk_size, tokenizer=default_model.get_tokenizer()
    )
    splits = splitter.par_split_documents(downloaded_docs, 12)
    texts = [split for split in splits if isinstance(split, SplitData) and split.text]
    doc_downloader.clear_downloaders()

    best_batch_size = 1
    best_time = 0

    times = {}

    start = best_batch_size

    while start < len(texts):
        time_taken = time_encode(default_model, texts, start)
        print(f"Batch size {start} took {time_taken:.4f} tokens per second")
        if time_taken == float("inf"):
            break
        times[start] = time_taken
        start *= 2

    for batch_size, batch_time in times.items():
        if batch_time > best_time:
            best_time = batch_time
            best_batch_size = batch_size

    print(
        f"Best batch size: {best_batch_size} with time {best_time:.4f} tokens per second"
    )
