from arxiv_vector_search.processors import TOKEN_OVERHEAD_FACTOR
from arxiv_vector_search.processors.splitter import SplitData
import math
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

TIME_TOL = 0.1
iters = 3


def time_encode(model_name, texts, batch_size):
    print(f"Testing batch size {batch_size}")
    embedder = Embedder(model_name, batch_size)
    total_time = 0
    try:
        print("Warming up...")
        attempt_size = 10 * batch_size
        for _ in range(iters):
            test_run = embedder.encode_text(
                texts[:attempt_size], batch_size, show_progress=True
            )
            attempt_size *= 10
            del test_run
    except torch.OutOfMemoryError:
        return float("inf")
    print("Running timed tests...")
    attempt_size = 1000 * batch_size
    for _ in range(iters):
        start_time = time.time()
        try:
            encodings = embedder.encode_text(
                texts[:attempt_size], batch_size, show_progress=True
            )
            del encodings
        except torch.OutOfMemoryError:
            return float("inf")
        end_time = time.time()
        total_time += end_time - start_time
    return total_time / (iters * attempt_size)


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    db_url = os.getenv("DATABASE_URL")
    db = Database(db_url)

    models = db.get_models()

    for i, model in enumerate(models):
        print(f"{i}: {model.name}")

    model = input("Select a model: ")

    try:
        model = models[int(model)].name
    except ValueError:
        print("Not a number, interpreting it as a model name")

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

    default_model = Embedder(model)
    chunk_size = 512
    if chunk_size > default_model.get_max_input_length() * TOKEN_OVERHEAD_FACTOR:
        chunk_size = math.floor(
            default_model.get_max_input_length() * TOKEN_OVERHEAD_FACTOR
        )
        print(f"Chunk size too large for model, reducing to {chunk_size} tokens")

    splitter = DocumentSplitter(chunk_size, tokenizer=default_model.get_tokenizer())
    splits = splitter.par_split_documents(downloaded_docs, 12)
    texts = [
        split.text for split in splits if isinstance(split, SplitData) and split.text
    ]
    del splits
    del splitter
    del downloaded_docs
    doc_downloader.clear_downloaders()
    del doc_downloader
    del ax_downloader
    del docs
    del default_model
    gc.collect()

    best_batch_size = 8
    best_time = float("inf")

    times = {}

    start = best_batch_size

    while start < len(texts):
        time_taken = time_encode(model, texts, start)
        gc.collect()
        if time_taken == float("inf"):
            break
        times[start] = time_taken
        start *= 2

    for batch_size, batch_time in times.items():
        if batch_time < best_time:
            best_time = batch_time
            best_batch_size = batch_size

    print(
        f"Best batch size: {best_batch_size} with time {best_time:.4f} seconds per text"
    )
