import math
from arxiv_vector_search.db import Database
from arxiv_vector_search.processors import Embedder, DocumentSplitter
from arxiv_vector_search.documents import (
    SplitDocument,
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

iters = 3


def time_encode(embedder, texts, batch_size):
    total_time = 0
    # try:
    #     attempt_size = max(100, math.ceil(len(texts) / batch_size) * 2)
    #     test_run = embedder.encode_text(
    #         texts[:attempt_size], batch_size, show_progress=False
    #     )
    #     del test_run
    # except torch.OutOfMemoryError:
    #     return float("inf")
    for _ in range(iters):
        random.shuffle(texts)
        start_time = time.time()
        try:
            encodings = embedder.encode_text(texts, batch_size, show_progress=True)
            del encodings
        except torch.OutOfMemoryError:
            return float("inf")
        end_time = time.time()
        total_time += end_time - start_time
    return total_time / iters


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

    embedder = Embedder(model, 32)
    docs = db.get_documents()
    random.shuffle(docs)

    num_docs = 250
    docs = docs[:num_docs]

    ax_downloader = ArxivDownloader()
    doc_downloader = DocumentDownloader()
    doc_downloader.register_downloader(DocumentType.ARXIV, ax_downloader)
    doc_downloader.add_documents(docs)

    downloaded_docs = doc_downloader.batch_download(40)
    downloaded_docs = [
        doc for doc in downloaded_docs if isinstance(doc, DownloadedDocument)
    ]

    splitter = DocumentSplitter()
    splits = splitter.par_split_documents(downloaded_docs, 12)
    texts = [
        chunk.text for doc in splits for chunk in doc if isinstance(doc, SplitDocument)
    ]
    del splits
    del splitter
    del downloaded_docs
    doc_downloader.clear_downloaders()
    del doc_downloader
    del ax_downloader
    del docs
    gc.collect(0)

    best_batch_size = 32
    best_time = float("inf")

    times = {}

    start = 1

    while start < len(texts):
        time_taken = time_encode(embedder, texts, start)
        gc.collect()
        if time_taken == float("inf"):
            break
        times[start] = time_taken
        start *= 2

    for batch_size, batch_time in times.items():
        if batch_time < best_time:
            best_time = batch_time
            best_batch_size = batch_size

    print(best_batch_size)
