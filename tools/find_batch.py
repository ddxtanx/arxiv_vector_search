import arxiv_vector_search
import os
import random
import time
import math

import torch
import logging

logging.basicConfig(level=logging.ERROR)

pdfs = [f for f in os.listdir("pdfs") if f.endswith(".pdf")][:1000]
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = arxiv_vector_search.create_model(model_name)
max_chunk_len = model.max_seq_length
splitter = arxiv_vector_search.create_splitter(max_chunk_len)
print(f"Model: {model_name}, Max chunk length: {max_chunk_len}")
chunks = map(lambda pdf: arxiv_vector_search.split_into_chunks(pdf, splitter), pdfs)
chunks = [chunk for page_chunks in chunks for chunk_list in page_chunks for chunk in chunk_list]

iters = 3

def time_encode(batch_size):
    total_time = 0
    for _ in range(iters):
        random.shuffle(chunks)
        start_time = time.time()
        try:
            model.encode(chunks, batch_size=batch_size, show_progress_bar=True)
        except torch.OutOfMemoryError:
            return float("inf")
        end_time = time.time()
        total_time += end_time - start_time
    return total_time / iters


best_batch_size = 1
best_time = float("inf")

times = {}

start = 1

while start < len(chunks):
    time_taken = time_encode(start)
    if time_taken == float("inf"):
        break
    times[start] = time_taken
    start *= 2

for batch_size, batch_time in times.items():
    if batch_time < best_time:
        best_time = batch_time
        best_batch_size = batch_size

print(best_batch_size)
