from arxiv_vector_search.documents import ReadError
from arxiv_vector_search.documents import DownloadedDocument
from matplotlib import pyplot as plt
from arxiv_vector_search.db import Database
from arxiv_vector_search.documents import DocumentDownloader, DocumentType
from arxiv_vector_search.documents.arxiv import ArxivDownloader
import random
import os
import numpy as np

if __name__ == "__main__":
    db_url = os.getenv("DATABASE_URL")
    db = Database(db_url)
    print("Fetching documents from database...")
    docs = db.get_documents()
    random.shuffle(docs)
    docs = docs[:5000]
    print(f"Fetched {len(docs)} documents.")

    down = DocumentDownloader()
    arxiv_down = ArxivDownloader()
    down.register_downloader(DocumentType.ARXIV, arxiv_down)
    down.add_documents(docs)

    print("Downloading documents...")
    downloaded = down.batch_download(40)
    good_docs = [d for d in downloaded if isinstance(d, DownloadedDocument)]
    print(f"Successfully downloaded {len(good_docs)} documents.")
    page_lens = {}
    print("Extracting page lengths...")
    for doc in good_docs:
        pages = doc.get_pages_text()
        if isinstance(pages, ReadError):
            continue
        for page in pages:
            page_len = len(page)
            if page_len not in page_lens:
                page_lens[page_len] = 0
            page_lens[page_len] += 1
    print(f"Extracted page lengths for {len(page_lens)} unique lengths.")
    down.clear_downloaders()
    bin_size = 32
    bins = np.arange(0, 10000 + bin_size, bin_size)
    print("Plotting histogram...")
    plt.hist(
        page_lens.keys(),
        weights=page_lens.values(),
        bins=bins,
        density=True,
        cumulative=True,
    )
    plt.xlabel("Page Length")
    plt.xlim(-10, 10010)
    plt.ylabel("Frequency")
    plt.title("Histogram of Page Lengths in ArXiv Documents")
    plt.show()
