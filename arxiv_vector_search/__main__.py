from arxiv_vector_search.documents import DownloadError
from arxiv_vector_search.documents.document import PagedDocument
from arxiv_vector_search.processors.splitter import SplitData, DocumentSplitter
from arxiv_vector_search.processors.embedder import (
    TOKEN_CHUNKSIZE,
    Embedder,
    TOKEN_OVERHEAD_FACTOR,
)
from arxiv_vector_search.db.tables import EmbeddingState
from arxiv_vector_search.documents.document import DownloadedDocument, DocumentType
from arxiv_vector_search.documents.arxiv import ArxivDownloader
from .db import Database
from .documents import DocumentDownloader
import sys
import os
from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
import gc
import codecs

batch_sizes = {
    "sentence-transformers/all-MiniLM-L6-v2": 512,
    "google/embeddinggemma-300m": 8,
    "WhereIsAI/UAE-Large-V1": 9,
    "Octen/Octen-Embedding-0.6B": 4,
    "ibm-granite/granite-embedding-small-english-r2": 64,
}

MIN_LEN = 128  # minimum character length for a chunk to be embedded. This is to filter out very short chunks that may not be useful for embedding and just waste space and compute.


@dataclass
class Args:
    query: str
    embed: bool
    model: str
    batch_size: int
    threads: int
    flush: bool
    flush_errors: bool
    update_arxiv_metadata: bool
    add_docs_as_missing: bool


def unescaped_input(prompt: str) -> str:
    """Get user input without interpreting escape sequences."""
    return codecs.decode(input(prompt), "unicode_escape")


if __name__ == "__main__":
    parser = ArgumentParser(description="Search arXiv papers using vector search.")
    _ = parser.add_argument("--query", type=str, default=None, help="The search query.")
    _ = parser.add_argument(
        "--embed",
        action=BooleanOptionalAction,
        help="Whether to embed the pdfs in the database.",
    )
    _ = parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The model to use for embedding.",
    )
    _ = parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="The number of pdfs to process in a single batch",
    )
    _ = parser.add_argument(
        "--threads",
        type=int,
        default=10,
        help="The number of threads to use for various tasks.",
    )
    _ = parser.add_argument(
        "--flush",
        action=BooleanOptionalAction,
        help="Whether to flush the database and start from scratch. Use with caution!",
    )
    _ = parser.add_argument(
        "--flush_errors",
        action=BooleanOptionalAction,
        help="Whether to reset the state of pdfs that errored during processing to missing, so they will be reprocessed. Use with caution!",
    )
    _ = parser.add_argument(
        "--update_arxiv_metadata",
        action=BooleanOptionalAction,
        help="Whether to update state with the arxiv metadata json file.",
    )
    _ = parser.add_argument(
        "--add_docs_as_missing",
        action=BooleanOptionalAction,
        help="Whether to add documents in the database that are not yet added as missing, so they will be processed",
    )

    args: Args = parser.parse_args()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL environment variable not set. Exiting.")
        sys.exit(1)
    db = Database(db_url)
    models = db.get_models()
    if args.model is None:
        if not models:
            print(
                "No models found in database. Please specify a model to use for embedding."
            )
            sys.exit(1)
        for i, model in enumerate(models):
            print(f"{i}: {model.name}")
        model_idx = input(
            "Enter the number corresponding to the model you want to use or the name of the model if it is not included above:"
        )
        if not model_idx.isdigit():
            args.model = model_idx
        else:
            while int(model_idx) < 0 or int(model_idx) >= len(models):
                model_idx = input("Invalid input. Please enter a valid number: ")
            args.model = models[int(model_idx)].name
    model_names_in_db = set(model.name for model in models)
    embedding_batch_size = 0
    model_doc_prefix = ""
    model_query_prefix = ""
    if args.model not in model_names_in_db:
        if args.model in batch_sizes:
            embedding_batch_size = batch_sizes[args.model]
        else:
            embedding_batch_size = int(
                input(
                    "Model not found in database and no default batch size found. Please enter a batch size for embedding: "
                )
            )
            model_doc_prefix = unescaped_input(
                "Enter a prefix to add to documents before embedding for this model (optional, can be left blank, and will be unescaped): "
            )
            model_query_prefix = unescaped_input(
                "Enter a prefix to add to queries before embedding for this model (optional, can be left blank, and will be unescaped): "
            )
    else:
        embedding_batch_size = next(
            model.batch_size for model in models if model.name == args.model
        )
    embedder = Embedder(
        args.model,
        embedding_batch_size,
        document_prefix=model_doc_prefix,
        query_prefix=model_query_prefix,
    )
    if args.model not in model_names_in_db:
        db.add_model(embedder)
    db.create_embedding_table_for_model(embedder)

    if args.flush:
        confirm = input(
            "Are you sure you want to flush the database and start from scratch? This will delete all data in the database. Type 'yes' to confirm: "
        )
        if confirm.lower() == "yes":
            db.delete_embeddings_for_model(embedder)
        else:
            print("Flush cancelled. Exiting.")
            sys.exit(0)

    if args.flush_errors:
        confirm = input(
            "Are you sure you want to reset the state of pdfs that errored during processing to missing, so they will be reprocessed? Type 'yes' to confirm: "
        )
        if confirm.lower() == "yes":
            db.flush_errors_for_model(embedder)
        else:
            print("Flush errors cancelled. Exiting.")
            sys.exit(0)

    if args.update_arxiv_metadata:
        from .documents.arxiv.arxiv_updater import create_arxiv_documents

        documents = create_arxiv_documents()
        db.add_documents(documents)

    if args.add_docs_as_missing:
        db.add_missing_metadata(embedder)

    if args.embed:
        downloader = DocumentDownloader()
        arxiv_downloader = ArxivDownloader()
        downloader.register_downloader(DocumentType.ARXIV, arxiv_downloader)
        chunk_size = TOKEN_CHUNKSIZE
        max_acceptable_chunk_size = int(
            embedder.get_max_input_length() * TOKEN_OVERHEAD_FACTOR
        )
        if chunk_size > max_acceptable_chunk_size:
            print(
                f"Default chunk size of {chunk_size} is too large for the model's max input length of {embedder.get_max_input_length()}. Setting chunk size to {max_acceptable_chunk_size}."
            )
            chunk_size = max_acceptable_chunk_size
        splitter = DocumentSplitter(
            chunk_size,
            tokenizer=embedder.get_tokenizer(),
            prefix=embedder.document_prefix,
        )
        while True:
            batch = db.get_missing_embeddings_for_model(
                embedder, limit=args.batch_size, offset=0
            )
            if not batch:
                break
            downloader.add_documents(batch)
            print(f"Processing batch of {len(batch)} documents.")
            downloaded = downloader.batch_download(4 * args.threads)
            print(
                f"Downloaded {len(downloaded)} documents. Processing splits and embeddings..."
            )
            dl_errs = []
            downloaded_docs = []
            for result in downloaded:
                if isinstance(result, DownloadError):
                    dl_errs.append(result)
                elif isinstance(result, DownloadedDocument):
                    downloaded_docs.append(result)
            # for err in dl_errs:
            #     print(f"Error downloading document {err.document.identifier}: {err}")
            print(
                f"Encountered {len(dl_errs)} download errors. Updating states in database..."
            )
            if dl_errs:
                db.update_embedding_metadata_states(
                    embedder,
                    [err.document for err in dl_errs],
                    EmbeddingState.DOWNLOAD_ERROR,
                )
            split_docs = splitter.par_split_documents(downloaded_docs, args.threads)
            downloader.clear_downloaders()
            del downloaded
            good_splits: list[SplitData] = []
            split_errs = []
            for split_doc in split_docs:
                if isinstance(split_doc, SplitData):
                    good_splits.append(split_doc)
                else:
                    split_errs.append(split_doc)
            num_successful_splits = len(good_splits)
            print(
                f"Split documents into {num_successful_splits} chunks. Encountered {len(split_errs)} split errors. Updating states in database..."
            )
            if split_errs:
                db.update_embedding_metadata_states_by_idents(
                    embedder,
                    [err.document_id for err in split_errs],
                    EmbeddingState.SPLIT_ERROR,
                )

            print(
                f"Generating embeddings for {num_successful_splits} document chunks. This may take a while..."
            )
            embeddings = embedder.embed_documents(
                good_splits,
                show_progress=True,
            )
            good_ids = set(split.identifier for split in good_splits)
            del split_docs
            del good_splits
            del split_errs
            print(
                f"Generated embeddings for {num_successful_splits} document chunks. Adding to database..."
            )
            if not embeddings:
                print(
                    "No embeddings generated for this batch. Moving on to next batch..."
                )
                continue
            db.add_embeddings(embeddings, embedder)
            del embeddings
            good_batch = [doc for doc in batch if doc.identifier in good_ids]
            del batch
            print(
                f"Embeddings added to database. Updating states for {len(good_batch)} documents to EMBEDDED."
            )
            db.update_embedding_metadata_states(
                embedder,
                good_batch,
                EmbeddingState.EMBEDDED,
            )
            print("Batch processing complete. Moving on to next batch...")
            gc.collect()

    if args.query:
        query_embedding = embedder.embed_queries([args.query])[0]
        print("Query embedding generated. Performing vector search...")
        results = db.query_embeddings(embedder, query_embedding, top_k=1000)
        print(
            f"Retrieved {len(results)} results. Processing and displaying top results..."
        )
        urls = set(result.document.get_url() for result in results)
        scores: dict[str, list[float]] = {}
        pages: dict[str, set[int]] = {}
        for result in results:
            url = result.document.get_url()
            if url not in scores:
                scores[url] = []
                pages[url] = set()
            scores[url].append(result.distance)
            pages[url].add(result.page_index)
        sorted_by_min = sorted(
            urls,
            key=lambda url: min(scores[url]),
        )
        top_10_by_min = sorted_by_min[:10]

        print("Top 10 results sorted by minimum distance:")
        for url in top_10_by_min:
            print(
                f"URL: {url}, Min Distance: {min(scores[url]):.4f}, Pages: {sorted(pages[url])}"
            )

        # top_10_by_avg = db.query_embeddings_avg(embedder, query_embedding, top_k=10)
        # print("\nTop 10 results sorted by average distance:")
        # for query_result in top_10_by_avg:
        #     url = query_result.document.get_url()
        #     print(f"URL: {url}, Avg Distance: {query_result.distance:.4f}")
