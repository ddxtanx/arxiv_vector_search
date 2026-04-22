from arxiv_vector_search.documents.document import SplitDocument
from arxiv_vector_search.db.tables import EmbeddingState
from arxiv_vector_search.documents.document import DownloadedDocument, DocumentType
from arxiv_vector_search.documents.arxiv import ArxivDownloader
from .db import Database
from .processors import Embedder, Embedding, DocumentSplitter
from .documents import DocumentDownloader
import sys
import os
from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass

batch_sizes = {
    "sentence-transformers/all-MiniLM-L6-v2": 512,
    "google/embeddinggemma-300m": 8,
    "WhereIsAI/UAE-Large-V1": 9,
    "Octen/Octen-Embedding-0.6B": 4,
    "ibm-granite/granite-embedding-small-english-r2": 64,
}


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
            "Enter the number corresponding to the model you want to use: "
        )
        while (
            not model_idx.isdigit()
            or int(model_idx) < 0
            or int(model_idx) >= len(models)
        ):
            model_idx = input("Invalid input. Please enter a valid number: ")
        args.model = models[int(model_idx)].name
    model_names_in_db = set(model.name for model in models)
    embedding_batch_size = 0
    if args.model not in model_names_in_db:
        if args.model in batch_sizes:
            embedding_batch_size = batch_sizes[args.model]
        else:
            embedding_batch_size = int(
                input(
                    "Model not found in database and no default batch size found. Please enter a batch size for embedding: "
                )
            )
    else:
        embedding_batch_size = next(
            model.batch_size for model in models if model.name == args.model
        )
    embedder = Embedder(args.model, embedding_batch_size)
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
        remaining_docs = db.get_missing_embeddings_for_model(embedder)
        downloader = DocumentDownloader()
        arxiv_downloader = ArxivDownloader()
        downloader.register_downloader(DocumentType.ARXIV, arxiv_downloader)
        splitter = DocumentSplitter()
        while remaining_docs:
            batch = remaining_docs[: args.batch_size]
            remaining_docs = remaining_docs[args.batch_size :]
            downloader.add_documents(batch)
            print(
                f"Processing batch of {len(batch)} documents. Remaining: {len(remaining_docs)}"
            )
            downloaded = downloader.batch_download(4 * args.threads)
            print(
                f"Downloaded {len(downloaded)} documents. Processing splits and embeddings..."
            )
            dl_errs = [
                doc for doc in downloaded if not isinstance(doc, DownloadedDocument)
            ]
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
            downloaded = [
                doc for doc in downloaded if isinstance(doc, DownloadedDocument)
            ]
            split_docs = splitter.par_split_documents(downloaded, args.threads)
            split_errs = [
                doc for doc in split_docs if not isinstance(doc, SplitDocument)
            ]
            print(
                f"Split documents into {len(split_docs)} chunks. Encountered {len(split_errs)} split errors. Updating states in database..."
            )
            if split_errs:
                db.update_embedding_metadata_states(
                    embedder,
                    [err.document for err in split_errs],
                    EmbeddingState.SPLIT_ERROR,
                )
            split_docs = [doc for doc in split_docs if isinstance(doc, SplitDocument)]

            print(
                f"Generating embeddings for {len(split_docs)} document chunks. This may take a while..."
            )
            embeddings = embedder.embed_documents(split_docs)
            print(
                f"Generated embeddings for {len(embeddings)} document chunks. Adding to database..."
            )
            db.add_embeddings(embeddings, embedder)
            print(
                f"Embeddings added to database. Updating states for {len(split_docs)} documents to EMBEDDED."
            )
            db.update_embedding_metadata_states(
                embedder,
                split_docs,
                EmbeddingState.EMBEDDED,
            )
            print("Batch processing complete. Moving on to next batch...")
            downloader.clear_downloaders()

    if args.query:
        query_embedding = embedder.model.encode(args.query)
        print("Query embedding generated. Performing vector search...")
        results = db.query_embeddings(embedder, query_embedding, top_k=1000)
        print(
            f"Retrieved {len(results)} results. Processing and displaying top results..."
        )
        urls = set(result.document.get_url() for result in results)
        scores: dict[str, list[float]] = {}
        pages: dict[str, list[int]] = {}
        for result in results:
            url = result.document.get_url()
            if url not in scores:
                scores[url] = []
                pages[url] = []
            scores[url].append(result.distance)
            pages[url].append(result.page_number)
        sorted_by_min = sorted(
            urls,
            key=lambda url: min(scores[url]),
        )
        sorted_by_avg = sorted(
            urls,
            key=lambda url: sum(scores[url]) / len(scores[url]),
        )
        top_10_by_min = sorted_by_min[:10]
        top_10_by_avg = sorted_by_avg[:10]

        print("Top 10 results sorted by minimum distance:")
        for url in top_10_by_min:
            print(
                f"URL: {url}, Min Distance: {min(scores[url]):.4f}, Pages: {sorted(pages[url])}"
            )
        print("\nTop 10 results sorted by average distance:")
        for url in top_10_by_avg:
            print(
                f"URL: {url}, Avg Distance: {sum(scores[url]) / len(scores[url]):.4f}, Pages: {sorted(pages[url])}"
            )
