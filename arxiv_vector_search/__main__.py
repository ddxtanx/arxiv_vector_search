from arxiv_vector_search.documents import DownloadError
from arxiv_vector_search.documents.document import PagedDocument
from arxiv_vector_search.processors.splitter import SplitData, DocumentSplitter
from arxiv_vector_search.processors.embedder import (
    TOKEN_CHUNKSIZE,
    Embedder,
    TOKEN_OVERHEAD_FACTOR,
)
from arxiv_vector_search.processors.llama_cpp_embedder import RemoteEmbedder
from arxiv_vector_search.db.tables import EmbeddingState
from arxiv_vector_search.documents.document import DownloadedDocument, DocumentType
from .db import Database
from .documents import DocumentDownloader
import sys
import os
from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
import gc
import codecs


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
    embedding_url: str
    embedding_api_key: str
    concurrency: int
    rerank_model: str
    rerank_candidates: int


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
        "--embedding_url",
        type=str,
        default=os.getenv("EMBEDDING_BASE_URL"),
        help=(
            "Base URL of an OpenAI-compatible embedding server (e.g. "
            "http://127.0.0.1:8194/v1 served by llama.cpp/llama-swap). When "
            "set, embeddings are requested over HTTP instead of locally with "
            "torch. Also settable via EMBEDDING_BASE_URL."
        ),
    )
    _ = parser.add_argument(
        "--embedding_api_key",
        type=str,
        default=os.getenv("EMBEDDING_API_KEY"),
        help="Optional API key for the embedding server (EMBEDDING_API_KEY).",
    )
    _ = parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=(
            "Concurrent embedding requests against a remote server "
            "(EMBED_CONCURRENCY, default 4). Keep low if the server also "
            "serves interactive traffic."
        ),
    )

    _ = parser.add_argument(
        "--rerank_model",
        type=str,
        default=os.getenv("RERANK_MODEL"),
        help=(
            "Reranker model id served by the same llama-swap router (e.g. "
            "QwenReranking). When set, --query re-scores the vector-search "
            "candidates with the cross-encoder. Needs chunk text stored "
            "(rows embedded with the current schema). RERANK_MODEL env also "
            "works."
        ),
    )
    _ = parser.add_argument(
        "--rerank_candidates",
        type=int,
        default=50,
        help="Number of vector-search candidates sent to the reranker (default 50).",
    )

    _ = parser.add_argument(
        "--split_engine",
        type=str,
        choices=["python", "rust"],
        default="python",
        help=(
            "Document splitting engine. 'rust' uses the rayon-parallel "
            "fastsplit binary (rust/fastsplit; ~2-3x faster on the extract+"
            "split stage, character-based chunking only — build the binary "
            "first). 'python' is the default MuPDF/langchain path."
        ),
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
    model_chunk_size = TOKEN_CHUNKSIZE
    if args.model not in model_names_in_db:
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
        model_chunk_size_input = input(
            f"Enter a chunk size for splitting documents for this model (optional, can be left blank for default {TOKEN_CHUNKSIZE}): "
        )
        if model_chunk_size_input.strip():
            model_chunk_size = int(model_chunk_size_input)
    else:
        # Reuse the settings stored when the model was first registered.
        # Previously only batch_size was reused, so re-embedding with an
        # existing model silently used different prefixes/chunk sizes than
        # the rows already in the table.
        model_row = next(m for m in models if m.name == args.model)
        embedding_batch_size = model_row.batch_size
        model_doc_prefix = model_row.document_prefix or ""
        model_query_prefix = model_row.query_prefix or ""
        model_chunk_size = model_row.chunk_size
    if args.embedding_url:
        embedder = RemoteEmbedder(
            args.model,
            base_url=args.embedding_url,
            api_key=args.embedding_api_key,
            batch_size=embedding_batch_size,
            document_prefix=model_doc_prefix,
            query_prefix=model_query_prefix,
            chunk_size=model_chunk_size,
            concurrency=args.concurrency,
        )
    else:
        embedder = Embedder(
            args.model,
            embedding_batch_size,
            document_prefix=model_doc_prefix,
            query_prefix=model_query_prefix,
            chunk_size=model_chunk_size,
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

    if args.embed:
        # Imported lazily: this pulls in google-cloud-storage, which is only
        # needed for the arXiv download path.
        from arxiv_vector_search.documents.arxiv import ArxivDownloader

        db.add_missing_metadata(embedder)
        downloader = DocumentDownloader()
        arxiv_downloader = ArxivDownloader()
        downloader.register_downloader(DocumentType.ARXIV, arxiv_downloader)
        if args.split_engine == "rust":
            from arxiv_vector_search.processors.rust_splitter import RustSplitEngine

            if embedder.get_tokenizer() is not None:
                print(
                    "Warning: the rust split engine chunks by characters only; "
                    "a tokenizer is configured but will not be used for chunk sizing."
                )
            splitter = RustSplitEngine(chunk_size=embedder.chunk_size)
        else:
            splitter = DocumentSplitter(
                embedder.chunk_size,
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
            del good_batch
            print("Batch processing complete. Moving on to next batch...")
            gc.collect()

    if args.query:
        query_embedding = embedder.embed_queries([args.query])[0]
        print("Query embedding generated. Performing vector search...")
        reranker = None
        if args.rerank_model:
            from arxiv_vector_search.processors.reranker import RemoteReranker

            reranker = RemoteReranker(
                args.rerank_model, base_url=args.embedding_url, api_key=args.embedding_api_key
            )
        if reranker is not None:
            candidates = db.query_embeddings(
                embedder, query_embedding, top_k=args.rerank_candidates
            )
            with_text = [r for r in candidates if r.text]
            if not with_text:
                print(
                    "Reranking requested but no stored chunk text is available "
                    "(rows predate the text column). Falling back to vector-only "
                    "results; re-embed to populate chunk text."
                )
            else:
                order = reranker.rerank(args.query, [r.text for r in with_text])
                print(
                    f"\nTop {min(10, len(order))} results after reranking {len(with_text)} candidates:"
                )
                for rank, (ci, score) in enumerate(order[:10], start=1):
                    r = with_text[ci]
                    print(
                        f"{rank:2d}. {r.get_url()} p{r.page_index} rerank={score:.4f} vec_dist={r.distance:.4f}"
                    )
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

        avg_results = db.query_embeddings_avg_by_doc(
            embedder, query_embedding, top_k=10
        )
        print("\nTop 10 results sorted by average distance:")
        for query_result in avg_results:
            url = query_result.document.get_url()
            print(f"URL: {url}, Avg Distance: {query_result.distance:.4f}")

        # top_10_by_avg = db.query_embeddings_avg(embedder, query_embedding, top_k=10)
        # print("\nTop 10 results sorted by average distance:")
        # for query_result in top_10_by_avg:
        #     url = query_result.document.get_url()
        #     print(f"URL: {url}, Avg Distance: {query_result.distance:.4f}")
