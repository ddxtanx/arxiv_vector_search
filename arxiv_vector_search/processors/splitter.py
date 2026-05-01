from transformers import PreTrainedTokenizerBase
from functools import lru_cache
from arxiv_vector_search.documents.document import PagedDocument
import traceback
from arxiv_vector_search.documents import (
    DownloadedDocument,
    ReadError,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from multiprocess import get_context

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 0.15
DEFAULT_CHUNK_FACTOR = 3.5


class SplitError(BaseException):
    document_id: str
    message: str

    def __init__(self, document_id: str, message: str):
        self.document_id = document_id
        self.message = message
        super().__init__(f"Error splitting document {document_id}: {message}")

    def __reduce__(self):
        return SplitError, (self.document_id, self.message)


class SplitData:
    identifier: str
    page_index: int
    chunk_index: int
    text: str

    def __init__(
        self,
        identifier: str,
        page_index: int,
        chunk_index: int,
        text: str,
    ):
        self.identifier = identifier
        self.page_index = page_index
        self.chunk_index = chunk_index
        self.text = text


class DocumentSplitter:
    chunk_size: int
    chunk_overlap: int
    recursive_splitter: RecursiveCharacterTextSplitter
    prefix_length: int

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        chunk_factor: float = DEFAULT_CHUNK_FACTOR,
        tokenizer: PreTrainedTokenizerBase | None = None,
        prefix: str = "",
    ):
        if not chunk_size:
            chunk_size = DEFAULT_CHUNK_SIZE
        if not chunk_overlap:
            chunk_overlap = int(chunk_size * DEFAULT_CHUNK_OVERLAP)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.prefix_length = len(prefix)
        kwargs = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "separators": [
                "\n\n\n",
                "\n\n",
                ". \n",
                ".\n",
                ". ",
                "? ",
                "! ",
                "; ",
                "\n",
                " ",
                "",
            ],
            "add_start_index": True,
            "length_function": lambda text: len(text) + self.prefix_length,
        }
        if tokenizer is not None:
            self.prefix_length = len(
                tokenizer.encode(prefix, add_special_tokens=False, verbose=False)
            )

            @lru_cache(maxsize=100000)
            def token_length_function(text: str) -> int:
                return (
                    len(tokenizer.encode(text, add_special_tokens=False, verbose=False))
                    + self.prefix_length
                )

            kwargs["length_function"] = token_length_function

        else:
            self.chunk_size = int(chunk_size * chunk_factor)
            self.chunk_overlap = int(chunk_overlap * chunk_factor)
            kwargs["chunk_size"] = self.chunk_size
            kwargs["chunk_overlap"] = self.chunk_overlap

        self.recursive_splitter = RecursiveCharacterTextSplitter(**kwargs)

    def split_document(
        self, document: DownloadedDocument
    ) -> list[SplitData] | ReadError | SplitError:
        paged_doc = PagedDocument.from_downloaded_document(document)
        if not isinstance(paged_doc, PagedDocument):
            return paged_doc
        document_text = paged_doc.get_text()
        if not document_text.strip():
            return SplitError(document.identifier, "Document text is empty")
        try:
            chunks = self.recursive_splitter.create_documents([document_text])
            return [
                SplitData(
                    identifier=paged_doc.identifier,
                    page_index=paged_doc.start_index_to_page_index(
                        doc.metadata["start_index"]
                    ),
                    chunk_index=i,
                    text=doc.page_content,
                )
                for i, doc in enumerate(chunks)
            ]
        except Exception as e:
            print(
                f"Error splitting document {document.identifier}: {traceback.format_exc()}"
            )
            return SplitError(document.identifier, str(e))

    def split_documents(
        self, documents: list[DownloadedDocument]
    ) -> list[SplitData | ReadError | SplitError]:
        results: list[SplitData | ReadError | SplitError] = []
        for document in documents:
            try:
                split_doc = self.split_document(document)
                if isinstance(split_doc, ReadError | SplitError):
                    results.append(split_doc)
                else:
                    results.extend(split_doc)
            except SplitError | ReadError as e:
                results.append(e)
        return results

    def par_split_documents(
        self, documents: list[DownloadedDocument], num_workers: int
    ) -> list[SplitData | ReadError | SplitError]:
        if num_workers <= 1:
            return self.split_documents(documents)
        chunk_size = max(1, len(documents) // num_workers)
        ctx = get_context("forkserver")
        with ctx.Pool(processes=num_workers) as pool:
            results = list(
                pool.imap(self.split_document, documents, chunksize=chunk_size)
            )
        results_unpacked: list[SplitData | ReadError | SplitError] = []
        for result in results:
            if isinstance(result, list):
                results_unpacked.extend(result)
            else:
                results_unpacked.append(result)
        return results_unpacked
