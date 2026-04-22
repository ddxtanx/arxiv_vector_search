from arxiv_vector_search.documents import (
    Document,
    DownloadedDocument,
    SplitDocument,
    ReadError,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from multiprocessing import Pool

DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = int(DEFAULT_CHUNK_SIZE * 0.15)
DEFAULT_CHUNK_FACTOR = 3


class SplitError(Exception):
    document: Document
    message: str

    def __init__(self, document: Document, message: str):
        self.document = document
        self.message = message
        super().__init__(f"Error splitting document {document.identifier}: {message}")


class DocumentSplitter:
    chunk_size: int
    chunk_overlap: int
    splitter: TextSplitter

    def __init__(
        self, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split_document(
        self, document: DownloadedDocument
    ) -> SplitDocument | ReadError | SplitError:
        pages_text = document.get_pages_text()
        if isinstance(pages_text, ReadError):
            raise pages_text
        try:
            splits = [self.splitter.split_text(page) for page in pages_text]
            return SplitDocument(
                document.identifier,
                document.document_type,
                splits,
            )
        except Exception as e:
            raise SplitError(document=document, message=str(e))

    def split_documents(
        self, documents: list[DownloadedDocument]
    ) -> list[SplitDocument | ReadError | SplitError]:
        results: list[SplitDocument | ReadError | SplitError] = []
        for document in documents:
            try:
                split_doc = self.split_document(document)
                results.append(split_doc)
            except SplitError | ReadError as e:
                results.append(e)
        return results

    def par_split_documents(
        self, documents: list[DownloadedDocument], num_workers: int
    ) -> list[SplitDocument | SplitError | ReadError]:
        with Pool(processes=num_workers) as pool:
            results = pool.map(self.split_document, documents)
        return results
