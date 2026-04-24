from arxiv_vector_search.documents import (
    Document,
    DownloadedDocument,
    SplitDocument,
    ReadError,
    DocumentSplitIterator,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from multiprocessing import get_context

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


class Splits:
    split_dict: dict[str, list[list[str]]]
    errs: list[SplitError | ReadError]

    def __init__(self):
        self.split_dict = {}
        self.errs = []

    def add_split_doc(self, split_doc: SplitDocument):
        self.split_dict[split_doc.identifier] = split_doc.splits

    def add_error(self, error: SplitError | ReadError):
        self.errs.append(error)

    def __iter__(self):
        return DocumentSplitIterator[str](self.split_dict)

    def get_errors(self) -> list[SplitError | ReadError]:
        return self.errs

    def get_num_successful_splits(self) -> int:
        return len(self.split_dict)

    def get_num_errors(self) -> int:
        return len(self.errs)


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

    def split_documents(self, documents: list[DownloadedDocument]) -> Splits:
        results: list[SplitDocument | ReadError | SplitError] = []
        for document in documents:
            try:
                split_doc = self.split_document(document)
                results.append(split_doc)
            except SplitError | ReadError as e:
                results.append(e)
        split_obj = Splits()
        for result in results:
            if isinstance(result, SplitDocument):
                split_obj.add_split_doc(result)
            else:
                split_obj.add_error(result)
        return split_obj

    def par_split_documents(
        self, documents: list[DownloadedDocument], num_workers: int
    ) -> Splits:
        ctx = get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            results = list(pool.imap(self.split_document, documents))
        split_obj = Splits()
        for result in results:
            if isinstance(result, SplitDocument):
                split_obj.add_split_doc(result)
            else:
                split_obj.add_error(result)
        return split_obj
