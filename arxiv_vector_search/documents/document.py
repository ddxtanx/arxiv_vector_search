from pymupdf import TEXTFLAGS_TEXT
from pymupdf import TEXT_COLLECT_VECTORS
from pymupdf import TEXT_PRESERVE_IMAGES
from pymupdf import TEXT_PRESERVE_LIGATURES
from pathlib import Path
from typing import TypeVar, Generic
import enum
import pymupdf

pymupdf.TOOLS.mupdf_display_errors(False)


class DocumentType(enum.Enum):
    ARXIV = "arxiv"
    DOI = "doi"
    URL = "url"


class Document:
    identifier: str
    document_type: DocumentType

    def get_url(self) -> str:
        raise NotImplementedError("Subclasses must implement get_url method")

    def get_filename(self) -> str:
        raise NotImplementedError("Subclasses must implement get_filename method")

    def get_parent_folders(self) -> list[str]:
        raise NotImplementedError("Subclasses must implement get_parent_folders method")


class ReadError(Exception):
    document_id: str
    message: str

    def __init__(self, document_id: str, message: str):
        self.document_id = document_id
        self.message = message
        super().__init__(f"Error reading document {document_id}: {message}")

    def __reduce__(self):
        return (ReadError, (self.document_id, self.message))


class DownloadedDocument(Document):
    path: Path

    def __init__(self, identifier: str, document_type: DocumentType, path: Path):
        self.identifier = identifier
        self.document_type = document_type
        self.path = path

    def get_pages_text(self) -> list[str] | ReadError:
        try:
            with pymupdf.open(self.path) as doc:
                return [str(page.get_text()) for page in doc]
        except Exception as e:
            return ReadError(self.identifier, str(e))

    @staticmethod
    def from_document(document: Document, path: Path) -> "DownloadedDocument":
        return DownloadedDocument(document.identifier, document.document_type, path)


class SplitData:
    identifier: str
    page_index: int
    chunk_index: int
    text: str

    def __init__(self, identifier: str, page_index: int, chunk_index: int, text: str):
        self.identifier = identifier
        self.page_index = page_index
        self.chunk_index = chunk_index
        self.text = text


class SplitDocument(Document):
    splits: list[list[str]]  # indexed as [page][chunk]
    cur_page: int
    cur_chunk: int

    def __init__(
        self, identifier: str, document_type: DocumentType, splits: list[list[str]]
    ):
        self.identifier = identifier
        self.document_type = document_type
        self.splits = splits
        self.cur_page = 0
        self.cur_chunk = 0

    def __iter__(self):
        self.cur_page = 0
        self.cur_chunk = 0
        return self

    def __next__(self) -> SplitData:
        if self.cur_page >= len(self.splits):
            raise StopIteration
        page_chunks = self.splits[self.cur_page]
        if self.cur_chunk >= len(page_chunks):
            self.cur_page += 1
            self.cur_chunk = 0
            return self.__next__()
        chunk_text = page_chunks[self.cur_chunk]
        split_data = SplitData(
            identifier=self.identifier,
            page_index=self.cur_page,
            chunk_index=self.cur_chunk,
            text=chunk_text,
        )
        self.cur_chunk += 1
        return split_data


T = TypeVar("T")


class DocumentSplitIterator(Generic[T]):
    backing_obj: dict[
        str, list[list[T]]
    ]  # indexed as [document_id][page_index][chunk_index]
    document_ids: list[str]
    cur_doc_index: int
    cur_page_index: int
    cur_chunk_index: int

    def __init__(self, backing_obj: dict[str, list[list[T]]]):
        self.backing_obj = backing_obj
        self.document_ids = list(backing_obj.keys())
        self.cur_doc_index = 0
        self.cur_page_index = 0
        self.cur_chunk_index = 0

    def __iter__(self):
        self.cur_doc_index = 0
        self.cur_page_index = 0
        self.cur_chunk_index = 0
        return self

    def __next__(self) -> tuple[str, int, int, T]:
        if self.cur_doc_index >= len(self.document_ids):
            raise StopIteration
        document_id = self.document_ids[self.cur_doc_index]
        pages = self.backing_obj[document_id]
        if self.cur_page_index >= len(pages):
            self.cur_doc_index += 1
            self.cur_page_index = 0
            self.cur_chunk_index = 0
            return self.__next__()
        page_chunks = pages[self.cur_page_index]
        if self.cur_chunk_index >= len(page_chunks):
            self.cur_page_index += 1
            self.cur_chunk_index = 0
            return self.__next__()
        chunk_data = page_chunks[self.cur_chunk_index]
        if chunk_data is None:
            raise ValueError(
                f"Chunk data is None for document {document_id}, page {self.cur_page_index}, chunk {self.cur_chunk_index}"
            )
        self.cur_chunk_index += 1
        return (document_id, self.cur_page_index, self.cur_chunk_index - 1, chunk_data)
