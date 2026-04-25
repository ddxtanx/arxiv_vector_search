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
    min_len: int

    def __init__(
        self,
        identifier: str,
        document_type: DocumentType,
        splits: list[list[str]],
        min_len: int = -1,
    ):
        self.identifier = identifier
        self.document_type = document_type
        self.splits = splits
        self.cur_page = 0
        self.cur_chunk = 0
        self.min_len = min_len

    def set_min_len(self, min_len: int):
        self.min_len = min_len

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
        if self.min_len > 0 and len(chunk_text) < self.min_len:
            self.cur_chunk += 1
            return self.__next__()
        split_data = SplitData(
            identifier=self.identifier,
            page_index=self.cur_page,
            chunk_index=self.cur_chunk,
            text=chunk_text,
        )
        self.cur_chunk += 1
        return split_data
