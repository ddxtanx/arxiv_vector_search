from pathlib import Path
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
    document: Document
    message: str

    def __init__(self, document: Document, message: str):
        self.document = document
        self.message = message
        super().__init__(f"Error reading document {document.identifier}: {message}")


class DownloadedDocument(Document):
    path: Path

    def __init__(self, identifier: str, document_type: DocumentType, path: Path):
        self.identifier = identifier
        self.document_type = document_type
        self.path = path

    def get_pages_text(self) -> list[str] | ReadError:
        try:
            doc = pymupdf.open(self.path)
            return [page.get_text() for page in doc]
        except Exception as e:
            return ReadError(document=self, message=str(e))

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
