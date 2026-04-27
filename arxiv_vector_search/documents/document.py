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


class ReadError(BaseException):
    document_id: str
    message: str

    def __init__(self, document_id: str, message: str):
        self.document_id = document_id
        self.message = message
        super().__init__(f"Error reading document {document_id}: {message}")


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


class PagedDocument(Document):
    pages: list[str]
    page_lens: list[int]
    full_text: str

    def __init__(self, identifier: str, document_type: DocumentType, pages: list[str]):
        self.identifier = identifier
        self.document_type = document_type
        self.pages = pages
        self.page_lens = [len(page) for page in pages]
        self.full_text = "\n\n".join(pages)

    @staticmethod
    def from_downloaded_document(
        document: DownloadedDocument,
    ) -> "PagedDocument | ReadError":
        pages_text = document.get_pages_text()
        if isinstance(pages_text, ReadError):
            return pages_text
        return PagedDocument(document.identifier, document.document_type, pages_text)

    def get_text(self) -> str:
        return self.full_text

    def start_index_to_page_index(self, start_index: int) -> int:
        cumulative_length = 0
        for i, page_len in enumerate(self.page_lens):
            cumulative_length += page_len + 2  # +2 for the "\n\n" separator
            if cumulative_length > start_index:
                return i
        return (
            len(self.page_lens) - 1
        )  # Return the last page index if start_index exceeds total length
