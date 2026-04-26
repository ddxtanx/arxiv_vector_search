from pathlib import Path
import enum
import pymupdf
import pymupdf4llm
import re

pymupdf.TOOLS.mupdf_display_errors(False)


class DocumentType(enum.Enum):
    ARXIV = "arxiv"
    DOI = "doi"
    URL = "url"


omit_regex = re.compile(
    r"\*\*==> [a-zA-Z]+ \[[0-9]+ x [0-9]+\] intentionally omitted <==\*\*\n\n"
)


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

    def get_text(self) -> str | ReadError:
        try:
            with pymupdf.open(self.path) as doc:
                page_chunk_list = pymupdf4llm.to_markdown(
                    doc,
                    extract_words=False,
                    footer=False,
                    header=False,
                    ignore_graphics=True,
                    ignore_images=True,
                    page_chunks=True,
                    show_progress=False,
                    use_ocr=False,
                )
        except Exception as e:
            return ReadError(self.identifier, str(e))
        all_text = "\n\n".join(page["text"] for page in page_chunk_list)
        all_text = omit_regex.sub("", all_text)
        return all_text

    @staticmethod
    def from_document(document: Document, path: Path) -> "DownloadedDocument":
        return DownloadedDocument(document.identifier, document.document_type, path)
