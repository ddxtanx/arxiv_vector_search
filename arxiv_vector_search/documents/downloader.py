from .document import Document, DocumentType, DownloadedDocument


class DownloadError(Exception):
    document: Document

    def __init__(self, document: Document, message: str):
        super().__init__(message)
        self.document = document


class Downloader:
    docs: list[Document]

    def add_document(self, document: Document):
        raise NotImplementedError("Subclasses must implement add_document method")

    def add_documents(self, documents: list[Document]):
        for document in documents:
            self.add_document(document)

    def batch_download(
        self, batch_size: int
    ) -> list[DownloadedDocument | DownloadError]:
        raise NotImplementedError("Subclasses must implement batch_download method")

    def clear(self):
        raise NotImplementedError("Subclasses must implement clear method")


class DocumentDownloader:
    documents: dict[DocumentType, list[Document]]
    downloaders: dict[DocumentType, Downloader]

    def __init__(self):
        self.documents = {
            DocumentType.ARXIV: [],
            DocumentType.DOI: [],
            DocumentType.URL: [],
        }
        self.downloaders = {}

    def register_downloader(self, document_type: DocumentType, downloader: Downloader):
        self.downloaders[document_type] = downloader

    def add_document(self, document: Document):
        self.documents[document.document_type].append(document)

    def add_documents(self, documents: list[Document]):
        for document in documents:
            self.add_document(document)

    def batch_download(
        self, batch_size: int
    ) -> list[DownloadedDocument | DownloadError]:
        results: list[DownloadedDocument | DownloadError] = []
        for doc_type, docs in self.documents.items():
            downloader = self.downloaders[doc_type]
            downloader.add_documents(docs)
            results.extend(downloader.batch_download(batch_size=batch_size))
        return results

    def clear_downloaders(self):
        for downloader in self.downloaders.values():
            downloader.clear()
        self.documents = {
            DocumentType.ARXIV: [],
            DocumentType.DOI: [],
            DocumentType.URL: [],
        }
