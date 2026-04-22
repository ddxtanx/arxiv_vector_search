import os

from google.cloud.storage.blob import Blob
from google.cloud.storage import Client, transfer_manager, Bucket
from arxiv_vector_search.documents import Downloader, DownloadedDocument, DownloadError
from .arxiv_document import ArxivDocument


class ArxivDownloader(Downloader):
    dl_path: str = "pdfs/arxiv/"
    client: Client
    bucket: Bucket
    docs: list[ArxivDocument]

    def __init__(self):
        self.client = Client()
        self.bucket = self.client.bucket("arxiv-dataset")
        self.docs = []
        os.makedirs(self.dl_path, exist_ok=True)

    def add_document(self, document: ArxivDocument):
        self.docs.append(document)

    def batch_download(
        self, batch_size: int
    ) -> list[DownloadedDocument | DownloadError]:
        blob_pairs = [
            (
                Blob(doc.get_gcloud_blob_name(), self.bucket),
                os.path.join(self.dl_path, doc.get_filename()),
            )
            for doc in self.docs
        ]
        download_results = transfer_manager.download_many(
            blob_pairs, max_workers=batch_size
        )
        results: list[DownloadedDocument | DownloadError] = []
        for doc, result in zip(self.docs, download_results):
            if isinstance(result, Exception):
                results.append(DownloadError(document=doc, message=str(result)))
            else:
                results.append(
                    DownloadedDocument.from_document(
                        document=doc,
                        path=os.path.join(self.dl_path, doc.get_filename()),
                    )
                )
        return results

    def clear(self):
        self.docs = []
        for path, _, files in os.walk(self.dl_path):
            for file in files:
                os.remove(os.path.join(path, file))
