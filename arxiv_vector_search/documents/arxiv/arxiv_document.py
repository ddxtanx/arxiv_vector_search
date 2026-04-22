from arxiv_vector_search.documents.document import Document, DocumentType


class ArxivDocument(Document):
    def __init__(self, identifier: str):
        self.identifier = identifier
        self.document_type = DocumentType.ARXIV

    def get_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.identifier}.pdf"

    def get_gcloud_blob_name(self) -> str:
        if "/" in self.identifier:
            subj, id = self.identifier.rsplit("/", 1)
            year_month = id[0:4]
            return f"arxiv/{subj}/pdf/{year_month}/{id}.pdf"
        else:
            year_month = self.identifier.split(".")[0]
            return f"arxiv/arxiv/pdf/{year_month}/{self.identifier}.pdf"

    def get_filename(self) -> str:
        return f"{self.identifier}.pdf"

    def get_parent_folders(self) -> list[str]:
        return self.identifier.split("/")[:-1] if "/" in self.identifier else []
