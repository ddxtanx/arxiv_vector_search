import traceback
from arxiv_vector_search.documents import (
    DownloadedDocument,
    ReadError,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from multiprocessing import get_context

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = int(DEFAULT_CHUNK_SIZE * 0.15)
DEFAULT_CHUNK_FACTOR = 4


class SplitError(Exception):
    document_id: str
    message: str

    def __init__(self, document_id: str, message: str):
        self.document_id = document_id
        self.message = message
        super().__init__(f"Error splitting document {document_id}: {message}")


class SplitData:
    identifier: str
    section: str
    chunk_index: int
    text: str

    def __init__(self, identifier: str, section: str, chunk_index: int, text: str):
        self.identifier = identifier
        self.section = section
        self.chunk_index = chunk_index
        self.text = text


class DocumentSplitter:
    chunk_size: int
    chunk_overlap: int
    md_splitter: MarkdownHeaderTextSplitter
    recursive_splitter: RecursiveCharacterTextSplitter

    def __init__(
        self,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        chunk_factor=DEFAULT_CHUNK_FACTOR,
    ):
        self.chunk_size = chunk_size * chunk_factor
        self.chunk_overlap = chunk_overlap * chunk_factor
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Section"),
                ("##", "Subsection"),
                ("###", "Subsubsection"),
                ("####", "Paragraph"),
            ]
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n\n",
                "\n\n",
                ". \n",
                ".\n",
                ". ",
                "? ",
                "! ",
                "; ",
                "\n",
                " ",
                "",
            ],
        )

    def split_document(
        self, document: DownloadedDocument
    ) -> list[SplitData] | ReadError | SplitError:
        doc_text = document.get_text()
        if isinstance(doc_text, ReadError):
            return doc_text
        try:
            split_datum = []
            md_splits = self.md_splitter.split_text(doc_text)
            for section in md_splits:
                section_titles = list(section.metadata.values())
                deepest_section = section_titles[-1] if section_titles else ""
                content = section.page_content
                if len(content) < self.chunk_size:
                    split_datum.append(
                        SplitData(
                            identifier=document.identifier,
                            section=deepest_section,
                            chunk_index=0,
                            text=content,
                        )
                    )
                else:
                    splits = self.recursive_splitter.split_text(content)
                    for i, split in enumerate(splits):
                        split_datum.append(
                            SplitData(
                                identifier=document.identifier,
                                section=deepest_section,
                                chunk_index=i,
                                text=split,
                            )
                        )
            return split_datum
        except Exception as e:
            print(
                f"Error splitting document {document.identifier}: {traceback.format_exc()}"
            )
            return SplitError(document.identifier, str(e))

    def split_documents(
        self, documents: list[DownloadedDocument]
    ) -> list[SplitData | ReadError | SplitError]:
        results: list[SplitData | ReadError | SplitError] = []
        for document in documents:
            try:
                split_doc = self.split_document(document)
                if isinstance(split_doc, ReadError | SplitError):
                    results.append(split_doc)
                else:
                    results.extend(split_doc)
            except SplitError | ReadError as e:
                results.append(e)
        return results

    def par_split_documents(
        self, documents: list[DownloadedDocument], num_workers: int
    ) -> list[SplitData | ReadError | SplitError]:
        ctx = get_context("forkserver")
        with ctx.Pool(processes=num_workers) as pool:
            results = list(pool.imap(self.split_document, documents))
        results_unpacked: list[SplitData | ReadError | SplitError] = []
        for result in results:
            if isinstance(result, list):
                results_unpacked.extend(result)
            else:
                results_unpacked.append(result)
        return results_unpacked
