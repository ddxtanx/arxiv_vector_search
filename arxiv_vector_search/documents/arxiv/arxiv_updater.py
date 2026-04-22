import kagglehub
from kagglehub import KaggleDatasetAdapter
from .arxiv_document import ArxivDocument


def create_arxiv_documents():
    df_stream = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "Cornell-University/arxiv",
        "arxiv-metadata-oai-snapshot.json",
        pandas_kwargs={
            "orient": "records",
            "lines": True,
            "chunksize": 10000,
            "dtype": {
                "id": str,
                "categories": str,
                "comments": str,
                "versions": object,
            },
        },
    )
    documents = []
    for df in df_stream:
        documents.extend(
            [
                ArxivDocument(
                    str(ident) + versions[-1]["version"],
                )
                for ident, comments, categories, versions in zip(
                    df["id"],
                    df["comments"],
                    df["categories"],
                    df["versions"],
                )
                if "math" in categories
                and (
                    not isinstance(comments, str) or "withdrawn" not in comments.lower()
                )
            ]
        )
    return documents
