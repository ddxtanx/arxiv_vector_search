"""Adapter for the experimental Rust split engine (rust/fastsplit).

`RustSplitEngine` mirrors DocumentSplitter's parallel interface: it shells
out to the fastsplit binary, which extracts PDF text (pdf_oxide), applies
the same cleanup passes, and chunks with a byte-for-byte port of the
langchain RecursiveCharacterTextSplitter used here (verified by
tools/validate_fastsplit.py), parallelized across documents with rayon.

Build:  cargo build --release --manifest-path rust/fastsplit/Cargo.toml

Caveats vs the Python path:
- The extractor is pdf_oxide, not MuPDF, so chunk *text* differs slightly
  (different engines, same cleanup passes). A document must be embedded
  end-to-end with one engine; don't mix engines within a model's rows.
- Requires the release binary to be built first; if missing, constructing
  the engine raises with build instructions.
"""

import json
import os
import subprocess
from pathlib import Path

from arxiv_vector_search.documents import DownloadedDocument, ReadError
from arxiv_vector_search.processors.splitter import SplitData, SplitError

DEFAULT_BINARY = (
    Path(__file__).resolve().parents[2] / "rust" / "fastsplit" / "target" / "release" / "fastsplit"
)


class RustSplitEngine:
    def __init__(
        self,
        binary: str | os.PathLike | None = None,
        chunk_size: int = 512,
        chunk_factor: float = 3.5,
        chunk_overlap: int | None = None,
        min_chunk_chars: int = 128,
    ):
        self.binary = Path(binary) if binary else DEFAULT_BINARY
        if not self.binary.exists():
            raise FileNotFoundError(
                f"fastsplit binary not found at {self.binary}. Build it with: "
                "cargo build --release --manifest-path rust/fastsplit/Cargo.toml"
            )
        self.chunk_chars = int(chunk_size * chunk_factor)
        self.chunk_overlap = (
            int(chunk_overlap * chunk_factor) if chunk_overlap is not None else int(chunk_size * 0.15 * chunk_factor)
        )
        self.min_chunk_chars = min_chunk_chars

    def _run(self, paths: list[str]) -> list[dict]:
        cmd = [
            str(self.binary),
            "--mode",
            "extract",
            "--chunk-chars",
            str(self.chunk_chars),
            "--chunk-overlap",
            str(self.chunk_overlap),
            "--min-chars",
            str(self.min_chunk_chars),
            *paths,
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"fastsplit failed ({proc.returncode}): {proc.stderr.decode()[:400]}")
        out = []
        for line in proc.stdout.split(b"\n"):
            if line.strip():
                out.append(json.loads(line.decode("utf-8", "surrogatepass")))
        return out

    def par_split_documents(
        self, documents: list[DownloadedDocument], num_workers: int
    ) -> list[SplitData | ReadError | SplitError]:
        # num_workers maps to rayon threads inside the binary.
        import os as _os

        env_backup = _os.environ.get("RAYON_NUM_THREADS")
        if num_workers > 0:
            _os.environ["RAYON_NUM_THREADS"] = str(num_workers)
        try:
            by_ident = {str(d.path): d.identifier for d in documents}
            results = self._run([str(d.path) for d in documents])
        finally:
            if env_backup is None:
                _os.environ.pop("RAYON_NUM_THREADS", None)
            else:
                _os.environ["RAYON_NUM_THREADS"] = env_backup
        seen = set()
        splits: list[SplitData | ReadError | SplitError] = []
        for o in results:
            # fastsplit derives identifiers from path stems; map back exactly
            ident = next((i for p, i in by_ident.items() if Path(p).stem == o["identifier"]), o["identifier"])
            seen.add(ident)
            splits.append(
                SplitData(
                    identifier=ident,
                    page_index=o["page_index"],
                    chunk_index=o["chunk_index"],
                    text=o["text"],
                )
            )
        for d in documents:
            if Path(str(d.path)).stem not in {o["identifier"] for o in results}:
                splits.append(
                    SplitError(d.identifier, "fastsplit produced no chunks (unreadable PDF?)")
                )
        return splits
