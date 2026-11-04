#!/usr/bin/env python3
"""Build HNSW cosine indexes (CONCURRENTLY) for model embedding tables.

The pipeline creates no vector index, so every query is an exact scan of the
whole embeddings table. For corpora of hundreds of thousands of chunks that
is seconds per query; an HNSW index over the halfvec column fixes it.

Usage:
    DATABASE_URL=postgresql://... python tools/add_hnsw_index.py [--model NAME]...
        [--m 16] [--ef-construction 64]

Notes:
- CREATE INDEX CONCURRENTLY runs outside a transaction (autocommit) and does
  not block writes to the table.
- halfvec(1024) fits well under HNSW's 2000-dim limit.
- One-off cost: roughly a few minutes per million rows at these defaults.
"""

import argparse
import os
import re

from sqlalchemy import create_engine, text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model name(s) to index (default: all models in the models table).",
    )
    parser.add_argument("--m", type=int, default=16, help="HNSW m (default 16).")
    parser.add_argument(
        "--ef-construction", type=int, default=64, help="HNSW ef_construction (default 64)."
    )
    args = parser.parse_args()

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL environment variable not set.")
    engine = create_engine(db_url, isolation_level="AUTOCOMMIT")

    with engine.connect() as conn:
        if args.model:
            models = args.model
        else:
            models = [
                row[0]
                for row in conn.execute(text("SELECT name FROM models ORDER BY name"))
            ]
        if not models:
            raise SystemExit("No models found.")
        for name in models:
            table = f"embeddings_{name.replace('/', '-')}"
            if not re.fullmatch(r"[A-Za-z0-9_-]+", table):
                print(f"Skipping model '{name}': unsafe table name '{table}'.")
                continue
            idx = f"{table}_embedding_hnsw"
            print(f"Building {idx} on {table} (m={args.m}, ef_construction={args.ef_construction})...")
            conn.execute(
                text(
                    f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {idx} "
                    f"ON {table} USING hnsw (embedding vector_cosine_ops) "
                    f"WITH (m = {int(args.m)}, ef_construction = {int(args.ef_construction)})"
                )
            )
            print(f"Done: {idx}")


if __name__ == "__main__":
    main()
