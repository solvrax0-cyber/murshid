"""Build a combined index from the existing corpus, raw PDFs, imports and uploads.
Usage: python -m scripts.build_index [--sample] [--show 5]"""
import argparse
import logging

from app import config
from app.index import HybridIndex

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="index the synthetic sample docs")
    ap.add_argument("--show", type=int, default=0, help="print N chunks to inspect chunking")
    a = ap.parse_args()
    folders = [config.SAMPLE_DIR, config.UPLOAD_DIR] if a.sample else config.source_folders()
    idx = HybridIndex.build(folders)  # missing folders are skipped
    types = {}
    for c in idx.chunks:
        types[c.chunk_type] = types.get(c.chunk_type, 0) + 1
    print(f"\n✔ {len(idx.chunks)} chunks from {folders} | {types} | dense={idx.has_dense}")
    for c in idx.chunks[: a.show]:
        print("-" * 70, f"\n[{c.label()}] ({c.chunk_type}, {len(c.text)} chars)\n{c.text[:400]}")
