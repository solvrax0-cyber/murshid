"""Hybrid index: BM25 (keyword, robust to exact course codes / article numbers)
+ dense multilingual embeddings in FAISS (semantic, bridges dialect <-> formal wording).
Results are fused with Reciprocal Rank Fusion, optionally re-ranked with a cross-encoder."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .arabic import tokenize
from .ingest import Chunk, ingest_folder, load_chunks, save_chunks

log = logging.getLogger(__name__)

CHUNKS_FILE = "chunks.jsonl"
FAISS_FILE = "dense.faiss"
META_FILE = "meta.json"


def _passage_text(c: Chunk) -> str:
    head = c.title + (f" المادة {c.article}" if c.article else "") + (f" {c.section}" if c.section else "")
    return f"{head}\n{c.text}"


class Embedder:
    """Wraps sentence-transformers; handles the e5 'query:'/'passage:' prefixes."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.e5 = "e5" in model_name.lower()

    def encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        if self.e5:
            texts = [("query: " if is_query else "passage: ") + t for t in texts]
        v = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype="float32")


_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder | None:
    global _EMBEDDER
    if not config.EMBEDDING_MODEL:
        return None
    if _EMBEDDER is None:
        try:
            _EMBEDDER = Embedder(config.EMBEDDING_MODEL)
        except Exception as e:  # offline / not installed -> keyword-only mode
            log.warning("dense embeddings disabled (%s); falling back to BM25 only", e)
            config.EMBEDDING_MODEL = ""
            return None
    return _EMBEDDER


@dataclass
class Hit:
    chunk: Chunk
    rrf: float
    bm25: float = 0.0
    dense: float = 0.0
    rerank: float | None = None
    coverage: float = 0.0


class HybridIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray | None):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(_passage_text(c)) for c in chunks]) if chunks else None
        self.faiss = None
        if vectors is not None and len(vectors):
            import faiss

            self.faiss = faiss.IndexFlatIP(vectors.shape[1])
            self.faiss.add(vectors)
        self._reranker = None

    # ---------------------------------------------------------- build / load
    @classmethod
    def build(cls, folders: list[Path], out_dir: Path = config.INDEX_DIR) -> "HybridIndex":
        chunks: list[Chunk] = []
        for f in folders:
            if f.exists():
                chunks.extend(ingest_folder(f))
        out_dir.mkdir(parents=True, exist_ok=True)
        save_chunks(chunks, out_dir / CHUNKS_FILE)
        vectors = None
        emb = get_embedder()
        if emb and chunks:
            vectors = emb.encode([_passage_text(c) for c in chunks], is_query=False)
            import faiss

            idx = faiss.IndexFlatIP(vectors.shape[1])
            idx.add(vectors)
            faiss.write_index(idx, str(out_dir / FAISS_FILE))
        elif (out_dir / FAISS_FILE).exists():
            (out_dir / FAISS_FILE).unlink()
        (out_dir / META_FILE).write_text(json.dumps({
            "embedding_model": config.EMBEDDING_MODEL if vectors is not None else "",
            "n_chunks": len(chunks), "sources": sorted({c.source for c in chunks}),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("index built: %d chunks, dense=%s", len(chunks), vectors is not None)
        return cls(chunks, vectors)

    @classmethod
    def load(cls, out_dir: Path = config.INDEX_DIR) -> "HybridIndex":
        chunks = load_chunks(out_dir / CHUNKS_FILE)
        vectors = None
        meta_path = out_dir / META_FILE
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if (out_dir / FAISS_FILE).exists() and meta.get("embedding_model") and get_embedder():
            if meta["embedding_model"] != config.EMBEDDING_MODEL:
                log.warning("index built with %s but EMBEDDING_MODEL=%s -> rebuild the index",
                            meta["embedding_model"], config.EMBEDDING_MODEL)
            import faiss

            idx = faiss.read_index(str(out_dir / FAISS_FILE))
            vectors = idx.reconstruct_n(0, idx.ntotal)
        return cls(chunks, vectors)

    # ---------------------------------------------------------- search
    def search(self, queries: list[str], k: int = config.TOP_K, candidates: int = config.CANDIDATES,
               doc_type: str | None = None) -> list[Hit]:
        if not self.chunks:
            return []
        queries = [q for q in dict.fromkeys(queries) if q and q.strip()]
        rrf: dict[int, float] = {}
        best_bm25: dict[int, float] = {}
        best_dense: dict[int, float] = {}
        K = 60  # RRF constant

        def allowed(i: int) -> bool:
            return doc_type is None or self.chunks[i].doc_type == doc_type

        # One ranking per retriever (max score over query variants), then RRF across retrievers.
        # (RRF over every variant would over-reward chunks that weakly match all variants.)
        for q in queries:
            toks = tokenize(q)
            if toks and self.bm25:
                scores = self.bm25.get_scores(toks)
                for i in np.argsort(-scores)[: candidates * 2]:
                    if scores[i] > 0 and allowed(int(i)):
                        best_bm25[int(i)] = max(best_bm25.get(int(i), 0), float(scores[i]))
        emb = get_embedder() if self.faiss is not None else None
        if emb:
            qv = emb.encode(queries, is_query=True)
            sims, ids = self.faiss.search(qv, min(candidates * 2, len(self.chunks)))
            for row_i, row_s in zip(ids, sims):
                for i, s in zip(row_i, row_s):
                    if i >= 0 and allowed(int(i)):
                        best_dense[int(i)] = max(best_dense.get(int(i), 0), float(s))
        for ranking in (best_bm25, best_dense):
            for r, (i, _) in enumerate(sorted(ranking.items(), key=lambda x: -x[1])[:candidates]):
                rrf[i] = rrf.get(i, 0) + 1 / (K + r + 1)

        qsets = [set(tokenize(q)) for q in queries]
        qsets = [s for s in qsets if s]
        hits = []
        for i, score in sorted(rrf.items(), key=lambda x: -x[1])[: max(k * 3, 10)]:
            ctoks = set(tokenize(_passage_text(self.chunks[i])))
            # term coverage = best share of any query variant's terms found in the chunk
            cov = max((len(s & ctoks) / len(s) for s in qsets), default=0.0)
            hits.append(Hit(self.chunks[i], score, best_bm25.get(i, 0.0), best_dense.get(i, 0.0), coverage=cov))

        if config.RERANK_MODEL and hits:
            hits = self._rerank(queries[0], hits)
        return hits[:k]

    def _rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        try:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(config.RERANK_MODEL)
            scores = self._reranker.predict([(query, h.chunk.text) for h in hits])
            for h, s in zip(hits, scores):
                h.rerank = float(s)
            return sorted(hits, key=lambda h: -(h.rerank or 0))
        except Exception as e:
            log.warning("reranker disabled: %s", e)
            config.RERANK_MODEL = ""
            return hits

    @property
    def has_dense(self) -> bool:
        return self.faiss is not None
