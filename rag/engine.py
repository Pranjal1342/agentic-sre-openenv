"""
rag/engine.py
Runtime RAG query engine — loads the pre-built FAISS index at startup
and serves low-latency semantic queries during episode execution.

Architecture (online/offline separation):
  Offline: offline_index.py builds and persists the FAISS index once.
  Online:  This module loads from disk at server startup.
           No external database, no internet required — satisfies the
           OpenEnv requirement for a clean `docker run` startup.

Query timing guidance (per blueprint):
  - During initial TRIAGE: surface relevant runbooks on first alert
  - During INVESTIGATION: validate hypothesis against known failure patterns
  - Pre-REMEDIATION: confirm safe command syntax before execution
"""

import os
import pickle
import logging
import pathlib
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

INDEX_DIR = pathlib.Path("assets/faiss_index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3


class RAGEngine:
    """
    Runtime hierarchical RAG query engine.

    Loads:
      - FAISS flat index (embeddings)
      - metadata.pkl (chunk texts + NetworkX graph)
      - SentenceTransformer for on-the-fly query embedding

    Usage:
        engine = RAGEngine()
        engine.load()
        results = engine.query("how to kill a blocking postgres pid", top_k=3)
    """

    def __init__(self, index_dir: pathlib.Path = INDEX_DIR) -> None:
        self._index_dir = index_dir
        self._index = None
        self._chunks: list[dict] = []
        self._graph = None
        self._model = None
        self._loaded = False

    def load(self) -> bool:
        """
        Load pre-built FAISS index and embedding model.
        Returns True on success, False if assets are not yet built.
        """
        faiss_path = self._index_dir / "index.faiss"
        meta_path = self._index_dir / "metadata.pkl"

        if not faiss_path.exists() or not meta_path.exists():
            logger.warning(
                "RAG index not found at %s. Run `python rag/offline_index.py` first.",
                self._index_dir,
            )
            return False

        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            self._index = faiss.read_index(str(faiss_path))
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self._chunks = meta["chunks"]
            self._graph = meta.get("graph")
            self._model = SentenceTransformer(EMBEDDING_MODEL)
            self._loaded = True
            logger.info(
                "RAGEngine loaded: %d chunks, %d vectors, model=%s",
                len(self._chunks), self._index.ntotal, EMBEDDING_MODEL,
            )
            return True
        except Exception as exc:
            logger.error("RAGEngine failed to load: %s", exc)
            return False

    def query(self, query_text: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """
        Semantic search over the knowledge base.

        Returns list of dicts: [{text, source, chunk_id, score}, ...]
        """
        if not self._loaded:
            logger.warning("RAGEngine not loaded — returning empty results.")
            return []

        import numpy as np
        q_embedding = self._model.encode([query_text], convert_to_numpy=True).astype("float32")
        distances, indices = self._index.search(q_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "score": round(float(1.0 / (1.0 + dist)), 4),  # Normalised similarity
            })
        return results

    def query_as_string(self, query_text: str, top_k: int = DEFAULT_TOP_K) -> str:
        """Convenience: return results as a formatted string for command_stdout."""
        results = self.query(query_text, top_k)
        if not results:
            return f"[RAG] No relevant runbooks found for: '{query_text}'"
        lines = [f"[RAG] Top {len(results)} results for: '{query_text}'\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"--- Result {i} | source: {r['source']} | score: {r['score']} ---")
            lines.append(r["text"])
            lines.append("")
        return "\n".join(lines)

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Module-level singleton — loaded once at server startup
_engine: Optional[RAGEngine] = None


def get_engine() -> RAGEngine:
    """Return the module-level RAGEngine singleton, loading if necessary."""
    global _engine
    if _engine is None:
        _engine = RAGEngine()
        _engine.load()
    return _engine
