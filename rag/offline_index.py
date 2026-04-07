"""
rag/offline_index.py
Offline FAISS index builder — run ONCE during Docker Stage 1 build.

Pipeline:
  1. Load SRE runbooks from knowledge_base/ using `unstructured` (title-chunking)
  2. Embed chunks via SentenceTransformer('all-MiniLM-L6-v2')
  3. Model hierarchical chunk relationships as a NetworkX directed graph
  4. Save FAISS index + metadata pickle to assets/faiss_index/

Output: assets/faiss_index/index.faiss + assets/faiss_index/metadata.pkl

Run with: python rag/offline_index.py
"""

import os
import pickle
import pathlib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = pathlib.Path("knowledge_base")
OUTPUT_DIR = pathlib.Path("assets/faiss_index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_STRATEGY = "by_title"   # unstructured layout-aware chunking


def _load_chunks_unstructured() -> list[dict]:
    """
    Load and chunk documents using `unstructured` with title-based strategy.
    Preserves document hierarchy (title → paragraph → table relationships).
    """
    try:
        from unstructured.partition.md import partition_md
        from unstructured.chunking.title import chunk_by_title
    except ImportError:
        logger.warning("unstructured not available — falling back to simple line splitting.")
        return _load_chunks_simple()

    chunks = []
    for fpath in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        logger.info("Parsing: %s", fpath.name)
        elements = partition_md(filename=str(fpath))
        title_chunks = chunk_by_title(elements, max_characters=512)
        for i, chunk in enumerate(title_chunks):
            chunks.append({
                "source": fpath.name,
                "chunk_id": f"{fpath.stem}_{i}",
                "text": str(chunk),
                "parent": fpath.stem,
            })
        logger.info("  → %d chunks from %s", len(title_chunks), fpath.name)
    return chunks


def _load_chunks_simple() -> list[dict]:
    """Fallback: plain line-splitting if unstructured is unavailable."""
    chunks = []
    for fpath in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        text = fpath.read_text(encoding="utf-8")
        sections = [s.strip() for s in text.split("\n\n") if s.strip()]
        for i, section in enumerate(sections):
            chunks.append({
                "source": fpath.name,
                "chunk_id": f"{fpath.stem}_{i}",
                "text": section[:512],
                "parent": fpath.stem,
            })
    return chunks


def _build_nx_graph(chunks: list[dict]):
    """
    Build a NetworkX directed graph modelling hierarchical chunk relationships.
    Document root → title chunks → sub-chunks.
    """
    import networkx as nx
    G = nx.DiGraph()
    parent_tracker: dict[str, str] = {}

    for chunk in chunks:
        cid = chunk["chunk_id"]
        G.add_node(cid, text=chunk["text"], source=chunk["source"])
        parent = chunk.get("parent")
        if parent:
            if parent not in G:
                G.add_node(parent, text="", source=chunk["source"])
            G.add_edge(parent, cid, relation="contains")
        parent_tracker[cid] = parent or ""

    logger.info("NetworkX graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def build_index() -> None:
    """Main entry point — build and persist FAISS index."""
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and chunk runbooks
    logger.info("Loading knowledge base from: %s", KNOWLEDGE_BASE_DIR)
    chunks = _load_chunks_unstructured()
    if not chunks:
        logger.error("No chunks found in knowledge_base/ — check that .md files exist.")
        return
    logger.info("Total chunks: %d", len(chunks))

    # 2. Embed chunks
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    logger.info("Embedding %d chunks…", len(texts))
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    # 3. Build NetworkX graph
    graph = _build_nx_graph(chunks)

    # 4. Build FAISS flat L2 index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)

    # 5. Persist assets
    faiss_path = OUTPUT_DIR / "index.faiss"
    meta_path = OUTPUT_DIR / "metadata.pkl"

    faiss.write_index(index, str(faiss_path))
    with open(meta_path, "wb") as f:
        pickle.dump({"chunks": chunks, "graph": graph}, f)

    logger.info("Index saved → %s", faiss_path)
    logger.info("Metadata saved → %s", meta_path)
    logger.info("Offline index build complete.")


if __name__ == "__main__":
    build_index()
