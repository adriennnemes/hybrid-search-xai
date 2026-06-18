"""
Indexing layer.

We build two indexes:
1) ChromaDB collection with embeddings (semantic search)
2) BM25 payload (lexical search)

We keep it simple:
- documents are chunks (title + abstract)
- embeddings are computed by SentenceTransformer
- BM25 uses simple tokenization (regex)

Important design choice to avoid overwriting between models.
We store BM25 payload per *collection name*:
- collection = "xai_papers" -> bm25__xai_papers.joblib
- collection = "xai_papers_mpnet"  -> bm25__xai_papers_mpnet.joblib

This keeps MiniLM and MPNet runs isolated.
"""

from pathlib import Path
from typing import List, Dict, Any
from chromadb.config import Settings as ChromaSettings
from ..storage.db import load_papers, DEFAULT_DB_FILE
from .embeddings import chunk_text, embed_texts
from ..core.config import settings
import os
import re
import joblib
import hashlib
import chromadb
import logging

logger = logging.getLogger(__name__)

# Store payload files inside app/data (same area as papers.json)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Use a collection name to separate indexes between models (MiniLM vs MPNet)
COLLECTION_NAME = settings.CHROMA_COLLECTION

_TOKEN_RE = re.compile(r"[a-z0-9_]+(?:-[a-z0-9_]+)*", re.IGNORECASE)


def _safe_filename(s: str) -> str:
    """
    Make a safe, stable filename from a string.
    - We use it to create BM25 file names per collection.
    - It prevents overwriting when running multiple models.
    """
    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s or "default"


# Store BM25 per collection to avoid overwriting between models/collections
BM25_PATH = DATA_DIR / f"bm25__{_safe_filename(COLLECTION_NAME)}.joblib"


def _get_chroma_client() -> chromadb.Client:
    """
    In Docker Compose we run Chroma as a separate service (HttpClient).
    If CHROMA_HOST is empty, fallback to a local PersistentClient.
    """
    host = (settings.CHROMA_HOST or "").strip()
    port = int(settings.CHROMA_PORT)

    # Local fallback path (only used when host is empty)
    persist_path = os.getenv("CHROMA_PATH", str(DATA_DIR / "chroma"))

    if host:
        return chromadb.HttpClient(host=host, port=port)

    return chromadb.PersistentClient(
        path=persist_path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection():
    """
    We explicitly request cosine space.
    Chroma returns distances; in cosine space: distance ≈ 1 - cosine_similarity
    """
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_indexes() -> Dict[str, Any]:
    """
    Clean start:
    - remove Chroma collection (for CURRENT collection name only)
    - remove BM25 file (for CURRENT collection name only)

    If we use different collection names for different models, this reset will not delete the other model's indexes.
    """
    logger.info("Resetting indexes (Chroma collection + BM25 payload)")

    client = _get_chroma_client()

    existing = [c.name for c in client.list_collections()]
    deleted_collection = False
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        deleted_collection = True

    deleted_bm25 = False
    if BM25_PATH.exists():
        BM25_PATH.unlink()
        deleted_bm25 = True

    return {
        "deleted_collection": deleted_collection,
        "deleted_bm25_file": deleted_bm25,
        "collection_name": COLLECTION_NAME,
        "bm25_path": str(BM25_PATH),
    }


def tokenize_for_bm25(text: str) -> List[str]:
    """
    BM25 tokenizer that preserves hyphenated terms and their parts.

    Example:
      "post-hoc explainability" -> ["post-hoc", "post", "hoc", "explainability"]
    """
    toks = _TOKEN_RE.findall((text or "").lower())
    out: List[str] = []

    for t in toks:
        out.append(t)
        if "-" in t:
            out.extend([p for p in t.split("-") if p])

    return out


def _to_scalar_str(value: Any) -> str:
    """
    Chroma metadata values must be scalar (str/int/float/bool/None).
    Authors/categories can be lists -> we convert them to a single string.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join([str(v).strip() for v in value if str(v).strip()])
    return str(value).strip()


def build_chunks_from_papers(
    papers: List[Dict],
    max_tokens: int = 224,
    overlap_tokens: int = 32,
) -> List[Dict]:
    """
    Create chunk records: {id, text, metadata}

    - Chroma metadata values must be scalar.
    - Chunk IDs must be stable + unique.
      We base them on arXiv id (or fallback) + chunk index.
    """
    chunks: List[Dict] = []

    for p in papers:
        # Prefer arxiv_id, fallback to id/abs_url/title
        arxiv_id = _to_scalar_str(p.get("arxiv_id")) or _to_scalar_str(p.get("id"))
        title = _to_scalar_str(p.get("title"))
        summary = _to_scalar_str(p.get("summary"))

        authors_str = _to_scalar_str(p.get("authors"))
        categories_str = _to_scalar_str(p.get("categories"))

        full_text = (title + "\n\n" + summary).strip()
        if not full_text:
            continue

        # Stable base id so reindex does not randomly change IDs
        base = arxiv_id or _to_scalar_str(p.get("abs_url")) or title
        hid = hashlib.md5((base or "").encode("utf-8")).hexdigest()[:12]

        for c in chunk_text(full_text, max_tokens=max_tokens, overlap_tokens=overlap_tokens):
            chunk_id = f"{hid}::chunk{c['chunk_index']}"

            chunks.append(
                {
                    "id": chunk_id,
                    "text": c["chunk_text"],
                    "metadata": {
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "authors": authors_str,
                        "categories": categories_str,
                        "published": _to_scalar_str(p.get("published")),
                        "updated": _to_scalar_str(p.get("updated")),
                        "abs_url": _to_scalar_str(p.get("abs_url")),
                        "pdf_url": _to_scalar_str(p.get("pdf_url")),
                        "source_query": _to_scalar_str(p.get("source_query")),
                        "source_scope": _to_scalar_str(p.get("source_scope")),
                        "chunk_index": int(c["chunk_index"]),
                        "start_token": int(c["start_token"]),
                        "end_token": int(c["end_token"]),
                    },
                }
            )

    return chunks


def upsert_chroma(chunks: List[Dict], batch_size: int = 64) -> int:
    """
    Upsert chunk docs + embeddings into Chroma.
    We do it in batches to avoid memory spikes.
    """
    if not chunks:
        return 0

    collection = get_collection()
    total = 0

    # Defensive dedup by ID (just in case)
    seen = set()
    unique_chunks: List[Dict] = []
    for c in chunks:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        unique_chunks.append(c)

    if len(unique_chunks) != len(chunks):
        logger.warning("Removed %s duplicate chunk IDs before upsert", len(chunks) - len(unique_chunks))

    for i in range(0, len(unique_chunks), batch_size):
        batch = unique_chunks[i : i + batch_size]
        ids = [b["id"] for b in batch]
        texts = [b["text"] for b in batch]
        metas = [b["metadata"] for b in batch]

        embs = embed_texts(texts)  # normalized embeddings

        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metas,
            embeddings=embs.tolist(),
        )
        total += len(batch)

    return total


def build_bm25_payload(chunks: List[Dict]) -> Dict[str, Any]:
    """
    We store the BM25 corpus as tokens + aligned arrays.
    The payload is saved to BM25_PATH which is per collection name, so different models do not overwrite each other.
    """
    docs = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    tokenized = [tokenize_for_bm25(d) for d in docs]

    payload = {
        "tokenized_corpus": tokenized,
        "ids": ids,
        "metas": metas,
        "docs": docs,
    }
    joblib.dump(payload, BM25_PATH)
    return payload


def load_bm25_payload() -> Dict[str, Any]:
    """
    Load BM25 payload for the current collection.
    If missing, return an empty payload.
    """
    if not BM25_PATH.exists():
        return {"tokenized_corpus": [], "ids": [], "metas": [], "docs": []}
    return joblib.load(BM25_PATH)


def rebuild_all(max_tokens: int = 224, overlap_tokens: int = 32) -> Dict[str, Any]:
    """
    Full rebuild:
    - read papers.json
    - chunk
    - push to Chroma (CURRENT collection name)
    - store BM25 payload (CURRENT collection name)

    This is safe to run for MPNet if you choose a different collection name.
    """
    logger.info("Starting full reindex (max_tokens=%s, overlap_tokens=%s)", max_tokens, overlap_tokens)

    papers = load_papers(DEFAULT_DB_FILE)
    logger.info("Loaded papers: %s", len(papers))

    chunks = build_chunks_from_papers(papers, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    logger.info("Built chunks: %s", len(chunks))

    n_upsert = upsert_chroma(chunks)
    build_bm25_payload(chunks)

    logger.info("Reindex done: chroma_upserted=%s, bm25_path=%s", n_upsert, BM25_PATH)

    return {
        "papers": len(papers),
        "chunks": len(chunks),
        "chroma_upserted": n_upsert,
        "bm25_saved_to": str(BM25_PATH),
        "collection_name": COLLECTION_NAME,
    }