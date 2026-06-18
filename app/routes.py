"""
API routes (endpoints) for the project.

Why this file exists:
- Keep main.py clean and focused only on app startup.
- Collect all HTTP endpoints in one place.
- Make the backend flow easy to understand and debug.

This module defines the full backend workflow:
ingest -> reindex -> search
"""

from pathlib import Path
from fastapi import APIRouter

# ----------------- Request / Response Schemas --------------

# These schemas define what clients can send and what the API returns.
from .schemas import IngestRequest, ReindexRequest, SearchRequest, SearchResponse

# -------------------- Core Service Functions ---------------

from .services.ingestion import ingest_topic
from .services.indexing import rebuild_all, reset_indexes
from .services.search import hybrid_search
from .services.embeddings import get_embedding_model_name
from .storage.db import DEFAULT_DB_FILE

# ------------------- Router Setup --------------------

# All API endpoints are registered on this router.
router = APIRouter()


# ---------------- Runtime Config Endpoint ------------------

@router.get("/config")
def get_runtime_config():
    """
    Exposes small runtime configuration values.

    Purpose:
    - Debug which embedding model is currently active
    - Verify backend configuration without opening logs
    """
    return {
        "embedding_model": get_embedding_model_name()
    }

# -------------------- Reset Endpoints --------------------

@router.post("/reset")
def reset_indexes_only():
    """
    Soft reset.

    Deletes:
    - Search indexes (ChromaDB + BM25)

    Keeps:
    - Raw paper metadata (papers.json)

    Use this when:
    - You want to rebuild indexes
    - But do NOT want to re-download data
    """
    reset_indexes()
    return {"status": "ok"}


@router.post("/reset_all")
def reset_everything():
    """
    Hard reset.

    Deletes:
    - Search indexes
    - Raw paper data file (papers.json)

    Use this when:
    - You want a completely fresh start
    - You want to re-ingest papers from scratch
    """
    reset_indexes()

    path = Path(DEFAULT_DB_FILE)
    if path.exists():
        path.unlink()

    return {"status": "ok"}


# -------------------- Ingestion Endpoint ----------------

@router.post("/ingest")
def ingest(req: IngestRequest):
    """
    Downloads paper metadata from arXiv and stores it locally.

    This step:
    - Collects raw data only
    - Doesn't build search indexes yet

    Typical workflow:
    ingest -> reindex -> search
    """
    ingest_topic(
        query=req.query,
        scope=req.scope,
        max_results=req.max_results
    )
    return {"status": "ok"}

# -------------------- Reindex Endpoint --------------------

@router.post("/reindex")
def reindex(req: ReindexRequest):
    """
    Builds or rebuilds search indexes from stored papers.

    Inside rebuild_all():
    - Chunk text into token-based segments
    - Compute embeddings and store them in ChromaDB
    - Build BM25 index for lexical search

    This must be run:
    - After ingest
    - After changing embedding model
    """
    rebuild_all(
        max_tokens=req.max_tokens,
        overlap_tokens=req.overlap_tokens
    )
    return {"status": "ok"}

# -------------------- Search Endpoint --------------------

@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    Performs hybrid search combining:

    - Semantic similarity (embeddings in ChromaDB)
    - Lexical similarity (BM25)
    - Score fusion controlled by alpha

    Returns:
    - Ranked results
    - Explanation metadata for transparency
    """
    items = hybrid_search(
        query=req.query,
        top_k=req.top_k,
        alpha=req.alpha
    )

    return {"items": items}