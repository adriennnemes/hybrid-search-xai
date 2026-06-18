"""
Pydantic models for API request and response validation.

This file defines the data structures used by the backend API.
The goal is to keep all request/response formats in one place,
so the main app logic stays clean and easier to maintain.

Think of this file as:
- the "API contract"
- the schema that describes what the frontend sends and receives
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


# -------------------- Ingestion request model --------------------

class IngestRequest(BaseModel):
    """
    Describes what the user sends when requesting new paper ingestion
    from the arXiv API.
    """

    # Search topic for arXiv (e.g. "explainable ai")
    query: str = Field(
        ...,
        min_length=1,
        description="Search topic for arXiv papers"
    )

    # Scope defines where arXiv should search:
    # - titles
    # - abstracts
    # - all metadata
    scope: str = Field(
        "abstracts",
        description="Search scope: titles | abstracts | all"
    )

    # Maximum number of papers to fetch
    max_results: int = Field(
        200,
        ge=1,
        le=2000,
        description="Max number of papers to ingest"
    )


# --------------------- Reindex request model ---------------------

class ReindexRequest(BaseModel):
    """
    Controls how text is chunked before indexing.
    These settings influence search quality and performance.
    """

    # Maximum number of tokens per text chunk
    max_tokens: int = Field(
        224,
        ge=64,
        le=512,
        description="Chunk size in tokens"
    )

    # Overlap between chunks to preserve context
    overlap_tokens: int = Field(
        32,
        ge=0,
        le=128,
        description="Token overlap between chunks"
    )


# -------------------- Search request model -------------------

class SearchRequest(BaseModel):
    """
    Defines what the user sends when performing a search query.
    """

    # The actual search text typed by the user
    query: str = Field(
        ...,
        min_length=1,
        description="User search query"
    )

    # How many results to return
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Number of results to return"
    )

    # Hybrid weighting parameter:
    # 1.0 = fully semantic
    # 0.0 = fully lexical (BM25)
    alpha: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Hybrid weighting between semantic and lexical search"
    )



# -------------------- Paper result model --------------------

class Paper(BaseModel):
    """
    Represents a single paper in the search results.
    This structure is returned to the frontend UI.
    """

    # Paper metadata
    title: str
    summary: str
    authors: List[str]

    # Publication info
    published: str
    updated: str

    # External links
    abs_url: str
    pdf_url: str

    # Final ranking score (after hybrid merging)
    score: float

    # Explainability block:
    # contains details about semantic vs lexical contributions
    explain: Dict[str, Any]


# -------------------- Search response wrapper -----------------------

class SearchResponse(BaseModel):
    """
    The full API response returned after a search request.
    """

    # List of ranked papers
    items: List[Paper]