"""
Embeddings + token-based chunking.

This module does two jobs:
1) Create embeddings with SentenceTransformer
2) Split long texts into token chunks (better retrieval quality)

Model switching:
- Controlled via Settings (ENV / .env -> app/core/config.py)
- Env var: EMBEDDING_MODEL

- If you change the embedding model, you MUST rebuild indexes:
  POST /reset_all
  POST /reindex

Reason:
- Chroma stores vectors created by the embedding model.
  Mixing vectors from different models in the same collection is invalid.
"""

import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from app.core.config import settings

# Keep defaults in ONE place (config.py), but we still document it here:
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Read chosen model from centralized config.
# settings.EMBEDDING_MODEL is already "safe" (never empty).
MODEL_NAME = settings.EMBEDDING_MODEL or DEFAULT_MODEL

# Load the model once at import/startup.
# This is fast for runtime requests, but requires restart to change the model.
_embedder = SentenceTransformer(MODEL_NAME)

print(f"[Embedding] Loaded model: {MODEL_NAME}")

# SentenceTransformer usually provides a tokenizer.
# If not, we fall back to a standard HF tokenizer.
_tokenizer = getattr(_embedder, "tokenizer", AutoTokenizer.from_pretrained(MODEL_NAME))


def get_embedding_model_name() -> str:
    """
    Return the model name currently loaded by the backend.

    Useful for debugging and for UI / notebooks.
    """
    return MODEL_NAME


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Create normalized embeddings for a list of texts.

    Why normalize?
    - With normalized vectors, cosine similarity = dot product.
    - This matches typical Chroma cosine configurations.

    Returns:
    - shape: (N, dim)
    - dtype: float32
    """
    if not texts:
        dim = _embedder.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    vecs = _embedder.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return vecs.astype(np.float32)


def chunk_text(text: str, max_tokens: int = 224, overlap_tokens: int = 32) -> List[Dict]:
    """
    Token-based chunking with overlap.

    Why overlap?
    - If we cut in the middle of an important sentence,
      overlap keeps context for the next chunk.

    Output format: list of dicts with metadata (index + token range)
    """
    text = (text or "").strip()
    if not text:
        return []

    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    # Ensure chunk size doesn't exceed embedding model max sequence length
    model_limit = getattr(_embedder, "max_seq_length", None)
    if isinstance(model_limit, int) and model_limit > 0:
        max_tokens = min(max_tokens, model_limit - 2)

    token_ids = _tokenizer.encode(text, add_special_tokens=False)

    chunks: List[Dict] = []
    start = 0
    idx = 0

    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunk_token_ids = token_ids[start:end]
        chunk_str = _tokenizer.decode(chunk_token_ids).strip()

        chunks.append(
            {
                "chunk_text": chunk_str,
                "chunk_index": idx,
                "start_token": start,
                "end_token": end,
            }
        )

        idx += 1
        if end == len(token_ids):
            break

        start = end - overlap_tokens

    return chunks