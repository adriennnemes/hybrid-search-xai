"""
Hybrid search.

We combine:
- semantic similarity from ChromaDB (cosine space)
- lexical BM25 score

Then we normalize both to [0, 1] and mix with alpha:
  final = alpha * semantic + (1 - alpha) * lexical

This keeps the ranking logic transparent.
"""

from typing import List, Dict, Any
from .embeddings import embed_texts
from rank_bm25 import BM25Okapi
from .indexing import get_collection, load_bm25_payload, tokenize_for_bm25
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    mn = float(scores.min())
    mx = float(scores.max())
    if mx - mn < 1e-9:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - mn) / (mx - mn)).astype(np.float32)


def semantic_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []

    logger.info("Semantic search: top_k=%d, query='%s'", top_k, query[:60])

    collection = get_collection()
    q_emb = embed_texts([query])[0].tolist()

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=int(top_k),
        include=["documents", "metadatas", "distances"],
    )

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out: List[Dict[str, Any]] = []
    for i in range(len(ids)):
        dist = float(dists[i]) if dists else 0.0
        sim = 1.0 - dist  # cosine space: distance ~ 1 - cosine_similarity
        out.append(
            {
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i],
                "score_semantic_raw": sim,
            }
        )
    return out

def lexical_search_bm25(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []

    logger.info("Lexical BM25 search: top_k=%d, query='%s'", top_k, query[:60])

    data = load_bm25_payload()
    tokenized = data["tokenized_corpus"]
    ids = data["ids"]
    metas = data["metas"]
    docs = data["docs"]

    if not tokenized or not ids:
        return []

    bm25 = BM25Okapi(tokenized)

    q_tokens = tokenize_for_bm25(query)
    if not q_tokens:
        return []

    scores = np.array(bm25.get_scores(q_tokens), dtype=np.float32)
    if scores.size == 0:
        return []

    k = max(1, min(int(top_k), scores.size))
    idx = np.argsort(scores)[::-1][:k]

    out: List[Dict[str, Any]] = []
    for i in idx:
        if scores[i] <= 0:
            continue
        out.append(
            {
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i],
                "score_lexical_raw": float(scores[i]),
            }
        )
    return out

def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.7) -> List[Dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []

    alpha = max(0.0, min(1.0, float(alpha)))

    logger.info("Hybrid search: top_k=%d, alpha=%.2f, query='%s'", top_k, alpha, query[:60])

    sem = semantic_search(query, top_k=max(20, top_k * 5))
    lex = lexical_search_bm25(query, top_k=max(50, top_k * 10))

    logger.info("Candidates: semantic=%d, lexical=%d", len(sem), len(lex))

    merged: Dict[str, Dict[str, Any]] = {}

    for r in sem:
        rid = r["id"]
        merged.setdefault(rid, {"id": rid, "metadata": r["metadata"], "document": r["document"]})
        merged[rid]["semantic_raw"] = r.get("score_semantic_raw", 0.0)

    for r in lex:
        rid = r["id"]
        merged.setdefault(rid, {"id": rid, "metadata": r["metadata"], "document": r["document"]})
        merged[rid]["lexical_raw"] = r.get("score_lexical_raw", 0.0)

    ids = list(merged.keys())
    sem_raw = np.array([merged[i].get("semantic_raw", 0.0) for i in ids], dtype=np.float32)
    lex_raw = np.array([merged[i].get("lexical_raw", 0.0) for i in ids], dtype=np.float32)


    # --- Debug: pool + overlap + raw score stats ---
    sem_only = sum(1 for i in ids if "semantic_raw" in merged[i] and "lexical_raw" not in merged[i])
    lex_only = sum(1 for i in ids if "lexical_raw" in merged[i] and "semantic_raw" not in merged[i])
    both = sum(1 for i in ids if "semantic_raw" in merged[i] and "lexical_raw" in merged[i])

    logger.info(
        "POOL DEBUG | merged=%d | both=%d | sem_only=%d | lex_only=%d",
        len(ids), both, sem_only, lex_only
    )

    logger.info(
        "RAW DEBUG  | sem_raw[min=%.4f max=%.4f] | lex_raw[min=%.4f max=%.4f]",
        float(sem_raw.min()) if sem_raw.size else -1.0,
        float(sem_raw.max()) if sem_raw.size else -1.0,
        float(lex_raw.min()) if lex_raw.size else -1.0,
        float(lex_raw.max()) if lex_raw.size else -1.0,
    )
    # --- End debug ---

    sem_n = _normalize(sem_raw)
    lex_n = _normalize(lex_raw)

    scored: List[Dict[str, Any]] = []
    for idx, rid in enumerate(ids):
        final = alpha * float(sem_n[idx]) + (1.0 - alpha) * float(lex_n[idx])
        item = merged[rid]

        item["score"] = final
        item["semantic_norm"] = float(sem_n[idx])
        item["lexical_norm"] = float(lex_n[idx])
        scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    scored = scored[: int(top_k)]

    out: List[Dict[str, Any]] = []
    for r in scored:
        m = r.get("metadata") or {}
        doc = (r.get("document") or "").strip()

        out.append(
            {
                "title": m.get("title", ""),
                "summary": doc[:800] + ("…" if len(doc) > 800 else ""),
                "authors": [a.strip() for a in (m.get("authors") or "").split(",") if a.strip()],
                "published": m.get("published", ""),
                "updated": m.get("updated", ""),
                "abs_url": m.get("abs_url", ""),
                "pdf_url": m.get("pdf_url", ""),
                "score": float(r["score"]),
                "explain": {
                    "alpha": alpha,
                    "semantic_component": float(r["semantic_norm"]),
                    "lexical_component": float(r["lexical_norm"]),
                    "semantic_raw_cosine": r.get("semantic_raw"),
                    "lexical_raw_bm25": r.get("lexical_raw"),
                    "semantic_metric": "cosine (via Chroma distance)",
                    "lexical_metric": "BM25",
                    "chunk_index": m.get("chunk_index"),
                    "token_span": [m.get("start_token"), m.get("end_token")],
                },
            }
        )

    logger.info("Returning %d results", len(out))
    return out