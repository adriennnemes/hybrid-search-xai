"""
Tiny local storage for raw papers (papers.json).

We store:
- raw arXiv metadata in JSON (easy to inspect)
We do NOT store embeddings here.
Embeddings go into ChromaDB, BM25 index goes into bm25.joblib.
"""

from pathlib import Path
from typing import List, Dict, Union
import json

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DB_FILE = DATA_DIR / "papers.json"


def load_papers(path: Union[str, Path] = DEFAULT_DB_FILE) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_papers(papers: List[Dict], path: Union[str, Path] = DEFAULT_DB_FILE) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

def db_stats(path: Union[str, Path] = DEFAULT_DB_FILE) -> Dict[str, int]:
    """
    Returns simple statistics about the paper database.
    This is used for sanity-checking during development.
    """
    papers = load_papers(path)
    return {
        "num_papers": len(papers)
    }