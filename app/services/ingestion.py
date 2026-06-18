"""
Data ingestion from arXiv (Atom API).

This module only downloads metadata:
- title
- abstract (summary)
- authors
- links (abstract/pdf)
- published/updated
- categories

We store everything into papers.json.
Later: indexing.py will chunk + embed + push into ChromaDB.
"""

from typing import List, Dict
from urllib.parse import quote_plus
from ..storage.db import load_papers, save_papers, DEFAULT_DB_FILE
import xml.etree.ElementTree as ET
import requests

ARXIV_API = "http://export.arxiv.org/api/query"


def _scope_to_arxiv_field(scope: str) -> str:
    scope = (scope or "abstracts").strip().lower()
    if scope == "titles":
        return "ti"
    if scope == "abstracts":
        return "abs"
    # "all" is still metadata search, not full PDF text
    return "all"


def arxiv_search(query: str, scope: str = "abstracts", max_results: int = 200) -> List[Dict]:
    query = (query or "").strip()
    if not query:
        return []

    field = _scope_to_arxiv_field(scope)
    search_query = f"{field}:{query}"

    url = f"{ARXIV_API}?search_query={quote_plus(search_query)}&start=0&max_results={int(max_results)}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    entries = root.findall("atom:entry", ns)

    papers: List[Dict] = []
    for e in entries:
        title = (e.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (e.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = (e.findtext("atom:published", default="", namespaces=ns) or "").strip()
        updated = (e.findtext("atom:updated", default="", namespaces=ns) or "").strip()

        authors = []
        for a in e.findall("atom:author", ns):
            name = (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
            if name:
                authors.append(name)

        abs_url = ""
        pdf_url = ""
        for link in e.findall("atom:link", ns):
            href = link.attrib.get("href", "")
            rel = link.attrib.get("rel", "")
            t = link.attrib.get("type", "")
            if rel == "alternate" and href.startswith("http"):
                abs_url = href
            if t == "application/pdf" and href.startswith("http"):
                pdf_url = href

        raw_id = (e.findtext("atom:id", default="", namespaces=ns) or "").strip()
        arxiv_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""

        categories = [c.attrib.get("term", "") for c in e.findall("atom:category", ns)]
        categories = [c for c in categories if c]

        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": summary,
                "authors": authors,
                "published": published,
                "updated": updated,
                "abs_url": abs_url,
                "pdf_url": pdf_url,
                "categories": categories,
                "source_query": query,
                "source_scope": scope,
            }
        )

    return papers


def _dedup(existing: List[Dict], new_items: List[Dict]) -> List[Dict]:
    """
    Deduplicate mostly by arxiv_id.
    If arxiv_id missing, fallback to abs_url or title.
    """
    seen = set()
    merged: List[Dict] = []

    def key(p: Dict) -> str:
        return p.get("arxiv_id") or p.get("abs_url") or p.get("title") or ""

    for p in existing:
        k = key(p)
        if k:
            seen.add(k)
        merged.append(p)

    for p in new_items:
        k = key(p)
        if not k or k in seen:
            continue
        seen.add(k)
        merged.append(p)

    return merged


def ingest_topic(query: str, scope: str = "abstracts", max_results: int = 200) -> Dict:
    """
    Main ingestion entry point used by the API.
    """
    existing = load_papers(DEFAULT_DB_FILE)
    new_items = arxiv_search(query=query, scope=scope, max_results=max_results)
    merged = _dedup(existing, new_items)
    save_papers(merged, DEFAULT_DB_FILE)

    return {
        "added": max(0, len(merged) - len(existing)),
        "total": len(merged),
        "query": query,
        "scope": scope,
    }