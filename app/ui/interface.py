"""
Gradio UI mounted under /ui.

It calls the backend via HTTP.
Since it runs inside the same container as FastAPI,
127.0.0.1:8000 works fine.
"""

from typing import List, Dict, Any, Generator
from ..core.config import settings
from datetime import datetime
import requests
import gradio as gr
import re

BACKEND_URL = "http://127.0.0.1:8000"

# Backend endpoints
SEARCH_ENDPOINT = f"{BACKEND_URL}/search"
INGEST_ENDPOINT = f"{BACKEND_URL}/ingest"
REINDEX_ENDPOINT = f"{BACKEND_URL}/reindex"
RESET_ENDPOINT = f"{BACKEND_URL}/reset"

# Timeouts
TIMEOUT_SECONDS = 30
PIPELINE_TIMEOUT_SECONDS = 600

# ------------- Formatting helpers --------------

# Convert arXiv ISO timestamps into a human-readable publication date
def format_published_date(published: str) -> str:
    if not published:
        return ""
    try:
        dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        return dt.strftime("%B %d, %Y")  # e.g. July 18, 2023
    except Exception:
        return published

def highlight_md(text: str, query: str) -> str:
    if not text:
        return ""
    query = (query or "").strip()
    if not query:
        return text

    q = query.lower().strip()

    patterns: List[re.Pattern] = []

    # If user query contains something like "post hoc" or "post-hoc" or "posthoc", highlight any of these variants in the text.
    if re.search(r"\bpost\s*-?\s*hoc\b", q) or "posthoc" in q:
        patterns.append(re.compile(r"\bpost\s*-?\s*hoc\b", flags=re.IGNORECASE))

    # Generic terms from the query (skip very short ones)
    terms = [t for t in re.split(r"\s+", query) if len(t) >= 2]
    # longest first to reduce partial overlaps
    terms = sorted(set(terms), key=len, reverse=True)

    for t in terms:
        tl = t.lower()
        # Avoid adding "post" or "hoc" separately if we already handle post-hoc as a group
        if patterns and tl in {"post", "hoc"}:
            continue
        # Word-boundary match to reduce "repost" style false positives
        patterns.append(re.compile(rf"\b{re.escape(t)}\b", flags=re.IGNORECASE))

    if not patterns:
        return text

    out = text
    for pat in patterns:
        out = pat.sub(lambda m: f"**{m.group(0)}**", out)

    return out


def _format_items(items: List[Dict[str, Any]], query: str) -> str:
    if not items:
        return "No results found."

    blocks: List[str] = []
    for i, it in enumerate(items, start=1):
        # Extract metadata from result
        title = highlight_md(it.get("title", "(no title)"), query)
        authors = it.get("authors", "")
        published = format_published_date(it.get("published", ""))
        summary = highlight_md(it.get("summary", "") or "", query)
        abs_url = it.get("abs_url", "")
        pdf_url = it.get("pdf_url", "")

        # Build markdown block for this result
        block = f"### {i}. {title}\n\n"

        if authors:
            # authors might be a list in some payloads -> make it readable
            if isinstance(authors, list):
                authors = ", ".join(str(a) for a in authors if str(a).strip())
            block += f"**Authors:** {authors}\n\n"

        if published:
            block += f"**Published:** {published}\n\n"

        if summary:
            block += f"{summary}\n\n"
        # Add links to abstract and PDF
        links = []
        if abs_url:
            links.append(f"[Abstract]({abs_url})")
        if pdf_url:
            links.append(f"[PDF]({pdf_url})")
        if links:
            block += " | ".join(links)

        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def _post_json(url: str, payload: dict, timeout_s: int) -> dict:
    resp = requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _get_json(url: str, timeout_s: int = 10) -> dict:
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


# -------------- Search (UI -> /search) --------------

def hybrid_search_ui(query: str, top_k: int, alpha: float) -> str:
    query = (query or "").strip()
    if not query:
        return "Please enter a query."

    payload = {"query": query, "top_k": int(top_k), "alpha": float(alpha)}

    try:
        data = _post_json(SEARCH_ENDPOINT, payload, TIMEOUT_SECONDS)
    except Exception as e:
        return f"Backend error: {e}"

    return _format_items(data.get("items", []), query)


# ------------- Pipeline (UI -> /ingest, /reindex) --------------

def ingest_ui(topic_query: str, scope: str, max_results: int) -> Generator[str, None, None]:
    topic_query = (topic_query or "").strip()
    if not topic_query:
        yield "Ingest not started: please provide an ingestion query/topic."
        return

    payload = {"query": topic_query, "scope": scope, "max_results": int(max_results)}

    yield "Ingest started. This can take some time."
    try:
        _post_json(INGEST_ENDPOINT, payload, PIPELINE_TIMEOUT_SECONDS)
        yield "Ingest finished. Next step: run reindex to build searchable indexes."
    except Exception as e:
        yield f"Ingest failed: {e}"


def reindex_ui(max_tokens: int, overlap_tokens: int) -> Generator[str, None, None]:
    payload = {"max_tokens": int(max_tokens), "overlap_tokens": int(overlap_tokens)}

    yield "Reindex started. This can take some time."
    try:
        _post_json(REINDEX_ENDPOINT, payload, PIPELINE_TIMEOUT_SECONDS)
        yield "Reindex finished. You can now search."
    except Exception as e:
        yield f"Reindex failed: {e}"


def pipeline_ui(
    topic_query: str,
    scope: str,
    max_results: int,
    max_tokens: int,
    overlap_tokens: int,
) -> Generator[str, None, None]:
    topic_query = (topic_query or "").strip()
    if not topic_query:
        yield "Pipeline not started: please provide an ingestion query/topic."
        return

    yield "Pipeline started: ingest then reindex."

    ingest_payload = {"query": topic_query, "scope": scope, "max_results": int(max_results)}
    try:
        _post_json(INGEST_ENDPOINT, ingest_payload, PIPELINE_TIMEOUT_SECONDS)
        yield "Step 1/2 finished: ingest."
    except Exception as e:
        yield f"Pipeline stopped: ingest failed: {e}"
        return

    reindex_payload = {"max_tokens": int(max_tokens), "overlap_tokens": int(overlap_tokens)}
    try:
        _post_json(REINDEX_ENDPOINT, reindex_payload, PIPELINE_TIMEOUT_SECONDS)
        yield "Pipeline finished: ingest and reindex completed."
    except Exception as e:
        yield f"Pipeline stopped: reindex failed: {e}"


# -------------- Reset (UI -> /reset) with confirmation -------------

def reset_ui(confirm: bool) -> str:
    if not confirm:
        return "Reset not executed. Please confirm the checkbox first."

    try:
        _post_json(RESET_ENDPOINT, {}, PIPELINE_TIMEOUT_SECONDS)
        return "Reset finished. Indexes cleared. Run reindex (or Ingest + Reindex) afterwards."
    except Exception as e:
        return f"Reset failed: {e}"


def build_demo() -> gr.Blocks:

    # Show active runtime configuration in the UI (model + collection)
    runtime_info = f"""
    **Active embedding model:** `{settings.EMBEDDING_MODEL}`  
    **Active Chroma collection:** `{settings.CHROMA_COLLECTION}`
    """

    with gr.Blocks(title="Research Paper Explorer") as demo:
        gr.Markdown(runtime_info)

        gr.Markdown(
            "# Research Paper Explorer\n"
            "Lexical and embedding-based retrieval over scientific literature\n\n"
            "Workflow: Ingest papers -> Reindex -> Search\n\n"
            "Tip: If you change the embedding model, run reindex again to rebuild the search index."
        )

        with gr.Tabs():
            # Search UI
            with gr.Tab("Search"):
                query = gr.Textbox(
                    label="Query",
                    placeholder="e.g. explainable ai, model interpretability, post hoc explanations",
                )
                top_k = gr.Slider(label="Top K", minimum=1, maximum=25, step=1, value=5)
                alpha = gr.Slider(
                    label="Alpha (semantic vs lexical)",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.7,
                    info="0.0 = mostly BM25 keyword, 1.0 = mostly semantic",
                )

                btn = gr.Button("Search")
                out = gr.Markdown()

                btn.click(hybrid_search_ui, inputs=[query, top_k, alpha], outputs=out)

            # Setup UI (pipeline controls)
            with gr.Tab("Setup"):
                gr.Markdown("## Pipeline controls")

                ingest_query = gr.Textbox(
                    label="Ingest query/topic",
                    placeholder="e.g. explainable AI",
                )

                with gr.Row():
                    scope = gr.Dropdown(
                        label="Scope",
                        choices=["all", "title", "abstract"],
                        value="all",
                        info="Where to search in arXiv metadata.",
                    )
                    max_results = gr.Slider(label="Max results", minimum=5, maximum=200, step=5, value=50)

                with gr.Row():
                    max_tokens = gr.Slider(label="Chunk max tokens", minimum=128, maximum=1024, step=32, value=224)
                    overlap_tokens = gr.Slider(label="Chunk overlap tokens", minimum=0, maximum=256, step=8, value=32)

                gr.Markdown(
                    "Recommended chunk settings:\n"
                    "- MiniLM (default): max_tokens=224, overlap_tokens=32\n"
                    "- MPNet: max_tokens=360, overlap_tokens=64\n"
                )

                status = gr.Markdown("Status: ready.")

                with gr.Row():
                    btn_ingest = gr.Button("Ingest")
                    btn_reindex = gr.Button("Reindex")
                    btn_pipeline = gr.Button("Ingest + Reindex")

                btn_ingest.click(ingest_ui, inputs=[ingest_query, scope, max_results], outputs=status)
                btn_reindex.click(reindex_ui, inputs=[max_tokens, overlap_tokens], outputs=status)
                btn_pipeline.click(
                    pipeline_ui,
                    inputs=[ingest_query, scope, max_results, max_tokens, overlap_tokens],
                    outputs=status,
                )

                gr.Markdown(
                    "Reset indexes\n\n"
                    "This clears indexes only (ChromaDB + BM25). Raw downloaded data remains.\n"
                    "After reset you must run reindex (or Ingest + Reindex)."
                )

                confirm = gr.Checkbox(
                    label="I confirm: I want to clear the search indexes and will reindex afterwards.",
                    value=False,
                )
                btn_reset = gr.Button("Reset indexes")
                reset_out = gr.Markdown()

                btn_reset.click(reset_ui, inputs=[confirm], outputs=reset_out)

    return demo