"""
Main entry point of the backend.

We keep this file small:
- create the FastAPI app
- register the API routes (router)
- mount the Gradio UI under /ui

All actual endpoints are defined in routes.py.
All request/response formats are in schemas.py.
"""

from fastapi import FastAPI
import gradio as gr

# -------------------- Logging setup --------------

from .core.logger import setup_logging

setup_logging()

# -------------------- Import our API router -----------------

from .routes import router

# ------------------- Import the Gradio UI builder -------------

from .ui.interface import build_demo

# -------------------- Create FastAPI app -------------------

app = FastAPI(
    title="Research Paper Explorer API",
    description=(
        "Hybrid lexical + semantic search over arXiv paper metadata.\n\n"
        "Typical workflow:\n"
        "1) POST /ingest   -> download metadata (title, abstract, authors, links)\n"
        "2) POST /reindex  -> build search indexes (ChromaDB + BM25)\n"
        "3) POST /search   -> query the collection\n\n"
        "UI is available under /ui."
    ),
)

# -------------------- Register routes --------------------

app.include_router(router)

# -------------------- Mount Gradio UI --------------------

demo = build_demo()
app = gr.mount_gradio_app(app, demo, path="/ui")