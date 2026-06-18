# Hybrid Semantic Search for Explainable AI Literature

## Project Overview

This project implements a hybrid search system for scientific literature by combining semantic retrieval using transformer embeddings with traditional keyword-based retrieval using BM25.

The system was developed to explore how different retrieval approaches perform when searching research papers related to Explainable Artificial Intelligence (XAI). By combining lexical and semantic search techniques, the application aims to improve retrieval quality compared to standalone methods.

The project covers the complete retrieval pipeline from data ingestion and indexing to search, evaluation, and user interaction.

## Objectives

- Build an end-to-end scientific literature search system
- Compare lexical, semantic, and hybrid retrieval approaches
- Evaluate different embedding models
- Explore the strengths and limitations of BM25 and dense retrieval
- Develop a reproducible search and indexing pipeline
- Improve search relevance for XAI-related research papers

## System Architecture

### Data Ingestion

Research papers are collected from the arXiv API.

The ingestion pipeline:

- Retrieves metadata and abstracts
- Cleans and preprocesses text
- Splits documents into searchable chunks
- Stores processed data for indexing

### Semantic Retrieval

Semantic search is implemented using Sentence Transformers.

Embedding models evaluated:

- all-MiniLM-L6-v2
- all-mpnet-base-v2

Document chunks are converted into vector embeddings and stored in ChromaDB.

Queries are embedded using the same model and matched through vector similarity search.

### Lexical Retrieval

Traditional keyword retrieval is implemented using BM25.

This approach ranks documents based on exact keyword matches and term importance within the corpus.

### Hybrid Retrieval

The final retrieval system combines:

- Semantic similarity scores
- BM25 relevance scores

A weighting parameter allows both approaches to be balanced dynamically.

This helps capture both semantic meaning and exact terminology, improving overall retrieval performance.

## Evaluation

The retrieval approaches were evaluated using manually created relevance judgments.

The comparison included:

- BM25 retrieval
- Semantic retrieval (MiniLM)
- Semantic retrieval (MPNet)
- Hybrid retrieval

Evaluation focused on retrieval quality and ranking performance using information retrieval metrics.

The experiments demonstrated how retrieval quality changes depending on the embedding model and retrieval strategy.

## User Interface

The application provides an interactive search interface built with Gradio.

Users can:

- Search scientific literature
- Select retrieval methods
- Adjust hybrid weighting parameters
- Compare search results across different approaches

## Technologies Used

### Machine Learning & NLP

- Sentence Transformers
- Transformer Embeddings
- Semantic Search
- Information Retrieval
- BM25

### Data Storage

- ChromaDB

### Backend & Application

- FastAPI
- Gradio
- Docker
- Docker Compose

### Data Processing

- Python
- Pandas
- NumPy

## Key Learnings

This project provided practical experience in:

- Information Retrieval (IR)
- Semantic Search
- Vector Databases
- Embedding Models
- Hybrid Search Systems
- Scientific Literature Retrieval
- Search Evaluation Methodologies
- API Development and Deployment

A key takeaway was that semantic retrieval and lexical retrieval solve different problems. Combining both approaches often produces more robust search results than relying on a single retrieval strategy.

## Repository Structure

```text
app/
├── api/
├── services/
├── ui/
├── ingest/
├── search/
├── evaluation/
└── data/

docker-compose.yml
Dockerfile
requirements.txt
```

## Future Improvements

Potential future extensions include:

- Additional embedding models
- Reranking approaches
- Query expansion techniques
- Larger scientific corpora
- Retrieval-Augmented Generation (RAG)
- Explainability features for search results
