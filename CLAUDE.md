# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands
- Install dependencies: `pip install -r requirements.txt`
- Build Knowledge Base: `python -m ingest.build_kb`
- Build support articles only: `export SUPPORT_ONLY_BUILD=1 && python -m ingest.build_kb`
- Run Web Chat UI: `python server.py` (or `uvicorn server:app --reload --port 8000`)
- Run CLI Chat: `python cli_chat.py`

## Architecture & Structure
The project is a Hybrid Retrieval-Augmented Generation (RAG) assistant for TestArchitect BIA/BIS documentation.

### High-Level Components
- **Ingestion (`ingest/`)**: Parses markdown files from `DOCS_ROOT` and `support_urls.txt`, then builds a structured SQLite knowledge base (`abt_kb.sqlite`).
- **Retrieval (`rag/retrieval.py`)**: Implements a hybrid search strategy combining:
    - FTS5 lexical ranking for keyword matches.
    - OpenAI embeddings for semantic similarity.
    - Graph-based expansion via the `relations` table.
- **Generation (`rag/chat.py`)**: Constructs a compact prompt from retrieved items and uses OpenAI Chat Completions to generate answers.
- **Entry Points**: 
    - `server.py`: HTTP API and Web UI.
    - `cli_chat.py`: Command-line interface.
- **Configuration (`config.py`)**: Manages settings and environment variables (e.g., `OPENAI_API_KEY`, `DOCS_ROOT`).

### Data Model (`abt_kb.sqlite`)
- `items`: Canonical structured records of documentation items.
- `items_fts`: FTS5 index for lexical search.
- `embeddings`: Vector embeddings for semantic search.
- `relations`: Graph edges representing relationships between items (e.g., `USES_SETTING`).
