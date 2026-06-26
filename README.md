# IR Search Engine — LoTTe

A Python-based Information Retrieval search engine built on the [LoTTe](https://ir-datasets.com) recreation/dev/search dataset (~200K documents). Implements three ranking models — **TF-IDF (VSM)**, **BM25**, and **Embedding** (SentenceTransformer) — with hybrid (serial & parallel) and Learning-to-Rank (LTR) extensions. Built with a **FastAPI** backend and **Streamlit** frontend following a Service-Oriented Architecture (SOA).

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Data Pipeline](#data-pipeline)
- [Running the System](#running-the-system)
- [API Endpoints](#api-endpoints)
- [Frontend](#frontend)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Architecture](#architecture)
- [Query Refinement](#query-refinement)
- [Index Files](#index-files)
- [License](#license)

---

## Features

- **Three Ranking Models**: TF-IDF (VSM), BM25, Embedding (SentenceTransformer `all-MiniLM-L6-v2`)
- **Hybrid Search**: Serial (BM25 first-stage → Embedding re-ranking) and Parallel (BM25 + Embedding with Reciprocal Rank Fusion)
- **Learning to Rank (LTR)**: Gradient-boosted tree model for multi-feature re-ranking
- **Query Refinement**: Spell correction, synonym expansion (WordNet), and history-based term boosting
- **Autocomplete Suggestions**: Trie-based prefix matching for query suggestions
- **Vector Store**: Optional FAISS-accelerated embedding search
- **Adjustable BM25 Parameters**: k1 and b sliders in the UI
- **SOA Architecture**: Decoupled services for preprocessing, indexing, retrieval, evaluation, and query refinement
- **Evaluation Metrics**: Precision@10, Recall@10, MAP@10, nDCG@10

---

## Project Structure

```
├── api_app.py                  # FastAPI application entry point
├── web_app.py                  # Streamlit frontend
├── start.py                    # Dataset loading demo script
│
├── routes/                     # FastAPI route definitions
│   ├── health_route.py
│   ├── search_route.py
│   ├── suggest_route.py
│   └── refine_route.py
│
├── controllers/                # Business logic layer
│   ├── health_controller.py
│   ├── search_controller.py
│   ├── suggest_controller.py
│   └── refine_controller.py
│
├── models/                     # Pydantic request/response schemas
│   ├── search_model.py
│   ├── suggest_model.py
│   └── refine_model.py
│
├── services/                   # SOA-style services
│   ├── dataset/                # Dataset loading (ir_datasets)
│   ├── preprocessing/          # Text preprocessing pipeline
│   ├── data_base/              # SQLite document storage
│   ├── indexing/               # Inverted index construction
│   ├── tfidf/                  # TF-IDF vectorization & search
│   ├── bm25/                   # BM25 scoring & search
│   ├── embedding/              # SentenceTransformer embeddings & search
│   ├── hybrid/                 # Hybrid search (fusion methods)
│   ├── ltr/                    # Learning to Rank
│   ├── evaluation/             # Evaluation metrics & scripts
│   └── query_refinement/       # Spell correction, synonyms, history boosting
│
├── scripts/                    # CLI entrypoints (build indices, evaluate)
├── tests/                      # Standalone test scripts
├── evaluations/                # Evaluation output (JSON reports)
├── indices/                    # Generated index files (gitignored)
├── processed_data/             # Preprocessed data (gitignored)
└── database/                   # SQLite database (gitignored)
```

---

## Prerequisites

- Python 3.9+
- Key dependencies:
  ```
  fastapi, uvicorn, streamlit, requests
  ir_datasets, nltk, contractions
  scikit-learn, scipy, numpy, joblib
  sentence-transformers, pydantic
  ```

> **Note:** There is no `requirements.txt`. Install all dependencies manually or create one from the list above.

---

## Data Pipeline

Run these scripts **in order** from the project root. Each script processes all documents by default; set `MAX_DOCUMENTS` to a small number (e.g. `1000`) for quick testing.

```bash
# 1. Load and preprocess dataset → JSONL output
python scripts/prepare_lotte_data.py

# 2. Store raw documents in SQLite
python scripts/load_raw_documents_to_db.py

# 3. Build inverted index (required for TF-IDF and BM25)
python scripts/build_inverted_index.py

# 4a. Build TF-IDF index
python scripts/build_tfidf_index.py

# 4b. Build BM25 index
python scripts/build_bm25_index.py

# 4c. Build Embedding index (downloads all-MiniLM-L6-v2 on first run)
python scripts/build_embedding.py
```

---

## Running the System

Start the API first, then the frontend:

```bash
# Terminal 1 — FastAPI backend (http://127.0.0.1:8000)
python api_app.py

# Terminal 2 — Streamlit frontend (http://127.0.0.1:8501)
streamlit run web_app.py
```

API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info |
| `/api/health` | GET | Health check |
| `/api/search` | POST | Search documents (TF-IDF, BM25, Embedding, Hybrid, LTR) |
| `/api/suggest` | POST | Autocomplete query suggestions |
| `/api/refine` | POST | Query refinement preview |

---

## Frontend

The Streamlit UI provides:

- **Search Mode**: "Basic Request only" or "Basic + Additional Features" (enables query refinement)
- **Ranking Model**: BM25, TF-IDF, Embedding, Hybrid Parallel, Hybrid Serial, or LTR
- **BM25/LTR Parameters**: Adjustable `k1` (0.5–3.0) and `b` (0.0–1.0) sliders
- **Hybrid Settings**: First-stage candidate count for serial mode
- **Vector Store Toggle**: FAISS-accelerated embedding search
- **Autocomplete**: Suggestion chips appear as you type (in refined mode)

---

## Evaluation

Run evaluation scripts to generate JSON reports:

```bash
python scripts/evaluate_tfidf_lotte.py
python scripts/evaluate_bm25_lotte.py
python scripts/evaluate_embedding_lotte.py
python scripts/evaluate_hybrid_lotte.py
python scripts/evaluate_ltr_lotte.py
```

Reports are saved to `evaluations/lotte/{model}/evaluation_report.json`.

**Metrics**: Precision@10, Recall@10, MAP@10, nDCG@10

Set `MAX_QUERIES = None` inside each script for full evaluation (default is a subset for faster iteration).

---

## Testing

Tests are standalone scripts (no pytest). Run from the project root:

```bash
python tests/test_tfidf_search.py
python tests/test_bm25_search.py
python tests/test_embedding_search.py
python tests/test_db.py
python tests/test_inverted_index.py
python tests/test_tfidf_index.py
python tests/test_bm25_index.py
python tests/test_lotte_qrels.py
python tests/test_single_lotte_query_evaluation.py
python tests/test_query_refinement.py
```

---

## Architecture

The system follows a **Service-Oriented Architecture (SOA)** with clear separation of concerns:

```
Streamlit Frontend (web_app.py)
        │
        ▼
FastAPI Gateway (api_app.py)
        │
        ▼
   ┌────┴────┐
   │ Routes  │  ← HTTP endpoint definitions
   └────┬────┘
        │
   ┌────┴──────┐
   │Controllers│  ← Business logic, service dispatch
   └────┬──────┘
        │
   ┌────┴────┐
   │ Models  │  ← Pydantic request/response schemas
   └─────────┘
        │
   ┌────┴──────────────────────────────────┐
   │              Services                  │
   ├──────────┬──────────┬─────────────────┤
   │ Dataset  │Preprocess│   Indexing       │
   │ Loading  │ Pipeline │ (Inverted Index) │
   ├──────────┼──────────┼─────────────────┤
   │  TF-IDF  │   BM25   │   Embedding      │
   ├──────────┼──────────┼─────────────────┤
   │  Hybrid  │   LTR    │ Query Refinement │
   └──────────┴──────────┴─────────────────┘
```

- Services are cached via `@lru_cache` in the controller
- BM25 uses a `Lock` for safe concurrent parameter updates
- Communication: REST API (JSON)

---

## Query Refinement

Applied automatically in "Basic + Additional Features" mode before dispatching to search services:

1. **Spell Correction** — Edit distance against NLTK `words` corpus
2. **Synonym Expansion** — WordNet synonyms (up to 2 per token)
3. **History-Based Boosting** — Repeats popular terms from past searches

The response includes `refined_query` and `enhanced` fields to show what changed.

---

## Index Files

Generated under `indices/lotte/{model}/` (gitignored — must rebuild locally):

| Model | Files |
|---|---|
| Inverted Index | `inverted_index.joblib`, `document_lengths.joblib`, `document_ids.json`, `report.json` |
| TF-IDF | `vectorizer.joblib`, `tfidf_matrix.npz`, `document_ids.json` |
| BM25 | `idf_by_term.joblib`, `document_lengths.joblib`, `document_ids.json`, `report.json` |
| Embedding | `document_embeddings.joblib`, `embedding_document_ids.joblib`, `report.json` |

---

## License

Academic project — Information Retrieval course, 2026.
