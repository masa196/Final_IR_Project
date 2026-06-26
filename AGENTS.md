# AGENTS.md — IR Search Engine (LoTTe)

## Project overview
Python-only IR search engine for the `lotte/recreation/dev/search` dataset (~200K docs). Three ranking models: TF-IDF (VSM), BM25, Embedding (SentenceTransformer `all-MiniLM-L6-v2`). FastAPI backend + Streamlit frontend, SOA-style architecture.

## Data pipeline (ordered)
```bash
python scripts/prepare_lotte_data.py          # 1. Load + preprocess (jsonl output)
python scripts/load_raw_documents_to_db.py    # 2. Store raw text in SQLite
python scripts/build_inverted_index.py        # 3. Build inverted index (prerequisite for TF-IDF + BM25)
python scripts/build_tfidf_index.py           # 4a. TF-IDF matrix
python scripts/build_bm25_index.py            # 4b. BM25 IDF + stats
python scripts/build_embedding.py             # 4c. Embedding matrix (SentenceTransformer, batched)
```

Each script defaults to processing **all documents** (`MAX_DOCUMENTS = None`). Set to a small number (e.g. `1000`) for quick iteration.

## Running the system
- **API**: `python api_app.py` — FastAPI on `http://127.0.0.1:8000`, docs at `/docs`
- **Frontend**: `streamlit run web_app.py` — Streamlit on `http://127.0.0.1:8501` (expects API at `http://127.0.0.1:8000`)
- **Order**: start API first, then frontend

## Testing
Tests are standalone Python scripts (NOT pytest). Run any with:
```bash
python tests/test_*.py
```
Key tests: `test_tfidf_search.py`, `test_bm25_search.py`, `test_embedding_search.py`, `test_db.py`, `test_inverted_index.py`, `test_tfidf_index.py`, `test_bm25_index.py`, `test_lotte_qrels.py`, `test_single_lotte_query_evaluation.py`

## Evaluation
```bash
python scripts/evaluate_bm25_lotte.py
python scripts/evaluate_tfidf_lotte.py
python scripts/evaluate_embedding_lotte.py
```
Outputs JSON reports to `evaluations/lotte/{model}/evaluation_report.json`. Metrics: Precision@10, Recall@10, MAP@10, nDCG@10. Set `MAX_QUERIES = None` for full evaluation.

## Architecture
- **`routes/`** — FastAPI route definitions  
- **`controllers/`** — Business logic (search dispatch, health)  
- **`models/`** — Pydantic request/response schemas  
- **`services/`** — SOA-style: `dataset/`, `preprocessing/`, `indexing/`, `data_base/`, `tfidf/`, `bm25/`, `embedding/`, `evaluation/`, `query_refinement/`  
- **`scripts/`** — CLI entrypoints for building indices and evaluation  
- **`tests/`** — Manual test scripts  

Services are cached (`@lru_cache`) in the controller. BM25 uses a `Lock` to safely update k1/b parameters on concurrent requests.

## Index files stored under `indices/lotte/{model}/`
- **inverted_index**: `inverted_index.joblib`, `document_lengths.joblib`, `document_ids.json`, `report.json`
- **tfidf**: `vectorizer.joblib`, `tfidf_matrix.npz`, `document_ids.json`
- **bm25**: `idf_by_term.joblib`, `document_lengths.joblib`, `document_ids.json`, `report.json`
- **embedding**: `document_embeddings.joblib`, `embedding_document_ids.joblib`, `report.json`

## Preprocessing pipeline (in order)
`lowercase` → `contractions.fix()` → URL removal → commas in numbers removed → decimal dots replaced with `_` → punctuation removal → NLTK `word_tokenize` → POS tagging → `WordNetLemmatizer` (with POS-aware lemmatization) → stopword removal → keep numbers (for LoTTe). Config: `preprocess_lotte()` calls `preprocess_text(text, keep_numbers=True)`.

## Import convention
All CLI scripts and tests append `PROJECT_ROOT` to `sys.path` at the top for imports to work. Always run from project root.

## .gitignore essentials
`database/`, `processed_data/`, `indices/`, `*.db`, `*.joblib`, `*.npz` are gitignored (large generated artifacts). These must be rebuilt locally.

## Notable
- **No requirements.txt / pyproject.toml** — Dependencies include `fastapi`, `streamlit`, `ir_datasets`, `nltk`, `contractions`, `scikit-learn`, `scipy`, `sentence-transformers`, `joblib`, `pydantic`, `numpy`, `uvicorn`
- **No test framework** — all tests are `if __name__ == "__main__": main()`
- Dataset loads from `ir_datasets` (`lotte/recreation/dev/search`). Second dataset (Quora) is wired in `start.py` but not fully integrated.
- BM25 UI in Streamlit lets users adjust `k1` (0.5–3.0, default 1.5) and `b` (0.0–1.0, default 0.75) via sliders.
- Encoding: `all-*` CORS in FastAPI, `ensure_ascii[::-1]=False` for JSON output.
- **Query refinement** (always-on): `services/query_refinement/` applies spell correction (NLTK `words` corpus + edit distance), synonym expansion (WordNet, 2 per token), and history-based term boosting (repeats popular past terms). Runs automatically in the controller before dispatching to search services. Response includes `refined_query` and `enhanced` fields.
