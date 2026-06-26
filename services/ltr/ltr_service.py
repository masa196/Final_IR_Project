import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from services.ltr.ltr_config import (
    FEATURE_NAMES,
    N_ESTIMATORS,
    MAX_DEPTH,
    LEARNING_RATE,
    MIN_SAMPLES_LEAF,
    RANDOM_STATE,
    CANDIDATE_K,
)
from services.ltr.feature_extraction_service import (
    LTRFeatureExtractor,
)
from services.bm25.bm25_search_service import (
    BM25SearchService,
)


def build_ltr_model(
    queries: list[tuple[str, str]],
    qrels_by_query_id: dict[str, dict[str, float]],
    inverted_index_directory: Path,
    bm25_directory: Path,
    embedding_directory: Path,
    output_directory: Path,
    candidate_k: int = CANDIDATE_K,
    max_queries: int | None = None,
    print_every: int = 10,
) -> dict:
    """
    Train an LTR model.

    queries:
        List of (query_id, query_text) tuples.
    qrels_by_query_id:
        Dict mapping query_id -> {doc_id: relevance_score, ...}.
    """
    print("======================================")
    print("BUILDING LTR MODEL (Learning to Rank)")
    print("======================================")

    print("Loading BM25 search service for candidate generation...")
    bm25_service = BM25SearchService(
        bm25_directory=bm25_directory,
        inverted_index_directory=inverted_index_directory,
        k1=1.5,
        b=0.75,
    )

    print("Loading index data for feature extractor...")
    inverted_index = joblib.load(
        inverted_index_directory / "inverted_index.joblib"
    )
    idf_by_term = joblib.load(
        bm25_directory / "idf_by_term.joblib"
    )
    raw_lengths = joblib.load(
        bm25_directory / "document_lengths.joblib"
    )
    document_lengths = {
        str(doc_id): int(length)
        for doc_id, length in raw_lengths.items()
    }
    avg_dl = (
        sum(document_lengths.values()) / len(document_lengths)
        if document_lengths
        else 1.0
    )

    print("Loading feature extractor...")
    feature_extractor = LTRFeatureExtractor(
        inverted_index=inverted_index,
        idf_by_term=idf_by_term,
        document_lengths=document_lengths,
        avg_document_length=avg_dl,
        embedding_directory=embedding_directory,
    )

    feature_vectors = []
    labels = []
    query_ids_used = []
    skipped_no_qrels = 0
    skipped_no_candidates = 0
    queries_processed = 0

    start_time = time.time()

    for query_id, query_text in queries:
        if max_queries is not None and queries_processed >= max_queries:
            break

        relevance_by_doc_id = qrels_by_query_id.get(query_id)
        if not relevance_by_doc_id:
            skipped_no_qrels += 1
            continue

        bm25_response = bm25_service.search(
            query=query_text,
            top_k=candidate_k,
            include_text=False,
        )

        candidate_doc_ids = bm25_response.get("result_ids", [])
        if not candidate_doc_ids:
            skipped_no_candidates += 1
            continue

        bm25_scores = {
            r["doc_id"]: r["score"]
            for r in bm25_response.get("results", [])
        }
        bm25_ranks = {
            r["doc_id"]: idx + 1
            for idx, r in enumerate(
                bm25_response.get("results", [])
            )
        }

        query_features = feature_extractor.extract_features(
            query_text=query_text,
            candidate_doc_ids=candidate_doc_ids,
            bm25_scores=bm25_scores,
            bm25_ranks=bm25_ranks,
        )

        query_labels = np.array(
            [
                float(relevance_by_doc_id.get(doc_id, 0.0))
                for doc_id in candidate_doc_ids
            ],
            dtype=np.float64,
        )

        feature_vectors.append(query_features)
        labels.append(query_labels)
        query_ids_used.append(query_id)

        queries_processed += 1

        if queries_processed % print_every == 0:
            elapsed = time.time() - start_time
            print(
                f"Processed queries: {queries_processed:,} "
                f"({elapsed:.1f}s)"
            )

    if not feature_vectors:
        raise ValueError(
            "No training data generated. "
            "Check that the queries have QRELs."
        )

    X = np.vstack(feature_vectors)
    y = np.concatenate(labels)

    print(f"\nTraining data shape: {X.shape}")
    print(f"Positive labels: {int(np.sum(y > 0)):,}")
    print(f"Negative labels: {int(np.sum(y == 0)):,}")
    print(f"Queries used: {len(query_ids_used):,}")
    print(f"Skipped (no QRELs): {skipped_no_qrels:,}")
    print(f"Skipped (no candidates): {skipped_no_candidates:,}")
    print(f"Feature dimension: {X.shape[1]}")

    print("\nTraining GradientBoostingRegressor...")
    train_start = time.time()

    model = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )

    model.fit(X, y)

    train_elapsed = time.time() - train_start
    print(f"Training completed in {train_elapsed:.1f}s")

    train_score = model.score(X, y)

    output_directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        output_directory / "ltr_model.joblib",
    )

    with (
        output_directory / "feature_names.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    report = {
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "GradientBoostingRegressor",
        "parameters": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "random_state": RANDOM_STATE,
        },
        "training_data": {
            "total_samples": int(X.shape[0]),
            "feature_dimension": int(X.shape[1]),
            "feature_names": FEATURE_NAMES,
            "positive_labels": int(np.sum(y > 0)),
            "negative_labels": int(np.sum(y == 0)),
            "positive_ratio": round(
                float(np.sum(y > 0) / len(y)), 4
            ),
        },
        "queries": {
            "candidate_k": candidate_k,
            "queries_used": len(query_ids_used),
            "skipped_no_qrels": skipped_no_qrels,
            "skipped_no_candidates": skipped_no_candidates,
        },
        "training_score": round(train_score, 4),
        "training_time_seconds": round(train_elapsed, 2),
        "total_time_seconds": round(
            time.time() - start_time, 2
        ),
    }

    with (
        output_directory / "report.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n✅ LTR MODEL BUILT SUCCESSFULLY")
    print(f"Model saved to: {output_directory / 'ltr_model.joblib'}")
    print(f"Report saved to: {output_directory / 'report.json'}")

    return report
