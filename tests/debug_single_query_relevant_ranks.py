import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# الوصول إلى جذر المشروع
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from services.dataset.dataset_loader import (
    load_lotte,
)

from services.evaluation.evaluation_service import (
    build_qrels_by_query_id,
    find_query_by_id,
)

from services.preprocessing.preprocessing_service import (
    preprocess_lotte,
)


# ==========================================================
# الإعدادات
# ==========================================================

QUERY_ID = "0"

TFIDF_DIRECTORY = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "tfidf"
)

VECTORIZER_FILE = (
    TFIDF_DIRECTORY
    / "vectorizer.joblib"
)

MATRIX_FILE = (
    TFIDF_DIRECTORY
    / "tfidf_matrix.npz"
)

DOCUMENT_IDS_FILE = (
    TFIDF_DIRECTORY
    / "document_ids.json"
)


# ==========================================================
# التشغيل
# ==========================================================

def main():
    print("======================================")
    print("DEBUGGING RELEVANT DOCUMENT RANKS")
    print("======================================")

    dataset = load_lotte()

    qrels_by_query_id = build_qrels_by_query_id(
        dataset
    )

    query = find_query_by_id(
        dataset=dataset,
        target_query_id=QUERY_ID,
    )

    relevance_by_doc_id = qrels_by_query_id.get(
        QUERY_ID,
        {},
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    tfidf_matrix = load_npz(
        MATRIX_FILE
    )

    with DOCUMENT_IDS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        document_ids = json.load(file)

    # ربط doc_id برقم الصف داخل TF-IDF Matrix.
    matrix_index_by_doc_id = {
        str(doc_id): matrix_index
        for matrix_index, doc_id
        in enumerate(document_ids)
    }

    query_tokens = preprocess_lotte(
        query.text
    )

    query_vector = vectorizer.transform(
        [
            " ".join(query_tokens)
        ]
    )

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix,
    ).flatten()

    # ترتيب جميع الصفوف من أعلى Score إلى أقل Score.
    ranked_matrix_indices = np.argsort(
        scores
    )[::-1]

    rank_by_matrix_index = {
        int(matrix_index): rank
        for rank, matrix_index
        in enumerate(
            ranked_matrix_indices,
            start=1,
        )
    }

    print(f"\nQuery ID:  {QUERY_ID}")
    print(f"Raw Query: {query.text}")
    print(f"Tokens:    {query_tokens}")

    print("\n======================================")
    print("EXPECTED DOCUMENT RANKS")
    print("======================================")

    for doc_id in relevance_by_doc_id:
        matrix_index = matrix_index_by_doc_id.get(
            str(doc_id)
        )

        if matrix_index is None:
            print(
                f"Doc ID: {doc_id} "
                f"→ ❌ NOT FOUND IN TF-IDF MATRIX"
            )
            continue

        rank = rank_by_matrix_index[
            matrix_index
        ]

        score = float(
            scores[matrix_index]
        )

        print(
            f"Doc ID: {doc_id} "
            f"→ Rank: {rank:,} "
            f"| Score: {score:.4f}"
        )


if __name__ == "__main__":
    main()