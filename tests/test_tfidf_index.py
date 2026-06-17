import json
from os import name
from pathlib import Path

import joblib
from scipy.sparse import load_npz


# ==========================================================
# المسارات
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
# الاختبار
# ==========================================================

def main():
    print("======================================")
    print("TESTING TF-IDF INDEX")
    print("======================================")

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

    vocabulary_size = len(
        vectorizer.vocabulary_
    )

    print(f"Documents IDs count: {len(document_ids):,}")

    print(
        f"Matrix shape: "
        f"{tfidf_matrix.shape[0]:,} rows × "
        f"{tfidf_matrix.shape[1]:,} columns"
    )

    print(
        f"Vocabulary size: "
        f"{vocabulary_size:,}"
    )

    print(
        f"Non-zero values: "
        f"{tfidf_matrix.nnz:,}"
    )

    # ======================================================
    # فحوصات أساسية
    # ======================================================

    assert (
        len(document_ids)
        == tfidf_matrix.shape[0]
    ), (
        "❌ Number of document IDs does not match "
        "the number of matrix rows."
    )

    assert (
        vocabulary_size
        == tfidf_matrix.shape[1]
    ), (
        "❌ Vocabulary size does not match "
        "the number of matrix columns."
    )

    print("\n✅ TF-IDF INDEX TEST PASSED")


if __name__ == "__main__":
    main()