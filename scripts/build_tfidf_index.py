import sys
from pathlib import Path
from unicodedata import name


# ==========================================================
# الوصول إلى جذر المشروع
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from services.tfidf.tfidf_service import (
    build_tfidf_index,
)


# ==========================================================
# المسارات
# ==========================================================

DOCUMENTS_FILE = (
    PROJECT_ROOT
    / "processed_data"
    / "lotte"
    / "documents.jsonl"
)

INVERTED_INDEX_FILE = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "inverted_index"
    / "inverted_index.joblib"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "tfidf"
)


# ==========================================================
# الإعدادات
# ==========================================================

# نجرب أولاً على 1000 وثيقة فقط.
# بعد نجاح الاختبار نضع None.
MAX_DOCUMENTS = None

PRINT_EVERY = 1000


# ==========================================================
# تشغيل البناء
# ==========================================================

def main():
    print("======================================")
    print("BUILDING LOTTE VSM TF-IDF")
    print("======================================")

    report = build_tfidf_index(
        documents_file=DOCUMENTS_FILE,
        inverted_index_file=INVERTED_INDEX_FILE,
        output_directory=OUTPUT_DIRECTORY,
        max_documents=MAX_DOCUMENTS,
        print_every=PRINT_EVERY,
    )

    print("\n======================================")
    print("✅ TF-IDF INDEX CREATED")
    print("======================================")

    print(
        f"Mode: "
        f"{report['mode']}"
    )

    print(
        f"Documents: "
        f"{report['documents_count']:,}"
    )

    print(
        f"Empty documents: "
        f"{report['empty_documents']:,}"
    )

    print(
        f"Vocabulary: "
        f"{report['vocabulary_size']:,}"
    )

    print(
        f"Matrix shape: "
        f"{report['matrix_rows']:,} rows × "
        f"{report['matrix_columns']:,} columns"
    )

    print(
        f"Non-zero values: "
        f"{report['non_zero_values']:,}"
    )

    print(
        f"Elapsed seconds: "
        f"{report['elapsed_seconds']}"
    )


if __name__ == "__main__":
    main()