import sys
from pathlib import Path


# ==========================================================
# الوصول إلى جذر المشروع
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from services.indexing.indexing_service import (
    build_inverted_index,
    save_inverted_index,
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

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "inverted_index"
)


# ==========================================================
# الإعدادات
# ==========================================================

# نجرب بالبداية على أول 1000 وثيقة فقط.
# بعد التأكد من صحة النتيجة نضع None.
MAX_DOCUMENTS = None

PRINT_EVERY = 1000


# ==========================================================
# تشغيل البناء
# ==========================================================

def main():
    print("======================================")
    print("BUILDING LOTTE INVERTED INDEX")
    print("======================================")

    index_data = build_inverted_index(
        documents_file=DOCUMENTS_FILE,
        max_documents=MAX_DOCUMENTS,
        print_every=PRINT_EVERY,
    )

    generated_files = save_inverted_index(
        index_data=index_data,
        output_directory=OUTPUT_DIRECTORY,
    )

    report = index_data["report"]

    print("\n======================================")
    print("✅ INVERTED INDEX CREATED")
    print("======================================")

    print(
        f"Documents:  "
        f"{report['documents_count']:,}"
    )

    print(
        f"Empty docs: "
        f"{report['empty_documents']:,}"
    )

    print(
        f"Vocabulary: "
        f"{report['vocabulary_size']:,}"
    )

    print(
        f"Total tokens: "
        f"{report['total_tokens']:,}"
    )

    print(
        f"Average document length: "
        f"{report['average_document_length']}"
    )

    print(
        f"Elapsed seconds: "
        f"{report['elapsed_seconds']}"
    )

    print("\nGenerated files:")

    for file_path in generated_files.values():
        print(file_path)


if __name__ == "__main__":
    main()