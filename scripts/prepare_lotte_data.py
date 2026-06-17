import json
import sys
import time
from pathlib import Path


# حتى يقدر السكربت الوصول إلى services عند تشغيله من جذر المشروع
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from services.dataset.dataset_loader import load_lotte
from services.preprocessing.preprocessing_service import preprocess_lotte



# ==========================================================
# إعدادات التشغيل
# ==========================================================

# بالبداية نجرب فقط على عدد صغير من الوثائق.
# بعد التأكد من صحة النتائج نضع القيمة None.
MAX_DOCUMENTS = None

# عدد Queries قليل، لذلك يمكن معالجة كامل Queries مباشرةً.
MAX_QUERIES = None

# طباعة تقدم التنفيذ كل 1000 وثيقة.
PRINT_EVERY = 1000


# ==========================================================
# دوال مساعدة
# ==========================================================

def write_json_line(file, record: dict):
    """
    حفظ سجل واحد كسطر JSON مستقل.
    """
    file.write(
        json.dumps(record, ensure_ascii=False) + "\n"
    )


def get_processing_limit(total: int, limit: int | None) -> int:
    """
    معرفة العدد المتوقع مع مراعاة وضع التجربة.
    """
    if limit is None:
        return total

    return min(total, limit)


# ==========================================================
# معالجة الوثائق
# ==========================================================

def process_documents(dataset, output_file, errors_file):
    saved_count = 0
    empty_count = 0
    error_count = 0

    expected_count = get_processing_limit(
        dataset.docs_count(),
        MAX_DOCUMENTS,
    )

    print("\n======================================")
    print("Processing Lotte documents...")
    print("======================================")
    print(f"Documents to process: {expected_count:,}")

    start_time = time.time()

    with output_file.open("w", encoding="utf-8") as file:
        for index, document in enumerate(
            dataset.docs_iter(),
            start=1,
        ):
            if MAX_DOCUMENTS is not None and index > MAX_DOCUMENTS:
                break

            try:
                tokens = preprocess_lotte(document.text)

                if not tokens:
                    empty_count += 1

                record = {
                    "doc_id": document.doc_id,
                    "text": document.text,
                    "tokens": tokens,
                }

                write_json_line(file, record)
                saved_count += 1

                # طباعة أول ثلاث عينات للمراجعة
                if saved_count <= 3:
                    print("\n------------------------------")
                    print(f"DOCUMENT SAMPLE {saved_count}")
                    print("------------------------------")
                    print("ID:")
                    print(document.doc_id)

                    print("\nRAW TEXT:")
                    print(document.text)

                    print("\nPROCESSED TOKENS:")
                    print(tokens)

                if saved_count % PRINT_EVERY == 0:
                    print(
                        f"\nProcessed documents: "
                        f"{saved_count:,}/{expected_count:,}"
                    )

            except Exception as error:
                error_count += 1

                errors_file.write(
                    "[DOCUMENT ERROR]\n"
                    f"doc_id: {getattr(document, 'doc_id', 'unknown')}\n"
                    f"text: {getattr(document, 'text', '')}\n"
                    f"error: {repr(error)}\n\n"
                )

    elapsed_seconds = round(time.time() - start_time, 2)

    return {
        "saved": saved_count,
        "empty_after_preprocessing": empty_count,
        "errors": error_count,
        "elapsed_seconds": elapsed_seconds,
    }


# ==========================================================
# معالجة Queries
# ==========================================================

def process_queries(dataset, output_file, errors_file):
    saved_count = 0
    empty_count = 0
    error_count = 0

    expected_count = get_processing_limit(
        dataset.queries_count(),
        MAX_QUERIES,
    )

    print("\n======================================")
    print("Processing Lotte queries...")
    print("======================================")
    print(f"Queries to process: {expected_count:,}")

    start_time = time.time()

    with output_file.open("w", encoding="utf-8") as file:
        for index, query in enumerate(
            dataset.queries_iter(),
            start=1,
        ):
            if MAX_QUERIES is not None and index > MAX_QUERIES:
                break

            try:
                tokens = preprocess_lotte(query.text)

                if not tokens:
                    empty_count += 1

                record = {
                    "query_id": query.query_id,
                    "text": query.text,
                    "tokens": tokens,
                }

                write_json_line(file, record)
                saved_count += 1

                # طباعة أول ثلاث عينات للمراجعة
                if saved_count <= 3:
                    print("\n------------------------------")
                    print(f"QUERY SAMPLE {saved_count}")
                    print("------------------------------")
                    print("ID:")
                    print(query.query_id)

                    print("\nRAW QUERY:")
                    print(query.text)

                    print("\nPROCESSED TOKENS:")
                    print(tokens)

            except Exception as error:
                error_count += 1

                errors_file.write(
                    "[QUERY ERROR]\n"
                    f"query_id: {getattr(query, 'query_id', 'unknown')}\n"
                    f"text: {getattr(query, 'text', '')}\n"
                    f"error: {repr(error)}\n\n"
                )

    elapsed_seconds = round(time.time() - start_time, 2)

    return {
        "saved": saved_count,
        "empty_after_preprocessing": empty_count,
        "errors": error_count,
        "elapsed_seconds": elapsed_seconds,
    }


# ==========================================================
# تشغيل البرنامج
# ==========================================================

def main():
    dataset = load_lotte()

    output_directory = PROJECT_ROOT / "processed_data" / "lotte"
    output_directory.mkdir(parents=True, exist_ok=True)

    documents_file = output_directory / "documents.jsonl"
    queries_file = output_directory / "queries.jsonl"
    errors_file_path = output_directory / "errors.log"
    report_file = output_directory / "report.json"

    print("✅ LOTTE Dataset loaded successfully")

    print("\nFull dataset size:")
    print(f"Documents: {dataset.docs_count():,}")
    print(f"Queries:   {dataset.queries_count():,}")

    with errors_file_path.open("w", encoding="utf-8") as errors_file:
        documents_report = process_documents(
            dataset,
            documents_file,
            errors_file,
        )

        queries_report = process_queries(
            dataset,
            queries_file,
            errors_file,
        )

    report = {
        "dataset": "lotte",
        "mode": (
            "full"
            if MAX_DOCUMENTS is None
            else "test"
        ),
        "full_dataset_documents": dataset.docs_count(),
        "full_dataset_queries": dataset.queries_count(),
        "documents": documents_report,
        "queries": queries_report,
    }

    with report_file.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n======================================")
    print("✅ LOTTE DATA PREPARATION FINISHED")
    print("======================================")

    print("\nDocuments:")
    print(f"Saved:  {documents_report['saved']:,}")
    print(
        f"Empty:  "
        f"{documents_report['empty_after_preprocessing']:,}"
    )
    print(f"Errors: {documents_report['errors']:,}")

    print("\nQueries:")
    print(f"Saved:  {queries_report['saved']:,}")
    print(
        f"Empty:  "
        f"{queries_report['empty_after_preprocessing']:,}"
    )
    print(f"Errors: {queries_report['errors']:,}")

    print("\nGenerated files:")
    print(documents_file)
    print(queries_file)
    print(errors_file_path)
    print(report_file)


if __name__ == "__main__":
    main()