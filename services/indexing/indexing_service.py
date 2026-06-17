import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib


# ==========================================================
# بناء Inverted Index
# ==========================================================

def build_inverted_index(
    documents_file: Path,
    max_documents: int | None = None,
    print_every: int = 1000,
):
    """
    قراءة الوثائق المعالجة من JSONL وبناء Inverted Index.

    شكل الفهرس:
    term -> [(doc_id, frequency), ...]
    """

    if not documents_file.exists():
        raise FileNotFoundError(
            f"Documents file was not found: {documents_file}"
        )

    inverted_index = defaultdict(list)

    # سنحتاج أطوال الوثائق لاحقاً عند بناء BM25
    document_lengths = {}

    # سنحتاج ترتيب IDs لاحقاً لربط صفوف TF-IDF بالوثائق
    document_ids = []

    documents_count = 0
    empty_documents_count = 0
    total_tokens = 0

    start_time = time.time()

    with documents_file.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if (
                max_documents is not None
                and line_number > max_documents
            ):
                break

            record = json.loads(line)

            doc_id = str(record["doc_id"])
            tokens = record["tokens"]

            document_ids.append(doc_id)

            document_length = len(tokens)

            document_lengths[doc_id] = document_length
            total_tokens += document_length

            if not tokens:
                empty_documents_count += 1

            # مثال:
            # ["game", "game", "rule"]
            #
            # تصبح:
            # {"game": 2, "rule": 1}
            term_frequencies = Counter(tokens)

            for term, frequency in term_frequencies.items():
                inverted_index[term].append(
                    (doc_id, frequency)
                )

            documents_count += 1

            if documents_count % print_every == 0:
                print(
                    f"Processed documents: "
                    f"{documents_count:,}"
                )

    elapsed_seconds = round(
        time.time() - start_time,
        2,
    )

    average_document_length = (
        total_tokens / documents_count
        if documents_count > 0
        else 0
    )

    return {
        "inverted_index": dict(inverted_index),
        "document_lengths": document_lengths,
        "document_ids": document_ids,
        "report": {
            "documents_count": documents_count,
            "empty_documents": empty_documents_count,
            "vocabulary_size": len(inverted_index),
            "total_tokens": total_tokens,
            "average_document_length": round(
                average_document_length,
                2,
            ),
            "elapsed_seconds": elapsed_seconds,
        },
    }


# ==========================================================
# حفظ Inverted Index
# ==========================================================

def save_inverted_index(
    index_data: dict,
    output_directory: Path,
):
    """
    حفظ الملفات الناتجة عن عملية الفهرسة.
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    inverted_index_file = (
        output_directory
        / "inverted_index.joblib"
    )

    document_lengths_file = (
        output_directory
        / "document_lengths.joblib"
    )

    document_ids_file = (
        output_directory
        / "document_ids.json"
    )

    report_file = (
        output_directory
        / "report.json"
    )

    joblib.dump(
        index_data["inverted_index"],
        inverted_index_file,
        compress=3,
    )

    joblib.dump(
        index_data["document_lengths"],
        document_lengths_file,
        compress=3,
    )

    with document_ids_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            index_data["document_ids"],
            file,
            indent=4,
            ensure_ascii=False,
        )

    with report_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            index_data["report"],
            file,
            indent=4,
            ensure_ascii=False,
        )

    return {
        "inverted_index_file": inverted_index_file,
        "document_lengths_file": document_lengths_file,
        "document_ids_file": document_ids_file,
        "report_file": report_file,
    }