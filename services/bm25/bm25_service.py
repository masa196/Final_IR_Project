import json
import math
from pathlib import Path

import joblib


# ==========================================================
# حساب IDF الخاص بـ BM25
# ==========================================================

def calculate_bm25_idf(
    number_of_documents: int,
    document_frequency: int,
) -> float:
    """
    حساب IDF في BM25.

    الكلمة النادرة تأخذ IDF أعلى.
    الكلمة المنتشرة في وثائق كثيرة تأخذ IDF أقل.
    """

    if document_frequency <= 0:
        return 0.0

    idf = math.log(
        1
        + (
            number_of_documents
            - document_frequency
            + 0.5
        )
        / (
            document_frequency
            + 0.5
        )
    )

    return idf


# ==========================================================
# بناء BM25 Index
# ==========================================================

def build_bm25_index(
    inverted_index_directory: Path,
    output_directory: Path,
) -> dict:
    """
    بناء ملفات BM25 اعتماداً على inverted index الموجود مسبقاً.

    لا نعيد preprocessing.
    لا نعيد قراءة LoTTe documents.
    فقط نستخرج الإحصائيات المطلوبة لـ BM25.
    """

    inverted_index_directory = Path(
        inverted_index_directory
    )

    output_directory = Path(
        output_directory
    )


    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("======================================")
    print("BUILDING BM25 INDEX")
    print("======================================")

    print("Loading inverted index...")

    inverted_index = joblib.load(
        inverted_index_directory
        / "inverted_index.joblib"
    )

    print("Loading document lengths...")

    raw_document_lengths = joblib.load(
        inverted_index_directory
        / "document_lengths.joblib"
    )

    document_lengths = {
        str(doc_id): int(length)
        for doc_id, length
        in raw_document_lengths.items()
    }

    print("Loading document ids...")

    with (
        inverted_index_directory
        / "document_ids.json"
    ).open("r", encoding="utf-8") as file:
        document_ids = [
            str(doc_id)
            for doc_id in json.load(file)
        ]

    number_of_documents = len(
        document_ids
    )

    average_document_length = (
        sum(document_lengths.values())
        / number_of_documents
    )

    print("Calculating BM25 IDF values...")

    idf_by_term = {}

    for term, posting_list in inverted_index.items():
        document_frequency = len(
            posting_list
        )

        idf_by_term[term] = calculate_bm25_idf(
            number_of_documents=number_of_documents,
            document_frequency=document_frequency,
        )

    report = {
        "number_of_documents": number_of_documents,
        "vocabulary_size": len(inverted_index),
        "average_document_length": average_document_length,
        "document_lengths_count": len(document_lengths),
        "idf_terms_count": len(idf_by_term),
    }

    print("Saving BM25 files...")

    joblib.dump(
        idf_by_term,
        output_directory / "idf_by_term.joblib",
    )

    joblib.dump(
        document_lengths,
        output_directory / "document_lengths.joblib",
    )

    with (
        output_directory
        / "document_ids.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            document_ids,
            file,
            ensure_ascii=False,
            indent=4,
        )

    with (
        output_directory
        / "report.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("✅ BM25 INDEX BUILT SUCCESSFULLY")

    return report