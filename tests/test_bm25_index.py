import json
from os import name
import sys
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


BM25_DIRECTORY = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "bm25"
)


def main():
    print("======================================")
    print("BM25 INDEX TEST")
    print("======================================")

    idf_file = BM25_DIRECTORY / "idf_by_term.joblib"
    lengths_file = BM25_DIRECTORY / "document_lengths.joblib"
    document_ids_file = BM25_DIRECTORY / "document_ids.json"
    report_file = BM25_DIRECTORY / "report.json"

    required_files = [
        idf_file,
        lengths_file,
        document_ids_file,
        report_file,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file: {file_path}"
            )

    idf_by_term = joblib.load(
        idf_file
    )

    document_lengths = joblib.load(
        lengths_file
    )

    with document_ids_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        document_ids = json.load(file)

    with report_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(file)

    print(f"Documents: {len(document_ids):,}")
    print(f"Document lengths: {len(document_lengths):,}")
    print(f"IDF terms: {len(idf_by_term):,}")

    print("\nReport:")
    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nSample IDF values:")

    sample_terms = [
        "bard",
        "sing",
        "spell",
        "attack",
        "dnd",
    ]

    for term in sample_terms:
        print(
            f"{term}: "
            f"{idf_by_term.get(term, 0.0):.4f}"
        )

    if len(document_ids) != len(document_lengths):
        raise ValueError(
            "document_ids count does not match document_lengths count."
        )

    print("\n✅ BM25 INDEX TEST PASSED")


if __name__ == "__main__":
    main()