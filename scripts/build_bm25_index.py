import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from services.bm25.bm25_service import (
    build_bm25_index,
)


INVERTED_INDEX_DIRECTORY = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "inverted_index"
)

BM25_DIRECTORY = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "bm25"
)


def main():
    report = build_bm25_index(
        inverted_index_directory=INVERTED_INDEX_DIRECTORY,
        output_directory=BM25_DIRECTORY,
    )

    print("\n======================================")
    print("BM25 REPORT")
    print("======================================")

    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()