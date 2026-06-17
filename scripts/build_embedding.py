import json
from pathlib import Path

from services.embedding.embedding_service import (
    build_embedding,
)


from services.data_base.database_service import (
    get_raw_documents_by_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INVERTED_INDEX_DIRECTORY = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "inverted_index"
)

EMBEDDING_DIRECTORY = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "embedding"
)


def main():
    report = build_embedding(
        inverted_index_directory=INVERTED_INDEX_DIRECTORY,
        output_directory=EMBEDDING_DIRECTORY,
        get_raw_documents_by_ids_func=get_raw_documents_by_ids,
        batch_size=10000,
    )

    print("\n======================================")
    print("EMBEDDING REPORT")
    print("======================================")

    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()