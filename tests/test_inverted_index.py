from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_FILE = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "inverted_index"
    / "inverted_index.joblib"
)


def main():
    inverted_index = joblib.load(
        INDEX_FILE
    )

    terms = [
        "game",
        "combat",
        "player",
    ]

    for term in terms:
        postings = inverted_index.get(
            term,
            [],
        )

        print("\n======================================")
        print(f"TERM: {term}")
        print("======================================")

        print(
            f"Documents containing term: "
            f"{len(postings)}"
        )

        print("\nFirst 10 postings:")

        for doc_id, frequency in postings[:10]:
            print(
                f"doc_id={doc_id}, "
                f"frequency={frequency}"
            )


if __name__ == "__main__":
    main()