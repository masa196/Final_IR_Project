import sys
from pathlib import Path
from unicodedata import name


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from services.bm25.bm25_search_service import (
    BM25SearchService,
)


BM25_DIRECTORY = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "bm25"
)

INVERTED_INDEX_DIRECTORY = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "inverted_index"
)


def main():
    print("======================================")
    print("BM25 SEARCH TEST")
    print("======================================")

    search_service = BM25SearchService(
        bm25_directory=BM25_DIRECTORY,
        inverted_index_directory=INVERTED_INDEX_DIRECTORY,
        k1=1.5,
        b=0.75,
    )

    query = input(
        "\nEnter query: "
    )

    response = search_service.search(
        query=query,
        top_k=10,
        include_text=True,
    )

    print("\n======================================")
    print("QUERY")
    print("======================================")

    print(f"Raw Query:       {response['query']}")
    print(f"Processed Query: {response['processed_query']}")

    print("\n======================================")
    print("TOP RESULTS")
    print("======================================")

    for rank, result in enumerate(
        response["results"],
        start=1,
    ):
        print("\n--------------------------------------")
        print(f"Rank:   {rank}")
        print(f"Doc ID: {result['doc_id']}")
        print(f"Score:  {result['score']:.4f}")
        print("Text:")
        if "text" in result and result["text"]:
            print(result["text"][:500])
        else:
            print("Text was not loaded.")
      


if __name__ == "__main__":
    main()