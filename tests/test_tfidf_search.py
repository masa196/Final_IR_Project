import sys
from pathlib import Path


# ==========================================================
# الوصول إلى جذر المشروع
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from services.tfidf.tfidf_search_service import (
    TfidfSearchService,
)


# ==========================================================
# مكان ملفات TF-IDF
# ==========================================================

TFIDF_DIRECTORY = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "tfidf"
)


# ==========================================================
# تشغيل اختبار البحث
# ==========================================================

def main():
    print("======================================")
    print("LOTTE TF-IDF SEARCH TEST")
    print("======================================")

    search_service = TfidfSearchService(
        tfidf_directory=TFIDF_DIRECTORY,
    )

    query = input(
        "\nEnter your query: "
    )

    response = search_service.search(
        query=query,
        top_k=10,
    )

    print("\nProcessed query:")
    print(
        response["processed_query"]
    )

    if not response["results"]:
        print(
            "\nNo matching documents found."
        )
        return

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
        print(result["text"][:500])


if __name__ == "__main__":
    main()