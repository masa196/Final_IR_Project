import sys
from pathlib import Path

# 1. إعداد مسار جذر المشروع لضمان عمل الـ imports بشكل صحيح
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 2. استيراد خدمة البحث الدلالي للـ Embedding المحدثة
from services.embedding.embedding_search_service import (
    EmbeddingSearchService,
)

# 3. تحديد مسار مجلد الـ Embedding الفهرس الحالي
EMBEDDING_DIRECTORY = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "embedding"
)


def main():
    print("======================================")
    print("EMBEDDING SEARCH TEST (WITH PARTIAL SORTING)")
    print("======================================")

    # 4. بناء غرض الخدمة وشحن الفهرس الحالي في الـ RAM
    search_service = EmbeddingSearchService(
        embedding_directory=EMBEDDING_DIRECTORY,
    )

    # 5. استقبال الاستعلام من المستخدم
    query = input(
        "\nEnter query: "
    )

    # 6. استدعاء دالة البحث الدلالي (التي تستخدم الآن الترتيب الجزئي np.argpartition السريع)
    response = search_service.search(
        query=query,
        top_k=10,
        include_text=True,
    )

    print("\n======================================")
    print("QUERY")
    print("======================================")

    print(f"Raw Query:       {response['query']}")

    print("\n======================================")
    print("TOP RESULTS")
    print("======================================")

    # 7. الدوران على النتائج المسترجعة وطباعتها بنفس التنسيق الحرفي لملف الـ BM25
    for rank, result in enumerate(
        response["results"],
        start=1,
    ):
        print("\n--------------------------------------")
        print(f"Rank:   {rank}")
        print(f"Doc ID: {result['doc_id']}")
        print(f"Score:  {result['score']:.4f}")  # سكور الـ Cosine Similarity المستخرج
        print("Text:")
        if "text" in result and result["text"]:
            print(result["text"][:500])  # طباعة أول 500 حرف لتناسق العرض بالعين
        else:
            print("Text was not loaded.")


if __name__ == "__main__":
    main()