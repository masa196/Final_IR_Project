import sys
from pathlib import Path


# ==========================================================
# الوصول إلى جذر المشروع
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from services.dataset.dataset_loader import (
    load_lotte,
)

from services.evaluation.evaluation_service import (
    build_qrels_by_query_id,
    find_query_by_id,
    evaluate_single_query,
)

from services.tfidf.tfidf_search_service import (
    TfidfSearchService,
)


# ==========================================================
# إعدادات الاختبار
# ==========================================================

QUERY_ID = "0"

TOP_K = 10

TFIDF_DIRECTORY = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "tfidf"
)


# ==========================================================
# تشغيل الاختبار
# ==========================================================

def main():
    print("======================================")
    print("SINGLE LOTTE QUERY EVALUATION")
    print("======================================")

    # --------------------------------------
    # تحميل Dataset المحلية
    # --------------------------------------

    dataset = load_lotte()

    # --------------------------------------
    # بناء قاموس Qrels
    # --------------------------------------

    qrels_by_query_id = build_qrels_by_query_id(
        dataset
    )

    # --------------------------------------
    # جلب Query حقيقية باستخدام ID
    # --------------------------------------

    query = find_query_by_id(
        dataset=dataset,
        target_query_id=QUERY_ID,
    )

    # --------------------------------------
    # جلب Qrels الخاصة بهذه Query
    # --------------------------------------

    relevance_by_doc_id = qrels_by_query_id.get(
        QUERY_ID,
        {},
    )

    # --------------------------------------
    # تحميل ملفات TF-IDF
    # --------------------------------------

    search_service = TfidfSearchService(
        tfidf_directory=TFIDF_DIRECTORY,
    )

    # --------------------------------------
    # تشغيل التقييم
    # --------------------------------------

    evaluation = evaluate_single_query(
        search_service=search_service,
        query_id=QUERY_ID,
        query_text=query.text,
        relevance_by_doc_id=relevance_by_doc_id,
        k=TOP_K,
        include_text=True,
    )

    # --------------------------------------
    # عرض Query
    # --------------------------------------

    print("\n======================================")
    print("QUERY")
    print("======================================")

    print(f"Query ID:       {evaluation['query_id']}")
    print(f"Raw Query:      {evaluation['query_text']}")
    print(
        f"Processed Query: "
        f"{evaluation['processed_query']}"
    )

    # --------------------------------------
    # عرض الوثائق الصحيحة
    # --------------------------------------

    print("\n======================================")
    print("EXPECTED RELEVANT DOCUMENTS FROM QRELS")
    print("======================================")

    for doc_id in evaluation["relevant_doc_ids"]:
        print(f"Doc ID: {doc_id}")

    # --------------------------------------
    # عرض النتائج المسترجعة
    # --------------------------------------

    print("\n======================================")
    print("TOP RETRIEVED DOCUMENTS")
    print("======================================")

    relevant_doc_ids = set(
        evaluation["relevant_doc_ids"]
    )

    for rank, result in enumerate(
        evaluation["results"],
        start=1,
    ):
        doc_id = str(
            result["doc_id"]
        )

        is_relevant = (
            "✅ RELEVANT"
            if doc_id in relevant_doc_ids
            else "❌ NOT RELEVANT"
        )

        print("\n--------------------------------------")
        print(f"Rank:   {rank}")
        print(f"Doc ID: {doc_id}")
        print(f"Score:  {result['score']:.4f}")
        print(f"Qrel:   {is_relevant}")
        print("Text:")
        print(result["text"][:300])

    # --------------------------------------
    # عرض المقاييس
    # --------------------------------------

    print("\n======================================")
    print("METRICS")
    print("======================================")

    for metric_name, metric_value in (
        evaluation["metrics"].items()
    ):
        print(
            f"{metric_name}: "
            f"{metric_value:.4f}"
        )


if __name__ == "__main__":
    main()