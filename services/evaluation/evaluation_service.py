from collections import defaultdict
from collections import defaultdict

from services.evaluation.evaluation_metrics_service import (
    precision_at_k,
    recall_at_k,
    average_precision_at_k,
    ndcg_at_k,
)
# ==========================================================
# بناء قاموس Qrels
# ==========================================================

def build_qrels_by_query_id(dataset) -> dict[
    str,
    dict[str, float],
]:
    """
    تجميع Qrels حسب query_id.

    الشكل الناتج:
    {
        "query_id": {
            "doc_id": relevance,
            ...
        },
        ...
    }
    """

    qrels_by_query_id = defaultdict(dict)

    for qrel in dataset.qrels_iter():
        query_id = str(
            qrel.query_id
        )

        doc_id = str(
            qrel.doc_id
        )

        relevance = float(
            qrel.relevance
        )

        qrels_by_query_id[query_id][doc_id] = (
            relevance
        )

    return dict(
        qrels_by_query_id
    )


# ==========================================================
# البحث عن Query محددة باستخدام Query ID
# ==========================================================

def find_query_by_id(
    dataset,
    target_query_id: str,
):
    """
    البحث عن Query واحدة داخل LoTTe باستخدام query_id.
    """

    target_query_id = str(
        target_query_id
    )

    for query in dataset.queries_iter():
        if str(query.query_id) == target_query_id:
            return query

    raise ValueError(
        f"Query ID was not found: {target_query_id}"
    )


# ==========================================================
# تقييم Query واحدة
# ==========================================================

def evaluate_single_query(
    search_service,
    query_id: str,
    query_text: str,
    relevance_by_doc_id: dict[str, float],
    k: int = 10,
    include_text: bool = False,
) -> dict:
    """
    تشغيل البحث على Query واحدة ثم حساب المقاييس.

    search_service:
        خدمة البحث باستخدام TF-IDF.

    query_id:
        رقم السؤال داخل LoTTe.

    query_text:
        السؤال الخام قبل التنظيف.

    relevance_by_doc_id:
        الوثائق الصحيحة ودرجات ملاءمتها من Qrels.

    k:
        عدد النتائج الأولى التي نريد تقييمها.
    """

    # --------------------------------------
    # تشغيل TF-IDF Search
    # --------------------------------------

    response = search_service.search(
        query=query_text,
        top_k=k,
        include_text=include_text,
    )

    # --------------------------------------
    # استخراج IDs الوثائق المسترجعة
    # --------------------------------------

    retrieved_doc_ids = [
        str(result["doc_id"])
        for result in response["results"]
    ]

    # --------------------------------------
    # استخراج IDs الوثائق الصحيحة من Qrels
    # --------------------------------------

    relevant_doc_ids = {
        str(doc_id)
        for doc_id, relevance
        in relevance_by_doc_id.items()
        if relevance > 0
    }

    # --------------------------------------
    # حساب المقاييس
    # --------------------------------------

    precision = precision_at_k(
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=relevant_doc_ids,
        k=k,
    )

    recall = recall_at_k(
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=relevant_doc_ids,
        k=k,
    )

    average_precision = average_precision_at_k(
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=relevant_doc_ids,
        k=k,
    )

    ndcg = ndcg_at_k(
        retrieved_doc_ids=retrieved_doc_ids,
        relevance_by_doc_id=relevance_by_doc_id,
        k=k,
    )

    # --------------------------------------
    # إعادة جميع النتائج بشكل منظم
    # --------------------------------------

    return {
        "query_id": str(query_id),
        "query_text": query_text,
        "processed_query": response.get("processed_query", "No processing required (Transformer Context)"),
        "retrieved_doc_ids": retrieved_doc_ids,
        "relevant_doc_ids": sorted(
            relevant_doc_ids
        ),
        "results": response["results"],
        "metrics": {
            f"precision@{k}": precision,
            f"recall@{k}": recall,
            f"ap@{k}": average_precision,
            f"ndcg@{k}": ndcg,
        },
    }


# ==========================================================
# تقييم جميع Queries
# ==========================================================

def evaluate_all_queries(
    dataset,
    search_service,
    k: int = 10,
    max_queries: int | None = None,
    print_every: int = 50,
) -> dict:
    """
    تقييم Search Service على جميع Queries الموجودة في LoTTe.

    dataset:
        Dataset القادمة من load_lotte().

    search_service:
        خدمة البحث، مثل TfidfSearchService.

    k:
        عدد النتائج الأولى المستخدمة في التقييم.
        مثلاً k=10 يعني Precision@10 و Recall@10 و AP@10 و nDCG@10.

    max_queries:
        للتجربة فقط.
        إذا كانت None نقيّم كل Queries.
        إذا كانت 10 نقيّم أول 10 Queries فقط.

    print_every:
        كل كم Query نطبع تقدم التنفيذ.
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    # --------------------------------------
    # بناء قاموس Qrels
    # --------------------------------------

    qrels_by_query_id = build_qrels_by_query_id(
        dataset
    )

    evaluated_queries_count = 0
    skipped_queries_count = 0

    precision_scores = []
    recall_scores = []
    average_precision_scores = []
    ndcg_scores = []

    per_query_results = []

    # --------------------------------------
    # المرور على Queries الحقيقية
    # --------------------------------------

    for query_index, query in enumerate(
        dataset.queries_iter(),
        start=1,
    ):
        if (
            max_queries is not None
            and evaluated_queries_count >= max_queries
        ):
            break

        query_id = str(
            query.query_id
        )

        relevance_by_doc_id = qrels_by_query_id.get(
            query_id,
            {},
        )

        # إذا ما في Qrels لهذا السؤال، لا نستطيع تقييمه.
        if not relevance_by_doc_id:
            skipped_queries_count += 1
            continue

        # ----------------------------------
        # تقييم Query واحدة
        # ----------------------------------

        evaluation = evaluate_single_query(
            search_service=search_service,
            query_id=query_id,
            query_text=query.text,
            relevance_by_doc_id=relevance_by_doc_id,
            k=k,
            include_text=False,
        )

        metrics = evaluation["metrics"]

        precision = metrics[f"precision@{k}"]
        recall = metrics[f"recall@{k}"]
        average_precision = metrics[f"ap@{k}"]
        ndcg = metrics[f"ndcg@{k}"]

        precision_scores.append(
            precision
        )

        recall_scores.append(
            recall
        )

        average_precision_scores.append(
            average_precision
        )

        ndcg_scores.append(
            ndcg
        )

        per_query_results.append(
            {
                "query_id": query_id,
                "query_text": query.text,
                "processed_query": evaluation["processed_query"],
                "retrieved_doc_ids": evaluation["retrieved_doc_ids"],
                "relevant_doc_ids": evaluation["relevant_doc_ids"],
                "metrics": metrics,
            }
        )

        evaluated_queries_count += 1

        if evaluated_queries_count % print_every == 0:
            print(
                f"Evaluated queries: "
                f"{evaluated_queries_count:,}"
            )

    # --------------------------------------
    # دالة صغيرة لحساب المتوسط
    # --------------------------------------

    def mean(scores: list[float]) -> float:
        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    # --------------------------------------
    # التقرير النهائي
    # --------------------------------------

    report = {
        "k": k,
        "evaluated_queries": evaluated_queries_count,
        "skipped_queries": skipped_queries_count,
        "metrics": {
            f"mean_precision@{k}": mean(
                precision_scores
            ),
            f"mean_recall@{k}": mean(
                recall_scores
            ),
            f"map@{k}": mean(
                average_precision_scores
            ),
            f"mean_ndcg@{k}": mean(
                ndcg_scores
            ),
        },
        "per_query_results": per_query_results,
    }

    return report
