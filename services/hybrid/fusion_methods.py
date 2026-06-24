from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[dict]],
    k: int = 60,
    top_k: int = 10,
) -> list[dict]:
    fusion_scores = defaultdict(float)

    for model_name, results in ranked_lists.items():
        for rank, item in enumerate(results, start=1):
            doc_id = item["doc_id"]
            fusion_scores[doc_id] += 1.0 / (k + rank)

    sorted_docs = sorted(
        fusion_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    return [
        {"doc_id": doc_id, "score": float(score)}
        for doc_id, score in sorted_docs
    ]


FUSION_METHODS = {
    "rrf": reciprocal_rank_fusion,
}
