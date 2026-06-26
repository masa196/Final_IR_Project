import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from services.dataset.dataset_loader import load_lotte
from services.hybrid.hybrid_search_service import HybridSearchService
from services.evaluation.evaluation_service import (
    evaluate_single_query,
    build_qrels_by_query_id,
)
from services.evaluation.evaluation_metrics_service import (
    precision_at_k,
    recall_at_k,
    average_precision_at_k,
    ndcg_at_k,
)


TOP_K = 10
MAX_QUERIES = None

EVALUATIONS_DIR = PROJECT_ROOT / "evaluations" / "lotte"


HYBRID_CONFIGS = [
    {
        "name": "hybrid_parallel_bm25_embedding_rrf",
        "mode": "parallel",
        "models": ["bm25", "embedding"],
        "fusion_method": "rrf",
    },
    {
        "name": "hybrid_serial_bm25_to_embedding",
        "mode": "serial",
        "first_stage": "bm25",
        "second_stage": "embedding",
        "first_stage_k": 200,
    },
]


def mean(scores):
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_parallel(
    hybrid_service, query_text, relevance_by_doc_id,
    config, k,
):
    response = hybrid_service.search_parallel(
        query=query_text,
        models=config["models"],
        fusion_method=config.get("fusion_method", "rrf"),
        top_k=k,
        include_text=False,
        k1=1.5,
        b=0.75,
    )
    retrieved_doc_ids = [r["doc_id"] for r in response["results"]]
    relevant_doc_ids = {
        str(did) for did, rel in relevance_by_doc_id.items() if rel > 0
    }
    return {
        "retrieved_doc_ids": retrieved_doc_ids,
        "relevant_doc_ids": sorted(relevant_doc_ids),
    }


def evaluate_serial(
    hybrid_service, query_text, relevance_by_doc_id,
    config, k,
):
    response = hybrid_service.search_serial(
        query=query_text,
        first_stage=config["first_stage"],
        second_stage=config["second_stage"],
        first_stage_k=config.get("first_stage_k", 200),
        top_k=k,
        include_text=False,
        k1=1.5,
        b=0.75,
    )
    retrieved_doc_ids = [r["doc_id"] for r in response["results"]]
    relevant_doc_ids = {
        str(did) for did, rel in relevance_by_doc_id.items() if rel > 0
    }
    return {
        "retrieved_doc_ids": retrieved_doc_ids,
        "relevant_doc_ids": sorted(relevant_doc_ids),
    }


def evaluate_config(hybrid_service, dataset, config, k=10, max_queries=None):
    print(f"\n{'='*60}")
    print(f"Evaluating: {config['name']}")
    print(f"{'='*60}")

    qrels_by_query_id = build_qrels_by_query_id(dataset)

    precision_scores = []
    recall_scores = []
    ap_scores = []
    ndcg_scores = []
    evaluated_count = 0
    skipped_count = 0
    per_query = []

    for query_index, query in enumerate(dataset.queries_iter(), 1):
        if max_queries is not None and evaluated_count >= max_queries:
            break

        query_id = str(query.query_id)
        relevance_by_doc_id = qrels_by_query_id.get(query_id, {})

        if not relevance_by_doc_id:
            skipped_count += 1
            continue

        if config["mode"] == "parallel":
            result = evaluate_parallel(
                hybrid_service, query.text, relevance_by_doc_id, config, k,
            )
        else:
            result = evaluate_serial(
                hybrid_service, query.text, relevance_by_doc_id, config, k,
            )

        retrieved = result["retrieved_doc_ids"]
        relevant = set(result["relevant_doc_ids"])

        precision_scores.append(precision_at_k(retrieved, relevant, k))
        recall_scores.append(recall_at_k(retrieved, relevant, k))
        ap_scores.append(average_precision_at_k(retrieved, relevant, k))
        ndcg_scores.append(ndcg_at_k(retrieved, relevance_by_doc_id, k))

        per_query.append({
            "query_id": query_id,
            "query_text": query.text,
            "retrieved_doc_ids": retrieved,
            "relevant_doc_ids": sorted(relevant),
            "metrics": {
                f"precision@{k}": precision_scores[-1],
                f"recall@{k}": recall_scores[-1],
                f"ap@{k}": ap_scores[-1],
                f"ndcg@{k}": ndcg_scores[-1],
            },
        })

        evaluated_count += 1
        if evaluated_count % 10 == 0:
            print(f"  Evaluated: {evaluated_count:,}")

    report = {
        "config": config,
        "k": k,
        "evaluated_queries": evaluated_count,
        "skipped_queries": skipped_count,
        "metrics": {
            f"mean_precision@{k}": mean(precision_scores),
            f"mean_recall@{k}": mean(recall_scores),
            f"map@{k}": mean(ap_scores),
            f"mean_ndcg@{k}": mean(ndcg_scores),
        },
        "per_query_results": per_query,
    }

    print(f"\nResults for {config['name']}:")
    for metric, value in report["metrics"].items():
        print(f"  {metric}: {value:.4f}")

    return report


def main():
    print("Loading LOTTE dataset...")
    dataset = load_lotte()
    print(f"Documents: {dataset.docs_count():,}")
    print(f"Queries:   {dataset.queries_count():,}")
    print(f"Qrels:     {dataset.qrels_count():,}")

    print("Loading HybridSearchService (this may take a few minutes)...")
    hybrid_service = HybridSearchService()

    all_reports = {}

    for config in HYBRID_CONFIGS:
        report = evaluate_config(
            hybrid_service, dataset, config,
            k=TOP_K, max_queries=MAX_QUERIES,
        )
        all_reports[config["name"]] = report

        config_dir = EVALUATIONS_DIR / config["name"]
        config_dir.mkdir(parents=True, exist_ok=True)
        output_file = config_dir / "evaluation_report.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        print(f"  Report saved: {output_file}")

    print("\n")
    print("=" * 60)
    print("HYBRID EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Config':<45} {'P@10':<8} {'R@10':<8} {'MAP@10':<8} {'nDCG@10':<8}")
    print("-" * 77)
    for name, report in all_reports.items():
        m = report["metrics"]
        print(
            f"{name:<45} "
            f"{m[f'mean_precision@{TOP_K}']:<8.4f} "
            f"{m[f'mean_recall@{TOP_K}']:<8.4f} "
            f"{m[f'map@{TOP_K}']:<8.4f} "
            f"{m[f'mean_ndcg@{TOP_K}']:<8.4f}"
        )

    print("\n✅ All hybrid evaluations completed!")


if __name__ == "__main__":
    main()
