import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from services.ltr.ltr_search_service import LTRSearchService
from services.evaluation.evaluation_metrics_service import (
    precision_at_k,
    recall_at_k,
    average_precision_at_k,
    ndcg_at_k,
)


QUERIES_FILE = (
    PROJECT_ROOT / "processed_data" / "lotte" / "queries.jsonl"
)

QRELS_EVAL_REPORT = (
    PROJECT_ROOT / "evaluations" / "lotte" / "bm25"
    / "evaluation_report.json"
)

LTR_DIRECTORY = (
    PROJECT_ROOT / "indices" / "lotte" / "ltr"
)

BM25_DIRECTORY = (
    PROJECT_ROOT / "indices" / "lotte" / "bm25"
)

INVERTED_INDEX_DIRECTORY = (
    PROJECT_ROOT / "indices" / "lotte" / "inverted_index"
)

EMBEDDING_DIRECTORY = (
    PROJECT_ROOT / "indices" / "lotte" / "embedding"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "evaluations" / "lotte" / "ltr"
)

OUTPUT_FILE = OUTPUT_DIRECTORY / "evaluation_report.json"

TOP_K = 10
MAX_QUERIES = None


def load_queries(queries_file: Path) -> list[tuple[str, str]]:
    queries = []
    with queries_file.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            queries.append((record["query_id"], record["text"]))
    return queries


def load_qrels_from_eval_report(
    eval_report_file: Path,
) -> dict[str, dict[str, float]]:
    with eval_report_file.open("r", encoding="utf-8") as f:
        report = json.load(f)

    qrels = {}
    for entry in report.get("per_query_results", []):
        query_id = entry["query_id"]
        qrels[query_id] = {
            doc_id: 1.0
            for doc_id in entry.get("relevant_doc_ids", [])
        }
    return qrels


def evaluate_query(
    search_service: LTRSearchService,
    query_text: str,
    relevance_by_doc_id: dict[str, float],
    k: int = 10,
) -> dict:
    response = search_service.search(
        query=query_text,
        top_k=k,
        include_text=False,
    )

    retrieved_doc_ids = [
        str(result["doc_id"])
        for result in response.get("results", [])
    ]

    relevant_doc_ids = {
        str(doc_id)
        for doc_id, relevance in relevance_by_doc_id.items()
        if relevance > 0
    }

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

    return {
        "retrieved_doc_ids": retrieved_doc_ids,
        "relevant_doc_ids": sorted(relevant_doc_ids),
        "results": response.get("results", []),
        "metrics": {
            f"precision@{k}": precision,
            f"recall@{k}": recall,
            f"ap@{k}": average_precision,
            f"ndcg@{k}": ndcg,
        },
    }


def main():
    print("======================================")
    print("EVALUATING LOTTE LTR")
    print("======================================")

    print(f"Loading queries from: {QUERIES_FILE}")
    queries = load_queries(QUERIES_FILE)
    print(f"Loaded {len(queries):,} queries")

    print(f"Loading QRELs from: {QRELS_EVAL_REPORT}")
    qrels = load_qrels_from_eval_report(QRELS_EVAL_REPORT)
    print(f"Loaded QRELs for {len(qrels):,} queries")

    print("\nInitializing LTR search service...")
    search_service = LTRSearchService(
        ltr_directory=LTR_DIRECTORY,
        inverted_index_directory=INVERTED_INDEX_DIRECTORY,
        bm25_directory=BM25_DIRECTORY,
        embedding_directory=EMBEDDING_DIRECTORY,
    )

    evaluated_count = 0
    skipped_count = 0
    precision_scores = []
    recall_scores = []
    average_precision_scores = []
    ndcg_scores = []
    per_query_results = []

    for query_id, query_text in queries:
        if MAX_QUERIES is not None and evaluated_count >= MAX_QUERIES:
            break

        relevance_by_doc_id = qrels.get(query_id)
        if not relevance_by_doc_id:
            skipped_count += 1
            continue

        result = evaluate_query(
            search_service=search_service,
            query_text=query_text,
            relevance_by_doc_id=relevance_by_doc_id,
            k=TOP_K,
        )

        metrics = result["metrics"]
        precision_scores.append(metrics[f"precision@{TOP_K}"])
        recall_scores.append(metrics[f"recall@{TOP_K}"])
        average_precision_scores.append(metrics[f"ap@{TOP_K}"])
        ndcg_scores.append(metrics[f"ndcg@{TOP_K}"])

        per_query_results.append({
            "query_id": query_id,
            "query_text": query_text,
            "retrieved_doc_ids": result["retrieved_doc_ids"],
            "relevant_doc_ids": result["relevant_doc_ids"],
            "metrics": metrics,
        })

        evaluated_count += 1

        if evaluated_count % 10 == 0:
            print(f"Evaluated queries: {evaluated_count:,}")

    def mean(scores):
        return sum(scores) / len(scores) if scores else 0.0

    report = {
        "k": TOP_K,
        "evaluated_queries": evaluated_count,
        "skipped_queries": skipped_count,
        "metrics": {
            f"mean_precision@{TOP_K}": mean(precision_scores),
            f"mean_recall@{TOP_K}": mean(recall_scores),
            f"map@{TOP_K}": mean(average_precision_scores),
            f"mean_ndcg@{TOP_K}": mean(ndcg_scores),
        },
        "per_query_results": per_query_results,
    }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n======================================")
    print("✅ LTR EVALUATION FINISHED")
    print("======================================")
    print(f"Evaluated queries: {evaluated_count:,}")
    print(f"Skipped queries:   {skipped_count:,}")

    print("\nMetrics:")
    for metric_name, metric_value in report["metrics"].items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nReport saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
