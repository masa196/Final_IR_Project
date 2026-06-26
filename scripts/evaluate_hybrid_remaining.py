import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from services.hybrid.hybrid_search_service import HybridSearchService
from services.evaluation.evaluation_metrics_service import (
    precision_at_k, recall_at_k, average_precision_at_k, ndcg_at_k,
)

TOP_K = 10
MAX_QUERIES = None
EVALUATIONS_DIR = PROJECT_ROOT / "evaluations" / "lotte"

config = {
    "name": "hybrid_serial_bm25_to_embedding",
    "mode": "serial",
    "first_stage": "bm25",
    "second_stage": "embedding",
    "first_stage_k": 200,
}

def mean(scores):
    return sum(scores) / len(scores) if scores else 0.0

def load_queries():
    queries = {}
    path = PROJECT_ROOT / "processed_data" / "lotte" / "queries.jsonl"
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            queries[entry["query_id"]] = entry["text"]
    return queries

def load_qrels():
    qrels = {}
    report_path = EVALUATIONS_DIR / "embedding" / "evaluation_report.json"
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    for entry in report["per_query_results"]:
        qrels[entry["query_id"]] = {str(did): 1.0 for did in entry["relevant_doc_ids"]}
    return qrels

def main():
    queries = load_queries()
    qrels = load_qrels()
    print(f"Queries: {len(queries)}, Qrels: {len(qrels)}")

    print("Loading HybridSearchService...")
    hybrid = HybridSearchService()

    print(f"\nEvaluating: {config['name']}")
    precisions, recalls, aps, ndcgs = [], [], [], []
    per_query = []
    evaluated = 0

    for qid in sorted(queries.keys()):
        if MAX_QUERIES is not None and evaluated >= MAX_QUERIES:
            break
        qid_s = str(qid)
        rel = qrels.get(qid_s, {})
        if not rel:
            continue

        response = hybrid.search_serial(
            query=queries[qid],
            first_stage=config["first_stage"],
            second_stage=config["second_stage"],
            first_stage_k=config.get("first_stage_k", 200),
            top_k=TOP_K,
            include_text=False,
            k1=1.5,
            b=0.75,
        )
        retrieved = [r["doc_id"] for r in response["results"]]
        relevant = {str(d) for d, r in rel.items() if r > 0}

        p = precision_at_k(retrieved, relevant, TOP_K)
        r = recall_at_k(retrieved, relevant, TOP_K)
        ap = average_precision_at_k(retrieved, relevant, TOP_K)
        ndcg = ndcg_at_k(retrieved, rel, TOP_K)

        precisions.append(p); recalls.append(r); aps.append(ap); ndcgs.append(ndcg)
        per_query.append({
            "query_id": qid_s, "query_text": queries[qid],
            "retrieved_doc_ids": retrieved,
            "relevant_doc_ids": sorted(relevant),
            "metrics": {f"precision@{TOP_K}": p, f"recall@{TOP_K}": r, f"ap@{TOP_K}": ap, f"ndcg@{TOP_K}": ndcg},
        })
        evaluated += 1
        if evaluated % 10 == 0:
            print(f"  Evaluated: {evaluated:,}")

    report = {
        "config": config, "k": TOP_K,
        "evaluated_queries": evaluated,
        "skipped_queries": len(queries) - evaluated - (0 if MAX_QUERIES is None else max(0, len(queries) - MAX_QUERIES)),
        "metrics": {
            f"mean_precision@{TOP_K}": mean(precisions),
            f"mean_recall@{TOP_K}": mean(recalls),
            f"map@{TOP_K}": mean(aps),
            f"mean_ndcg@{TOP_K}": mean(ndcgs),
        },
        "per_query_results": per_query,
    }

    config_dir = EVALUATIONS_DIR / config["name"]
    config_dir.mkdir(parents=True, exist_ok=True)
    output_file = config_dir / "evaluation_report.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"\nSaved: {output_file}")
    for metric, value in report["metrics"].items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
