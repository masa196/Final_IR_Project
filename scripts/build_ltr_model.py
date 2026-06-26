import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from services.ltr.ltr_service import build_ltr_model


QUERIES_FILE = (
    PROJECT_ROOT / "processed_data" / "lotte" / "queries.jsonl"
)

QRELS_EVAL_REPORT = (
    PROJECT_ROOT / "evaluations" / "lotte" / "bm25"
    / "evaluation_report.json"
)

INVERTED_INDEX_DIRECTORY = (
    PROJECT_ROOT / "indices" / "lotte" / "inverted_index"
)

BM25_DIRECTORY = (
    PROJECT_ROOT / "indices" / "lotte" / "bm25"
)

EMBEDDING_DIRECTORY = (
    PROJECT_ROOT / "indices" / "lotte" / "embedding"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "indices" / "lotte" / "ltr"
)

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


def main():
    print("======================================")
    print("BUILDING LTR MODEL")
    print("======================================")

    print(f"Loading queries from: {QUERIES_FILE}")
    queries = load_queries(QUERIES_FILE)
    print(f"Loaded {len(queries):,} queries")

    print(f"Loading QRELs from: {QRELS_EVAL_REPORT}")
    qrels = load_qrels_from_eval_report(QRELS_EVAL_REPORT)
    print(f"Loaded QRELs for {len(qrels):,} queries")

    report = build_ltr_model(
        queries=queries,
        qrels_by_query_id=qrels,
        inverted_index_directory=INVERTED_INDEX_DIRECTORY,
        bm25_directory=BM25_DIRECTORY,
        embedding_directory=EMBEDDING_DIRECTORY,
        output_directory=OUTPUT_DIRECTORY,
        max_queries=MAX_QUERIES,
    )

    print("\n======================================")
    print("✅ LTR MODEL BUILD FINISHED")
    print("======================================")
    print(
        f"Training samples: "
        f"{report['training_data']['total_samples']:,}"
    )
    print(
        f"Queries used: "
        f"{report['queries']['queries_used']:,}"
    )
    print(
        f"Training R² score: "
        f"{report['training_score']}"
    )


if __name__ == "__main__":
    main()
