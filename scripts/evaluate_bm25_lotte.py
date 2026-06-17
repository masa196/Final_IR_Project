import json
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

from services.bm25.bm25_search_service import (
    BM25SearchService,
)

from services.evaluation.evaluation_service import (
    evaluate_all_queries,
)


# ==========================================================
# الإعدادات
# ==========================================================

TOP_K = 10


# بعد نجاح التجربة نضع None لتقييم كل Queries.
MAX_QUERIES = None

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

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "evaluations"
    / "lotte"
    / "bm25"
)

OUTPUT_FILE = (
    OUTPUT_DIRECTORY
    / "evaluation_report.json"
)


# ==========================================================
# تشغيل التقييم
# ==========================================================

def main():
    print("======================================")
    print("EVALUATING LOTTE BM25")
    print("======================================")

    dataset = load_lotte()

    print(f"Documents: {dataset.docs_count():,}")
    print(f"Queries:   {dataset.queries_count():,}")
    print(f"Qrels:     {dataset.qrels_count():,}")

    search_service = BM25SearchService(
        bm25_directory=BM25_DIRECTORY,
        inverted_index_directory=INVERTED_INDEX_DIRECTORY,
        k1=1.5,
        b=0.90,
    )

    report = evaluate_all_queries(
        dataset=dataset,
        search_service=search_service,
        k=TOP_K,
        max_queries=MAX_QUERIES,
        print_every=10,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n======================================")
    print("✅ BM25 EVALUATION FINISHED")
    print("======================================")

    print(
        f"Evaluated queries: "
        f"{report['evaluated_queries']:,}"
    )

    print(
        f"Skipped queries:   "
        f"{report['skipped_queries']:,}"
    )

    print("\nMetrics:")

    for metric_name, metric_value in (
        report["metrics"].items()
    ):
        print(
            f"{metric_name}: "
            f"{metric_value:.4f}"
        )

    print("\nReport saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()