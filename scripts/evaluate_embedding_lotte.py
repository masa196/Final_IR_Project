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

# استيراد خدمة البحث الدلالي للـ Embedding بدلاً من BM25
from services.embedding.embedding_search_service import (
    EmbeddingSearchService,
)

from services.evaluation.evaluation_service import (
    evaluate_all_queries,
)


# ==========================================================
# الإعدادات
# ==========================================================

TOP_K = 10

# نضع None لتقييم كل Queries بشكل كامل
MAX_QUERIES = None

# مسار مجلد الفهرس الخاص بالـ Embedding الحالي
EMBEDDING_DIRECTORY = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "embedding"
)

# مسار مجلد المخرجات المخصص لحفظ تقرير الـ Embedding لعدم خلطه مع BM25
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "evaluations"
    / "lotte"
    / "embedding"
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
    print("EVALUATING LOTTE EMBEDDING")
    print("======================================")

    # 1. تحميل الـ Dataset (نفس البيانات تماماً لضمان عدالة المقارنة)
    dataset = load_lotte()

    print(f"Documents: {dataset.docs_count():,}")
    print(f"Queries:   {dataset.queries_count():,}")
    print(f"Qrels:     {dataset.qrels_count():,}")

    # 2. بناء خدمة بحث الـ Embedding وشحنها في الـ RAM
    search_service = EmbeddingSearchService(
        embedding_directory=EMBEDDING_DIRECTORY,
    )

    # 3. استدعاء دالة التقييم الشاملة الموحدة (الموجودة في نظام التقييم لديكِ)
    # نمرر الخيار الموفر للوقت تلقائياً داخل الدالة لعدم شحن النصوص من SQLite أثناء حساب المقاييس
    report = evaluate_all_queries(
        dataset=dataset,
        search_service=search_service,
        k=TOP_K,
        max_queries=MAX_QUERIES,
        print_every=10,  # يطبع التقدم كل 10 أسئلة
    )

    # 4. إنشاء مجلد الحفظ إن لم يكن موجوداً
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 5. حفظ التقرير الشامل بصيغة JSON
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
    print("✅ EMBEDDING EVALUATION FINISHED")
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

    # 6. طباعة مقاييس الأداء النهائية المقاسة بدقة (MAP, nDCG, Recall, Precision)
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