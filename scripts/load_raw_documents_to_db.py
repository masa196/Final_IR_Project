import json
from os import name
import sys
from pathlib import Path


# ==========================================================
# الوصول إلى مجلد المشروع
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from services.data_base.database_service import (
    DATABASE_PATH,
    create_documents_table,
    open_database,
    save_raw_documents,
)


# ==========================================================
# الإعدادات
# ==========================================================

DOCUMENTS_FILE = (
    PROJECT_ROOT
    / "processed_data"
    / "lotte"
    / "documents.jsonl"
)

# عدد الوثائق التي يتم حفظها دفعة واحدة
BATCH_SIZE = 1000

# عند التشغيل يتم تنظيف الجدول القديم
# ثم إدخال البيانات من جديد
CLEAR_EXISTING_DOCUMENTS = True


# ==========================================================
# تشغيل النقل
# ==========================================================

def main():
    if not DOCUMENTS_FILE.exists():
        raise FileNotFoundError(
            "documents.jsonl was not found. "
            "Run prepare_lotte_data.py first."
        )

    connection = open_database()

    try:
        create_documents_table(connection)

        if CLEAR_EXISTING_DOCUMENTS:
            connection.execute(
                "DELETE FROM documents"
            )
            connection.commit()

        batch = []
        saved_count = 0

        with DOCUMENTS_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):
                record = json.loads(line)

                batch.append(
                    (
                        str(record["doc_id"]),
                        record["text"],
                    )
                )

                if len(batch) >= BATCH_SIZE:
                    save_raw_documents(
                        connection,
                        batch,
                    )

                    connection.commit()

                    saved_count += len(batch)
                    batch.clear()

                    print(
                        f"Saved documents: "
                        f"{saved_count:,}"
                    )

        # حفظ آخر دفعة إذا كان عددها أقل من 1000
        if batch:
            save_raw_documents(
                connection,
                batch,
            )

            connection.commit()

            saved_count += len(batch)

        database_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM documents
            """
        ).fetchone()[0]

    finally:
        connection.close()

    print("\n======================================")
    print("✅ RAW DOCUMENTS SAVED TO SQLITE")
    print("======================================")

    print(f"Saved documents: {saved_count:,}")
    print(f"Rows in database: {database_count:,}")
    print(f"Database file: {DATABASE_PATH}")


if __name__ == "__main__":
    main()