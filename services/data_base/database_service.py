import sqlite3
from pathlib import Path


# ==========================================================
# مسار قاعدة البيانات
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_PATH = (
    PROJECT_ROOT
    / "database"
    / "lotte.db"
)


# ==========================================================
# فتح الاتصال بقاعدة البيانات
# ==========================================================

def open_database():
    """
    فتح اتصال مع SQLite الموجودة في جذر المشروع.
    """

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Database file was not found: {DATABASE_PATH}"
        )

    return sqlite3.connect(DATABASE_PATH)


# ==========================================================
# إنشاء الجدول
# ==========================================================

def create_documents_table(connection):
    """
    إنشاء جدول الوثائق إذا لم يكن موجوداً مسبقاً.
    """

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            raw_text TEXT NOT NULL
        )
        """
    )

    connection.commit()


# ==========================================================
# حفظ مجموعة وثائق
# ==========================================================

def save_raw_documents(connection, documents):
    """
    حفظ النصوص الأصلية داخل SQLite.
    """

    connection.executemany(
        """
        INSERT OR REPLACE INTO documents (
            doc_id,
            raw_text
        )
        VALUES (?, ?)
        """,
        documents,
    )

    connection.commit()


# ==========================================================
# جلب النصوص الأصلية حسب IDs
# ==========================================================

def get_raw_documents_by_ids(doc_ids):
    """
    جلب النصوص الأصلية من قاعدة البيانات
    مع الحفاظ على ترتيب IDs المطلوب.
    """

    if not doc_ids:
        return []

    doc_ids = [
        str(doc_id)
        for doc_id in doc_ids
    ]

    placeholders = ", ".join(
        "?"
        for _ in doc_ids
    )

    connection = open_database()

    try:
        rows = connection.execute(
            f"""
            SELECT doc_id, raw_text
            FROM documents
            WHERE doc_id IN ({placeholders})
            """,
            doc_ids,
        ).fetchall()

    finally:
        connection.close()

    text_by_id = {
        str(doc_id): raw_text
        for doc_id, raw_text in rows
    }

    return [
        {
            "doc_id": doc_id,
            "text": text_by_id[doc_id],
        }
        for doc_id in doc_ids
        if doc_id in text_by_id
    ]