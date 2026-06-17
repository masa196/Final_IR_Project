import json
from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

from services.preprocessing.preprocessing_service import (
    preprocess_lotte,
)

from services.data_base.database_service import (
    get_raw_documents_by_ids,
)


# ==========================================================
# خدمة البحث باستخدام TF-IDF
# ==========================================================

class TfidfSearchService:
    """
    تحميل ملفات TF-IDF والبحث داخلها.

    نحمل الملفات مرة واحدة فقط عند إنشاء الخدمة،
    ثم نستطيع تنفيذ أكثر من Query بدون إعادة التحميل.
    """

    def __init__(
        self,
        tfidf_directory: Path,
    ):
        vectorizer_file = (
            tfidf_directory
            / "vectorizer.joblib"
        )

        matrix_file = (
            tfidf_directory
            / "tfidf_matrix.npz"
        )

        document_ids_file = (
            tfidf_directory
            / "document_ids.json"
        )

        # التأكد من وجود الملفات قبل تحميلها
        if not vectorizer_file.exists():
            raise FileNotFoundError(
                f"Vectorizer file was not found: "
                f"{vectorizer_file}"
            )

        if not matrix_file.exists():
            raise FileNotFoundError(
                f"TF-IDF matrix file was not found: "
                f"{matrix_file}"
            )

        if not document_ids_file.exists():
            raise FileNotFoundError(
                f"Document IDs file was not found: "
                f"{document_ids_file}"
            )

        print("Loading TF-IDF search files...")

        self.vectorizer = joblib.load(
            vectorizer_file
        )

        self.tfidf_matrix = load_npz(
            matrix_file
        )

        with document_ids_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.document_ids = json.load(file)

        print("✅ TF-IDF search files loaded")

    def search(
        self,
        query: str,
        top_k: int = 10,
        include_text: bool = True,
    ):
        """
        البحث باستخدام TF-IDF.

        include_text=True:
            يرجع النصوص الأصلية من SQLite.

        include_text=False:
            يرجع doc_ids و scores فقط، وهذا أسرع للتقييم.
        """

        query_tokens = preprocess_lotte(
            query
        )

        if not query_tokens:
            return {
                "query": query,
                "processed_query": query_tokens,
                "result_ids": [],
                "results": [],
            }

        query_as_text = " ".join(
            query_tokens
        )

        query_vector = self.vectorizer.transform(
            [
                query_as_text
            ]
        )

        scores = cosine_similarity(
            query_vector,
            self.tfidf_matrix,
        ).flatten()

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        ranked_results = []

        for matrix_index in top_indices:
            score = float(
                scores[matrix_index]
            )

            if score <= 0:
                continue

            doc_id = str(
                self.document_ids[matrix_index]
            )

            ranked_results.append(
                {
                    "doc_id": doc_id,
                    "score": score,
                }
            )

        result_ids = [
            result["doc_id"]
            for result in ranked_results
        ]

        if include_text and result_ids:
            raw_documents = get_raw_documents_by_ids(
                result_ids
            )

            raw_text_by_id = {
                str(document["doc_id"]): document["text"]
                for document in raw_documents
            }

            for result in ranked_results:
                result["text"] = raw_text_by_id.get(
                    str(result["doc_id"]),
                    "Original text was not found in SQLite.",
                )

        return {
            "query": query,
            "processed_query": query_tokens,
            "result_ids": result_ids,
            "results": ranked_results,
        }