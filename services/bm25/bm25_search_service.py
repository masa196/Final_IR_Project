import json
import heapq
from collections import defaultdict
from pathlib import Path

import joblib

from services.preprocessing.preprocessing_service import (
    preprocess_lotte,
)

from services.data_base.database_service import (
    get_raw_documents_by_ids,
)


class BM25SearchService:
    """
    خدمة البحث باستخدام BM25.

    تعتمد على:
    - inverted_index.joblib من inverted_index directory
    - idf_by_term.joblib من bm25 directory
    - document_lengths.joblib من bm25 directory
    - report.json من bm25 directory
    """

    def __init__(
        self,
        bm25_directory: Path,
        inverted_index_directory: Path,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.bm25_directory = Path(
            bm25_directory
        )

        self.inverted_index_directory = Path(
            inverted_index_directory
        )

        self.k1 = k1
        self.b = b

        print("Loading BM25 search files...")

        self.inverted_index = joblib.load(
            self.inverted_index_directory
            / "inverted_index.joblib"
        )

        self.idf_by_term = joblib.load(
            self.bm25_directory
            / "idf_by_term.joblib"
        )

        self.document_lengths = joblib.load(
            self.bm25_directory
            / "document_lengths.joblib"
        )

        with (
            self.bm25_directory
            / "document_ids.json"
        ).open("r", encoding="utf-8") as file:
            self.document_ids = [
                str(doc_id)
                for doc_id in json.load(file)
            ]

        with (
            self.bm25_directory
            / "report.json"
        ).open("r", encoding="utf-8") as file:
            self.report = json.load(file)

        self.number_of_documents = int(
            self.report["number_of_documents"]
        )

        self.average_document_length = float(
            self.report["average_document_length"]
        )

        print("✅ BM25 search files loaded")
        print(
            f"Documents: "
            f"{self.number_of_documents:,}"
        )
        print(
            f"Average document length: "
            f"{self.average_document_length:.2f}"
        )

    # ======================================================
    # حساب مساهمة كلمة واحدة داخل وثيقة واحدة
    # ======================================================

    def _calculate_term_score(
        self,
        term_frequency: int,
        document_length: int,
        idf: float,
    ) -> float:
        """
        حساب BM25 score لكلمة واحدة داخل وثيقة واحدة.
        """

        numerator = (
            term_frequency
            * (
                self.k1
                + 1
            )
        )

        denominator = (
            term_frequency
            + self.k1
            * (
                1
                - self.b
                + self.b
                * (
                    document_length
                    / self.average_document_length
                )
            )
        )

        if denominator == 0:
            return 0.0

        return idf * (
            numerator / denominator
        )

    # ======================================================
    # البحث
    # ======================================================

    def search(
        self,
        query: str,
        top_k: int = 10,
        include_text: bool = True,
    ) -> dict:
        """
        البحث باستخدام BM25.

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

        scores_by_doc_id = defaultdict(float)

        # حالياً نحسب كل term مرة واحدة.
        # هذا مناسب لأن Queries قصيرة غالباً.
        unique_query_terms = set(
            query_tokens
        )

        for term in unique_query_terms:
            posting_list = self.inverted_index.get(
                term,
                [],
            )

            if not posting_list:
                continue

            idf = self.idf_by_term.get(
                term,
                0.0,
            )

            if idf <= 0:
                continue

            for doc_id, term_frequency in posting_list:
                doc_id = str(
                    doc_id
                )

                document_length = self.document_lengths.get(
                    doc_id,
                    0,
                )

                if document_length <= 0:
                    continue

                term_score = self._calculate_term_score(
                    term_frequency=int(term_frequency),
                    document_length=int(document_length),
                    idf=float(idf),
                )

                scores_by_doc_id[doc_id] += term_score

        ranked_items = heapq.nlargest(
            top_k,
            scores_by_doc_id.items(),
            key=lambda item: item[1],
        )

        ranked_results = [
            {
                "doc_id": doc_id,
                "score": float(score),
            }
            for doc_id, score in ranked_items
            if score > 0
        ]

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