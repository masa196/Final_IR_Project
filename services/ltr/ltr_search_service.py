import json
import heapq
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from services.ltr.ltr_config import (
    FEATURE_NAMES,
    CANDIDATE_K,
)
from services.ltr.feature_extraction_service import (
    LTRFeatureExtractor,
)
from services.preprocessing.preprocessing_service import (
    preprocess_lotte,
)
from services.data_base.database_service import (
    get_raw_documents_by_ids,
)


class LTRSearchService:
    def __init__(
        self,
        ltr_directory: Path,
        inverted_index_directory: Path,
        bm25_directory: Path,
        embedding_directory: Path,
    ):
        self.ltr_directory = Path(ltr_directory)
        self.inverted_index_directory = Path(
            inverted_index_directory
        )
        self.bm25_directory = Path(bm25_directory)
        self.embedding_directory = Path(embedding_directory)

        self.k1 = 1.5
        self.b = 0.75

        print("Loading LTR search service...")

        self.model = joblib.load(
            self.ltr_directory / "ltr_model.joblib"
        )

        with (
            self.ltr_directory / "feature_names.json"
        ).open("r", encoding="utf-8") as f:
            self.feature_names = json.load(f)

        print("Loading BM25 data for candidate generation...")

        self.inverted_index = joblib.load(
            self.inverted_index_directory / "inverted_index.joblib"
        )
        self.idf_by_term = joblib.load(
            self.bm25_directory / "idf_by_term.joblib"
        )
        self.document_lengths = joblib.load(
            self.bm25_directory / "document_lengths.joblib"
        )
        self.document_lengths = {
            str(doc_id): int(length)
            for doc_id, length in self.document_lengths.items()
        }

        with (
            self.bm25_directory / "document_ids.json"
        ).open("r", encoding="utf-8") as f:
            self.document_ids = [
                str(doc_id) for doc_id in json.load(f)
            ]

        with (
            self.bm25_directory / "report.json"
        ).open("r", encoding="utf-8") as f:
            report = json.load(f)

        self.number_of_documents = int(
            report["number_of_documents"]
        )
        self.average_document_length = float(
            report["average_document_length"]
        )

        print("Initializing feature extractor...")
        self.feature_extractor = LTRFeatureExtractor(
            inverted_index=self.inverted_index,
            idf_by_term=self.idf_by_term,
            document_lengths=self.document_lengths,
            avg_document_length=self.average_document_length,
            embedding_directory=self.embedding_directory,
        )

        print("✅ LTR search service loaded")
        print(f"Documents: {self.number_of_documents:,}")
        print(
            f"Feature dimension: {len(self.feature_names)}"
        )

    def _compute_bm25_score(
        self,
        term_frequency: int,
        document_length: int,
        idf: float,
    ) -> float:
        numerator = term_frequency * (self.k1 + 1)
        denominator = term_frequency + self.k1 * (
            1
            - self.b
            + self.b
            * (document_length / self.average_document_length)
        )
        if denominator == 0:
            return 0.0
        return idf * (numerator / denominator)

    def _get_bm25_candidates(
        self,
        query_tokens: list[str],
        candidate_k: int = CANDIDATE_K,
    ) -> tuple[list[str], dict[str, float], dict[str, int]]:
        scores_by_doc_id: dict[str, float] = defaultdict(float)
        unique_terms = set(query_tokens)

        for term in unique_terms:
            posting_list = self.inverted_index.get(term, [])
            if not posting_list:
                continue
            idf = self.idf_by_term.get(term, 0.0)
            if idf <= 0:
                continue
            for doc_id, term_frequency in posting_list:
                doc_id = str(doc_id)
                document_length = self.document_lengths.get(
                    doc_id, 0
                )
                if document_length <= 0:
                    continue
                term_score = self._compute_bm25_score(
                    term_frequency=int(term_frequency),
                    document_length=int(document_length),
                    idf=float(idf),
                )
                scores_by_doc_id[doc_id] += term_score

        ranked_items = heapq.nlargest(
            candidate_k,
            scores_by_doc_id.items(),
            key=lambda item: item[1],
        )

        doc_ids = []
        scores = {}
        ranks = {}
        for idx, (doc_id, score) in enumerate(ranked_items):
            doc_ids.append(doc_id)
            scores[doc_id] = float(score)
            ranks[doc_id] = idx + 1

        return doc_ids, scores, ranks

    def search(
        self,
        query: str,
        top_k: int = 10,
        include_text: bool = True,
    ) -> dict:
        query_tokens = preprocess_lotte(query)

        if not query_tokens:
            return {
                "query": query,
                "processed_query": query_tokens,
                "result_ids": [],
                "results": [],
            }

        candidate_doc_ids, bm25_scores, bm25_ranks = (
            self._get_bm25_candidates(
                query_tokens, candidate_k=CANDIDATE_K
            )
        )

        if not candidate_doc_ids:
            return {
                "query": query,
                "processed_query": query_tokens,
                "result_ids": [],
                "results": [],
            }

        features = self.feature_extractor.extract_features(
            query_text=query,
            candidate_doc_ids=candidate_doc_ids,
            bm25_scores=bm25_scores,
            bm25_ranks=bm25_ranks,
        )

        ltr_scores = self.model.predict(features)

        scored_candidates = list(
            zip(candidate_doc_ids, ltr_scores)
        )

        top_indices = np.argsort(
            [score for _, score in scored_candidates]
        )[::-1][:top_k]

        ranked_results = []
        for idx in top_indices:
            doc_id = candidate_doc_ids[idx]
            ltr_score = float(ltr_scores[idx])
            if ltr_score < 0:
                ltr_score = 0.0
            ranked_results.append(
                {
                    "doc_id": doc_id,
                    "score": ltr_score,
                }
            )

        result_ids = [
            r["doc_id"] for r in ranked_results
        ]

        if include_text and result_ids:
            raw_documents = get_raw_documents_by_ids(
                result_ids
            )
            raw_text_by_id = {
                str(doc["doc_id"]): doc["text"]
                for doc in raw_documents
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
