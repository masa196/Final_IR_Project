from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from services.ltr.ltr_config import FEATURE_NAMES
from services.preprocessing.preprocessing_service import (
    preprocess_lotte,
)


class LTRFeatureExtractor:
    def __init__(
        self,
        inverted_index: dict,
        idf_by_term: dict[str, float],
        document_lengths: dict[str, int],
        avg_document_length: float,
        embedding_directory: Path,
    ):
        self.inverted_index = inverted_index
        self.idf_by_term = idf_by_term
        self.document_lengths = document_lengths
        self.avg_document_length = avg_document_length

        print("  Loading document embeddings...")
        self.document_embeddings = joblib.load(
            Path(embedding_directory) / "document_embeddings.joblib"
        )
        embedding_doc_ids = joblib.load(
            Path(embedding_directory) / "embedding_document_ids.joblib"
        )
        self.embedding_index_map = {
            str(doc_id): idx
            for idx, doc_id in enumerate(embedding_doc_ids)
        }

        print("  Loading SentenceTransformer model...")
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def extract_features(
        self,
        query_text: str,
        candidate_doc_ids: list[str],
        bm25_scores: dict[str, float],
        bm25_ranks: dict[str, int],
    ) -> np.ndarray:
        query_tokens = preprocess_lotte(query_text)
        if not query_tokens:
            return np.zeros(
                (len(candidate_doc_ids), len(FEATURE_NAMES))
            )

        query_length = len(query_tokens)
        unique_query_terms = set(query_tokens)

        embedding_similarities = self._compute_embedding_similarities(
            query_text, candidate_doc_ids
        )

        overlap_and_idf = self._compute_overlap_and_idf_stats(
            unique_query_terms, candidate_doc_ids
        )

        feature_matrix = []
        for rank_idx, doc_id in enumerate(candidate_doc_ids):
            dl = self.document_lengths.get(doc_id, 0)
            norm_dl = (
                dl / self.avg_document_length
                if self.avg_document_length > 0
                else 1.0
            )

            info = overlap_and_idf.get(
                doc_id,
                {
                    "overlap": 0,
                    "mean_idf": 0.0,
                    "max_idf": 0.0,
                    "min_idf": 0.0,
                    "sum_idf": 0.0,
                },
            )
            overlap_count = info["overlap"]
            overlap_ratio = (
                overlap_count / query_length
                if query_length > 0
                else 0.0
            )

            row = [
                float(bm25_scores.get(doc_id, 0.0)),
                float(embedding_similarities[rank_idx]),
                float(overlap_count),
                float(overlap_ratio),
                float(dl),
                float(norm_dl),
                float(query_length),
                float(info["mean_idf"]),
                float(info["max_idf"]),
                float(info["min_idf"]),
                float(info["sum_idf"]),
                float(bm25_ranks.get(doc_id, 999)),
            ]
            feature_matrix.append(row)

        return np.array(feature_matrix, dtype=np.float64)

    def _compute_embedding_similarities(
        self,
        query_text: str,
        doc_ids: list[str],
    ) -> np.ndarray:
        query_embedding = self.embedding_model.encode(
            [query_text],
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        candidate_indices = []
        valid_mask = []
        for doc_id in doc_ids:
            idx = self.embedding_index_map.get(doc_id)
            if idx is not None:
                candidate_indices.append(idx)
                valid_mask.append(True)
            else:
                candidate_indices.append(0)
                valid_mask.append(False)

        candidate_embeddings = self.document_embeddings[
            candidate_indices
        ]

        similarities = cosine_similarity(
            query_embedding, candidate_embeddings
        )[0]

        similarities[~np.array(valid_mask)] = 0.0
        return similarities

    def _compute_overlap_and_idf_stats(
        self,
        unique_query_terms: set[str],
        candidate_doc_ids: list[str],
    ) -> dict[str, dict]:
        candidate_set = set(candidate_doc_ids)
        result = {
            doc_id: {"overlap": 0, "idfs": []}
            for doc_id in candidate_doc_ids
        }

        for term in unique_query_terms:
            posting_list = self.inverted_index.get(term, [])
            if not posting_list:
                continue
            idf = self.idf_by_term.get(term, 0.0)
            for doc_id_str, _ in posting_list:
                doc_id_str = str(doc_id_str)
                if doc_id_str in candidate_set:
                    info = result[doc_id_str]
                    info["overlap"] += 1
                    if idf > 0:
                        info["idfs"].append(idf)

        output = {}
        for doc_id, info in result.items():
            idfs = info["idfs"]
            if idfs:
                output[doc_id] = {
                    "overlap": info["overlap"],
                    "mean_idf": float(np.mean(idfs)),
                    "max_idf": float(np.max(idfs)),
                    "min_idf": float(np.min(idfs)),
                    "sum_idf": float(np.sum(idfs)),
                }
            else:
                output[doc_id] = {
                    "overlap": info["overlap"],
                    "mean_idf": 0.0,
                    "max_idf": 0.0,
                    "min_idf": 0.0,
                    "sum_idf": 0.0,
                }

        return output

    def get_feature_dimension(self) -> int:
        return len(FEATURE_NAMES)

    def get_feature_names(self) -> list[str]:
        return list(FEATURE_NAMES)
