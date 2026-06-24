from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from services.preprocessing.preprocessing_service import (
    preprocess_lotte,
)
from services.data_base.database_service import (
    get_raw_documents_by_ids,
)
from services.hybrid.fusion_methods import FUSION_METHODS
from services.bm25.bm25_search_service import BM25SearchService
from services.embedding.embedding_search_service import EmbeddingSearchService


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BM25_DIRECTORY = PROJECT_ROOT / "indices" / "lotte" / "bm25"
INVERTED_INDEX_DIRECTORY = PROJECT_ROOT / "indices" / "lotte" / "inverted_index"
EMBEDDING_DIRECTORY = PROJECT_ROOT / "indices" / "lotte" / "embedding"


class HybridSearchService:

    def __init__(self):
        self.bm25_service = BM25SearchService(
            bm25_directory=BM25_DIRECTORY,
            inverted_index_directory=INVERTED_INDEX_DIRECTORY,
        )
        self.embedding_service = EmbeddingSearchService(
            embedding_directory=EMBEDDING_DIRECTORY,
        )

    # ==========================================================
    # دوال مساعدة للبحث الفردي
    # ==========================================================

    def _search_bm25(self, query: str, top_k: int, k1: float = 1.5, b: float = 0.75) -> list[dict]:
        self.bm25_service.k1 = k1
        self.bm25_service.b = b
        return self.bm25_service.search(query, top_k=top_k, include_text=False)["results"]

    def _search_embedding1(self, query: str, top_k: int) -> list[dict]:
        return self.embedding_service.search(query, top_k=top_k, include_text=False)["results"]

    # ==========================================================
    # دالة مساعدة لجلب النصوص
    # ==========================================================

    def _fetch_texts(self, results: list[dict]) -> list[dict]:
        doc_ids = [r["doc_id"] for r in results]
        if not doc_ids:
            return results
        raw_docs = get_raw_documents_by_ids(doc_ids)
        text_by_id = {str(d["doc_id"]): d["text"] for d in raw_docs}
        for r in results:
            r["text"] = text_by_id.get(
                str(r["doc_id"]),
                "Original text was not found in SQLite.",
            )
        return results

    # ==========================================================
    # البحث المتوازي (Parallel Hybrid)
    # ==========================================================

    def search_parallel(
        self,
        query: str,
        models: list[str],
        fusion_method: str = "rrf",
        top_k: int = 10,
        include_text: bool = True,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> dict:
        intermediate_top_k = top_k * 10

        results_by_model = {}
        for model_name in models:
            if model_name == "bm25":
                results_by_model["bm25"] = self._search_bm25(query, intermediate_top_k, k1, b)
            elif model_name == "embedding":
                results_by_model["embedding"] = self._search_embedding1(query, intermediate_top_k)

        fuse_fn = FUSION_METHODS.get(fusion_method, FUSION_METHODS["rrf"])
        fused_results = fuse_fn(results_by_model, top_k)

        if include_text:
            fused_results = self._fetch_texts(fused_results)

        return {
            "query": query,
            "processed_query": preprocess_lotte(query),
            "result_ids": [r["doc_id"] for r in fused_results],
            "results": fused_results,
            "hybrid_type": "parallel",
            "fusion_method": fusion_method,
            "models_used": models,
        }

    # ==========================================================
    # البحث التسلسلي (Serial Hybrid)
    # ==========================================================

    def search_serial(
        self,
        query: str,
        first_stage: str = "bm25",
        second_stage: str = "embedding",
        first_stage_k: int = 200,
        top_k: int = 10,
        include_text: bool = True,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> dict:
        candidate_results = self._run_first_stage(
            query, first_stage, first_stage_k, k1, b,
        )

        if not candidate_results:
            return {
                "query": query,
                "processed_query": preprocess_lotte(query),
                "result_ids": [],
                "results": [],
                "hybrid_type": "serial",
                "first_stage": first_stage,
                "second_stage": second_stage,
            }

        candidate_ids = [r["doc_id"] for r in candidate_results]
        reranked = self._rerank_with_model(
            query, candidate_ids, second_stage, k1, b,
        )

        reranked = reranked[:top_k]

        if include_text:
            reranked = self._fetch_texts(reranked)

        return {
            "query": query,
            "processed_query": preprocess_lotte(query),
            "result_ids": [r["doc_id"] for r in reranked],
            "results": reranked,
            "hybrid_type": "serial",
            "first_stage": first_stage,
            "second_stage": second_stage,
        }

    def _run_first_stage(
        self, query: str, model_name: str, top_k: int,
        k1: float, b: float,
    ) -> list[dict]:
        if model_name == "bm25":
            return self._search_bm25(query, top_k, k1, b)
        elif model_name == "embedding":
            return self._search_embedding1(query, top_k)
        return []

    def _rerank_with_model(
        self, query: str, candidate_ids: list[str],
        model_name: str, k1: float, b: float,
    ) -> list[dict]:
        if model_name in ("embedding",):
            return self._rerank_with_embedding(query, candidate_ids)
        elif model_name == "bm25":
            return self._rerank_with_bm25(query, candidate_ids, k1, b)
        return [{"doc_id": cid, "score": 0.0} for cid in candidate_ids]

    def _rerank_with_embedding(
        self, query: str, candidate_ids: list[str],
    ) -> list[dict]:
        matrix = self.embedding_service.document_embeddings
        doc_ids = self.embedding_service.document_ids
        model = self.embedding_service.model

        id_to_idx = {str(did): i for i, did in enumerate(doc_ids)}
        candidate_indices = []
        valid_ids = []
        for cid in candidate_ids:
            idx = id_to_idx.get(str(cid))
            if idx is not None:
                candidate_indices.append(idx)
                valid_ids.append(cid)

        if not candidate_indices:
            return []

        query_emb = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        candidate_embs = matrix[candidate_indices]
        scores = cosine_similarity(query_emb, candidate_embs)[0]
        top_order = np.argsort(scores)[::-1]

        return [
            {"doc_id": str(valid_ids[i]), "score": float(scores[i])}
            for i in top_order
        ]

    def _rerank_with_bm25(
        self, query: str, candidate_ids: list[str],
        k1: float, b: float,
    ) -> list[dict]:
        self.bm25_service.k1 = k1
        self.bm25_service.b = b

        query_tokens = preprocess_lotte(query)
        if not query_tokens:
            return [{"doc_id": cid, "score": 0.0} for cid in candidate_ids]

        candidate_set = set(candidate_ids)
        scores_by_doc = defaultdict(float)

        for term in set(query_tokens):
            posting_list = self.bm25_service.inverted_index.get(term, [])
            if not posting_list:
                continue
            idf = self.bm25_service.idf_by_term.get(term, 0.0)
            if idf <= 0:
                continue
            for doc_id, tf in posting_list:
                doc_id = str(doc_id)
                if doc_id not in candidate_set:
                    continue
                doc_len = self.bm25_service.document_lengths.get(doc_id, 0)
                if doc_len <= 0:
                    continue
                scores_by_doc[doc_id] += self.bm25_service._calculate_term_score(
                    term_frequency=int(tf),
                    document_length=int(doc_len),
                    idf=float(idf),
                )

        ranked = sorted(scores_by_doc.items(), key=lambda x: x[1], reverse=True)
        return [
            {"doc_id": doc_id, "score": float(score)}
            for doc_id, score in ranked if score > 0
        ]


