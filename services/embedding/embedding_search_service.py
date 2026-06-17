import json
import heapq
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from services.data_base.database_service import (
    get_raw_documents_by_ids,
)


class EmbeddingSearchService:


    def __init__(
        self,
        embedding_directory: Path,
    ):
        self.embedding_directory = Path(
            embedding_directory
        )

        print("Loading Embedding search files...")

        self.document_embeddings = joblib.load(
            self.embedding_directory
            / "document_embeddings.joblib"
        )

        self.document_ids = joblib.load(
            self.embedding_directory
            / "embedding_document_ids.joblib"
        )

        with (
            self.embedding_directory
            / "report.json"
        ).open("r", encoding="utf-8") as file:
            self.report = json.load(file)

        self.number_of_documents = int(
            self.report["number_of_documents"]
        )

        self.embedding_dimension = int(
            self.report["embedding_dimension"]
        )

        print("Loading SentenceTransformer model for query encoding...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("✅ Embedding search files loaded")
        print(
            f"Documents: "
            f"{self.number_of_documents:,}"
        )
        print(
            f"Embedding Dimension: "
            f"{self.embedding_dimension}"
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

       
        query_embedding = self.model.encode(
            [query], 
            show_progress_bar=False, 
            convert_to_numpy=True
        )


        similarity_scores = cosine_similarity(
            query_embedding, 
            self.document_embeddings
        )[0]


        unordered_top_indices = np.argpartition(similarity_scores, -top_k)[-top_k:]

       
        top_indices = unordered_top_indices[np.argsort(similarity_scores[unordered_top_indices])[::-1]]

       
        ranked_results = [
            {
                "doc_id": str(self.document_ids[idx]),
                "score": float(similarity_scores[idx]),
            }
            for idx in top_indices
            if similarity_scores[idx] > 0
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
            "result_ids": result_ids,
            "results": ranked_results,
        }