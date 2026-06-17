import sys
from pathlib import Path
from functools import lru_cache
from threading import Lock

from models.search_model import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from services.embedding.embedding_search_service import (
    EmbeddingSearchService,
)

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from services.tfidf.tfidf_search_service import TfidfSearchService
from services.bm25.bm25_search_service import BM25SearchService


TFIDF_DIRECTORY = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "tfidf"
)

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

EMBEDDING_DIRECTORY = (
    PROJECT_ROOT
    / "indices"
    / "lotte"
    / "embedding"
)


bm25_lock = Lock()


@lru_cache(maxsize=1)
def load_tfidf_service():
    return TfidfSearchService(
        tfidf_directory=TFIDF_DIRECTORY,
    )


@lru_cache(maxsize=1)
def load_bm25_service():
    return BM25SearchService(
        bm25_directory=BM25_DIRECTORY,
        inverted_index_directory=INVERTED_INDEX_DIRECTORY,
        k1=1.5,
        b=0.75,
    )

@lru_cache(maxsize=1)
def load_embedding_service():
    return EmbeddingSearchService(
        embedding_directory=EMBEDDING_DIRECTORY,
    )


def normalize_model_name(model_name: str) -> str:
    normalized = (
        model_name
        .strip()
        .lower()
        .replace("_", "-")
        .replace(" ", "")
    )

    if normalized in ["tfidf", "tf-idf"]:
        return "tfidf"

    if normalized == "bm25":
        return "bm25"
    
    if normalized == "embedding":
         return "embedding"

    raise ValueError(
        "Unsupported model. Use 'tfidf' or 'bm25' or 'embedding'."
    )


def search_documents(request: SearchRequest) -> SearchResponse:
    model_name = normalize_model_name(request.model)

    if model_name == "tfidf":
        search_service = load_tfidf_service()

        raw_response = search_service.search(
            query=request.query,
            top_k=request.top_k,
            include_text=request.include_text,
        )

        return build_search_response(
            request=request,
            model_name="tfidf",
            raw_response=raw_response,
        )

    if model_name == "bm25":
        search_service = load_bm25_service()

        k1 = float(request.k1)
        b = float(request.b)

        # BM25 service is cached, so we update parameters before searching.
        # The lock avoids parameter conflicts if multiple requests arrive together.
        with bm25_lock:
            search_service.k1 = k1
            search_service.b = b

            raw_response = search_service.search(
                query=request.query,
                top_k=request.top_k,
                include_text=request.include_text,
            )

        return build_search_response(
            request=request,
            model_name="bm25",
            raw_response=raw_response,
            k1=k1,
            b=b,
        )
    
    if model_name == "embedding":
       search_service = load_embedding_service()

    raw_response = search_service.search(
        query=request.query,
        top_k=request.top_k,
        include_text=request.include_text,
    )

    return build_search_response(
        request=request,
        model_name="embedding",
        raw_response=raw_response,
    )

    raise ValueError(
        "Unsupported model. Use 'tfidf' or 'bm25' or 'embedding'."
    )


def build_search_response(
    request: SearchRequest,
    model_name: str,
    raw_response: dict,
    k1: float | None = None,
    b: float | None = None,
) -> SearchResponse:
    results = []

    for rank, item in enumerate(
        raw_response.get("results", []),
        start=1,
    ):
        results.append(
            SearchResult(
                rank=rank,
                doc_id=str(item.get("doc_id")),
                score=float(item.get("score", 0.0)),
                text=item.get("text"),
            )
        )

    return SearchResponse(
        query=request.query,
        model=model_name,
        top_k=request.top_k,
        processed_query=raw_response.get("processed_query", []),
        results=results,
        k1=k1,
        b=b,
    )