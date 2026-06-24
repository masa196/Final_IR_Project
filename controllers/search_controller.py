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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from services.tfidf.tfidf_search_service import TfidfSearchService
from services.bm25.bm25_search_service import BM25SearchService


TFIDF_DIRECTORY = PROJECT_ROOT / "indices" / "lotte" / "tfidf"
BM25_DIRECTORY = PROJECT_ROOT / "indices" / "lotte" / "bm25"
INVERTED_INDEX_DIRECTORY = PROJECT_ROOT / "indices" / "lotte" / "inverted_index"
EMBEDDING_DIRECTORY = PROJECT_ROOT / "indices" / "lotte" / "embedding"


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


@lru_cache(maxsize=1)
def load_hybrid_service():
    from services.hybrid.hybrid_search_service import HybridSearchService
    return HybridSearchService()


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

    if normalized in ["hybrid-parallel", "hybridparallel"]:
        return "hybrid_parallel"

    if normalized in ["hybrid-serial", "hybridserial"]:
        return "hybrid_serial"

    raise ValueError(
        "Unsupported model. Use 'tfidf', 'bm25', 'embedding', "
        "'hybrid_parallel', or 'hybrid_serial'."
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

    if model_name == "hybrid_parallel":
        hybrid_service = load_hybrid_service()

        models = request.hybrid_models or ["bm25", "embedding"]
        fusion_method = request.hybrid_fusion_method or "rrf"

        raw_response = hybrid_service.search_parallel(
            query=request.query,
            models=models,
            fusion_method=fusion_method,
            top_k=request.top_k,
            include_text=request.include_text,
            k1=float(request.k1),
            b=float(request.b),
        )

        return build_search_response(
            request=request,
            model_name="hybrid_parallel",
            raw_response=raw_response,
            k1=request.k1,
            b=request.b,
        )

    if model_name == "hybrid_serial":
        hybrid_service = load_hybrid_service()

        first_stage = request.hybrid_first_stage or "bm25"
        second_stage = request.hybrid_second_stage or "embedding"
        first_stage_k = request.hybrid_first_stage_k or 200

        raw_response = hybrid_service.search_serial(
            query=request.query,
            first_stage=first_stage,
            second_stage=second_stage,
            first_stage_k=first_stage_k,
            top_k=request.top_k,
            include_text=request.include_text,
            k1=float(request.k1),
            b=float(request.b),
        )

        return build_search_response(
            request=request,
            model_name="hybrid_serial",
            raw_response=raw_response,
            k1=request.k1,
            b=request.b,
        )

    raise ValueError(
        "Unsupported model. Use 'tfidf', 'bm25', 'embedding', "
        "'hybrid_parallel', or 'hybrid_serial'."
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

    hybrid_type = raw_response.get("hybrid_type")
    fusion_method = raw_response.get("fusion_method")
    models_used = raw_response.get("models_used")

    return SearchResponse(
        query=request.query,
        model=model_name,
        top_k=request.top_k,
        processed_query=raw_response.get("processed_query", []),
        results=results,
        k1=k1,
        b=b,
        hybrid_type=hybrid_type,
        fusion_method=fusion_method,
        models_used=models_used,
    )
