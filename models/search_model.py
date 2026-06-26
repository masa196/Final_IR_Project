from typing import List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="User search query",
    )

    model: str = Field(
        default="bm25",

        description="Ranking model: bm25 or tfidf or embedding or ltr",

        description="Ranking model: bm25, tfidf, embedding, hybrid_parallel, hybrid_serial, or ltr",

    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of returned results",
    )

    include_text: bool = Field(
        default=True,
        description="Whether to include raw document text in results",
    )

    use_refinement: bool = Field(
        default=True,
        description="Whether to apply query refinement (spell correction, synonyms, boosting)",
    )

    k1: Optional[float] = Field(
        default=1.5,
        description="BM25 k1 parameter",
    )

    b: Optional[float] = Field(
        default=0.75,
        description="BM25 b parameter",
    )

    hybrid_models: Optional[List[str]] = Field(
        default=None,
        description="Models to use in hybrid: ['bm25', 'embedding']",
    )

    hybrid_fusion_method: Optional[str] = Field(
        default="rrf",
        description="Fusion method: rrf",
    )

    hybrid_first_stage: Optional[str] = Field(
        default="bm25",
        description="First stage model for serial hybrid",
    )

    hybrid_second_stage: Optional[str] = Field(
        default="embedding",
        description="Second stage model for serial hybrid",
    )

    hybrid_first_stage_k: int = Field(
        default=200,
        ge=10,
        le=1000,
        description="Number of candidates retrieved in first stage (serial hybrid)",
    )


class SearchResult(BaseModel):
    rank: int
    doc_id: str
    score: float
    text: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    model: str
    top_k: int
    processed_query: List[str]
    results: List[SearchResult]
    refined_query: Optional[str] = None
    enhanced: bool = Field(default=False, description="Whether query enhancements were applied")

    k1: Optional[float] = None
    b: Optional[float] = None

    hybrid_type: Optional[str] = None
    fusion_method: Optional[str] = None
    models_used: Optional[List[str]] = None