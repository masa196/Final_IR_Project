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