from pydantic import BaseModel, Field


class RefineRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="User search query to refine",
    )
    use_history: bool = Field(
        default=True,
        description="Whether to apply history-based term boosting",
    )


class RefineResponse(BaseModel):
    original_query: str
    refined_query: str
    corrected: str
    expanded_tokens: list[str]
    boosted_tokens: list[str]
    enhanced: bool
